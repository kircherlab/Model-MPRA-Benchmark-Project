#!/usr/bin/env python3
"""
Quick NGN2 analysis on native Linux filesystem for speed.
Reads from /tmp/ngn2_data/, writes to /tmp/ngn2_data/analysis/.
"""

import re, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

DATA = Path("/tmp/ngn2_data")
OUT = DATA / "analysis"
OUT.mkdir(exist_ok=True)

TRACK_MODELS = {
    "enformer": DATA / "enformer_scores.parquet",
    "basenji":  DATA / "basenji_scores.parquet",
    "alphagenome": DATA / "alphagenome_scores.parquet",
}
ND_MODELS = {
    "hyenadna": DATA / "hyenadna_scores.parquet",
    "dnabert2": DATA / "dnabert2_scores.parquet",
}


def _roc_auc(y_true, y_score):
    desc = np.argsort(y_score)[::-1]
    y_true = np.asarray(y_true)[desc]
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    return float(np.trapz(tps / n_pos, fps / n_neg))


def load_gt(vcf_path):
    records = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            chrom = p[0].replace("chr", "")
            vid = f"{chrom}:{p[1]}:{p[3]}>{p[4]}"
            info = p[7]
            l = re.search(r"LOG2FC=([-+]?\d*\.?\d+)", info)
            q = re.search(r"QVAL=([-+]?\d*\.?\d+)", info)
            records.append((vid, float(l.group(1)) if l else 0.0, float(q.group(1)) if q else 0.0))
    gt = pd.DataFrame(records, columns=["variant_id", "log2fc", "qval"]).set_index("variant_id")
    return gt


def aggregate_track(parquet_path, gt_set):
    """Stream-aggregate tidy parquet. Returns per-variant DataFrame."""
    pf = pq.ParquetFile(str(parquet_path))
    has_assay = "assay_type" in pf.schema.names
    cols = ["variant_id", "raw_score"] + (["assay_type"] if has_assay else [])

    # Detect chr prefix
    sample = next(pf.iter_batches(batch_size=5, columns=["variant_id"])).to_pandas()
    needs_strip = sample["variant_id"].iloc[0].startswith("chr")

    # Build filter set
    filter_set = gt_set | {f"chr{v}" for v in gt_set} if needs_strip else gt_set

    pf2 = pq.ParquetFile(str(parquet_path))
    aggs, assay_aggs = [], []
    t0 = time.time()

    for i, batch in enumerate(pf2.iter_batches(batch_size=2_000_000, columns=cols)):
        df = batch.to_pandas()
        df = df[df["variant_id"].isin(filter_set)]
        if df.empty:
            continue
        if needs_strip:
            df["variant_id"] = df["variant_id"].str[3:]  # strip "chr" fast

        df["abs_score"] = df["raw_score"].abs()
        g = df.groupby("variant_id", sort=False)
        agg = g["raw_score"].agg(["sum", "count"])
        agg["abs_sum"] = g["abs_score"].sum()
        idx_max = g["abs_score"].idxmax()
        agg["max_sad"] = df.loc[idx_max, "raw_score"].values
        aggs.append(agg)

        if has_assay:
            ag = df.groupby(["variant_id", "assay_type"], sort=False)["raw_score"].agg(["sum", "count"])
            assay_aggs.append(ag)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    batch {i+1}: {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"    Done: {i+1} batches in {elapsed:.0f}s")

    if not aggs:
        return pd.DataFrame(), None

    combined = pd.concat(aggs)
    summed = combined[["sum", "count", "abs_sum"]].groupby(level=0).sum()
    summary = pd.DataFrame({
        "mean_sad": summed["sum"] / summed["count"],
        "mean_abs_sad": summed["abs_sum"] / summed["count"],
    })
    max_df = combined[["max_sad"]].reset_index()
    max_df["abs_max"] = max_df["max_sad"].abs()
    best = max_df.loc[max_df.groupby("variant_id")["abs_max"].idxmax()]
    summary["max_sad"] = best.set_index("variant_id")["max_sad"].reindex(summary.index)

    assay_df = None
    if assay_aggs:
        ac = pd.concat(assay_aggs).groupby(level=[0, 1]).sum()
        ac["mean_sad"] = ac["sum"] / ac["count"]
        assay_df = ac["mean_sad"].unstack(level="assay_type")

    return summary, assay_df


def main():
    print("Loading ground truth ...")
    gt = load_gt(DATA / "80k_normalized.vcf")
    gt_set = set(gt.index)
    print(f"  {len(gt)} variants, log2FC range [{gt['log2fc'].min():.2f}, {gt['log2fc'].max():.2f}]")

    all_metrics = []
    model_scores = {}

    # Track-based models
    for model, path in TRACK_MODELS.items():
        if not path.exists():
            print(f"SKIP {model}")
            continue
        print(f"\nProcessing {model} ...")
        summary, assay_df = aggregate_track(path, gt_set)
        joined = gt.join(summary, how="inner")
        print(f"  Joined: {len(joined)} variants")

        if len(joined) == 0:
            continue

        for sc in ["mean_sad", "mean_abs_sad", "max_sad"]:
            sp_r, sp_p = spearmanr(joined["log2fc"], joined[sc])
            pe_r, _ = pearsonr(joined["log2fc"], joined[sc])
            nz = joined["log2fc"] != 0
            dir_acc = np.mean(np.sign(joined.loc[nz, sc]) == np.sign(joined.loc[nz, "log2fc"])) if nz.sum() else np.nan
            y_bin = ((joined["log2fc"].abs() > 0.5) & (joined["qval"] > 2)).astype(int)
            auroc = _roc_auc(y_bin.values, joined[sc].abs().values) if 5 <= y_bin.sum() < len(y_bin) else np.nan

            all_metrics.append({
                "model": model, "score_type": sc, "n": len(joined),
                "spearman_r": sp_r, "pearson_r": pe_r,
                "dir_accuracy": dir_acc, "auroc": auroc,
            })

        # Per-assay spearman
        if assay_df is not None:
            aj = gt.join(assay_df, how="inner")
            for assay in assay_df.columns:
                sp_r, _ = spearmanr(aj["log2fc"], aj[assay])
                all_metrics.append({"model": model, "score_type": f"assay:{assay}", "spearman_r": sp_r, "n": len(aj)})

        model_scores[model] = joined

    # ND models
    for model, path in ND_MODELS.items():
        if not path.exists():
            print(f"SKIP {model}")
            continue
        print(f"\nProcessing {model} ...")
        nd = pd.read_parquet(str(path))
        nd["variant_id"] = nd["variant_id"].str.replace("^chr", "", regex=True)
        nd = nd.set_index("variant_id")
        joined = gt.join(nd, how="inner")
        print(f"  Joined: {len(joined)} variants")

        if len(joined) == 0:
            continue

        for sc in ["ND_influence_score", "ND_mean", "ND_max", "ND_mean_ABS", "ND_max_ABS"]:
            if sc not in joined.columns:
                continue
            sp_r, _ = spearmanr(joined["log2fc"], joined[sc])
            pe_r, _ = pearsonr(joined["log2fc"], joined[sc])
            nz = joined["log2fc"] != 0
            dir_acc = np.mean(np.sign(joined.loc[nz, sc]) == np.sign(joined.loc[nz, "log2fc"])) if nz.sum() else np.nan
            y_bin = ((joined["log2fc"].abs() > 0.5) & (joined["qval"] > 2)).astype(int)
            auroc = _roc_auc(y_bin.values, joined[sc].abs().values) if 5 <= y_bin.sum() < len(y_bin) else np.nan

            all_metrics.append({
                "model": model, "score_type": sc, "n": len(joined),
                "spearman_r": sp_r, "pearson_r": pe_r,
                "dir_accuracy": dir_acc, "auroc": auroc,
            })
        model_scores[model] = joined

    # Save metrics
    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(OUT / "ngn2_metrics.csv", index=False)
    print(f"\nMetrics saved: {OUT / 'ngn2_metrics.csv'}")
    print(mdf[mdf["score_type"].isin(["mean_sad", "mean_abs_sad", "max_sad", "ND_influence_score"])].to_string())

    # ── PLOTS ────────────────────────────────────────────────────────────────

    # Plot 1: Spearman bar chart — best score per model
    best = (mdf.dropna(subset=["spearman_r"])
            .assign(abs_sp=lambda d: d["spearman_r"].abs())
            .sort_values("abs_sp", ascending=False)
            .drop_duplicates("model", keep="first")
            .sort_values("spearman_r", ascending=True))

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(best) * 0.7)))
    colors = ["#2196F3" if m in TRACK_MODELS else "#FF9800" for m in best["model"]]
    bars = ax.barh(best["model"] + "\n(" + best["score_type"] + ")", best["spearman_r"], color=colors)
    ax.set_xlabel("Spearman ρ (vs MPRA log2FC)")
    ax.set_title("NGN2 — Model Performance Comparison")
    ax.axvline(0, color="gray", lw=0.5)
    for bar, val in zip(bars, best["spearman_r"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#2196F3", label="Seq-to-Function"),
                       Patch(fc="#FF9800", label="Foundation Model")], loc="lower right")
    plt.tight_layout()
    fig.savefig(OUT / "ngn2_spearman.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {OUT / 'ngn2_spearman.png'}")

    # Plot 2: Multi-metric
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, len(best) * 0.6)))
    for i, (col, title) in enumerate([("spearman_r", "Spearman ρ"), ("auroc", "AUROC"), ("dir_accuracy", "Dir. Accuracy")]):
        ax = axes[i]
        vals = best.set_index("model")[col].dropna()
        c = ["#2196F3" if m in TRACK_MODELS else "#FF9800" for m in vals.index]
        ax.barh(vals.index, vals.values, color=c)
        ax.set_xlabel(title)
        ax.set_title(title)
        if col in ("auroc", "dir_accuracy"):
            ax.axvline(0.5, color="red", lw=0.8, ls="--", label="random")
            ax.legend(fontsize=8)
    plt.suptitle("NGN2 — Multi-Metric Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT / "ngn2_multi_metric.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {OUT / 'ngn2_multi_metric.png'}")

    # Plot 3: Scatter — each model vs MPRA
    score_map = {"enformer": "mean_sad", "basenji": "mean_sad", "alphagenome": "mean_sad",
                 "hyenadna": "ND_influence_score", "dnabert2": "ND_influence_score"}
    avail = [m for m in score_map if m in model_scores]
    n = len(avail)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
        if n == 1:
            axes = [axes]
        for i, model in enumerate(avail):
            ax = axes[i]
            df = model_scores[model]
            sc = score_map[model]
            if sc not in df.columns:
                sc = [c for c in df.columns if c not in ("log2fc", "qval")][0]
            x, y = df["log2fc"].values, df[sc].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            ax.scatter(x, y, alpha=0.05, s=2, rasterized=True)
            r, _ = spearmanr(x, y)
            ax.set_xlabel("MPRA log2FC")
            ax.set_ylabel(sc)
            ax.set_title(f"{model}\nρ = {r:.4f}")
            ax.axhline(0, color="gray", lw=0.3)
            ax.axvline(0, color="gray", lw=0.3)
        plt.suptitle("NGN2 — Predictions vs MPRA", fontsize=13)
        plt.tight_layout()
        fig.savefig(OUT / "ngn2_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot: {OUT / 'ngn2_scatter.png'}")

    # Plot 4: Per-assay heatmap
    assay_rows = mdf[mdf["score_type"].str.startswith("assay:")].copy()
    if not assay_rows.empty:
        assay_rows["assay"] = assay_rows["score_type"].str.replace("assay:", "")
        pivot = assay_rows.pivot(index="model", columns="assay", values="spearman_r")
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(2.5, len(pivot) * 0.8)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Spearman ρ")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > 0.08 else "black")
        ax.set_title("NGN2 — Spearman ρ by Assay Type")
        plt.tight_layout()
        fig.savefig(OUT / "ngn2_assay_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot: {OUT / 'ngn2_assay_heatmap.png'}")

    print(f"\nAll done! Results in {OUT}/")


if __name__ == "__main__":
    main()
