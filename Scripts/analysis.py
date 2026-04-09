#!/usr/bin/env python3
"""
Benchmark Analysis — Model predictions vs MPRA ground truth.

RAM-efficient: streams large track-based Parquets via PyArrow batches.
Produces comparison plots for all experiments and models.

Usage:
    python Scripts/analysis.py                    # all experiments
    python Scripts/analysis.py --experiment ngn2  # single experiment
"""

import argparse
import re
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
DATA_DIR = Path("Data")
OUTPUT_DIR = Path("results/analysis")

EXPERIMENTS = {
    "hek293t": {"vcf": "Data/VCF/IGVFFI4134MFLL.vcf"},
    "hepg2":   {"vcf": "Data/VCF/IGVFFI4378PZYI.vcf"},
    "ngn2":    {"vcf": "Data/VCF/80k_normalized.vcf"},
}

# Models that produce tidy (long) output: 1 row per variant × track
TRACK_MODELS = ["enformer", "basenji", "alphagenome"]
# Models that produce wide output: 1 row per variant
ND_MODELS = ["hyenadna", "dnabert2"]

BATCH_SIZE = 500_000  # rows per PyArrow batch for streaming


def normalize_variant_id(vid):
    """Normalize variant_id to consistent format without 'chr' prefix.

    Handles: chr1:123:A>G → 1:123:A>G, chr1:123:A:G → 1:123:A>G
    """
    # Strip chr prefix
    if vid.startswith("chr"):
        vid = vid[3:]
    # Normalize separator: some use : between ref/alt, others use >
    # Format: chrom:pos:ref>alt or chrom:pos:ref:alt
    parts = vid.split(":")
    if len(parts) == 4:
        # chrom:pos:ref:alt → chrom:pos:ref>alt
        return f"{parts[0]}:{parts[1]}:{parts[2]}>{parts[3]}"
    return vid


# ── sklearn-free AUROC / AUPRC ────────────────────────────────────────────


def _roc_auc_score(y_true, y_score):
    """Compute AUROC without sklearn."""
    desc = np.argsort(y_score)[::-1]
    y_true = np.asarray(y_true)[desc]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Prepend origin
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    return float(np.trapz(tpr, fpr))


def _average_precision_score(y_true, y_score):
    """Compute average precision without sklearn."""
    desc = np.argsort(y_score)[::-1]
    y_true = np.asarray(y_true)[desc]
    n_pos = y_true.sum()
    if n_pos == 0:
        return np.nan
    tps = np.cumsum(y_true)
    precision = tps / np.arange(1, len(y_true) + 1)
    recall = tps / n_pos
    # Compute AP as sum of precision * delta-recall
    recall = np.r_[0, recall]
    precision = np.r_[0, precision]
    delta_recall = np.diff(recall)
    return float(np.sum(precision[1:] * delta_recall))


# ── Ground Truth Loading ─────────────────────────────────────────────────────


def load_ground_truth(vcf_path):
    """Parse VCF to extract variant_id → LOG2FC mapping.

    Returns DataFrame with columns: variant_id, log2fc, pval, qval
    """
    records = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom = parts[0].replace("chr", "")
            pos = parts[1]
            ref = parts[3]
            alt = parts[4]
            info = parts[7]

            variant_id = f"{chrom}:{pos}:{ref}>{alt}"

            log2fc = 0.0
            pval = 0.0
            qval = 0.0
            m = re.search(r"LOG2FC=([-+]?\d*\.?\d+)", info)
            if m:
                log2fc = float(m.group(1))
            m = re.search(r"PVAL=([-+]?\d*\.?\d+)", info)
            if m:
                pval = float(m.group(1))
            m = re.search(r"QVAL=([-+]?\d*\.?\d+)", info)
            if m:
                qval = float(m.group(1))

            records.append((variant_id, log2fc, pval, qval))

    gt = pd.DataFrame(records, columns=["variant_id", "log2fc", "pval", "qval"])
    gt = gt.set_index("variant_id")
    return gt


# ── Score Loading ────────────────────────────────────────────────────────────


def load_nd_scores(parquet_path):
    """Load foundation model scores (small, fits in RAM).

    Returns DataFrame indexed by variant_id with ND score columns.
    """
    df = pd.read_parquet(parquet_path)
    df.index = df["variant_id"].map(normalize_variant_id)
    df.index.name = "variant_id"
    df = df.drop(columns=["variant_id"], errors="ignore")
    return df


def aggregate_track_scores_streaming(parquet_path, gt_index=None):
    """Stream tidy Parquet and aggregate per variant.

    Computes per-variant:
        - mean_sad:  mean(raw_score) across all tracks
        - mean_abs_sad: mean(|raw_score|) across all tracks
        - max_sad:   max(raw_score) by absolute value (signed)
        - per assay_type: mean(raw_score) grouped by assay_type

    Only keeps variants present in gt_index (if provided) to save memory.

    Returns: (summary_df, assay_df)
        summary_df: variant_id × {mean_sad, mean_abs_sad, max_sad}
        assay_df:   variant_id × assay_type → mean_sad (pivoted)
    """
    pf = pq.ParquetFile(parquet_path)
    has_assay = "assay_type" in pf.schema.names
    columns = ["variant_id", "raw_score"]
    if has_assay:
        columns.append("assay_type")

    # Pre-build filter sets in BOTH chr-prefix formats to avoid per-row normalization
    gt_set_no_chr = set(gt_index) if gt_index is not None else None
    gt_set_chr = {f"chr{v}" for v in gt_index} if gt_index is not None else None
    gt_set_all = (gt_set_no_chr | gt_set_chr) if gt_index is not None else None

    # Detect format from first batch to build a fast lookup
    first_batch = next(pf.iter_batches(batch_size=10, columns=["variant_id"]))
    sample_vid = first_batch.to_pandas()["variant_id"].iloc[0]
    needs_chr_strip = sample_vid.startswith("chr")
    del first_batch

    n_batches = 0
    batch_aggs = []
    batch_assay_aggs = []
    pf2 = pq.ParquetFile(parquet_path)  # re-open to restart iteration

    for batch in pf2.iter_batches(batch_size=2_000_000, columns=columns):
        df = batch.to_pandas()
        n_batches += 1

        # Fast filter using pre-built set (no per-row normalization)
        if gt_set_all is not None:
            df = df[df["variant_id"].isin(gt_set_all)]
        if df.empty:
            continue

        # Normalize variant_ids ONLY for the filtered subset (much smaller)
        if needs_chr_strip:
            df["variant_id"] = df["variant_id"].str.replace("^chr", "", regex=True)

        # Vectorized per-variant aggregation
        df["abs_score"] = df["raw_score"].abs()
        g = df.groupby("variant_id", sort=False)
        agg = g["raw_score"].agg(["sum", "count"])
        agg["abs_sum"] = g["abs_score"].sum()
        idx_max = g["abs_score"].idxmax()
        agg["max_sad"] = df.loc[idx_max, "raw_score"].values
        batch_aggs.append(agg)

        if has_assay:
            ag = df.groupby(["variant_id", "assay_type"], sort=False)["raw_score"].agg(["sum", "count"])
            batch_assay_aggs.append(ag)

        if n_batches % 50 == 0:
            print(f"    ... processed {n_batches} batches ({n_batches * 2_000_000 / 1e6:.0f}M rows)")

    print(f"    Done: {n_batches} batches, combining ...")

    # Combine all batch aggregates
    if not batch_aggs:
        return pd.DataFrame(), None

    combined = pd.concat(batch_aggs)
    # sum/count/abs_sum are additive across batches; max_sad needs special handling
    summed = combined[["sum", "count", "abs_sum"]].groupby(level=0).sum()
    summary = pd.DataFrame({
        "mean_sad": summed["sum"] / summed["count"],
        "mean_abs_sad": summed["abs_sum"] / summed["count"],
    })

    # For max_sad: pick the value with the largest absolute across all batches
    max_df = combined[["max_sad"]].copy()
    max_df["abs_max"] = max_df["max_sad"].abs()
    # Reset index to avoid duplicate-label issues, then groupby variant_id
    max_df = max_df.reset_index()
    best_rows = max_df.loc[max_df.groupby("variant_id")["abs_max"].idxmax()]
    max_series = best_rows.set_index("variant_id")["max_sad"]
    summary["max_sad"] = max_series.reindex(summary.index)

    summary.index.name = "variant_id"
    print(f"    {len(summary)} variants aggregated")

    # Build per-assay pivot
    assay_df = None
    if batch_assay_aggs:
        ac = pd.concat(batch_assay_aggs).groupby(level=[0, 1]).sum()
        ac["mean_sad"] = ac["sum"] / ac["count"]
        assay_df = ac["mean_sad"].unstack(level="assay_type")

    return summary, assay_df


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred, label=""):
    """Compute correlation and classification metrics.

    Returns dict with spearman_r, spearman_p, pearson_r, auroc, auprc, dir_acc.
    Classification: |log2fc| > 0.5 and qval > 2 (i.e. -log10(q) > 2 → q < 0.01)
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 10:
        return None

    sp_r, sp_p = spearmanr(y_true, y_pred)
    pe_r, pe_p = pearsonr(y_true, y_pred)

    # Directional accuracy: fraction where sign(pred) == sign(true)
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        dir_acc = np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))
    else:
        dir_acc = np.nan

    return {
        "label": label,
        "n_variants": len(y_true),
        "spearman_r": sp_r,
        "spearman_p": sp_p,
        "pearson_r": pe_r,
        "dir_accuracy": dir_acc,
    }


def compute_classification_metrics(log2fc, qval, y_pred):
    """AUROC/AUPRC for binary classification of functional variants.

    Positive class: |log2fc| > 0.5 AND qval > 2 (-log10 scale, i.e. q < 0.01).
    """
    mask = np.isfinite(log2fc) & np.isfinite(y_pred) & np.isfinite(qval)
    log2fc = log2fc[mask]
    qval = qval[mask]
    y_pred = y_pred[mask]

    y_binary = ((np.abs(log2fc) > 0.5) & (qval > 2)).astype(int)

    if y_binary.sum() < 5 or y_binary.sum() == len(y_binary):
        return {"auroc": np.nan, "auprc": np.nan, "n_pos": int(y_binary.sum()), "n_total": len(y_binary)}

    pred_abs = np.abs(y_pred)
    auroc = _roc_auc_score(y_binary, pred_abs)
    auprc = _average_precision_score(y_binary, pred_abs)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "n_pos": int(y_binary.sum()),
        "n_total": len(y_binary),
    }


# ── Analysis per experiment ──────────────────────────────────────────────────


def analyze_experiment(experiment, out_dir):
    """Run full analysis for one experiment."""

    print(f"\n{'='*70}")
    print(f"  Experiment: {experiment}")
    print(f"{'='*70}")

    exp_dir = RESULTS_DIR / experiment
    vcf_path = EXPERIMENTS[experiment]["vcf"]

    if not os.path.exists(vcf_path):
        print(f"  SKIP: VCF not found: {vcf_path}")
        return None

    # 1. Load ground truth
    print(f"  Loading ground truth from {vcf_path} ...")
    gt = load_ground_truth(vcf_path)
    print(f"  Ground truth: {len(gt)} variants, LOG2FC range [{gt['log2fc'].min():.3f}, {gt['log2fc'].max():.3f}]")

    all_metrics = []
    model_scores = {}  # model → DataFrame of per-variant scores for plots

    # 2. Load track-based model scores
    for model in TRACK_MODELS:
        parquet_path = exp_dir / model / f"{model}_scores.parquet"
        if not parquet_path.exists():
            print(f"  SKIP {model}: {parquet_path} not found")
            continue

        print(f"\n  Processing {model} (streaming) ...")
        summary, assay_df = aggregate_track_scores_streaming(str(parquet_path), gt_index=gt.index)

        # Join with ground truth
        joined = gt.join(summary, how="inner")
        print(f"    Joined: {len(joined)} variants")

        if len(joined) == 0:
            print(f"    WARNING: No variants matched for {model}. Skipping.")
            continue

        # Compute metrics for different aggregation strategies
        for score_col in ["mean_sad", "mean_abs_sad", "max_sad"]:
            m = compute_metrics(joined["log2fc"].values, joined[score_col].values,
                                label=f"{model}_{score_col}")
            if m:
                clf = compute_classification_metrics(
                    joined["log2fc"].values, joined["qval"].values, joined[score_col].values)
                m.update(clf)
                m["model"] = model
                m["score_type"] = score_col
                all_metrics.append(m)

        # Per-assay metrics
        if assay_df is not None:
            assay_joined = gt.join(assay_df, how="inner")
            for assay in assay_df.columns:
                m = compute_metrics(assay_joined["log2fc"].values, assay_joined[assay].values,
                                    label=f"{model}_{assay}")
                if m:
                    m["model"] = model
                    m["score_type"] = f"assay:{assay}"
                    all_metrics.append(m)

        model_scores[model] = joined[["log2fc", "qval", "mean_sad", "mean_abs_sad", "max_sad"]]

    # 3. Load ND model scores
    for model in ND_MODELS:
        parquet_path = exp_dir / model / f"{model}_scores.parquet"
        if not parquet_path.exists():
            print(f"  SKIP {model}: {parquet_path} not found")
            continue

        print(f"\n  Processing {model} ...")
        nd = load_nd_scores(str(parquet_path))

        joined = gt.join(nd, how="inner")
        print(f"    Joined: {len(joined)} variants")

        if len(joined) == 0:
            print(f"    WARNING: No variants matched for {model}. Skipping.")
            continue

        nd_score_cols = ["ND_influence_score", "ND_mean", "ND_max", "ND_mean_ABS", "ND_max_ABS"]
        for score_col in nd_score_cols:
            if score_col not in joined.columns:
                continue
            m = compute_metrics(joined["log2fc"].values, joined[score_col].values,
                                label=f"{model}_{score_col}")
            if m:
                clf = compute_classification_metrics(
                    joined["log2fc"].values, joined["qval"].values, joined[score_col].values)
                m.update(clf)
                m["model"] = model
                m["score_type"] = score_col
                all_metrics.append(m)

        model_scores[model] = joined[["log2fc", "qval"] + [c for c in nd_score_cols if c in joined.columns]]

    # 4. Compile results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / f"{experiment}_metrics.csv", index=False)
    print(f"\n  Metrics saved: {out_dir / f'{experiment}_metrics.csv'}")

    # 5. Plots
    plot_model_comparison(metrics_df, experiment, out_dir, model_scores)

    return metrics_df


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_model_comparison(metrics_df, experiment, out_dir, model_scores):
    """Generate comparison plots for one experiment."""

    # --- Plot 1: Spearman bar chart (best score type per model) ---
    # Pick the best-performing score_type per model (by |spearman_r|)
    best = (metrics_df
            .assign(abs_sp=lambda d: d["spearman_r"].abs())
            .sort_values("abs_sp", ascending=False)
            .drop_duplicates(subset="model", keep="first")
            .sort_values("spearman_r", ascending=True))

    fig, ax = plt.subplots(figsize=(8, max(4, len(best) * 0.8)))
    colors = []
    for _, row in best.iterrows():
        if row["model"] in TRACK_MODELS:
            colors.append("#2196F3")  # blue for seq2func
        else:
            colors.append("#FF9800")  # orange for foundation
    ax.barh(best["model"] + "\n(" + best["score_type"] + ")", best["spearman_r"], color=colors)
    ax.set_xlabel("Spearman ρ (vs MPRA log2FC)")
    ax.set_title(f"{experiment.upper()} — Model Performance Comparison")
    ax.axvline(0, color="gray", linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2196F3", label="Sequence-to-Function"),
                       Patch(facecolor="#FF9800", label="Foundation Model")]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / f"{experiment}_spearman_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {experiment}_spearman_comparison.png")

    # --- Plot 2: Multi-metric comparison (Spearman, AUROC, Dir. Accuracy) ---
    if "auroc" in best.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(best) * 0.7)))

        for i, (metric, title) in enumerate([
            ("spearman_r", "Spearman ρ"),
            ("auroc", "AUROC"),
            ("dir_accuracy", "Directional Accuracy"),
        ]):
            ax = axes[i]
            vals = best.set_index("model")[metric].dropna()
            if vals.empty:
                continue
            c = ["#2196F3" if m in TRACK_MODELS else "#FF9800" for m in vals.index]
            ax.barh(vals.index, vals.values, color=c)
            ax.set_xlabel(title)
            ax.set_title(title)
            if metric == "dir_accuracy":
                ax.axvline(0.5, color="red", linewidth=0.8, linestyle="--", label="random")
                ax.legend()
            elif metric == "auroc":
                ax.axvline(0.5, color="red", linewidth=0.8, linestyle="--", label="random")
                ax.legend()

        plt.suptitle(f"{experiment.upper()} — Multi-Metric Comparison", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{experiment}_multi_metric.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {experiment}_multi_metric.png")

    # --- Plot 3: Per-assay-type Spearman heatmap (track models only) ---
    assay_rows = metrics_df[metrics_df["score_type"].str.startswith("assay:")]
    if not assay_rows.empty:
        assay_rows = assay_rows.copy()
        assay_rows["assay"] = assay_rows["score_type"].str.replace("assay:", "")
        pivot = assay_rows.pivot(index="model", columns="assay", values="spearman_r")

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(3, len(pivot) * 0.8)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Spearman ρ")
        ax.set_title(f"{experiment.upper()} — Spearman by Assay Type")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > 0.15 else "black")

        plt.tight_layout()
        fig.savefig(out_dir / f"{experiment}_assay_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {experiment}_assay_heatmap.png")

    # --- Plot 4: Scatter plots (best score per model vs MPRA) ---
    best_score_map = {
        "enformer": "mean_sad",
        "basenji": "mean_sad",
        "alphagenome": "mean_sad",
        "hyenadna": "ND_influence_score",
        "dnabert2": "ND_influence_score",
    }
    available_models = [m for m in model_scores.keys() if m in best_score_map]
    if available_models:
        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        for i, model in enumerate(available_models):
            ax = axes[i]
            score_col = best_score_map[model]
            df = model_scores[model]
            if score_col not in df.columns:
                # Fallback: first numeric column after log2fc/qval
                score_col = [c for c in df.columns if c not in ("log2fc", "qval")][0]

            x = df["log2fc"].values
            y = df[score_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            ax.scatter(x, y, alpha=0.05, s=2, rasterized=True)
            sp_r, _ = spearmanr(x, y)
            ax.set_xlabel("MPRA log2FC")
            ax.set_ylabel(f"{model} {score_col}")
            ax.set_title(f"{model}\nρ = {sp_r:.4f}")
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

        plt.suptitle(f"{experiment.upper()} — Model Predictions vs MPRA", fontsize=13)
        plt.tight_layout()
        fig.savefig(out_dir / f"{experiment}_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {experiment}_scatter.png")


# ── Cross-experiment comparison ──────────────────────────────────────────────


def plot_cross_experiment(all_results, out_dir):
    """Compare models across experiments."""
    if not all_results:
        return

    combined = pd.concat(all_results.values(), keys=all_results.keys(), names=["experiment"])
    combined = combined.reset_index(level="experiment")

    # Best score_type per model per experiment
    best = (combined
            .assign(abs_sp=lambda d: d["spearman_r"].abs())
            .sort_values("abs_sp", ascending=False)
            .drop_duplicates(subset=["experiment", "model"], keep="first"))

    pivot = best.pivot(index="model", columns="experiment", values="spearman_r")

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), max(4, len(pivot) * 0.8)))
    x = np.arange(len(pivot.index))
    width = 0.25
    for i, exp in enumerate(pivot.columns):
        vals = pivot[exp].values
        c = ["#2196F3", "#4CAF50", "#FF9800"][i % 3]
        ax.bar(x + i * width, vals, width, label=exp, color=c, alpha=0.85)

    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Cross-Experiment Model Comparison (best score type)")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "cross_experiment_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Cross-experiment plot saved: cross_experiment_spearman.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark analysis")
    parser.add_argument("--experiment", "-e", nargs="*",
                        help="Experiment(s) to analyze (default: all)")
    args = parser.parse_args()

    experiments = args.experiment if args.experiment else list(EXPERIMENTS.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exp in experiments:
        if exp not in EXPERIMENTS:
            print(f"Unknown experiment: {exp}")
            continue
        metrics_df = analyze_experiment(exp, OUTPUT_DIR)
        if metrics_df is not None:
            all_results[exp] = metrics_df

    if len(all_results) > 1:
        plot_cross_experiment(all_results, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(f"  All done. Results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
