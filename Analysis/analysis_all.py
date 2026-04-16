#!/usr/bin/env python3
"""
Comprehensive MPRA benchmark analysis across all cell types.

For each cell type:
  - Streams large Parquet files from /tmp (native ext4, fast)
  - Computes Spearman (all variants), Spearman (significant only),
    AUROC/AUPRC (classify significant vs non-significant),
    directional accuracy (on significant variants)
  - Plots: Spearman bars, AUROC bars, scatter (balanced), assay heatmap,
           effect-size threshold curve

"Significant" definition:
  HEK293T / HepG2 : QVAL > 1.3  (i.e. adjusted-p < 0.05, -log10 scale)
  NGN2            : |log2FC| > 0.5  (no QVAL in VCF)

Usage:
  python3 Scripts/analysis_all.py [--cell-types ngn2 hek293t hepg2]
"""

import re, os, sys, time, shutil, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/mnt/c/Users/user/Desktop/Benchmark_results/Pipeline_v2")
VCF_DIR  = BASE / "Data/VCF"
TMP_ROOT = Path("/tmp/mpra_analysis")

EXPERIMENTS = {
    "ngn2": {
        "vcf":         VCF_DIR / "80k_normalized.vcf",
        "results_dir": BASE / "results/ngn2",
        "has_qval":    False,
        "sig_lfc_thr": 0.5,   # |log2FC| > 0.5 => "significant"
        "label":       "NGN2",
    },
    "hek293t": {
        "vcf":         VCF_DIR / "IGVFFI4134MFLL.vcf",
        "results_dir": BASE / "results/hek293t",
        "has_qval":    True,
        "qval_thr":    1.3,   # -log10(0.05)
        "label":       "HEK293T",
    },
    "hepg2": {
        "vcf":         VCF_DIR / "IGVFFI4378PZYI.vcf",
        "results_dir": BASE / "results/hepg2",
        "has_qval":    True,
        "qval_thr":    1.3,
        "label":       "HepG2",
    },
}

TRACK_MODELS = ["enformer", "basenji", "alphagenome"]
ND_MODELS    = ["hyenadna", "dnabert2"]

MODEL_COLORS = {
    "enformer":    "#1565C0",
    "basenji":     "#0288D1",
    "alphagenome": "#00796B",
    "hyenadna":    "#E65100",
    "dnabert2":    "#F57C00",
}
CAT_COLORS = {"Seq-to-Function": "#2196F3", "Foundation Model": "#FF9800"}

# ── Math helpers ──────────────────────────────────────────────────────────────

def _roc_auc(y_true, y_score):
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n_pos = int(y_true.sum());  n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    tps = np.cumsum(y_true);    fps = np.cumsum(1 - y_true)
    return float(np.trapezoid(tps / n_pos, fps / n_neg))


def _avg_precision(y_true, y_score):
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.sum() == 0:
        return np.nan
    order  = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    tp   = np.cumsum(y_true)
    prec = tp / np.arange(1, len(y_true) + 1)
    return float(np.sum(prec * y_true) / y_true.sum())  # noqa: AUPRC


def balanced_sample(df, log2fc_col="log2fc", n_per_bin=300, n_bins=30):
    """
    Scatter-plot sampling: equal density across |log2FC| bins.
    Keeps all extreme outliers, subsamples near-zero variants so the
    distribution appears ~flat rather than spike-at-zero.
    """
    vals  = df[log2fc_col].abs()
    upper = max(vals.quantile(0.995), 0.01)
    edges = np.linspace(0, upper, n_bins + 1)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask   = (vals >= lo) & (vals < hi)
        subset = df[mask]
        if len(subset) == 0:
            continue
        parts.append(subset.sample(min(len(subset), n_per_bin), random_state=42))
    parts.append(df[vals >= upper])
    return pd.concat(parts).drop_duplicates()

# ── Ground truth loader ───────────────────────────────────────────────────────

def load_gt(cfg):
    """Parse VCF → DataFrame[variant_id, log2fc, qval, sig]."""
    records = []
    with open(cfg["vcf"]) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p    = line.rstrip("\n").split("\t")
            chrom = p[0].replace("chr", "")
            vid  = f"{chrom}:{p[1]}:{p[3]}>{p[4]}"
            info = p[7]
            l = re.search(r"LOG2FC=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", info)
            q = re.search(r"QVAL=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",   info)
            lfc  = float(l.group(1)) if l else 0.0
            qval = float(q.group(1)) if q else 0.0
            records.append((vid, lfc, qval))

    gt = pd.DataFrame(records, columns=["variant_id", "log2fc", "qval"])
    gt = gt.set_index("variant_id")

    if cfg["has_qval"]:
        gt["sig"] = (gt["qval"] > cfg["qval_thr"]).astype(int)
    else:
        gt["sig"] = (gt["log2fc"].abs() > cfg["sig_lfc_thr"]).astype(int)

    return gt

# ── Parquet aggregation ───────────────────────────────────────────────────────

def aggregate_track(parquet_path, gt_set, batch_size=2_000_000):
    """Stream-aggregate tidy track-score parquet. Returns (summary_df, assay_df)."""
    pf = pq.ParquetFile(str(parquet_path))
    has_assay = "assay_type" in pf.schema.names
    cols = ["variant_id", "raw_score"] + (["assay_type"] if has_assay else [])

    # Detect chr prefix from first few rows
    sample     = next(pf.iter_batches(batch_size=5, columns=["variant_id"])).to_pandas()
    needs_strip = sample["variant_id"].iloc[0].startswith("chr")
    filter_set  = gt_set | {f"chr{v}" for v in gt_set}

    aggs, assay_aggs = [], []
    t0 = time.time()

    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols)):
        df = batch.to_pandas()
        df = df[df["variant_id"].isin(filter_set)]
        if df.empty:
            continue
        if needs_strip:
            df["variant_id"] = df["variant_id"].str[3:]

        df["abs_score"] = df["raw_score"].abs()
        g   = df.groupby("variant_id", sort=False)
        agg = g["raw_score"].agg(["sum", "count"])
        agg["abs_sum"]  = g["abs_score"].sum()
        idx_max = g["abs_score"].idxmax()
        agg["max_sad"] = df.loc[idx_max, "raw_score"].values
        aggs.append(agg)

        if has_assay:
            ag = df.groupby(["variant_id", "assay_type"], sort=False)["raw_score"].agg(["sum", "count"])
            assay_aggs.append(ag)

        if (i + 1) % 20 == 0:
            print(f"    batch {i+1}: {time.time()-t0:.0f}s elapsed")

    total = time.time() - t0
    print(f"    Done: {i+1} batches in {total:.0f}s")

    if not aggs:
        return pd.DataFrame(), None

    combined = pd.concat(aggs)
    summed   = combined[["sum", "count", "abs_sum"]].groupby(level=0).sum()
    summary  = pd.DataFrame({
        "mean_sad":     summed["sum"]     / summed["count"],
        "mean_abs_sad": summed["abs_sum"] / summed["count"],
    })
    max_df = combined[["max_sad"]].reset_index()
    max_df["abs_max"] = max_df["max_sad"].abs()
    best   = max_df.loc[max_df.groupby("variant_id")["abs_max"].idxmax()]
    best   = best.drop_duplicates(subset="variant_id", keep="first")
    summary["max_sad"] = best.set_index("variant_id")["max_sad"].reindex(summary.index)

    assay_df = None
    if assay_aggs:
        ac = pd.concat(assay_aggs).groupby(level=[0, 1]).sum()
        ac["mean_sad"] = ac["sum"] / ac["count"]
        assay_df = ac["mean_sad"].unstack(level="assay_type")

    return summary, assay_df


def load_nd(parquet_path, gt_set):
    """Load ND-model parquet (small enough for full in-memory read)."""
    nd = pd.read_parquet(str(parquet_path))
    nd["variant_id"] = nd["variant_id"].str.replace("^chr", "", regex=True)
    nd = nd.set_index("variant_id")
    nd = nd[nd.index.isin(gt_set)]
    # deduplicate if any
    nd = nd[~nd.index.duplicated(keep="first")]
    return nd

# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(joined, score_col, model, score_type):
    """Compute full suite of metrics for one (model, score_type) pair."""
    x = joined["log2fc"].values
    y = joined[score_col].values
    s = joined["sig"].values

    mask = np.isfinite(y)
    x, y, s = x[mask], y[mask], s[mask]
    n = len(x)

    sp_all,  _ = spearmanr(x, y)
    pe_all,  _ = pearsonr(x, y)

    # Significant-only metrics
    sig_mask = s.astype(bool)
    n_sig    = sig_mask.sum()
    sp_sig   = pearsonr_sig = dir_acc = np.nan
    if n_sig >= 10:
        sp_sig, _ = spearmanr(x[sig_mask], y[sig_mask])
        dir_acc   = float(np.mean(np.sign(y[sig_mask]) == np.sign(x[sig_mask])))

    # Classification metrics
    auroc = auprc = np.nan
    if 5 <= s.sum() < n:
        auroc = _roc_auc(s, np.abs(y))
        auprc = _avg_precision(s, np.abs(y))

    return {
        "model":       model,
        "score_type":  score_type,
        "n":           n,
        "n_sig":       int(n_sig),
        "spearman_r":  float(sp_all),
        "pearson_r":   float(pe_all),
        "spearman_sig": float(sp_sig),
        "dir_acc_sig": float(dir_acc),
        "auroc":       float(auroc),
        "auprc":       float(auprc),
    }


def effect_size_curve(joined, score_col, n_points=30):
    """
    Returns (thresholds, spearman_values) for |log2FC| cutoff sweep.
    At each cutoff, only keep variants with |log2FC| >= cutoff.
    """
    abs_lfc = joined["log2fc"].abs()
    max_thr = abs_lfc.quantile(0.99)
    thresholds = np.linspace(0, max_thr, n_points)
    rhos = []
    for thr in thresholds:
        sub = joined[abs_lfc >= thr]
        if len(sub) < 20:
            rhos.append(np.nan)
            continue
        r, _ = spearmanr(sub["log2fc"], sub[score_col])
        rhos.append(r)
    return thresholds, np.array(rhos)

# ── Plotting ──────────────────────────────────────────────────────────────────

def _bar_h(ax, models, values, title, xlabel, vline=None, fmt=".3f"):
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    bars   = ax.barh(models, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, color="gray", lw=0.5)
    if vline is not None:
        ax.axvline(vline, color="red", lw=0.8, ls="--", alpha=0.6)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(bar.get_width() + 0.001 * np.sign(bar.get_width() + 1e-9),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:{fmt}}", va="center", fontsize=8)


def plot_overview(mdf, label, out_dir, track_models):
    """Summary bar chart: Spearman (all + sig), AUROC side by side."""
    # Best row per model (highest |spearman_r| for full set)
    best = (mdf.dropna(subset=["spearman_r"])
            .assign(_abs=lambda d: d["spearman_r"].abs())
            .sort_values("_abs", ascending=False)
            .drop_duplicates("model", keep="first")
            .sort_values("spearman_r"))

    models = best["model"].tolist()
    sp_all  = best["spearman_r"].values
    sp_sig  = best.set_index("model")["spearman_sig"].reindex(models).values
    auroc_v = best.set_index("model")["auroc"].reindex(models).values
    auprc_v = best.set_index("model")["auprc"].reindex(models).values
    n_cols  = 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, max(3, len(models) * 0.7 + 1)))
    fig.suptitle(f"{label} — Performance Overview", fontsize=14, fontweight="bold")

    _bar_h(axes[0], models, sp_all,  "Spearman ρ (all variants)",    "Spearman ρ")
    _bar_h(axes[1], models, sp_sig,  "Spearman ρ (significant only)", "Spearman ρ")
    _bar_h(axes[2], models, auroc_v, "AUROC (sig. classification)",   "AUROC", vline=0.5)
    _bar_h(axes[3], models, auprc_v, "AUPRC (sig. classification)",   "AUPRC")

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(fc=MODEL_COLORS.get(m, "#888"), label=m) for m in models]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / f"{label.lower()}_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_dir}/{label.lower()}_overview.png")


def plot_scatter(model_scores, score_map, label, out_dir):
    """Scatter plot: balanced-sampled predictions vs MPRA log2FC."""
    avail = [m for m in score_map if m in model_scores]
    if not avail:
        return
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{label} — Predictions vs MPRA (balanced sample)", fontsize=12)

    for ax, model in zip(axes, avail):
        df = model_scores[model]
        sc = score_map[model]
        if sc not in df.columns:
            sc = [c for c in df.columns if c not in ("log2fc", "qval", "sig")][0]

        bdf = balanced_sample(df[["log2fc", sc, "sig"]].dropna(), n_per_bin=400)
        # colour by significance
        colors = np.where(bdf["sig"] == 1, "#E53935", "#90A4AE")
        ax.scatter(bdf["log2fc"], bdf[sc], c=colors, alpha=0.4, s=3, rasterized=True)
        r, _ = spearmanr(df["log2fc"], df[sc])
        ax.set_xlabel("MPRA log2FC")
        ax.set_ylabel(sc)
        ax.set_title(f"{model}\nρ = {r:.4f}")
        ax.axhline(0, color="gray", lw=0.3)
        ax.axvline(0, color="gray", lw=0.3)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#E53935',
                   label='significant', markersize=6),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#90A4AE',
                   label='non-significant', markersize=6),
        ], fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / f"{label.lower()}_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_dir}/{label.lower()}_scatter.png")


def plot_assay_heatmap(mdf, label, out_dir):
    """Per-assay Spearman heatmap for track models."""
    assay_rows = mdf[mdf["score_type"].str.startswith("assay:")].copy()
    if assay_rows.empty:
        return
    assay_rows["assay"] = assay_rows["score_type"].str.replace("assay:", "", regex=False)
    pivot = assay_rows.pivot_table(index="model", columns="assay", values="spearman_r", aggfunc="first")

    vmax = max(0.1, pivot.abs().max().max())
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.3), max(2.5, len(pivot) * 0.9)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > vmax * 0.5 else "black")
    ax.set_title(f"{label} — Spearman ρ by Assay Type (all variants)")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label.lower()}_assay_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_dir}/{label.lower()}_assay_heatmap.png")


def plot_effect_size_curve(esc_data, label, out_dir):
    """Performance vs |log2FC| threshold: shows how Spearman improves on stronger variants."""
    if not esc_data:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, (thresholds, rhos) in esc_data.items():
        color = MODEL_COLORS.get(model, "#888888")
        ax.plot(thresholds, rhos, label=model, color=color, lw=2)

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("|log2FC| cutoff (only variants ≥ cutoff included)")
    ax.set_ylabel("Spearman ρ vs log2FC")
    ax.set_title(f"{label} — Performance vs Effect-Size Threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{label.lower()}_effect_size_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_dir}/{label.lower()}_effect_size_curve.png")


def plot_sig_lfc_dist(gt, label, out_dir):
    """Show the log2FC distribution, highlighting significant variants."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{label} — log2FC Distribution", fontsize=12)

    ax = axes[0]
    sig = gt["sig"] == 1
    bins = np.linspace(gt["log2fc"].quantile(0.001), gt["log2fc"].quantile(0.999), 80)
    ax.hist(gt.loc[~sig, "log2fc"], bins=bins, alpha=0.5, label=f"non-sig (n={( ~sig).sum():,})", color="#90CAF9")
    ax.hist(gt.loc[ sig, "log2fc"], bins=bins, alpha=0.8, label=f"significant (n={sig.sum():,})", color="#E53935")
    ax.set_xlabel("log2FC"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    ax.set_title("Full distribution")

    ax = axes[1]
    if "qval" in gt.columns and gt["qval"].max() > 0:
        ax.hist(gt["qval"], bins=50, color="#4CAF50", edgecolor="white", linewidth=0.3)
        ax.axvline(1.3, color="red", ls="--", label="0.05 threshold")
        ax.set_xlabel("QVAL (-log10)"); ax.set_ylabel("count")
        ax.set_title("QVAL distribution"); ax.legend()
    else:
        ax.hist(gt["log2fc"].abs(), bins=50, color="#4CAF50", edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="red", ls="--", label="|log2FC|=0.5")
        ax.set_xlabel("|log2FC|"); ax.set_ylabel("count")
        ax.set_title("|log2FC| distribution"); ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / f"{label.lower()}_lfc_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_dir}/{label.lower()}_lfc_dist.png")

# ── Per-experiment runner ─────────────────────────────────────────────────────

def run_experiment(ct, cfg, tmp_dir):
    """Run full analysis for one cell type. Data must already be in tmp_dir."""
    label    = cfg["label"]
    out_dir  = BASE / "results" / ct / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Ground truth
    print("Loading ground truth ...")
    gt     = load_gt(cfg)
    gt_set = set(gt.index)
    n_sig  = gt["sig"].sum()
    print(f"  n={len(gt):,}  n_sig={n_sig:,} ({100*n_sig/len(gt):.1f}%)")
    print(f"  log2FC range [{gt['log2fc'].min():.3f}, {gt['log2fc'].max():.3f}]  std={gt['log2fc'].std():.4f}")

    plot_sig_lfc_dist(gt, label, out_dir)

    all_metrics   = []
    model_scores  = {}
    esc_data      = {}   # effect-size-curve data
    score_map     = {}   # model -> primary score column for scatter

    # ── Track models ─────────────────────────────────────────────────────────
    for model in TRACK_MODELS:
        path = tmp_dir / f"{model}_scores.parquet"
        if not path.exists():
            print(f"\n  SKIP {model} (not found)")
            continue
        print(f"\n  Processing {model} ...")
        summary, assay_df = aggregate_track(path, gt_set)
        if summary.empty:
            continue
        joined = gt.join(summary, how="inner")
        print(f"  Joined: {len(joined):,} variants")
        model_scores[model] = joined
        score_map[model]    = "mean_sad"

        for sc in ["mean_sad", "mean_abs_sad", "max_sad"]:
            all_metrics.append(compute_metrics(joined, sc, model, sc))

        # Per-assay
        if assay_df is not None:
            aj = gt.join(assay_df, how="inner")
            for assay in assay_df.columns:
                if aj[assay].isna().all():
                    continue
                r, _ = spearmanr(aj["log2fc"], aj[assay].fillna(0))
                all_metrics.append({"model": model, "score_type": f"assay:{assay}", "spearman_r": float(r), "n": len(aj)})

        # Effect-size curve
        thrs, rhos = effect_size_curve(joined, "mean_sad")
        esc_data[model] = (thrs, rhos)

    # ── ND models ────────────────────────────────────────────────────────────
    for model in ND_MODELS:
        path = tmp_dir / f"{model}_scores.parquet"
        if not path.exists():
            print(f"\n  SKIP {model} (not found)")
            continue
        print(f"\n  Processing {model} ...")
        nd     = load_nd(path, gt_set)
        joined = gt.join(nd, how="inner")
        print(f"  Joined: {len(joined):,} variants")
        if joined.empty:
            continue
        model_scores[model] = joined

        nd_cols = [c for c in nd.columns if c.startswith("ND_")]
        for sc in nd_cols:
            all_metrics.append(compute_metrics(joined, sc, model, sc))
        primary_sc = "ND_influence_score" if "ND_influence_score" in nd_cols else (nd_cols[0] if nd_cols else None)
        if primary_sc:
            score_map[model] = primary_sc
            thrs, rhos = effect_size_curve(joined, primary_sc)
            esc_data[model] = (thrs, rhos)

    # ── Save metrics ─────────────────────────────────────────────────────────
    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(out_dir / f"{ct}_metrics.csv", index=False)
    print(f"\n  Metrics saved: {out_dir}/{ct}_metrics.csv")

    # Print summary table
    summary_cols = ["model", "score_type", "n", "n_sig", "spearman_r", "spearman_sig", "auroc", "auprc"]
    primary_types = ["mean_sad", "max_sad", "ND_influence_score", "ND_max_ABS"]
    disp = mdf[mdf["score_type"].isin(primary_types)].reindex(columns=summary_cols, fill_value=np.nan)
    print(disp.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Only include whole-variant score_types for overview plot
    overview_mdf = mdf[~mdf["score_type"].str.startswith("assay:")].copy()
    plot_overview(overview_mdf, label, out_dir, TRACK_MODELS)
    plot_scatter(model_scores, score_map, label, out_dir)
    plot_assay_heatmap(mdf, label, out_dir)
    plot_effect_size_curve(esc_data, label, out_dir)

    return mdf, model_scores

# ── Cross-experiment comparison ───────────────────────────────────────────────

def plot_cross_comparison(all_results, out_dir):
    """Spearman and AUROC bars across all cell types for each model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_types = {"mean_sad", "ND_influence_score"}

    rows = []
    for ct, (mdf, _) in all_results.items():
        label = EXPERIMENTS[ct]["label"]
        sub   = mdf[mdf["score_type"].isin(primary_types)].copy()
        sub["celltype"] = label
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)

    metrics = [("spearman_r", "Spearman ρ (all)", None),
               ("spearman_sig", "Spearman ρ (sig. only)", None),
               ("auroc", "AUROC", 0.5)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    fig.suptitle("Cross Cell-Type Comparison", fontsize=14, fontweight="bold")

    celltypes = [EXPERIMENTS[ct]["label"] for ct in all_results]
    models    = [m for m in (TRACK_MODELS + ND_MODELS) if m in combined["model"].unique()]
    x         = np.arange(len(models))
    width     = 0.8 / len(celltypes)
    ct_colors = ["#1565C0", "#C62828", "#2E7D32"]

    for ax_i, (col, title, vline) in enumerate(metrics):
        ax = axes[ax_i]
        for j, (ct, color) in enumerate(zip(all_results.keys(), ct_colors)):
            label = EXPERIMENTS[ct]["label"]
            vals  = []
            for model in models:
                row = combined[(combined["celltype"] == label) & (combined["model"] == model)]
                if row.empty or col not in row.columns:
                    vals.append(np.nan)
                else:
                    # take the row with highest absolute spearman_r
                    r = row.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).iloc[0][col]
                    vals.append(float(r))
            bars = ax.bar(x + j * width, vals, width=width * 0.9, label=label, color=color, alpha=0.8)

        ax.set_xticks(x + width * (len(celltypes) - 1) / 2)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(col)
        ax.set_title(title)
        if vline:
            ax.axhline(vline, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "cross_celltype_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_dir}/cross_celltype_comparison.png")

# ── Data copy helpers ─────────────────────────────────────────────────────────

def ensure_data_in_tmp(ct, cfg, tmp_dir):
    """Copy parquet files from NTFS results dir to /tmp if not already there."""
    src_root = cfg["results_dir"]
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for model in TRACK_MODELS + ND_MODELS:
        # Find parquet in src_root/<model>/
        for subdir in [src_root / model, src_root]:
            src = subdir / f"{model}_scores.parquet"
            if src.exists():
                dst = tmp_dir / f"{model}_scores.parquet"
                if dst.exists():
                    print(f"    {model}: already in /tmp, skipping copy")
                else:
                    size_gb = src.stat().st_size / 1e9
                    print(f"    {model}: copying {size_gb:.1f} GB ...", flush=True)
                    t0 = time.time()
                    shutil.copy2(str(src), str(dst))
                    print(f"           done in {time.time()-t0:.0f}s")
                break

    # Copy VCF if needed (small)
    vcf_dst = tmp_dir / cfg["vcf"].name
    if not vcf_dst.exists():
        shutil.copy2(str(cfg["vcf"]), str(vcf_dst))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MPRA benchmark analysis")
    parser.add_argument("--cell-types", nargs="+", default=["ngn2", "hek293t", "hepg2"],
                        choices=list(EXPERIMENTS.keys()),
                        help="Cell types to process (default: all)")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip copying to /tmp (use if data is already there)")
    args = parser.parse_args()

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ct in args.cell_types:
        cfg     = EXPERIMENTS[ct]
        tmp_dir = TMP_ROOT / ct

        if not args.no_copy:
            print(f"\nCopying {cfg['label']} data to /tmp ...")
            ensure_data_in_tmp(ct, cfg, tmp_dir)
        else:
            # For NGN2, data is in legacy location /tmp/ngn2_data/
            if ct == "ngn2" and not tmp_dir.exists():
                legacy = Path("/tmp/ngn2_data")
                if legacy.exists():
                    print(f"  Using legacy /tmp/ngn2_data for NGN2")
                    tmp_dir = legacy

        mdf, model_scores = run_experiment(ct, cfg, tmp_dir)
        all_results[ct]   = (mdf, model_scores)

    # Cross-experiment comparison (only if >1 celltype)
    if len(all_results) > 1:
        print("\n\nGenerating cross-celltype comparison ...")
        cross_dir = BASE / "results" / "analysis"
        plot_cross_comparison(all_results, cross_dir)
        # Merge all metrics
        frames = []
        for ct, (mdf, _) in all_results.items():
            mdf["celltype"] = ct
            frames.append(mdf)
        pd.concat(frames).to_csv(cross_dir / "all_metrics.csv", index=False)
        print(f"  All metrics: {cross_dir}/all_metrics.csv")

    print("\nDone.")

if __name__ == "__main__":
    main()
