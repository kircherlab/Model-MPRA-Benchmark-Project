#!/usr/bin/env python3
"""
Cell-type Filtered MPRA Benchmark Analysis
============================================
Re-computes benchmark metrics (Spearman, AUROC) using only model tracks
whose biosample_name matches the cell type being evaluated.

For each cell type × track model combination:
  - Streams the parquet, keeps only rows where biosample_name contains
    a cell-type-relevant keyword
  - Aggregates per-variant: mean_sad, max_abs_sad across matched tracks
  - Computes the same Spearman / AUROC metrics as analysis_all.py
  - Also computes n_matched_biosamples for transparency

Biosample keyword strategy
--------------------------
  HepG2    : "hepg2"
  HEK293T  : "hek293"
  NGN2     : "neuron" | "neural" | "glutamat"
              (NGN2 produces excitatory glutamatergic neurons;
               AlphaGenome has "glutamatergic neuron" exactly)

Output
------
  results/<celltype>/analysis/<celltype>_filtered_metrics.csv
  results/<celltype>/analysis/<celltype>_filtered_scatter.png
  results/<celltype>/analysis/<celltype>_filtered_overview.png
  results/analysis/cross_celltype_filtered_comparison.png

Usage
-----
  python3 Scripts/celltype_filtered_analysis.py [--cell-types ngn2 hek293t hepg2] [--no-copy]
"""

import re, sys, time, argparse
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
        "sig_lfc_thr": 0.5,
        "label":       "NGN2",
    },
    "hek293t": {
        "vcf":         VCF_DIR / "IGVFFI4134MFLL.vcf",
        "results_dir": BASE / "results/hek293t",
        "has_qval":    True,
        "qval_thr":    1.3,
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

MODEL_COLORS = {
    "enformer":    "#1565C0",
    "basenji":     "#0288D1",
    "alphagenome": "#00796B",
}

# ── Cell-type biosample keyword filters ──────────────────────────────────────
# Regex patterns for vectorised str.contains(case=False) filtering.
# One pattern per (cell-type, model) combination.
BIOSAMPLE_FILTERS = {
    "hepg2": {
        # Enformer/Basenji: "HepG2", "3xFLAG-XX:HepG2 genetically modified..."
        "enformer":    "hepg2",
        "basenji":     "hepg2",
        # AlphaGenome: clean name "HepG2"
        "alphagenome": "hepg2",
    },
    "hek293t": {
        "enformer":    "hek293",
        "basenji":     "hek293",
        "alphagenome": "hek293",
    },
    "ngn2": {
        # NGN2 protocol = excitatory glutamatergic neurons from iPSC
        "enformer":    r"neuron|neural|glutamat",
        "basenji":     r"neuron|neural|glutamat",
        "alphagenome": r"neuron|neural|glutamat",
    },
}


# ── Math helpers (same as analysis_all.py) ────────────────────────────────────

def _roc_auc(y_true, y_score):
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n_pos = int(y_true.sum());  n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order  = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    tps = np.cumsum(y_true);  fps = np.cumsum(1 - y_true)
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
    return float(np.sum(prec * y_true) / y_true.sum())


def compute_metrics(joined, score_col):
    """Compute Spearman (all), Spearman (sig only), AUROC, AUPRC."""
    x, y = joined["log2FC"].values, joined[score_col].values
    sig  = joined["sig"].values.astype(bool)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, sig = x[mask], y[mask], sig[mask]

    sp_all, _ = spearmanr(x, y)

    sp_sig = np.nan
    if sig.sum() >= 10:
        sp_sig, _ = spearmanr(x[sig], y[sig])

    auroc = _roc_auc(sig.astype(float), np.abs(y))
    auprc = _avg_precision(sig.astype(float), np.abs(y))

    return {
        "n":            len(x),
        "n_sig":        int(sig.sum()),
        "spearman_r":   float(sp_all),
        "spearman_sig": float(sp_sig),
        "auroc":        float(auroc),
        "auprc":        float(auprc),
    }


def balanced_sample(df, log2fc_col="log2FC", n_per_bin=400, n_bins=30):
    """Sample ≤ n_per_bin rows per |log2FC| quantile bin to flatten near-zero."""
    df = df.copy()
    df["_abs"] = df[log2fc_col].abs()
    bins = pd.qcut(df["_abs"], q=n_bins, duplicates="drop", labels=False)
    parts = []
    for _, g in df.groupby(bins, observed=True):
        parts.append(g.sample(min(n_per_bin, len(g)), random_state=42))
    return pd.concat(parts).drop(columns="_abs")


# ── Ground truth loading (same as analysis_all.py) ───────────────────────────

_LOG2FC_RE = re.compile(r"LOG2FC=([^;]+)")
_QVAL_RE   = re.compile(r"QVAL=([^;]+)")

def load_gt(cfg):
    vcf_path  = cfg["vcf"]
    has_qval  = cfg["has_qval"]
    qval_thr  = cfg.get("qval_thr",    1.3)
    lfc_thr   = cfg.get("sig_lfc_thr", 0.5)

    records = []
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            chrom, pos, _, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
            info  = cols[7] if len(cols) > 7 else ""
            vid   = f"{chrom}:{pos}:{ref}>{alt}"
            m_lfc = _LOG2FC_RE.search(info)
            if m_lfc is None:
                continue
            lfc   = float(m_lfc.group(1))
            qval  = np.nan
            if has_qval:
                m_q = _QVAL_RE.search(info)
                if m_q:
                    qval = float(m_q.group(1))
            records.append({"variant_id": vid, "log2FC": lfc, "QVAL": qval})

    gt = pd.DataFrame(records).drop_duplicates(subset="variant_id")
    if has_qval:
        gt["sig"] = gt["QVAL"] > qval_thr
    else:
        gt["sig"] = gt["log2FC"].abs() > lfc_thr
    return gt


# ── Filtered streaming aggregation ───────────────────────────────────────────

def aggregate_filtered(parquet_path, gt_set, bio_pattern, batch_size=2_000_000):
    """
    Stream parquet, keep only rows where biosample_name matches bio_pattern
    (case-insensitive regex) and variant_id is in gt_set.
    Aggregate per-variant: mean_sad, max_abs_sad.

    Returns (scores_df, n_matched_biosamples, matched_biosample_names)
    """
    pf = pq.ParquetFile(parquet_path)
    matched_biosamples = set()
    aggs = []
    t0 = time.time()

    for n_batches_, batch in enumerate(pf.iter_batches(
            batch_size=batch_size,
            columns=["variant_id", "biosample_name", "raw_score"])):
        df = batch.to_pandas()

        # Vectorised biosample filter (C-level, ~50x faster than .apply(lambda))
        bio_mask = df["biosample_name"].str.contains(
            bio_pattern, case=False, na=False, regex=True)
        matched_biosamples.update(df.loc[bio_mask, "biosample_name"].unique())
        df = df[bio_mask]
        if df.empty:
            continue

        # Filter to variants in ground truth
        df = df[df["variant_id"].isin(gt_set)]
        if df.empty:
            continue

        # Vectorised per-variant aggregation
        df["abs_score"] = df["raw_score"].abs()
        g = df.groupby("variant_id", sort=False)
        agg = g.agg(
            sum=("raw_score",   "sum"),
            count=("raw_score", "count"),
            max_abs=("abs_score", "max"),
        )
        aggs.append(agg)

        n_batches = n_batches_ + 1
        if n_batches % 20 == 0:
            print(f"    batch {n_batches}: {time.time()-t0:.0f}s elapsed", flush=True)

    if not aggs:
        return None, 0, set()

    combined = pd.concat(aggs)
    summed   = combined[["sum", "count"]].groupby(level=0).sum()
    max_vals = combined["max_abs"].groupby(level=0).max()

    scores_df = pd.DataFrame({
        "variant_id":  summed.index,
        "mean_sad":    (summed["sum"] / summed["count"]).values,
        "max_abs_sad": max_vals.values,
    })
    return scores_df, len(matched_biosamples), matched_biosamples


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_filtered_scatter(all_joined, ct, out_dir):
    """Scatter plot: filtered mean_sad vs log2FC for each model."""
    models  = [m for m in TRACK_MODELS if m in all_joined]
    n_cols  = len(models)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"{EXPERIMENTS[ct]['label']} — Cell-type Filtered Predictions vs MPRA\n"
                 f"(only cell-type matched biosamples)", fontsize=11)

    for ax, model in zip(axes, models):
        joined = all_joined[model]
        samp   = balanced_sample(joined, n_per_bin=400, n_bins=30)
        c      = ["#D32F2F" if s else "#B0BEC5" for s in samp["sig"]]
        sp, _  = spearmanr(samp["log2FC"], samp["mean_sad"])
        ax.scatter(samp["log2FC"], samp["mean_sad"], c=c, s=8, alpha=0.5, linewidths=0)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("MPRA log2FC"); ax.set_ylabel("mean_sad (filtered)")
        ax.set_title(f"{model}\nρ = {sp:.4f}", fontsize=10)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#D32F2F',ms=5,label='significant'),
                            Line2D([0],[0],marker='o',color='w',markerfacecolor='#B0BEC5',ms=5,label='non-significant')],
                  fontsize=7)

    plt.tight_layout()
    out = out_dir / f"{ct}_filtered_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_filtered_overview(metrics_rows, ct, out_dir):
    """
    4-panel bar chart: Spearman (all), Spearman (sig), AUROC, AUPRC
    for filtered vs all-tracks (if all-tracks CSV exists).
    """
    label = EXPERIMENTS[ct]["label"]
    all_csv = BASE / f"results/{ct}/analysis/{ct}_metrics.csv"

    filt_df = pd.DataFrame(metrics_rows)
    # For filtered: use mean_sad rows only (clearest comparison point)
    filt_sub = filt_df[filt_df["score_type"] == "mean_sad"].copy()

    panels = [
        ("spearman_r",   "Spearman ρ (all)",     None),
        ("spearman_sig", "Spearman ρ (sig only)", None),
        ("auroc",        "AUROC",                 0.5),
        ("auprc",        "AUPRC",                 None),
    ]

    if all_csv.exists():
        all_df  = pd.read_csv(all_csv)
        all_sub = all_df[(all_df["score_type"] == "mean_sad")].copy()
        has_all = True
    else:
        has_all = False

    models = filt_sub["model"].tolist()
    x      = np.arange(len(models))
    width  = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"{label} — All tracks vs Cell-type Filtered Tracks", fontsize=12, fontweight="bold")

    for ax, (col, title, vline) in zip(axes, panels):
        fvals = [filt_sub.loc[filt_sub["model"]==m, col].values[0]
                 if m in filt_sub["model"].values else np.nan for m in models]

        if has_all:
            avals = [all_sub.loc[all_sub["model"]==m, col].values[0]
                     if m in all_sub["model"].values else np.nan for m in models]
            bars_a = ax.bar(x - width/2, avals, width=width*0.9,
                            label="All tracks", color="#90A4AE", alpha=0.85)
            bars_f = ax.bar(x + width/2, fvals, width=width*0.9,
                            label="CT-filtered",
                            color=[MODEL_COLORS.get(m, "gray") for m in models], alpha=0.85)
            for bar, v in zip(bars_a, avals):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6, color="#555")
            for bar, v in zip(bars_f, fvals):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        else:
            ax.bar(x, fvals, width=0.6,
                   color=[MODEL_COLORS.get(m, "gray") for m in models], alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col)
        if vline is not None:
            ax.axhline(vline, color="red", lw=0.8, ls="--", alpha=0.7)
        if has_all:
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = out_dir / f"{ct}_filtered_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_cross_filtered(all_ct_results, out_dir):
    """Cross-celltype bar chart for filtered Spearman (all) and AUROC."""
    out_dir.mkdir(parents=True, exist_ok=True)

    celltypes = list(all_ct_results.keys())
    ct_colors = ["#1565C0", "#C62828", "#2E7D32"]
    models    = TRACK_MODELS

    panels = [
        ("spearman_r",   "Spearman ρ (all variants)",  None),
        ("spearman_sig", "Spearman ρ (sig. only)",      None),
        ("auroc",        "AUROC (classification)",      0.5),
    ]

    x     = np.arange(len(models))
    width = 0.8 / len(celltypes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cross Cell-Type — Cell-type Filtered Tracks", fontsize=13, fontweight="bold")

    for ax, (col, title, vline) in zip(axes, panels):
        for j, (ct, color) in enumerate(zip(celltypes, ct_colors)):
            label = EXPERIMENTS[ct]["label"]
            rows  = all_ct_results[ct]
            df    = pd.DataFrame(rows)
            df_sub = df[df["score_type"] == "mean_sad"]
            vals  = []
            for m in models:
                row = df_sub[df_sub["model"] == m]
                vals.append(float(row[col].values[0]) if not row.empty and pd.notna(row[col].values[0]) else np.nan)
            bars = ax.bar(x + j * width, vals, width=width*0.9,
                          label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(x + width * (len(celltypes)-1) / 2)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(title, fontsize=10); ax.set_ylabel(col)
        if vline is not None:
            ax.axhline(vline, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "cross_celltype_filtered_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


# ── Main per-celltype runner ──────────────────────────────────────────────────

def run_filtered(ct, cfg, tmp_dir):
    label    = cfg["label"]
    out_dir  = BASE / f"results/{ct}/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label} — cell-type filtered analysis")
    print(f"{'='*60}")

    # Load ground truth
    print("Loading ground truth ...")
    gt = load_gt(cfg)
    gt_set = set(gt["variant_id"])
    n_sig  = gt["sig"].sum()
    print(f"  n={len(gt):,}  n_sig={n_sig:,} ({100*n_sig/len(gt):.1f}%)")

    metrics_rows = []
    all_joined   = {}   # model → joined df for scatter

    for model in TRACK_MODELS:
        parquet = tmp_dir / f"{model}_scores.parquet"
        if not parquet.exists():
            print(f"\n  SKIP {model} (not found at {parquet})")
            continue

        bio_pattern = BIOSAMPLE_FILTERS[ct][model]
        print(f"\n  Processing {model} (filtered) ...", flush=True)

        scores_df, n_bio, bio_names = aggregate_filtered(parquet, gt_set, bio_pattern)

        if scores_df is None or scores_df.empty:
            print(f"    WARNING: no matching biosamples found — skipping {model}")
            continue

        print(f"    Matched biosamples: {n_bio}")
        if n_bio <= 30:
            for b in sorted(bio_names):
                print(f"      - {b}")

        # Join with ground truth
        joined = gt.merge(scores_df, on="variant_id", how="inner")
        joined = joined.drop_duplicates(subset="variant_id")
        print(f"    Joined: {len(joined):,} variants")

        all_joined[model] = joined

        # Metrics
        for sc in ["mean_sad", "max_abs_sad"]:
            try:
                m = compute_metrics(joined, sc)
                m.update({"model": model, "score_type": sc,
                           "n_biosamples": n_bio})
                metrics_rows.append(m)
                print(f"    [{sc}] Spearman(all)={m['spearman_r']:.4f} "
                      f"Spearman(sig)={m['spearman_sig']:.4f}  "
                      f"AUROC={m['auroc']:.4f}")
            except Exception as e:
                print(f"    [{sc}] ERROR: {e}")

    if not metrics_rows:
        print(f"  No results for {ct}, skipping plots.")
        return []

    # Save metrics CSV
    mdf = pd.DataFrame(metrics_rows)
    col_order = ["model","score_type","n_biosamples","n","n_sig",
                 "spearman_r","spearman_sig","auroc","auprc"]
    mdf = mdf[[c for c in col_order if c in mdf.columns]]
    out_csv = out_dir / f"{ct}_filtered_metrics.csv"
    mdf.to_csv(out_csv, index=False)
    print(f"\n  Metrics saved: {out_csv}")
    print(mdf.to_string(index=False))

    # Plots
    try:
        plot_filtered_scatter(all_joined, ct, out_dir)
    except Exception as e:
        print(f"  scatter plot failed: {e}")
    try:
        plot_filtered_overview(metrics_rows, ct, out_dir)
    except Exception as e:
        print(f"  overview plot failed: {e}")

    return metrics_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cell-type filtered MPRA analysis")
    parser.add_argument("--cell-types", nargs="+", default=["ngn2", "hek293t", "hepg2"],
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--no-copy", action="store_true",
                        help="Data already in /tmp/mpra_analysis/<ct>/")
    args = parser.parse_args()

    all_ct_results = {}

    for ct in args.cell_types:
        cfg     = EXPERIMENTS[ct]
        tmp_dir = TMP_ROOT / ct

        # NGN2 legacy path
        if ct == "ngn2" and not tmp_dir.exists():
            legacy = Path("/tmp/ngn2_data")
            if legacy.exists():
                tmp_dir = legacy

        if not tmp_dir.exists():
            print(f"  WARNING: {tmp_dir} not found — skipping {ct}")
            continue

        rows = run_filtered(ct, cfg, tmp_dir)
        if rows:
            all_ct_results[ct] = rows

    # Cross-celltype comparison (if >1 cell type done)
    if len(all_ct_results) > 1:
        print("\n\nGenerating cross-celltype filtered comparison ...")
        cross_dir = BASE / "results" / "analysis"
        plot_cross_filtered(all_ct_results, cross_dir)

        # Merged CSV
        frames = []
        for ct, rows in all_ct_results.items():
            df = pd.DataFrame(rows)
            df["celltype"] = ct
            frames.append(df)
        merged = pd.concat(frames, ignore_index=True)
        out_csv = cross_dir / "all_filtered_metrics.csv"
        merged.to_csv(out_csv, index=False)
        print(f"  Merged CSV: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
