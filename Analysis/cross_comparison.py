#!/usr/bin/env python3
"""
Generate the cross-celltype comparison plot from previously saved metrics CSVs.
Run from project root:
  python3 Scripts/cross_comparison.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mnt/c/Users/user/Desktop/Benchmark_results/Pipeline_v2")

EXPERIMENTS = {
    "ngn2":    {"label": "NGN2",    "csv": BASE / "results/ngn2/analysis/ngn2_metrics.csv"},
    "hek293t": {"label": "HEK293T", "csv": BASE / "results/hek293t/analysis/hek293t_metrics.csv"},
    "hepg2":   {"label": "HepG2",   "csv": BASE / "results/hepg2/analysis/hepg2_metrics.csv"},
}

TRACK_MODELS = ["enformer", "basenji", "alphagenome"]
ND_MODELS    = ["hyenadna", "dnabert2"]

def main():
    out_dir = BASE / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics CSVs
    all_mdf = {}
    for ct, cfg in EXPERIMENTS.items():
        if not cfg["csv"].exists():
            print(f"  WARNING: missing {cfg['csv']}, skipping {ct}")
            continue
        mdf = pd.read_csv(cfg["csv"])
        all_mdf[ct] = mdf
        print(f"  Loaded {ct}: {len(mdf)} rows")

    if len(all_mdf) < 2:
        print("Need at least 2 cell types. Exiting.")
        return

    # ── Plot 1: Overview bars (Spearman, Spearman sig, AUROC) ─────────────────
    primary_types = {"mean_sad", "ND_influence_score"}
    rows = []
    for ct, mdf in all_mdf.items():
        label = EXPERIMENTS[ct]["label"]
        sub   = mdf[mdf["score_type"].isin(primary_types)].copy()
        sub["celltype"] = label
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)

    metrics = [
        ("spearman_r",   "Spearman ρ (all variants)",  None),
        ("spearman_sig", "Spearman ρ (sig. only)",      None),
        ("auroc",        "AUROC (sig. classification)", 0.5),
    ]

    celltypes = list(all_mdf.keys())
    models    = [m for m in (TRACK_MODELS + ND_MODELS) if m in combined["model"].unique()]
    x         = np.arange(len(models))
    ct_width  = 0.8 / len(celltypes)
    ct_colors = ["#1565C0", "#C62828", "#2E7D32"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    fig.suptitle("Cross Cell-Type Comparison", fontsize=14, fontweight="bold")

    for ax_i, (col, title, vline) in enumerate(metrics):
        ax = axes[ax_i]
        for j, (ct, color) in enumerate(zip(celltypes, ct_colors)):
            label = EXPERIMENTS[ct]["label"]
            vals  = []
            for model in models:
                row = combined[(combined["celltype"] == label) & (combined["model"] == model)]
                if row.empty or col not in row.columns or row[col].isna().all():
                    vals.append(np.nan)
                else:
                    # Best score_type by absolute spearman_r
                    r = row.sort_values("spearman_r", key=lambda s: s.abs(),
                                        ascending=False).iloc[0][col]
                    vals.append(float(r) if pd.notna(r) else np.nan)
            bars = ax.bar(x + j * ct_width, vals, width=ct_width * 0.9,
                          label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(x + ct_width * (len(celltypes) - 1) / 2)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(col)
        ax.set_title(title)
        if vline is not None:
            ax.axhline(vline, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "cross_celltype_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # ── Plot 2: AUROC max across score types ──────────────────────────────────
    # Best AUROC per (model, celltype) regardless of score_type
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle("Best AUROC per Model × Cell Type", fontsize=13, fontweight="bold")

    for j, (ct, color) in enumerate(zip(celltypes, ct_colors)):
        label = EXPERIMENTS[ct]["label"]
        mdf   = all_mdf[ct]
        vals  = []
        for model in models:
            sub = mdf[(mdf["model"] == model) & mdf["auroc"].notna()]
            if sub.empty:
                vals.append(np.nan)
            else:
                vals.append(float(sub["auroc"].max()))
        bars = ax2.bar(x + j * ct_width, vals, width=ct_width * 0.9,
                       label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

    ax2.set_xticks(x + ct_width * (len(celltypes) - 1) / 2)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Best AUROC per model")
    ax2.axhline(0.5, color="red", lw=0.8, ls="--", alpha=0.7, label="chance")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.45, None)

    plt.tight_layout()
    out2 = out_dir / "cross_celltype_auroc.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")

    # ── Merge all metrics CSV ─────────────────────────────────────────────────
    frames = []
    for ct, mdf in all_mdf.items():
        mdf2 = mdf.copy()
        mdf2["celltype"] = ct
        frames.append(mdf2)
    merged = pd.concat(frames, ignore_index=True)
    out_csv = out_dir / "all_metrics.csv"
    merged.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
