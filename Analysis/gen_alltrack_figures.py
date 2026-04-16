#!/usr/bin/env python3
"""
Lightweight script to generate all-track assay figures from existing parquets.

Only does a simple per-(assay_type, variant_idx) groupby — no per-biosample
splitting — so it is much faster than the full 'detailed' analysis pass.

Outputs per cell type (in the same analysis/ or analysis_sigonly/ directory):
  {ct}_{model}_alltrack_assay_correlation_{st}_unified.png
  {ct}_{model}_alltrack_assay_scatter_{st}_unified.png
  {ct}_alltrack_assay_barchart_all.png
  {ct}_alltrack_assay_barchart_sig.png
  {ct}_alltrack_assay_metrics_unified.csv
"""

import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

# ── Shared constants (mirror unified_celltype_analysis.py) ───────────────────
BASE     = Path("/mnt/c/Users/user/Desktop/Benchmark_results/Pipeline_v2")
VCF_DIR  = BASE / "Data/VCF"

EXPERIMENTS = {
    "ngn2": {
        "vcf": VCF_DIR / "80k_normalized.vcf",
        "has_qval": False,
        "sig_lfc_thr": 0.5,
        "label": "NGN2",
    },
    "hek293t": {
        "vcf": VCF_DIR / "IGVFFI4134MFLL.vcf",
        "has_qval": True,
        "qval_thr": 1.0,
        "label": "HEK293T",
    },
    "hepg2": {
        "vcf": VCF_DIR / "IGVFFI4378PZYI.vcf",
        "has_qval": True,
        "qval_thr": 1.0,
        "label": "HepG2",
    },
}

TRACK_MODELS = ["enformer", "basenji", "alphagenome"]

MODEL_COLORS = {
    "enformer":    "#1565C0",
    "basenji":     "#0288D1",
    "alphagenome": "#00796B",
}

_LOG2FC_RE = re.compile(r"LOG2FC=([^;]+)")
_QVAL_RE   = re.compile(r"QVAL=([^;]+)")

# ── Helpers ──────────────────────────────────────────────────────────────────

def plot_data_path(out_path):
    return out_path.with_name(f"{out_path.stem}__plot_data.csv")


def load_gt(cfg):
    records = []
    with open(cfg["vcf"]) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
            info = cols[7] if len(cols) > 7 else ""
            chrom = chrom[3:] if chrom.startswith("chr") else chrom
            vid = f"{chrom}:{pos}:{ref}>{alt}"
            m_lfc = _LOG2FC_RE.search(info)
            if m_lfc is None:
                continue
            lfc = float(m_lfc.group(1))
            qval = np.nan
            if cfg["has_qval"]:
                m_q = _QVAL_RE.search(info)
                if m_q is not None:
                    qval = float(m_q.group(1))
            records.append({"variant_id": vid, "log2FC": lfc, "QVAL": qval})
    gt = pd.DataFrame(records).drop_duplicates(subset="variant_id").reset_index(drop=True)
    if cfg["has_qval"]:
        gt["sig"] = gt["QVAL"] > cfg.get("qval_thr", 1.3)
    else:
        gt["sig"] = gt["log2FC"].abs() > cfg.get("sig_lfc_thr", 0.5)
    return gt


def compute_metrics(log2fc, sig, score, use_abs_lfc=False):
    mask = np.isfinite(log2fc) & np.isfinite(score)
    x, y, s = log2fc[mask], score[mask], sig[mask]
    if len(x) < 10:
        return None
    x_corr = np.abs(x) if use_abs_lfc else x
    sp_all, _ = spearmanr(x_corr, y)
    sp_sig = np.nan
    if s.sum() >= 10:
        sp_sig, _ = spearmanr(x_corr[s], y[s])
    return {
        "n": int(len(x)),
        "n_sig": int(s.sum()),
        "spearman_r": float(sp_all),
        "spearman_sig": float(sp_sig),
    }


def build_all_assay_acc(parquet_path, gt_index, n_variants, batch_size):
    """Single parquet pass: group by (assay_type, variant_idx) only — no biosample filter."""
    pf = pq.ParquetFile(parquet_path)
    acc = {}  # assay_type -> {sum, abs_sum, count}
    t0 = time.time()
    for n_batch, batch in enumerate(pf.iter_batches(
            batch_size=batch_size, columns=["variant_id", "raw_score", "assay_type"])):
        df = batch.to_pandas()
        df["variant_id"] = df["variant_id"].str.replace(r"^chr", "", regex=True)
        vidx = df["variant_id"].map(gt_index)
        keep = vidx.notna()
        if not keep.any():
            continue
        df = df.loc[keep].copy()
        df["variant_idx"] = vidx.loc[keep].astype(np.int32)
        df["assay_type"] = df["assay_type"].fillna("UNKNOWN")
        finite_mask = np.isfinite(df["raw_score"].to_numpy(dtype=np.float32))
        df = df.loc[finite_mask].copy()
        if df.empty:
            continue
        df["abs_score"] = df["raw_score"].abs()
        agg = (
            df.groupby(["assay_type", "variant_idx"], sort=False, observed=True)
            .agg(raw_sum=("raw_score", "sum"), abs_sum=("abs_score", "sum"), cnt=("raw_score", "count"))
            .reset_index()
        )
        for assay, kdf in agg.groupby("assay_type", sort=False):
            key = str(assay)
            if key not in acc:
                acc[key] = {
                    "sum":     np.zeros(n_variants, dtype=np.float32),
                    "abs_sum": np.zeros(n_variants, dtype=np.float32),
                    "count":   np.zeros(n_variants, dtype=np.uint16),
                }
            vidx_arr = kdf["variant_idx"].to_numpy(dtype=np.int32)
            acc[key]["sum"][vidx_arr]     += kdf["raw_sum"].to_numpy(dtype=np.float32)
            acc[key]["abs_sum"][vidx_arr] += kdf["abs_sum"].to_numpy(dtype=np.float32)
            acc[key]["count"][vidx_arr]   += kdf["cnt"].to_numpy(dtype=np.uint16)
        if (n_batch + 1) % 20 == 0:
            print(f"      batch {n_batch+1}: {time.time()-t0:.0f}s", flush=True)
    return acc


def assay_score_matrix(acc, score_type="mean_sad"):
    if not acc:
        return pd.DataFrame()
    n = len(next(iter(acc.values()))["count"])
    cols = {}
    for assay in sorted(acc):
        v = acc[assay]
        valid = v["count"] > 0
        score = np.full(n, np.nan, dtype=np.float32)
        if score_type == "mean_sad":
            score[valid] = v["sum"][valid] / v["count"][valid]
        elif score_type == "mean_abs_sad":
            score[valid] = v["abs_sum"][valid] / v["count"][valid]
        cols[assay] = score
    return pd.DataFrame(cols)


# ── Plot functions (self-contained copies) ───────────────────────────────────

def plot_assay_correlation_heatmap(acc, ct_label, model, score_type, out_path):
    import matplotlib.cm as cm
    pair_acc = {("_all_", assay): v for assay, v in acc.items()}
    assays = sorted({k[1] for k in pair_acc})
    if len(assays) < 2:
        return
    n_variants = len(next(iter(pair_acc.values()))["count"])
    cols = {}
    for assay in assays:
        s = np.zeros(n_variants, dtype=np.float32)
        c = np.zeros(n_variants, dtype=np.uint16)
        for (_, a), v in pair_acc.items():
            if a == assay:
                if score_type == "mean_sad":
                    s += v["sum"]
                else:
                    s += v["abs_sum"]
                c += v["count"]
        score = np.full(n_variants, np.nan, dtype=np.float32)
        valid = c > 0
        score[valid] = s[valid] / c[valid]
        cols[assay] = score
    score_df = pd.DataFrame(cols)
    corr = score_df.corr(method="spearman", min_periods=200)
    if corr.empty:
        return
    data = corr.to_numpy(dtype=float)
    cmap = cm.RdBu_r.copy()
    cmap.set_bad("#E0E0E0")
    n = len(corr.columns)
    fig_w = max(5.5, 1.2 + 0.8 * n)
    fig_h = max(5.0, 1.0 + 0.8 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(n));  ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns.tolist(), rotation=35, ha="right")
    ax.set_yticklabels(corr.index.tolist())
    ax.set_title(f"{ct_label} | {model} | assay-assay Spearman corr ({score_type})\n(all tracks, no biosample filter)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Spearman correlation")
    cell_font = max(5, min(9, 90 // n))
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=cell_font, color="white" if v > 0.5 or v < -0.2 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(plot_data_path(out_path), index=True)


def plot_assay_scatter_grid(acc, gt_log2fc, gt_sig, ct_label, model, score_type, out_path):
    assay_score_df = assay_score_matrix(acc, score_type)
    assays = list(assay_score_df.columns)
    n_assays = len(assays)
    if n_assays == 0:
        return
    col_c = MODEL_COLORS.get(model, "#607D8B")
    finite_lfc = np.isfinite(gt_log2fc)
    sig = gt_sig.astype(bool) & finite_lfc

    row_defs = [
        ("All",                         lambda: np.ones(len(gt_log2fc), bool),  col_c,     "#B0BEC5"),
        ("Sig only (α=0.1)",             lambda: sig,                             col_c,     col_c),
        ("Pred < 0",                     None,                                    "#C62828",  None),
        ("Pred > 0",                     None,                                    "#1B5E20",  None),
        ("logFC < 0",                    lambda: finite_lfc & (gt_log2fc < 0),    "#E65100",  None),
        ("logFC > 0",                    lambda: finite_lfc & (gt_log2fc > 0),    "#1565C0",  None),
        ("Sig + logFC < 0",              lambda: sig & (gt_log2fc < 0),           "#BF360C",  None),
        ("Sig + logFC > 0",              lambda: sig & (gt_log2fc > 0),           "#0D47A1",  None),
    ]
    n_rows = len(row_defs)
    fig, axes = plt.subplots(n_rows, n_assays,
                             figsize=(3.5 * n_assays, 3.5 * n_rows),
                             squeeze=False)
    for ci, assay in enumerate(assays):
        y_full = assay_score_df[assay].to_numpy(dtype=np.float32)
        for ri, (row_label, mask_fn, c_main, c_no_sig) in enumerate(row_defs):
            ax = axes[ri][ci]
            if ri in (2, 3):   # pred-based mask
                pred_mask = (y_full < 0) if ri == 2 else (y_full > 0)
                mask = finite_lfc & pred_mask
            elif mask_fn is not None:
                mask = mask_fn()
            else:
                mask = np.zeros(len(gt_log2fc), bool)
            valid = mask & np.isfinite(y_full)
            x_v, y_v = gt_log2fc[valid], y_full[valid]
            if len(x_v) >= 10:
                if ri == 0 and c_no_sig is not None:
                    sig_v = sig[valid]
                    ax.scatter(x_v[~sig_v], y_v[~sig_v], s=2, alpha=0.3, color="#B0BEC5", rasterized=True)
                    ax.scatter(x_v[sig_v],  y_v[sig_v],  s=3, alpha=0.5, color=c_main,   rasterized=True)
                else:
                    ax.scatter(x_v, y_v, s=2, alpha=0.4, color=c_main, rasterized=True)
                sp, _ = spearmanr(x_v, y_v)
                ax.set_title(f"ρ={sp:.3f}  n={len(x_v):,}", fontsize=7)
            else:
                ax.set_title("n<10", fontsize=7)
            ax.axhline(0, color="grey", lw=0.5, ls="--")
            ax.axvline(0, color="grey", lw=0.5, ls="--")
            if ri == 0:
                ax.set_xlabel(f"{assay}", fontsize=8, labelpad=1)
            if ci == 0:
                ax.set_ylabel(f"{row_label}\nSAD score", fontsize=7)
    fig.suptitle(f"{ct_label} | {model} | {score_type} (all tracks, no biosample filter)", fontsize=10, y=1.001)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_alltrack_assay_barchart(alltrack_df, ct_label, out_dir, ct,
                                 bar_width_scale=1.0, fig_width=None, fig_height=4.0):
    if alltrack_df.empty:
        return
    assays  = sorted(alltrack_df["assay_type"].unique())
    models  = [m for m in TRACK_MODELS if m in alltrack_df["model"].unique()]
    n_assays, n_models = len(assays), len(models)
    if n_assays == 0 or n_models == 0:
        return
    assay_idx = {a: i for i, a in enumerate(assays)}
    width    = (0.8 / n_models) * bar_width_scale
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * (0.8 / n_models)

    for metric, metric_label, suffix in [
        ("spearman_r",   "Spearman ρ (all variants)",        "all"),
        ("spearman_sig", "Spearman ρ (sig only, α=0.1)",     "sig"),
    ]:
        _fw = fig_width if fig_width is not None else max(4, 1.0 * n_assays + 1.5 * n_models)
        fig, ax = plt.subplots(figsize=(_fw, fig_height))
        ax.axhline(0, color="black", lw=0.7)
        for mi, model in enumerate(models):
            mdf = alltrack_df[alltrack_df["model"] == model]
            xs, ys = [], []
            for assay in assays:
                row = mdf[mdf["assay_type"] == assay]
                val = float(row[metric].values[0]) if len(row) > 0 and np.isfinite(row[metric].values[0]) else 0.0
                xs.append(assay_idx[assay] + offsets[mi])
                ys.append(val)
            bars = ax.bar(xs, ys, width=width * 0.9,
                          color=MODEL_COLORS.get(model, "#607D8B"),
                          label=model, alpha=0.88, edgecolor="white", linewidth=0.4)
            for bar, y in zip(bars, ys):
                if y != 0.0:
                    va = "bottom" if y >= 0 else "top"
                    ax.text(bar.get_x() + bar.get_width() / 2, y + (0.005 if y >= 0 else -0.005),
                            f"{y:.2f}", ha="center", va=va, fontsize=7.5)
        ax.set_xticks(np.arange(n_assays))
        ax.set_xticklabels(assays, fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_xlabel("Assay type  (all tracks, no biosample filter)", fontsize=9)
        ax.set_title(f"{ct_label} | Per-assay model performance — {metric_label}", fontsize=10)
        ax.legend(title="Model", fontsize=9, title_fontsize=9)
        ax.set_ylim(min(-0.05, ax.get_ylim()[0] - 0.05), max(0.75, ax.get_ylim()[1] + 0.08))
        plt.tight_layout()
        out = out_dir / f"{ct}_alltrack_assay_barchart_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    for ct in args.cell_types:
        cfg   = EXPERIMENTS[ct]
        label = cfg["label"]
        _dir  = "analysis_sigonly" if args.sig_only else "analysis"
        out_dir = BASE / f"results/{ct}/{_dir}"
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_root = Path(args.parquet_dir) / ct if args.parquet_dir else BASE / "results" / ct

        print(f"\n{'='*60}")
        print(f"{label}  →  {out_dir}")

        gt = load_gt(cfg)
        log2fc = gt["log2FC"].to_numpy(dtype=np.float32)
        sig_arr = gt["sig"].to_numpy(dtype=bool)
        sig_mask = None
        if args.sig_only:
            sig_mask = sig_arr.copy()
            log2fc  = log2fc[sig_mask]
            sig_arr = sig_arr[sig_mask]
        gt_idx = {v: i for i, v in enumerate(gt["variant_id"].values)}

        alltrack_rows = []

        for model in TRACK_MODELS:
            parquet = parquet_root / model / f"{model}_scores.parquet"
            if not parquet.exists():
                print(f"  SKIP {model}: {parquet} not found")
                continue

            print(f"\n  {model} — building all-track assay accumulator ...", flush=True)
            acc = build_all_assay_acc(parquet, gt_idx, len(gt), args.batch_size)

            # Apply sig_only mask to accumulator arrays
            if sig_mask is not None:
                for v in acc.values():
                    for k in list(v):
                        if isinstance(v[k], np.ndarray):
                            v[k] = v[k][sig_mask]

            # Collect per-assay Spearman metrics
            for assay, v in acc.items():
                cnt   = v["count"]
                valid = cnt > 0
                score = np.full(len(log2fc), np.nan, dtype=np.float32)
                score[valid] = v["sum"][valid] / cnt[valid]
                m = compute_metrics(log2fc, sig_arr, score)
                if m is not None:
                    alltrack_rows.append({"model": model, "assay_type": assay, **m})

            # Per-model figures
            for st in ["mean_sad", "mean_abs_sad"]:
                out = out_dir / f"{ct}_{model}_alltrack_assay_correlation_{st}_unified.png"
                plot_assay_correlation_heatmap(acc, label, model, st, out)
                print(f"  Saved: {out}")
                out = out_dir / f"{ct}_{model}_alltrack_assay_scatter_{st}_unified.png"
                plot_assay_scatter_grid(acc, log2fc, sig_arr, label, model, st, out)
                print(f"  Saved: {out}")

        if alltrack_rows:
            df = pd.DataFrame(alltrack_rows)
            csv_out = out_dir / f"{ct}_alltrack_assay_metrics_unified.csv"
            df.to_csv(csv_out, index=False)
            print(f"\n  Saved: {csv_out}")
            plot_alltrack_assay_barchart(df, label, out_dir, ct,
                                         bar_width_scale=args.bar_width_scale,
                                         fig_width=args.fig_width,
                                         fig_height=args.fig_height)


def regen_barcharts(args):
    """Re-plot barcharts from existing CSVs — no parquet scan needed."""
    for ct in args.cell_types:
        cfg   = EXPERIMENTS[ct]
        label = cfg["label"]
        _dir  = "analysis_sigonly" if args.sig_only else "analysis"
        src_dir = BASE / f"results/{ct}/{_dir}"
        csv_path = src_dir / f"{ct}_alltrack_assay_metrics_unified.csv"
        if not csv_path.exists():
            print(f"  SKIP {ct}: CSV not found at {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        out_dir = src_dir / args.barchart_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {label} → {out_dir}")
        plot_alltrack_assay_barchart(df, label, out_dir, ct,
                                     bar_width_scale=args.bar_width_scale,
                                     fig_width=args.fig_width,
                                     fig_height=args.fig_height)


def parse_args():
    p = argparse.ArgumentParser(description="Generate all-track assay figures (fast, no biosample filter)")
    p.add_argument("--cell-types", nargs="+", default=["hepg2", "hek293t", "ngn2"],
                   choices=list(EXPERIMENTS.keys()))
    p.add_argument("--parquet-dir", type=str, default="",
                   help="Root dir for parquets: {parquet-dir}/{ct}/{model}/{model}_scores.parquet")
    p.add_argument("--batch-size", type=int, default=2_000_000)
    p.add_argument("--sig-only", action="store_true",
                   help="Restrict metrics and plots to significant variants (α=0.1 / |lfc|>0.5)")
    p.add_argument("--bar-width-scale", type=float, default=1.0,
                   help="Scale factor for bar width (e.g. 0.5 = half width). Default: 1.0")
    p.add_argument("--fig-width", type=float, default=None,
                   help="Figure width in inches. Default: auto (scales with n_assays × n_models).")
    p.add_argument("--fig-height", type=float, default=4.0,
                   help="Figure height in inches. Default: 4.0")
    p.add_argument("--barchart-subdir", type=str, default="manuscript",
                   help="Subdirectory inside analysis/ to write re-generated barcharts. Default: manuscript")
    p.add_argument("--regen-barcharts", action="store_true",
                   help="Skip parquet scanning; re-plot barcharts from existing CSVs only.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.regen_barcharts:
        regen_barcharts(args)
    else:
        run(args)
