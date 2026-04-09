#!/usr/bin/env python3
"""
Unified cell-type analysis for sequence-to-function track models.

Goals:
- Single script with command-line switches to enable/disable analysis blocks.
- Single parquet pass per celltype x model to avoid repeated loading.
- Optional detailed biosample/assay analysis from the same pass.
- Explicit control of near-zero downsampling used in scatter plots.

Analyses:
- baseline: all tracks aggregated per variant (mean_sad, max_abs_sad)
- filtered: only cell-type matched biosamples aggregated per variant
- detailed: per-biosample, per-assay, and biosample x assay metrics (filtered only)
"""

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


BASE = Path("/mnt/c/Users/user/Desktop/Benchmark_results/Pipeline_v2")
VCF_DIR = BASE / "Data/VCF"
TMP_ROOT = Path("/tmp/mpra_analysis")

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
        "qval_thr": 1.0,  # alpha=0.1 → -log10(0.1)=1.0
        "label": "HEK293T",
    },
    "hepg2": {
        "vcf": VCF_DIR / "IGVFFI4378PZYI.vcf",
        "has_qval": True,
        "qval_thr": 1.0,  # alpha=0.1 → -log10(0.1)=1.0
        "label": "HepG2",
    },
}

TRACK_MODELS = ["enformer", "basenji", "alphagenome"]

MODEL_COLORS = {
    "enformer": "#1565C0",
    "basenji": "#0288D1",
    "alphagenome": "#00796B",
}

BIOSAMPLE_FILTERS = {
    "hepg2": {
        "enformer": "hepg2",
        "basenji": "hepg2",
        "alphagenome": "hepg2",
    },
    "hek293t": {
        "enformer": "hek293",
        "basenji": "hek293",
        "alphagenome": "hek293",
    },
    "ngn2": {
        "enformer": r"neuron|neural|glutamat",
        "basenji": r"neuron|neural|glutamat",
        "alphagenome": r"neuron|neural|glutamat",
    },
}

DEFAULT_EXCLUDE_BIOSAMPLE_REGEX = (
    r"genetically modified|crispr|stable transfection|transduction|"
    r"knockout|\bko\b|overexpress|inducible|3xflag|tagged"
)

_LOG2FC_RE = re.compile(r"LOG2FC=([^;]+)")
_QVAL_RE = re.compile(r"QVAL=([^;]+)")

SCORE_DEFINITIONS = {
    "mean_sad": "mean(SAD)",
    "mean_abs_sad": "mean(abs(SAD))",
    "max_abs_sad": "max(abs(SAD))",
}


def plot_data_path(out_path):
    return out_path.with_name(f"{out_path.stem}__plot_data.csv")


def attach_score_definition(df):
    if "score_type" in df.columns:
        df = df.copy()
        df["score_definition"] = df["score_type"].map(SCORE_DEFINITIONS).fillna("")
    return df


def _roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    return float(np.trapezoid(tps / n_pos, fps / n_neg))


def _avg_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.sum() == 0:
        return np.nan
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    prec = tp / np.arange(1, len(y_true) + 1)
    return float(np.sum(prec * y_true) / y_true.sum())


def compute_metrics(log2fc, sig, score, use_abs_lfc=False):
    mask = np.isfinite(log2fc) & np.isfinite(score)
    x = log2fc[mask]
    y = score[mask]
    s = sig[mask]

    if len(x) < 10:
        return None

    # For magnitude-based scores (mean_abs_SAD, max_abs_SAD) correlate against |log2FC|.
    # Signed target would give near-zero correlation even for a perfect magnitude predictor.
    x_corr = np.abs(x) if use_abs_lfc else x

    sp_all, _ = spearmanr(x_corr, y)
    sp_sig = np.nan
    if s.sum() >= 10:
        sp_sig, _ = spearmanr(x_corr[s], y[s])

    # Directional accuracy only meaningful for signed scores
    dir_acc = np.nan
    if not use_abs_lfc and s.sum() >= 10:
        dir_acc = float(np.mean(np.sign(x[s]) == np.sign(y[s])))

    return {
        "n": int(len(x)),
        "n_sig": int(s.sum()),
        "spearman_r": float(sp_all),
        "spearman_sig": float(sp_sig),
        "auroc": float(_roc_auc(s.astype(float), np.abs(y))),
        "auprc": float(_avg_precision(s.astype(float), np.abs(y))),
        "directional_accuracy": dir_acc,
    }


def load_gt(cfg):
    records = []
    with open(cfg["vcf"]) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
            info = cols[7] if len(cols) > 7 else ""
            # Normalise: strip 'chr' prefix so gt_idx is always chr-free
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


def empty_acc(n):
    return {
        "sum": np.zeros(n, dtype=np.float32),
        "abs_sum": np.zeros(n, dtype=np.float32),
        "count": np.zeros(n, dtype=np.uint16),
        "max_abs": np.zeros(n, dtype=np.float32),
    }


def update_acc(acc, idx, raw_score):
    finite = np.isfinite(raw_score)
    if not finite.any():
        return
    idx = idx[finite]
    raw_score = raw_score[finite]

    np.add.at(acc["sum"], idx, raw_score)
    np.add.at(acc["count"], idx, 1)
    abs_score = np.abs(raw_score)
    np.add.at(acc["abs_sum"], idx, abs_score)
    np.maximum.at(acc["max_abs"], idx, abs_score)


def scores_from_acc(acc, mode):
    valid = acc["count"] > 0
    out = np.full_like(acc["sum"], np.nan, dtype=np.float32)
    if mode == "mean_sad":
        out[valid] = acc["sum"][valid] / acc["count"][valid]
    elif mode == "mean_abs_sad":
        out[valid] = acc["abs_sum"][valid] / acc["count"][valid]
    elif mode == "max_abs_sad":
        out[valid] = acc["max_abs"][valid]
    else:
        raise ValueError(f"unknown score mode: {mode}")
    return out


def balanced_sample_idx(log2fc, n_per_bin=400, n_bins=30):
    """Downsample near-zero dense region by equalizing |log2FC| quantile bins."""
    df = pd.DataFrame({"x": log2fc})
    df["abs_x"] = df["x"].abs()
    bins = pd.qcut(df["abs_x"], q=n_bins, duplicates="drop", labels=False)
    keep = []
    for _, g in df.groupby(bins, observed=True):
        keep.append(g.sample(min(n_per_bin, len(g)), random_state=42).index.values)
    if not keep:
        return np.arange(len(df))
    return np.concatenate(keep)


def plot_scatter(log2fc, score, sig, title, out_path, use_downsample, n_bins, n_per_bin):
    mask = np.isfinite(log2fc) & np.isfinite(score)
    x = log2fc[mask]
    y = score[mask]
    s = sig[mask]
    if len(x) == 0:
        return

    if use_downsample:
        idx = balanced_sample_idx(x, n_per_bin=n_per_bin, n_bins=n_bins)
        x, y, s = x[idx], y[idx], s[idx]
        sample_note = f"Downsampled by |log2FC| quantile bins: bins={n_bins}, max/bin={n_per_bin}"
    else:
        sample_note = "No downsampling"

    sp, _ = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    colors = np.where(s, "#D32F2F", "#B0BEC5")
    ax.scatter(x, y, c=colors, s=8, alpha=0.55, linewidths=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("MPRA log2FC")
    ax.set_ylabel("Predicted score")
    ax.set_title(f"{title}\nSpearman={sp:.4f}\n{sample_note}", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "log2FC": x,
        "predicted_score": y,
        "sig": s.astype(int),
        "used_downsampling": int(use_downsample),
        "scatter_bins": int(n_bins),
        "scatter_per_bin": int(n_per_bin),
    }).to_csv(plot_data_path(out_path), index=False)


# ---------------------------------------------------------------------------
# Effect-size bin analysis (Q1.5)
# ---------------------------------------------------------------------------

def compute_bin_metrics(log2fc, sig, score, n_bins=10, use_abs_lfc=False):
    """Return a DataFrame with Spearman per |log2FC| quantile bin."""
    mask = np.isfinite(log2fc) & np.isfinite(score)
    x, y, s = log2fc[mask], score[mask], sig[mask]
    abs_x = np.abs(x)
    x_corr = abs_x if use_abs_lfc else x
    bin_edges = np.quantile(abs_x, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-9  # include the minimum
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        sel = (abs_x > lo) & (abs_x <= hi)
        if sel.sum() < 10:
            continue
        sp, _ = spearmanr(x_corr[sel], y[sel])
        rows.append({
            "bin": i + 1,
            "abs_log2fc_lo": float(lo),
            "abs_log2fc_hi": float(hi),
            "abs_log2fc_mid": float((lo + hi) / 2),
            "n": int(sel.sum()),
            "n_sig": int(s[sel].sum()),
            "spearman_r": float(sp),
        })
    return pd.DataFrame(rows)


def plot_bin_metrics(bin_df, out_path, title=""):
    if bin_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bin_df["abs_log2fc_mid"], bin_df["spearman_r"], marker="o", color="steelblue")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("|log2FC| bin midpoint")
    ax.set_ylabel("Spearman r")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    bin_df.to_csv(plot_data_path(out_path), index=False)


def aggregate_single_pass(
    parquet_path,
    gt_index,
    n_variants,
    bio_pattern,
    run_filtered,
    run_detailed,
    batch_size,
    include_regex=None,
    exclude_regex=None,
    allowed_assays=None,
):
    """One parquet pass that can feed baseline + filtered + detailed analyses."""
    pf = pq.ParquetFile(parquet_path)

    all_acc = empty_acc(n_variants)
    filt_acc = empty_acc(n_variants) if run_filtered else None

    matched_biosamples = set()
    n_passed_filtered_rows = 0
    n_excluded_by_name = 0
    n_excluded_by_assay = 0
    pair_acc = {} if run_detailed else None
    all_assay_acc = {} if run_detailed else None  # keyed by assay_type, no biosample filter

    cols = ["variant_id", "raw_score"]
    if run_filtered or run_detailed:
        cols += ["biosample_name", "assay_type"]

    t0 = time.time()
    for n_batches_, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols)):
        df = batch.to_pandas()

        # Normalise: strip 'chr' prefix so parquet IDs match gt_index keys
        df["variant_id"] = df["variant_id"].str.replace(r"^chr", "", regex=True)
        vidx = df["variant_id"].map(gt_index)
        keep = vidx.notna()
        if not keep.any():
            continue

        df = df.loc[keep].copy()
        df["variant_idx"] = vidx.loc[keep].astype(np.int32)

        idx_all = df["variant_idx"].to_numpy(dtype=np.int32, copy=False)
        raw_all = df["raw_score"].to_numpy(dtype=np.float32, copy=False)
        update_acc(all_acc, idx_all, raw_all)

        if run_filtered or run_detailed:
            # Normalise CHIP_HISTONE → CHIP so AlphaGenome tracks merge with Basenji/Enformer
            df["assay_type"] = df["assay_type"].str.replace("CHIP_HISTONE", "CHIP", regex=False)
            # --- All-track assay accumulation (no biosample filter) ---
            if run_detailed:
                adf = df[["variant_idx", "assay_type", "raw_score"]].copy()
                adf["assay_type"] = adf["assay_type"].fillna("UNKNOWN")
                _finite_mask = np.isfinite(adf["raw_score"].to_numpy(dtype=np.float32))
                adf = adf.loc[_finite_mask]
                if not adf.empty:
                    adf = adf.copy()
                    adf["abs_score"] = adf["raw_score"].abs()
                    _agg = (
                        adf.groupby(["assay_type", "variant_idx"], sort=False, observed=True)
                        .agg(raw_sum=("raw_score", "sum"), abs_sum=("abs_score", "sum"), cnt=("raw_score", "count"))
                        .reset_index()
                    )
                    for _assay, _kdf in _agg.groupby("assay_type", sort=False):
                        _key = str(_assay)
                        if _key not in all_assay_acc:
                            all_assay_acc[_key] = {
                                "sum": np.zeros(n_variants, dtype=np.float32),
                                "abs_sum": np.zeros(n_variants, dtype=np.float32),
                                "count": np.zeros(n_variants, dtype=np.uint16),
                            }
                        _vidx = _kdf["variant_idx"].to_numpy(dtype=np.int32)
                        all_assay_acc[_key]["sum"][_vidx] += _kdf["raw_sum"].to_numpy(dtype=np.float32)
                        all_assay_acc[_key]["abs_sum"][_vidx] += _kdf["abs_sum"].to_numpy(dtype=np.float32)
                        all_assay_acc[_key]["count"][_vidx] += _kdf["cnt"].to_numpy(dtype=np.uint16)

            bio_mask = df["biosample_name"].str.contains(bio_pattern, case=False, na=False, regex=True)
            if bio_mask.any():
                fdf = df.loc[bio_mask].copy()

                if include_regex:
                    keep_inc = fdf["biosample_name"].str.contains(include_regex, case=False, na=False, regex=True)
                    fdf = fdf.loc[keep_inc]

                if exclude_regex:
                    drop_exc = fdf["biosample_name"].str.contains(exclude_regex, case=False, na=False, regex=True)
                    n_excluded_by_name += int(drop_exc.sum())
                    fdf = fdf.loc[~drop_exc]

                if allowed_assays:
                    assay_series = fdf["assay_type"].fillna("UNKNOWN").astype(str).str.upper()
                    keep_assay = assay_series.isin(allowed_assays)
                    n_excluded_by_assay += int((~keep_assay).sum())
                    fdf = fdf.loc[keep_assay]

                if fdf.empty:
                    continue

                n_passed_filtered_rows += len(fdf)
                matched_biosamples.update(fdf["biosample_name"].dropna().unique())

                if run_filtered:
                    idx_f = fdf["variant_idx"].to_numpy(dtype=np.int32, copy=False)
                    raw_f = fdf["raw_score"].to_numpy(dtype=np.float32, copy=False)
                    update_acc(filt_acc, idx_f, raw_f)

                if run_detailed:
                    fdf = fdf.copy()
                    fdf["assay_type"] = fdf["assay_type"].fillna("UNKNOWN")
                    # Pre-filter non-finite scores before aggregation
                    fdf = fdf[np.isfinite(fdf["raw_score"].to_numpy(dtype=np.float32))]
                    if not fdf.empty:
                        fdf["abs_score"] = fdf["raw_score"].abs()
                        # Single vectorized groupby replaces the per-(bio,assay) Python loop.
                        # After groupby each (bio, assay, variant_idx) key is unique so direct
                        # indexing is safe and ~10x faster than np.add.at.
                        agg = (
                            fdf.groupby(
                                ["biosample_name", "assay_type", "variant_idx"],
                                sort=False,
                                observed=True,
                            )
                            .agg(
                                raw_sum=("raw_score", "sum"),
                                abs_sum=("abs_score", "sum"),
                                cnt=("raw_score", "count"),
                            )
                            .reset_index()
                        )
                        for (bio, assay), kdf in agg.groupby(
                            ["biosample_name", "assay_type"], sort=False
                        ):
                            key = (str(bio), str(assay))
                            if key not in pair_acc:
                                pair_acc[key] = {
                                    "sum": np.zeros(n_variants, dtype=np.float32),
                                    "abs_sum": np.zeros(n_variants, dtype=np.float32),
                                    "count": np.zeros(n_variants, dtype=np.uint16),
                                }
                            vidx = kdf["variant_idx"].to_numpy(dtype=np.int32)
                            pair_acc[key]["sum"][vidx] += kdf["raw_sum"].to_numpy(dtype=np.float32)
                            pair_acc[key]["abs_sum"][vidx] += kdf["abs_sum"].to_numpy(dtype=np.float32)
                            pair_acc[key]["count"][vidx] += kdf["cnt"].to_numpy(dtype=np.uint16)

        n_batches = n_batches_ + 1
        if n_batches % 20 == 0:
            print(f"    batch {n_batches}: {time.time()-t0:.0f}s elapsed", flush=True)

    stats = {
        "passed_filtered_rows": int(n_passed_filtered_rows),
        "excluded_by_name": int(n_excluded_by_name),
        "excluded_by_assay": int(n_excluded_by_assay),
    }
    return all_acc, filt_acc, matched_biosamples, pair_acc, all_assay_acc, stats


def pair_metrics_df(pair_acc, gt_log2fc, gt_sig, model, score_types=("mean_sad", "mean_abs_sad")):
    rows = []
    for (biosample_name, assay_type), v in pair_acc.items():
        cnt = v["count"]
        valid = cnt > 0
        if valid.sum() < 10:
            continue
        for st in score_types:
            score = np.full_like(gt_log2fc, np.nan, dtype=np.float32)
            if st == "mean_sad":
                score[valid] = v["sum"][valid] / cnt[valid]
            elif st == "mean_abs_sad":
                score[valid] = v["abs_sum"][valid] / cnt[valid]
            else:
                continue
            _use_abs = (st != "mean_sad")
            m = compute_metrics(gt_log2fc, gt_sig, score, use_abs_lfc=_use_abs)
            if m is None:
                continue
            m.update({
                "model": model,
                "score_type": st,
                "biosample_name": biosample_name,
                "assay_type": assay_type,
            })
            rows.append(m)
    return pd.DataFrame(rows)


def grouped_metrics_from_pair(
    pair_acc,
    gt_log2fc,
    gt_sig,
    model,
    by="biosample_name",
    score_types=("mean_sad", "mean_abs_sad"),
):
    pos = 0 if by == "biosample_name" else 1
    groups = sorted({k[pos] for k in pair_acc.keys()})
    n = len(gt_log2fc)

    rows = []
    for g in groups:
        sum_acc = np.zeros(n, dtype=np.float32)
        abs_sum_acc = np.zeros(n, dtype=np.float32)
        cnt_acc = np.zeros(n, dtype=np.uint16)
        for key, v in pair_acc.items():
            if key[pos] == g:
                sum_acc += v["sum"]
                abs_sum_acc += v["abs_sum"]
                cnt_acc += v["count"]
        valid = cnt_acc > 0
        if valid.sum() < 10:
            continue
        for st in score_types:
            score = np.full(n, np.nan, dtype=np.float32)
            if st == "mean_sad":
                score[valid] = sum_acc[valid] / cnt_acc[valid]
            elif st == "mean_abs_sad":
                score[valid] = abs_sum_acc[valid] / cnt_acc[valid]
            else:
                continue
            _use_abs = (st != "mean_sad")
            m = compute_metrics(gt_log2fc, gt_sig, score, use_abs_lfc=_use_abs)
            if m is None:
                continue
            m.update({"model": model, "score_type": st, by: g})
            rows.append(m)

    return pd.DataFrame(rows)


def assay_score_matrix_from_pair(pair_acc, score_type="mean_sad"):
    assays = sorted({k[1] for k in pair_acc.keys()})
    if not assays:
        return pd.DataFrame()

    n = len(next(iter(pair_acc.values()))["count"])
    cols = {}
    for assay in assays:
        sum_acc = np.zeros(n, dtype=np.float32)
        cnt_acc = np.zeros(n, dtype=np.uint16)
        for (bio, a), v in pair_acc.items():
            if a == assay:
                if score_type == "mean_sad":
                    sum_acc += v["sum"]
                elif score_type == "mean_abs_sad":
                    sum_acc += v["abs_sum"]
                cnt_acc += v["count"]
        score = np.full(n, np.nan, dtype=np.float32)
        valid = cnt_acc > 0
        score[valid] = sum_acc[valid] / cnt_acc[valid]
        cols[assay] = score

    return pd.DataFrame(cols)


def assay_score_matrix_from_assay_acc(all_assay_acc, score_type="mean_sad"):
    """Build per-variant score matrix from all_assay_acc (keyed by assay_type, no biosample filter)."""
    if not all_assay_acc:
        return pd.DataFrame()
    n = len(next(iter(all_assay_acc.values()))["count"])
    cols = {}
    for assay in sorted(all_assay_acc):
        v = all_assay_acc[assay]
        cnt = v["count"]
        valid = cnt > 0
        score = np.full(n, np.nan, dtype=np.float32)
        if score_type == "mean_sad":
            score[valid] = v["sum"][valid] / cnt[valid]
        elif score_type == "mean_abs_sad":
            score[valid] = v["abs_sum"][valid] / cnt[valid]
        cols[assay] = score
    return pd.DataFrame(cols)


def plot_assay_scatter_grid(assay_score_df, gt_log2fc, gt_sig, ct_label, model, score_type, out_path):
    """8-row × n_assays scatter grid.
    Row 0: all variants (sig=model colour, non-sig=grey).
    Row 1: significant variants only (alpha=0.1).
    Row 2: negative predictions only (score < 0).
    Row 3: positive predictions only (score > 0).
    Row 4: negative logFC only (log2FC < 0).
    Row 5: positive logFC only (log2FC > 0).
    Row 6: sig & negative logFC (log2FC < 0).
    Row 7: sig & positive logFC (log2FC > 0).
    x = MPRA log2FC,  y = per-assay mean SAD score.
    """
    assays = [c for c in assay_score_df.columns]
    n_assays = len(assays)
    if n_assays == 0:
        return

    finite_lfc = np.isfinite(gt_log2fc)
    sig = gt_sig.astype(bool) & finite_lfc
    col_c = MODEL_COLORS.get(model, "#607D8B")

    # (row_label, filter_key, scatter_colour)
    row_defs = [
        ("All variants",                              "all",         col_c),
        ("Sig-only α=0.1",                            "sig",         col_c),
        ("Negative predictions\n(score < 0)",         "neg_score",   "#C62828"),
        ("Positive predictions\n(score > 0)",         "pos_score",   "#1B5E20"),
        ("Negative logFC\n(log2FC < 0)",              "neg_lfc",     "#E65100"),
        ("Positive logFC\n(log2FC > 0)",              "pos_lfc",     "#1565C0"),
        ("Sig + Negative logFC\n(sig & log2FC < 0)",  "sig_neg_lfc", "#BF360C"),
        ("Sig + Positive logFC\n(sig & log2FC > 0)",  "sig_pos_lfc", "#0D47A1"),
    ]
    n_rows = len(row_defs)

    fig, axes = plt.subplots(n_rows, n_assays,
                             figsize=(3.5 * n_assays, 3.5 * n_rows), squeeze=False)

    for row, (row_label, filt, row_col) in enumerate(row_defs):
        for col, assay in enumerate(assays):
            ax = axes[row, col]
            scores = assay_score_df[assay].to_numpy(dtype=np.float32)
            valid = finite_lfc & np.isfinite(scores)
            x_v = gt_log2fc[valid]
            y_v = scores[valid]
            s_v = sig[valid]

            xp, yp, sp = x_v, y_v, np.nan  # defaults
            if filt == "all":
                xp, yp = x_v, y_v
                ax.scatter(x_v[~s_v], y_v[~s_v], c="#BDBDBD", s=4, alpha=0.25, linewidths=0, rasterized=True)
                ax.scatter(x_v[s_v],  y_v[s_v],  c=row_col,  s=6, alpha=0.65, linewidths=0, rasterized=True)
                sp = spearmanr(x_v[s_v], y_v[s_v])[0] if s_v.sum() >= 5 else np.nan
            elif filt == "sig":
                mask = s_v
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=6, alpha=0.55, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan
            elif filt == "neg_score":
                mask = y_v < 0
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=5, alpha=0.5, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan
            elif filt == "pos_score":
                mask = y_v > 0
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=5, alpha=0.5, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan
            elif filt == "neg_lfc":
                mask = x_v < 0
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=5, alpha=0.5, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan
            elif filt == "pos_lfc":
                mask = x_v > 0
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=5, alpha=0.5, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan
            elif filt == "sig_neg_lfc":
                mask = s_v & (x_v < 0)
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=6, alpha=0.55, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan
            elif filt == "sig_pos_lfc":
                mask = s_v & (x_v > 0)
                xp, yp = x_v[mask], y_v[mask]
                ax.scatter(xp, yp, c=row_col, s=6, alpha=0.55, linewidths=0, rasterized=True)
                sp = spearmanr(xp, yp)[0] if len(xp) >= 5 else np.nan

            n_pts = len(xp)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax.axvline(0, color="k", lw=0.5, alpha=0.3)
            sp_str = f"{sp:.3f}" if np.isfinite(sp) else "n/a"
            ax.set_title(f"{assay}\nρ={sp_str} (n={n_pts:,})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("MPRA log2FC", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{row_label}\n{score_type}", fontsize=7)

    fig.suptitle(f"{ct_label} | {model} | Per-assay scatter ({score_type})", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_df = assay_score_df.copy()
    out_df.insert(0, "log2FC", gt_log2fc)
    out_df.insert(1, "sig", gt_sig.astype(int))
    out_df.to_csv(plot_data_path(out_path), index=False)


def plot_assay_correlation_heatmap(pair_acc, ct_label, model, score_type, out_path):
    score_df = assay_score_matrix_from_pair(pair_acc, score_type=score_type)
    if score_df.shape[1] < 2:
        return

    score_df.to_csv(plot_data_path(out_path).with_name(f"{out_path.stem}__assay_scores.csv"), index=False)

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
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns.tolist(), rotation=35, ha="right")
    ax.set_yticklabels(corr.index.tolist())
    ax.set_title(f"{ct_label} | {model} | assay-assay Spearman corr ({score_type})", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Spearman correlation")
    # Annotate each cell with its value
    font_size = max(5, min(9, 90 // n))
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=font_size, color="white" if abs(v) > 0.55 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    corr.to_csv(plot_data_path(out_path), index=True)


def plot_biosample_vs_baseline(bio_df, all_metric, filt_metric, ct_label, model, out_path):
    if bio_df.empty:
        return
    show = bio_df.sort_values("spearman_sig", ascending=False).head(30)

    fig_h = max(6, 0.28 * len(show) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(show))
    ax.barh(y, show["spearman_sig"].values, color=MODEL_COLORS.get(model, "#607D8B"), alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(show["biosample_name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman(sig)")
    ax.set_title(f"{ct_label} | {model} | Biosample Spearman(sig) vs baselines")

    if np.isfinite(all_metric):
        ax.axvline(all_metric, color="#546E7A", lw=1.5, ls="--", label=f"All-track mean: {all_metric:.3f}")
    if np.isfinite(filt_metric):
        ax.axvline(filt_metric, color="#E65100", lw=1.5, ls="-.", label=f"CT-filtered mean: {filt_metric:.3f}")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_df = show.copy()
    out_df["baseline_all_track_mean_spearman_sig"] = all_metric
    out_df["baseline_ct_filtered_mean_spearman_sig"] = filt_metric
    out_df.to_csv(plot_data_path(out_path), index=False)


def plot_assay_performance(assay_df, ct_label, model, score_type, out_path):
    if assay_df.empty:
        return
    show = assay_df.sort_values("spearman_sig", ascending=False).copy()
    if len(show) == 0:
        return

    fig_h = max(4.5, 0.45 * len(show) + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, fig_h), sharey=True)

    y = np.arange(len(show))
    axes[0].barh(y, show["spearman_r"].values, color="#607D8B", alpha=0.85)
    axes[0].set_title("Spearman(all)")
    axes[0].set_xlabel("Correlation")

    axes[1].barh(y, show["spearman_sig"].values, color=MODEL_COLORS.get(model, "#607D8B"), alpha=0.85)
    axes[1].set_title("Spearman(sig)")
    axes[1].set_xlabel("Correlation")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(show["assay_type"].tolist())
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(show["assay_type"].tolist())
    axes[0].invert_yaxis()

    fig.suptitle(f"{ct_label} | {model} | assay performance ({score_type})", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    show.to_csv(plot_data_path(out_path), index=False)


def plot_pair_heatmap(pair_df, ct_label, model, out_path):
    if pair_df.empty:
        return
    piv = pair_df.pivot_table(index="biosample_name", columns="assay_type", values="spearman_sig", aggfunc="mean")
    if piv.empty:
        return

    rank = piv.max(axis=1, skipna=True).fillna(-np.inf)
    piv = piv.loc[rank.sort_values(ascending=False).index]
    if len(piv) > 60:
        piv = piv.iloc[:60]

    data = piv.to_numpy(dtype=float)
    cmap = cm.RdBu_r.copy()
    cmap.set_bad("#E0E0E0")

    fig_w = max(8, 1.5 + 0.9 * len(piv.columns))
    fig_h = max(6, 2.0 + 0.22 * len(piv.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.4, vmax=0.8)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index.tolist(), fontsize=7)
    ax.set_title(f"{ct_label} | {model} | Spearman(sig) by biosample x assay", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman(sig)")
    # Annotate cells with values (skip if grid is too large to be readable)
    n_rows, n_cols = data.shape
    if n_rows <= 40:
        cell_font = max(4, min(7, 120 // max(n_rows, n_cols)))
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=cell_font, color="white" if v > 0.5 or v < -0.2 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    piv.to_csv(plot_data_path(out_path), index=True)


def plot_alltrack_assay_barchart(alltrack_df, ct_label, out_path):
    """Grouped bar chart: x=assay_type, groups=model, bars split by all vs sig variants.

    alltrack_df columns: model, assay_type, spearman_r (all), spearman_sig (sig-only).
    Produces one figure per score dimension (all / sig).
    """
    if alltrack_df.empty:
        return

    assays = sorted(alltrack_df["assay_type"].unique())
    models = [m for m in TRACK_MODELS if m in alltrack_df["model"].unique()]
    n_assays = len(assays)
    n_models = len(models)
    if n_assays == 0 or n_models == 0:
        return

    assay_idx = {a: i for i, a in enumerate(assays)}
    width = 0.8 / n_models          # bar width
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for metric, metric_label in [("spearman_r", "Spearman ρ (all variants)"),
                                   ("spearman_sig", "Spearman ρ (sig only, α=0.1)")]:
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * n_assays * n_models), 5))
        ax.axhline(0, color="black", lw=0.7)

        for mi, model in enumerate(models):
            mdf = alltrack_df[alltrack_df["model"] == model]
            xs = []
            ys = []
            for assay in assays:
                row = mdf[mdf["assay_type"] == assay]
                xs.append(assay_idx[assay] + offsets[mi])
                ys.append(float(row[metric].values[0]) if len(row) > 0 and np.isfinite(row[metric].values[0]) else 0.0)

            bars = ax.bar(xs, ys, width=width * 0.9,
                          color=MODEL_COLORS.get(model, "#607D8B"),
                          label=model, alpha=0.88, edgecolor="white", linewidth=0.4)
            # Annotate each bar with its value
            for bar, y in zip(bars, ys):
                if y != 0.0:
                    va = "bottom" if y >= 0 else "top"
                    offset_y = 0.005 if y >= 0 else -0.005
                    ax.text(bar.get_x() + bar.get_width() / 2, y + offset_y,
                            f"{y:.2f}", ha="center", va=va, fontsize=7.5, rotation=0)

        ax.set_xticks(np.arange(n_assays))
        ax.set_xticklabels(assays, fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_xlabel("Assay type (all tracks, no biosample filter)", fontsize=9)
        ax.set_title(f"{ct_label} | Per-assay model performance ({metric_label})", fontsize=10)
        ax.legend(title="Model", fontsize=9, title_fontsize=9)
        ax.set_ylim(min(-0.05, ax.get_ylim()[0] - 0.05), max(0.75, ax.get_ylim()[1] + 0.08))
        plt.tight_layout()

        suffix = "all" if metric == "spearman_r" else "sig"
        fig.savefig(out_path.with_name(out_path.stem.replace("__SUFFIX__", suffix) + ".png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    alltrack_df.to_csv(plot_data_path(out_path.with_name(
        out_path.stem.replace("__SUFFIX__", "all") + ".png")), index=False)


def plot_assay_global_vs_filtered_barchart(
    alltrack_df, filtered_df, ct_label, out_path,
    baseline_df=None, filtered_overall_df=None,
):
    """Grouped bar chart: global (all-track) vs CT-filtered per model.

    First group ('All assays'): undifferentiated baseline vs filtered (mean_sad,
    drawn from baseline_df / filtered_overall_df when supplied).
    Subsequent groups: one per assay type (CAGE, CHIP, DNASE, …).
    Two bars per group: global hatched/light, CT-filtered solid.
    One subplot per model, shared Y axis.
    A vertical separator line is drawn between 'All assays' and the per-assay groups.
    """
    if alltrack_df.empty and filtered_df.empty:
        return

    models = [m for m in TRACK_MODELS
              if m in alltrack_df["model"].values or m in filtered_df["model"].values]
    per_assay = sorted(set(alltrack_df["assay_type"].tolist() + filtered_df["assay_type"].tolist()))
    groups = ["All assays"] + per_assay   # first group = undifferentiated
    n_groups = len(groups)
    n_models = len(models)
    if n_groups == 0 or n_models == 0:
        return

    for metric, metric_label, suffix in [
        ("spearman_r",   "Spearman ρ (all variants)",    "all"),
        ("spearman_sig", "Spearman ρ (sig only, α=0.1)", "sig"),
    ]:
        panel_w = max(4.0, n_groups * 1.3 + 1.0)
        fig, axes = plt.subplots(1, n_models, figsize=(panel_w * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            color = MODEL_COLORS.get(model, "#607D8B")
            global_vals, filt_vals = [], []

            for grp in groups:
                if grp == "All assays":
                    # Undifferentiated score from baseline / filtered rows
                    gv = np.nan
                    fv = np.nan
                    if baseline_df is not None and not baseline_df.empty:
                        b = baseline_df[(baseline_df["model"] == model) &
                                        (baseline_df["score_type"] == "mean_sad")]
                        if len(b) and np.isfinite(b[metric].values[0]):
                            gv = float(b[metric].values[0])
                    if filtered_overall_df is not None and not filtered_overall_df.empty:
                        f = filtered_overall_df[(filtered_overall_df["model"] == model) &
                                                (filtered_overall_df["score_type"] == "mean_sad")]
                        if len(f) and np.isfinite(f[metric].values[0]):
                            fv = float(f[metric].values[0])
                else:
                    g = alltrack_df[(alltrack_df["model"] == model) &
                                    (alltrack_df["assay_type"] == grp)]
                    f = filtered_df[(filtered_df["model"] == model) &
                                    (filtered_df["assay_type"] == grp)]
                    gv = (float(g[metric].values[0])
                          if len(g) and np.isfinite(g[metric].values[0]) else np.nan)
                    fv = (float(f[metric].values[0])
                          if len(f) and np.isfinite(f[metric].values[0]) else np.nan)
                global_vals.append(gv)
                filt_vals.append(fv)

            x = np.arange(n_groups)
            w = 0.35
            bars_g = ax.bar(x - w / 2,
                            [v if np.isfinite(v) else 0 for v in global_vals], w,
                            label="Global (all tracks)", color=color, alpha=0.40,
                            edgecolor=color, linewidth=1.0, hatch="//")
            bars_f = ax.bar(x + w / 2,
                            [v if np.isfinite(v) else 0 for v in filt_vals], w,
                            label="CT-filtered", color=color, alpha=0.88,
                            edgecolor="white", linewidth=0.4)

            for bars, vals in [(bars_g, global_vals), (bars_f, filt_vals)]:
                for bar, y in zip(bars, vals):
                    if np.isfinite(y) and y != 0:
                        va = "bottom" if y >= 0 else "top"
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                y + (0.006 if y >= 0 else -0.006),
                                f"{y:.2f}", ha="center", va=va, fontsize=7)

            # Vertical separator after 'All assays'
            ax.axvline(0.5, color="grey", lw=0.9, linestyle="--", alpha=0.6)

            all_finite = [v for v in global_vals + filt_vals if np.isfinite(v)]
            ymin = min(-0.05, min(all_finite) - 0.08) if all_finite else -0.05
            ymax = max(0.50, max(all_finite) + 0.12) if all_finite else 0.50
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(groups, fontsize=9, rotation=20, ha="right")
            ax.set_title(model, fontsize=11)
            ax.set_xlabel("Assay type", fontsize=9)
            ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=8, loc="upper right")

        axes[0].set_ylabel(metric_label, fontsize=10)
        fig.suptitle(
            f"{ct_label} | Global vs CT-filtered — overall + per-assay mean SAD ({metric_label})",
            fontsize=11,
        )
        plt.tight_layout()
        out_fig = out_path.with_name(out_path.stem.replace("__SUFFIX__", suffix) + ".png")
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_fig}")

    # Save merged data CSV (per-assay only; overall row added manually)
    rows_overall = []
    for model in models:
        go_r = go_s = fo_r = fo_s = np.nan
        if baseline_df is not None and not baseline_df.empty:
            b = baseline_df[(baseline_df["model"] == model) & (baseline_df["score_type"] == "mean_sad")]
            if len(b):
                go_r = float(b["spearman_r"].values[0])
                go_s = float(b["spearman_sig"].values[0])
        if filtered_overall_df is not None and not filtered_overall_df.empty:
            f = filtered_overall_df[(filtered_overall_df["model"] == model) &
                                    (filtered_overall_df["score_type"] == "mean_sad")]
            if len(f):
                fo_r = float(f["spearman_r"].values[0])
                fo_s = float(f["spearman_sig"].values[0])
        rows_overall.append(dict(model=model, assay_type="All assays",
                                 global_r=go_r, global_sig=go_s,
                                 filtered_r=fo_r, filtered_sig=fo_s))
    df_overall = pd.DataFrame(rows_overall)

    df_per_assay = pd.merge(
        alltrack_df[["model", "assay_type", "spearman_r", "spearman_sig", "n", "n_sig"]]
        .rename(columns={"spearman_r": "global_r", "spearman_sig": "global_sig",
                         "n": "global_n", "n_sig": "global_n_sig"}),
        filtered_df[["model", "assay_type", "spearman_r", "spearman_sig", "n", "n_sig"]]
        .rename(columns={"spearman_r": "filtered_r", "spearman_sig": "filtered_sig",
                         "n": "filtered_n", "n_sig": "filtered_n_sig"}),
        on=["model", "assay_type"],
        how="outer",
    )
    merged = pd.concat([df_overall, df_per_assay], ignore_index=True)
    csv_path = out_path.with_name(out_path.stem.replace("__SUFFIX__", "all") + "__plot_data.csv")
    merged.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def run_celltype(ct, args):
    cfg = EXPERIMENTS[ct]
    label = cfg["label"]
    _analysis_dir = "analysis_sigonly" if args.sig_only else "analysis"
    out_dir = BASE / f"results/{ct}/{_analysis_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.parquet_dir) / ct if args.parquet_dir else BASE / "results" / ct
    if not results_dir.exists():
        print(f"WARNING: {results_dir} not found. Skipping {ct}.")
        return

    run_baseline = "baseline" in args.analyses
    run_filtered = "filtered" in args.analyses or "detailed" in args.analyses
    run_detailed = "detailed" in args.analyses

    allowed_assays = None
    if args.assay_whitelist:
        allowed_assays = {a.strip().upper() for a in args.assay_whitelist.split(",") if a.strip()}

    print("\n" + "=" * 70)
    print(f"{label} | unified analysis")
    print(f"Analyses: {', '.join(args.analyses)}")
    print("=" * 70)

    gt = load_gt(cfg)          # always full — never filtered before aggregate
    _full_log2fc = gt["log2FC"].to_numpy(dtype=np.float32)
    _full_sig    = gt["sig"].to_numpy(dtype=bool)
    if args.sig_only:
        _sig_mask = _full_sig.copy()
        print(f"  --sig-only: {len(gt):,} → {int(_sig_mask.sum()):,} significant variants (alpha=0.1)")
        gt_log2fc = _full_log2fc[_sig_mask]
        gt_sig    = _full_sig[_sig_mask]
    else:
        _sig_mask = None
        gt_log2fc = _full_log2fc
        gt_sig    = _full_sig
    gt_idx = {v: i for i, v in enumerate(gt["variant_id"].values)}

    print(f"Ground truth variants: {len(gt):,}, significant: {int(_full_sig.sum()):,}")
    print(
        f"Scatter downsampling around 0: {'ON' if args.scatter_downsample else 'OFF'}"
        + (f" (bins={args.scatter_bins}, max_per_bin={args.scatter_per_bin})" if args.scatter_downsample else "")
    )

    baseline_rows = []
    filtered_rows = []
    biosample_frames = []
    assay_frames = []
    pair_frames = []
    bin_frames = []
    alltrack_assay_rows = []       # per-model per-assay Spearman rows for summary barchart (all tracks)
    filtered_assay_rows = []       # per-model per-assay Spearman rows for CT-filtered biosamples

    for model in TRACK_MODELS:
        parquet = results_dir / model / f"{model}_scores.parquet"
        if not parquet.exists():
            print(f"\n  SKIP {model}: missing {parquet}")
            continue

        print(f"\n  Processing {model} (single pass) ...", flush=True)
        bio_pattern = BIOSAMPLE_FILTERS[ct][model]

        all_acc, filt_acc, matched_biosamples, pair_acc, all_assay_acc, filt_stats = aggregate_single_pass(
            parquet_path=parquet,
            gt_index=gt_idx,
            n_variants=len(gt),
            bio_pattern=bio_pattern,
            run_filtered=run_filtered,
            run_detailed=run_detailed,
            batch_size=args.batch_size,
            include_regex=args.include_biosample_regex,
            exclude_regex=args.exclude_biosample_regex,
            allowed_assays=allowed_assays,
        )

        # ── Save score caches for extended_analyses.py ─────────────────────────
        if args.save_score_caches:
            from collections import defaultdict
            _cache_dir = TMP_ROOT / ct
            _cache_dir.mkdir(parents=True, exist_ok=True)
            # Baseline cache
            np.savez_compressed(
                _cache_dir / f"{model}_baseline_cache.npz",
                sum=all_acc["sum"],
                count=all_acc["count"],
            )
            # Filtered cache
            if filt_acc is not None:
                np.savez_compressed(
                    _cache_dir / f"{model}_filtered_cache.npz",
                    sum=filt_acc["sum"],
                    abs_sum=filt_acc["abs_sum"],
                    count=filt_acc["count"],
                )
            # Per-assay cache: aggregate pair_acc {(bio, assay): {sum, abs_sum, count}}
            # by summing across all biosamples that share the same assay type
            if pair_acc:
                _assay_sum   = defaultdict(lambda: np.zeros(len(gt), dtype=np.float64))
                _assay_abs   = defaultdict(lambda: np.zeros(len(gt), dtype=np.float64))
                _assay_cnt   = defaultdict(lambda: np.zeros(len(gt), dtype=np.uint32))
                for (bio, assay), v in pair_acc.items():
                    _assay_sum[assay]  += v["sum"]
                    _assay_abs[assay]  += v["abs_sum"]
                    _assay_cnt[assay]  += v["count"].astype(np.uint32)
                _assay_scores = {}
                for assay in _assay_sum:
                    _valid = _assay_cnt[assay] > 0
                    _mean_sad     = np.full(len(gt), np.nan, dtype=np.float32)
                    _mean_abs_sad = np.full(len(gt), np.nan, dtype=np.float32)
                    _mean_sad[_valid]     = (_assay_sum[assay][_valid] / _assay_cnt[assay][_valid]).astype(np.float32)
                    _mean_abs_sad[_valid] = (_assay_abs[assay][_valid] / _assay_cnt[assay][_valid]).astype(np.float32)
                    _assay_scores[assay] = {"mean_sad": _mean_sad, "mean_abs_sad": _mean_abs_sad}
                # np.savez can only store plain arrays; pack the dict as an object array
                np.savez(_cache_dir / f"{model}_assay_cache.npz", assay_scores=np.array(_assay_scores, dtype=object))
            print(f"    Score caches saved → {_cache_dir}/{model}_*_cache.npz", flush=True)
        # ───────────────────────────────────────────────────────────────────────

        # If --sig-only: mask accumulator arrays in-place to the sig subset so all
        # downstream compute/plot code receives correctly-sized arrays.
        if _sig_mask is not None:
            for _acc in [all_acc, filt_acc]:
                if _acc is not None:
                    for _k in list(_acc.keys()):
                        if isinstance(_acc[_k], np.ndarray):
                            _acc[_k] = _acc[_k][_sig_mask]
            if pair_acc:
                for _pkey in list(pair_acc.keys()):
                    for _k in list(pair_acc[_pkey].keys()):
                        if isinstance(pair_acc[_pkey][_k], np.ndarray):
                            pair_acc[_pkey][_k] = pair_acc[_pkey][_k][_sig_mask]
            if all_assay_acc:
                for _akey in list(all_assay_acc.keys()):
                    for _k in list(all_assay_acc[_akey].keys()):
                        if isinstance(all_assay_acc[_akey][_k], np.ndarray):
                            all_assay_acc[_akey][_k] = all_assay_acc[_akey][_k][_sig_mask]

        base_sp_sig = {"mean_sad": np.nan, "mean_abs_sad": np.nan}
        filt_sp_sig = {"mean_sad": np.nan, "mean_abs_sad": np.nan}

        if run_baseline:
            for st in ["mean_sad", "mean_abs_sad", "max_abs_sad"]:
                score = scores_from_acc(all_acc, st)
                _use_abs = (st != "mean_sad")
                m = compute_metrics(gt_log2fc, gt_sig, score, use_abs_lfc=_use_abs)
                if m is None:
                    continue
                m.update({"model": model, "score_type": st})
                baseline_rows.append(m)
                if st in base_sp_sig:
                    base_sp_sig[st] = m["spearman_sig"]
                # Effect-size bin analysis
                bdf = compute_bin_metrics(gt_log2fc, gt_sig, score, use_abs_lfc=_use_abs)
                if not bdf.empty:
                    bdf["model"] = model
                    bdf["score_type"] = st
                    bdf["subset"] = "baseline"
                    bin_frames.append(bdf)
                if not args.no_plots and not bdf.empty:
                    out = out_dir / f"{ct}_{model}_bins_{st}_baseline_unified.png"
                    plot_bin_metrics(bdf, out, f"{label} | {model} | {st} | baseline – Spearman by |log2FC| bin")
            if not args.no_plots:
                out = out_dir / f"{ct}_{model}_baseline_scatter_mean_sad.png"
                plot_scatter(
                    gt_log2fc,
                    scores_from_acc(all_acc, "mean_sad"),
                    gt_sig,
                    f"{label} | {model} | all-track mean_sad",
                    out,
                    args.scatter_downsample,
                    args.scatter_bins,
                    args.scatter_per_bin,
                )
                out = out_dir / f"{ct}_{model}_baseline_scatter_mean_abs_sad.png"
                plot_scatter(
                    gt_log2fc,
                    scores_from_acc(all_acc, "mean_abs_sad"),
                    gt_sig,
                    f"{label} | {model} | all-track mean_abs_sad",
                    out,
                    args.scatter_downsample,
                    args.scatter_bins,
                    args.scatter_per_bin,
                )

        if run_filtered:
            n_bio = len(matched_biosamples)
            for st in ["mean_sad", "mean_abs_sad", "max_abs_sad"]:
                score = scores_from_acc(filt_acc, st)
                _use_abs = (st != "mean_sad")
                m = compute_metrics(gt_log2fc, gt_sig, score, use_abs_lfc=_use_abs)
                if m is None:
                    continue
                m.update({"model": model, "score_type": st, "n_biosamples": n_bio})
                filtered_rows.append(m)
                if st in filt_sp_sig:
                    filt_sp_sig[st] = m["spearman_sig"]
                # Effect-size bin analysis
                bdf = compute_bin_metrics(gt_log2fc, gt_sig, score, use_abs_lfc=_use_abs)
                if not bdf.empty:
                    bdf["model"] = model
                    bdf["score_type"] = st
                    bdf["subset"] = "filtered"
                    bin_frames.append(bdf)
                if not args.no_plots and not bdf.empty:
                    out = out_dir / f"{ct}_{model}_bins_{st}_filtered_unified.png"
                    plot_bin_metrics(bdf, out, f"{label} | {model} | {st} | filtered – Spearman by |log2FC| bin")
            if not args.no_plots:
                out = out_dir / f"{ct}_{model}_filtered_scatter_mean_sad_unified.png"
                plot_scatter(
                    gt_log2fc,
                    scores_from_acc(filt_acc, "mean_sad"),
                    gt_sig,
                    f"{label} | {model} | filtered mean_sad",
                    out,
                    args.scatter_downsample,
                    args.scatter_bins,
                    args.scatter_per_bin,
                )
                out = out_dir / f"{ct}_{model}_filtered_scatter_mean_abs_sad_unified.png"
                plot_scatter(
                    gt_log2fc,
                    scores_from_acc(filt_acc, "mean_abs_sad"),
                    gt_sig,
                    f"{label} | {model} | filtered mean_abs_sad",
                    out,
                    args.scatter_downsample,
                    args.scatter_bins,
                    args.scatter_per_bin,
                )

        if run_detailed:
            if pair_acc:
                pair_df = pair_metrics_df(pair_acc, gt_log2fc, gt_sig, model)
                bio_df = grouped_metrics_from_pair(pair_acc, gt_log2fc, gt_sig, model, by="biosample_name")
                assay_df = grouped_metrics_from_pair(pair_acc, gt_log2fc, gt_sig, model, by="assay_type")

                if not pair_df.empty:
                    pair_frames.append(pair_df)
                if not bio_df.empty:
                    biosample_frames.append(bio_df)
                if not assay_df.empty:
                    assay_frames.append(assay_df)
                    # Collect CT-filtered per-assay rows for global vs CT-filtered comparison
                    for _, _row in assay_df[assay_df["score_type"] == "mean_sad"].iterrows():
                        filtered_assay_rows.append({
                            "model": model,
                            "assay_type": _row["assay_type"],
                            "spearman_r": _row.get("spearman_r", np.nan),
                            "spearman_sig": _row.get("spearman_sig", np.nan),
                            "n": _row.get("n", np.nan),
                            "n_sig": _row.get("n_sig", np.nan),
                        })

                if not args.no_plots and not bio_df.empty:
                    for st in ["mean_sad", "mean_abs_sad"]:
                        bio_sub = bio_df[bio_df["score_type"] == st]
                        if bio_sub.empty:
                            continue
                        out = out_dir / f"{ct}_{model}_biosample_vs_mean_spearman_sig_{st}_unified.png"
                        plot_biosample_vs_baseline(
                            bio_sub,
                            base_sp_sig.get(st, np.nan),
                            filt_sp_sig.get(st, np.nan),
                            label,
                            model,
                            out,
                        )

                if not args.no_plots and not assay_df.empty:
                    for st in ["mean_sad", "mean_abs_sad"]:
                        assay_sub = assay_df[assay_df["score_type"] == st]
                        if assay_sub.empty:
                            continue
                        out = out_dir / f"{ct}_{model}_assay_performance_{st}_unified.png"
                        plot_assay_performance(assay_sub, label, model, st, out)
                if not args.no_plots and not pair_df.empty:
                    for st in ["mean_sad", "mean_abs_sad"]:
                        pair_sub = pair_df[pair_df["score_type"] == st]
                        if pair_sub.empty:
                            continue
                        out = out_dir / f"{ct}_{model}_biosample_assay_spearman_sig_heatmap_{st}_unified.png"
                        plot_pair_heatmap(pair_sub, label, model, out)

                    for st in ["mean_sad", "mean_abs_sad"]:
                        out = out_dir / f"{ct}_{model}_assay_correlation_{st}_unified.png"
                        plot_assay_correlation_heatmap(pair_acc, label, model, st, out)

                if not args.no_plots:
                    for st in ["mean_sad", "mean_abs_sad"]:
                        sdf = assay_score_matrix_from_pair(pair_acc, score_type=st)
                        if not sdf.empty:
                            out = out_dir / f"{ct}_{model}_assay_scatter_{st}_unified.png"
                            plot_assay_scatter_grid(sdf, gt_log2fc, gt_sig, label, model, st, out)

        # ── All-track assay plots (no biosample filter, all tracks by assay type)
        if run_detailed and all_assay_acc:
            # Collect per-assay Spearman for summary barchart
            for assay, v in all_assay_acc.items():
                cnt = v["count"]
                valid = cnt > 0
                score = np.full(len(gt_log2fc), np.nan, dtype=np.float32)
                score[valid] = v["sum"][valid] / cnt[valid]
                m = compute_metrics(gt_log2fc, gt_sig, score)
                if m is not None:
                    alltrack_assay_rows.append({
                        "model": model,
                        "assay_type": assay,
                        "spearman_r": m["spearman_r"],
                        "spearman_sig": m["spearman_sig"],
                        "n": m["n"],
                        "n_sig": m["n_sig"],
                    })

            if not args.no_plots:
                _pair_from_all = {("_all_tracks_", assay): v for assay, v in all_assay_acc.items()}
                for st in ["mean_sad", "mean_abs_sad"]:
                    out = out_dir / f"{ct}_{model}_alltrack_assay_correlation_{st}_unified.png"
                    plot_assay_correlation_heatmap(_pair_from_all, label, model, st, out)
                    sdf_all = assay_score_matrix_from_assay_acc(all_assay_acc, score_type=st)
                    if not sdf_all.empty:
                        out = out_dir / f"{ct}_{model}_alltrack_assay_scatter_{st}_unified.png"
                        plot_assay_scatter_grid(sdf_all, gt_log2fc, gt_sig, label, model, st, out)

        if run_filtered:
            print(f"    Matched biosamples for {model}: {len(matched_biosamples)}")
            print(
                "    Filtered rows kept="
                f"{filt_stats['passed_filtered_rows']:,}, "
                f"excluded_by_name={filt_stats['excluded_by_name']:,}, "
                f"excluded_by_assay={filt_stats['excluded_by_assay']:,}"
            )

    if baseline_rows:
        df = attach_score_definition(pd.DataFrame(baseline_rows))
        out = out_dir / f"{ct}_metrics_unified.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

    if alltrack_assay_rows:
        alltrack_df = pd.DataFrame(alltrack_assay_rows)
        out_csv = out_dir / f"{ct}_alltrack_assay_metrics_unified.csv"
        alltrack_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        if not args.no_plots:
            out_stem = out_dir / f"{ct}_alltrack_assay_barchart___SUFFIX__"
            plot_alltrack_assay_barchart(alltrack_df, label, out_stem)
            print(f"Saved: {out_dir}/{ct}_alltrack_assay_barchart_all.png + _sig.png")

        # Global vs CT-filtered comparison figure
        if filtered_assay_rows and not args.no_plots:
            filtered_assay_df = pd.DataFrame(filtered_assay_rows)
            out_csv2 = out_dir / f"{ct}_assay_global_vs_filtered_metrics.csv"
            filtered_assay_df.to_csv(out_csv2, index=False)
            print(f"Saved: {out_csv2}")
            out_stem2 = out_dir / f"{ct}_assay_global_vs_filtered___SUFFIX__"
            plot_assay_global_vs_filtered_barchart(
                alltrack_df, filtered_assay_df, label, out_stem2,
                baseline_df=pd.DataFrame(baseline_rows) if baseline_rows else None,
                filtered_overall_df=pd.DataFrame(filtered_rows) if filtered_rows else None,
            )

    if filtered_rows:
        df = attach_score_definition(pd.DataFrame(filtered_rows))
        out = out_dir / f"{ct}_filtered_metrics_unified.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

    if biosample_frames:
        df = pd.concat(biosample_frames, ignore_index=True).sort_values(["model", "score_type", "spearman_sig"], ascending=[True, True, False])
        df = attach_score_definition(df)
        out = out_dir / f"{ct}_filtered_biosample_metrics_unified.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

    if assay_frames:
        df = pd.concat(assay_frames, ignore_index=True).sort_values(["model", "score_type", "spearman_sig"], ascending=[True, True, False])
        df = attach_score_definition(df)
        out = out_dir / f"{ct}_filtered_assay_metrics_unified.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

    if pair_frames:
        df = pd.concat(pair_frames, ignore_index=True).sort_values(["model", "score_type", "spearman_sig"], ascending=[True, True, False])
        df = attach_score_definition(df)
        out = out_dir / f"{ct}_filtered_biosample_assay_metrics_unified.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

    if bin_frames:
        df = attach_score_definition(pd.concat(bin_frames, ignore_index=True))
        out = out_dir / f"{ct}_bin_metrics_unified.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified one-pass parquet analysis")
    parser.add_argument(
        "--cell-types",
        nargs="+",
        default=["hepg2", "hek293t", "ngn2"],
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=["baseline", "filtered", "detailed"],
        choices=["baseline", "filtered", "detailed"],
        help="Enable one or more analysis blocks.",
    )
    parser.add_argument("--batch-size", type=int, default=5_000_000)
    parser.add_argument(
        "--parquet-dir", type=str, default="",
        help="Override root directory for model parquets. Expects {parquet-dir}/{ct}/{model}/{model}_scores.parquet. "
             "Defaults to BASE/results/. Use a native WSL path for faster I/O.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of cell types to process in parallel (1-3). Uses multiprocessing.",
    )

    parser.add_argument(
        "--scatter-downsample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downsample dense near-zero region in scatter plots via |log2FC| quantile bins.",
    )
    parser.add_argument("--scatter-bins", type=int, default=30)
    parser.add_argument("--scatter-per-bin", type=int, default=400)

    parser.add_argument(
        "--exclude-biosample-regex",
        default=DEFAULT_EXCLUDE_BIOSAMPLE_REGEX,
        help="Regex for biosample names to exclude after cell-type matching.",
    )
    parser.add_argument(
        "--include-biosample-regex",
        default="",
        help="Optional additional regex that biosample names must match.",
    )
    parser.add_argument(
        "--assay-whitelist",
        default="",
        help="Comma-separated assay whitelist, e.g. DNASE,CAGE,CHIP.",
    )

    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation and save CSVs only.")
    parser.add_argument(
        "--save-score-caches",
        action="store_true",
        default=False,
        help="Save per-variant score numpy caches to TMP_ROOT/{ct}/{model}_*_cache.npz for use by extended_analyses.py.",
    )
    parser.add_argument(
        "--sig-only",
        action="store_true",
        default=False,
        help="Restrict all analyses to only significant variants (padj < alpha). "
             "Outputs go to results/{ct}/analysis_sigonly/.",
    )
    parser.add_argument(
        "--regen-assay-scatter",
        action="store_true",
        default=False,
        help="Fast mode: regenerate per-assay scatter plots from existing .npz caches "
             "(no parquet re-read). Requires --save-score-caches to have been run previously.",
    )
    return parser.parse_args()


def regen_assay_scatter(args):
    """Load assay .npz caches and regenerate per-assay scatter plots without re-reading parquets."""
    for ct in args.cell_types:
        cfg = EXPERIMENTS[ct]
        label = cfg["label"]
        _analysis_dir = "analysis_sigonly" if args.sig_only else "analysis"
        out_dir = BASE / f"results/{ct}/{_analysis_dir}"
        out_dir.mkdir(parents=True, exist_ok=True)

        gt = load_gt(cfg)
        if args.sig_only:
            n_before = len(gt)
            gt = gt[gt["sig"]].reset_index(drop=True)
            print(f"  --sig-only [{label}]: {n_before:,} → {len(gt):,} significant variants")
        gt_log2fc = gt["log2FC"].to_numpy(dtype=np.float32)
        gt_sig = gt["sig"].to_numpy(dtype=bool)

        for model in TRACK_MODELS:
            cache_assay = TMP_ROOT / ct / f"{model}_assay_cache.npz"
            if not cache_assay.exists():
                print(f"  SKIP {ct}/{model}: no assay cache at {cache_assay}")
                continue
            print(f"  Regen assay scatter: {ct}/{model} ...", flush=True)
            raw = np.load(cache_assay, allow_pickle=True)
            assay_scores = raw["assay_scores"].item()  # dict: assay -> {mean_sad, mean_abs_sad}
            for st in ["mean_sad", "mean_abs_sad"]:
                cols = {assay: v[st].astype(np.float32) for assay, v in sorted(assay_scores.items())
                        if st in v}
                if not cols:
                    continue
                # Align length to gt (in case cache was built on different gt)
                n_cache = next(iter(cols.values())).shape[0]
                if n_cache != len(gt):
                    print(f"    WARNING: cache length {n_cache} != gt length {len(gt)}, skipping {st}")
                    continue
                sdf = pd.DataFrame(cols)
                out = out_dir / f"{ct}_{model}_assay_scatter_{st}_unified.png"
                plot_assay_scatter_grid(sdf, gt_log2fc, gt_sig, label, model, st, out)
                print(f"    Saved: {out}")


def main():
    args = parse_args()

    if args.regen_assay_scatter:
        regen_assay_scatter(args)
        print("\nDone (regen-assay-scatter).")
        return

    cell_types = args.cell_types
    n_workers = min(max(1, args.workers), len(cell_types))
    if n_workers > 1:
        print(f"Running {len(cell_types)} cell types in parallel with {n_workers} workers.")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(run_celltype, ct, args): ct for ct in cell_types}
            for fut in as_completed(futs):
                ct = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"ERROR processing {ct}: {e}", flush=True)
    else:
        for ct in cell_types:
            run_celltype(ct, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
