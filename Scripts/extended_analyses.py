#!/usr/bin/env python3
"""
Extended analyses for the MPRA benchmark study.

Implements the following research questions from the manuscript:
  Q2.3 - Cross-cell-type rank consistency  (HepG2 vs HEK293T)
  Q1.2 - Linear integration of assay scores (Ridge/Lasso)
  Q1.3 - Most informative assay types       (from Ridge coefficients)
  Q3.1 - Performance by genomic region       (promoter / enhancer / other)
  Q2.2 - Cell-type prediction from model scores

Requirements:
  pip install pyarrow pandas numpy matplotlib scipy scikit-learn

For Q3.1 (genomic regions), download the ENCODE cCRE annotations once:
  wget -O Data/GRCh38-cCREs.bed \\
    https://downloads.wenglab.org/V3/GRCh38-cCREs.bed

Usage:
  python3 Scripts/extended_analyses.py --analyses cross_celltype linear_model genomic_regions celltype_prediction
  python3 Scripts/extended_analyses.py --analyses cross_celltype --no-plots
"""

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration  (mirrored from unified_celltype_analysis.py)
# ---------------------------------------------------------------------------

BASE = Path("/mnt/c/Users/user/Desktop/Benchmark_results/Pipeline_v2")
VCF_DIR = BASE / "Data/VCF"
TMP_ROOT = Path("/tmp/mpra_analysis")
# Resolved at runtime from --parquet-dir arg (defaults to BASE/results)
PARQUET_ROOT = BASE / "results"

EXPERIMENTS = {
    "hepg2": {
        "vcf": VCF_DIR / "IGVFFI4378PZYI.vcf",
        "has_qval": True,
        "qval_thr": 1.0,  # alpha=0.1
        "label": "HepG2",
    },
    "hek293t": {
        "vcf": VCF_DIR / "IGVFFI4134MFLL.vcf",
        "has_qval": True,
        "qval_thr": 1.0,  # alpha=0.1
        "label": "HEK293T",
    },
    "ngn2": {
        "vcf": VCF_DIR / "80k_normalized.vcf",
        "has_qval": False,
        "sig_lfc_thr": 0.5,
        "label": "NGN2",
    },
}

# Models that produce per-track SAD scores
SEQ2FUNC_MODELS = ["enformer", "basenji", "alphagenome"]

# Foundation models — one embedding-based score per variant
FOUNDATION_MODELS = ["hyenadna", "dnabert2"]

ALL_MODELS = SEQ2FUNC_MODELS + FOUNDATION_MODELS

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

DEFAULT_EXCLUDE_RE = (
    r"genetically modified|crispr|stable transfection|transduction|"
    r"knockout|\bko\b|overexpress|inducible|3xflag|tagged"
)

MODEL_COLORS = {
    "enformer": "#1565C0",
    "basenji": "#0288D1",
    "alphagenome": "#00796B",
    "hyenadna": "#E65100",
    "dnabert2": "#AD1457",
}

# Foundation model score columns to evaluate  (ND = nucleotide dependency)
# Signed scores capture directionality, absolute scores capture magnitude
FM_SCORE_COLS = {
    "ND_influence_score": "ND_influence_score",
    "ND_mean": "ND_mean (signed)",
    "ND_mean_ABS": "ND_mean_ABS (|embedding delta|)",
}

_LOG2FC_RE = re.compile(r"LOG2FC=([^;]+)")
_QVAL_RE = re.compile(r"QVAL=([^;]+)")


def plot_data_path(out_path):
    return out_path.with_name(f"{out_path.stem}__plot_data.csv")


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_gt(cfg):
    records = []
    with open(cfg["vcf"]) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
            info = cols[7] if len(cols) > 7 else ""
            # Normalise: strip 'chr' prefix so IDs are always chr-free
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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


def compute_metrics(log2fc, sig, score):
    mask = np.isfinite(log2fc) & np.isfinite(score)
    x = log2fc[mask]
    y = score[mask]
    s = sig[mask]
    if len(x) < 10:
        return None
    sp_all, _ = spearmanr(x, y)
    sp_sig = np.nan
    dir_acc = np.nan
    if s.sum() >= 10:
        sp_sig, _ = spearmanr(x[s], y[s])
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


# ---------------------------------------------------------------------------
# Score loading / caching (seq2func)
# ---------------------------------------------------------------------------

def _cache_path(ct, model, kind):
    """Return path for cached per-variant score arrays."""
    return TMP_ROOT / ct / f"{model}_{kind}_cache.npz"


def _compute_seq2func_scores(ct, model, gt_index, n_variants, batch_size,
                              exclude_regex, bio_pattern):
    """Single-pass parquet read: returns per-variant baseline & filtered
    accumulators **and** a per-assay score dict.

    Returns (all_sum, all_count, filt_sum, filt_abs_sum, filt_count,
             assay_scores) where assay_scores is a dict:
       assay_type -> (sum_array, abs_sum_array, count_array)
    """
    parquet_path = PARQUET_ROOT / ct / model / f"{model}_scores.parquet"
    if not parquet_path.exists():
        return None
    pf = pq.ParquetFile(parquet_path)

    all_sum = np.zeros(n_variants, dtype=np.float64)
    all_count = np.zeros(n_variants, dtype=np.uint32)
    filt_sum = np.zeros(n_variants, dtype=np.float64)
    filt_abs_sum = np.zeros(n_variants, dtype=np.float64)
    filt_count = np.zeros(n_variants, dtype=np.uint32)

    # Per-assay accumulators (cell-type matched tracks only)
    assay_acc = {}

    cols = ["variant_id", "raw_score", "biosample_name", "assay_type"]
    t0 = time.time()
    for nb, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols)):
        df = batch.to_pandas()
        # Normalise: strip 'chr' prefix so parquet IDs match gt_index keys
        df["variant_id"] = df["variant_id"].str.replace(r"^chr", "", regex=True)
        vidx = df["variant_id"].map(gt_index)
        keep = vidx.notna()
        if not keep.any():
            continue
        df = df.loc[keep].copy()
        df["vidx"] = vidx.loc[keep].astype(np.int32)

        raw = df["raw_score"].to_numpy(dtype=np.float64)
        idx = df["vidx"].to_numpy(dtype=np.int32)
        finite = np.isfinite(raw)

        # Baseline accumulator (all tracks)
        if finite.any():
            fi, fr = idx[finite], raw[finite]
            np.add.at(all_sum, fi, fr)
            np.add.at(all_count, fi, 1)

        # Filtered by cell-type biosample match
        bio_mask = df["biosample_name"].str.contains(
            bio_pattern, case=False, na=False, regex=True
        )
        if exclude_regex:
            drop = df["biosample_name"].str.contains(
                exclude_regex, case=False, na=False, regex=True
            )
            bio_mask = bio_mask & ~drop

        if bio_mask.any():
            fdf = df.loc[bio_mask]
            fidx = fdf["vidx"].to_numpy(dtype=np.int32)
            fraw = fdf["raw_score"].to_numpy(dtype=np.float64)
            ff = np.isfinite(fraw)
            if ff.any():
                fi2, fr2 = fidx[ff], fraw[ff]
                np.add.at(filt_sum, fi2, fr2)
                np.add.at(filt_abs_sum, fi2, np.abs(fr2))
                np.add.at(filt_count, fi2, 1)

            # Per-assay accumulation (cell-type matched only)
            fdf = fdf.copy()
            fdf["assay_type"] = fdf["assay_type"].fillna("UNKNOWN")
            fdf = fdf[np.isfinite(fdf["raw_score"].to_numpy(dtype=np.float64))]
            if not fdf.empty:
                fdf["abs_score"] = fdf["raw_score"].abs()
                agg = (
                    fdf.groupby(["assay_type", "vidx"], sort=False, observed=True)
                    .agg(s=("raw_score", "sum"), a=("abs_score", "sum"), c=("raw_score", "count"))
                    .reset_index()
                )
                for assay, kdf in agg.groupby("assay_type", sort=False):
                    if assay not in assay_acc:
                        assay_acc[assay] = {
                            "sum": np.zeros(n_variants, dtype=np.float64),
                            "abs_sum": np.zeros(n_variants, dtype=np.float64),
                            "count": np.zeros(n_variants, dtype=np.uint32),
                        }
                    vi = kdf["vidx"].to_numpy(dtype=np.int32)
                    assay_acc[assay]["sum"][vi] += kdf["s"].to_numpy(dtype=np.float64)
                    assay_acc[assay]["abs_sum"][vi] += kdf["a"].to_numpy(dtype=np.float64)
                    assay_acc[assay]["count"][vi] += kdf["c"].to_numpy(dtype=np.uint32)

        if (nb + 1) % 50 == 0:
            print(f"      batch {nb+1}: {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"      done {nb+1} batches in {elapsed:.0f}s", flush=True)

    # Derive assay score arrays: variants × assay_type
    assay_scores = {}
    for assay, acc in assay_acc.items():
        valid = acc["count"] > 0
        mean_sad = np.full(n_variants, np.nan, dtype=np.float64)
        mean_abs_sad = np.full(n_variants, np.nan, dtype=np.float64)
        mean_sad[valid] = acc["sum"][valid] / acc["count"][valid]
        mean_abs_sad[valid] = acc["abs_sum"][valid] / acc["count"][valid]
        assay_scores[assay] = {"mean_sad": mean_sad, "mean_abs_sad": mean_abs_sad}

    return all_sum, all_count, filt_sum, filt_abs_sum, filt_count, assay_scores


def get_seq2func_variant_scores(ct, model, gt_index, n_variants, batch_size,
                                 exclude_regex):
    """Get per-variant scores for a seq2func model. Uses cache if available."""
    bio_pattern = BIOSAMPLE_FILTERS[ct][model]
    cache_base = _cache_path(ct, model, "baseline")
    cache_filt = _cache_path(ct, model, "filtered")
    cache_assay = TMP_ROOT / ct / f"{model}_assay_cache.npz"

    # Try cache
    if cache_base.exists() and cache_filt.exists() and cache_assay.exists():
        print(f"    Loading cached scores for {model} / {ct}", flush=True)
        bl = np.load(cache_base)
        fl = np.load(cache_filt)
        all_sum, all_count = bl["sum"], bl["count"]
        filt_sum, filt_abs_sum, filt_count = fl["sum"], fl["abs_sum"], fl["count"]
        # Load assay cache
        al = np.load(cache_assay, allow_pickle=True)
        assay_scores = al["assay_scores"].item()
    else:
        print(f"    Computing scores for {model} / {ct} (parquet read) ...", flush=True)
        result = _compute_seq2func_scores(
            ct, model, gt_index, n_variants, batch_size, exclude_regex, bio_pattern
        )
        if result is None:
            return None
        all_sum, all_count, filt_sum, filt_abs_sum, filt_count, assay_scores = result
        # Save cache
        np.savez_compressed(cache_base, sum=all_sum, count=all_count)
        np.savez_compressed(cache_filt, sum=filt_sum, abs_sum=filt_abs_sum, count=filt_count)
        np.savez(cache_assay, assay_scores=assay_scores)
        print(f"    Cached to {cache_base.parent}", flush=True)

    # Derive per-variant score vectors
    valid_all = all_count > 0
    valid_filt = filt_count > 0

    baseline_mean_sad = np.full(n_variants, np.nan, dtype=np.float64)
    baseline_mean_sad[valid_all] = all_sum[valid_all] / all_count[valid_all]

    filtered_mean_sad = np.full(n_variants, np.nan, dtype=np.float64)
    filtered_mean_sad[valid_filt] = filt_sum[valid_filt] / filt_count[valid_filt]

    filtered_mean_abs_sad = np.full(n_variants, np.nan, dtype=np.float64)
    filtered_mean_abs_sad[valid_filt] = filt_abs_sum[valid_filt] / filt_count[valid_filt]

    return {
        "baseline_mean_sad": baseline_mean_sad,
        "filtered_mean_sad": filtered_mean_sad,
        "filtered_mean_abs_sad": filtered_mean_abs_sad,
        "assay_scores": assay_scores,
    }


# ---------------------------------------------------------------------------
# Score loading (foundation models)
# ---------------------------------------------------------------------------

def get_foundation_model_scores(ct, model, gt_index, n_variants):
    """Load foundation model scores — one row per variant, no parquet streaming needed."""
    parquet_path = PARQUET_ROOT / ct / model / f"{model}_scores.parquet"
    if not parquet_path.exists():
        print(f"    SKIP {model}/{ct}: no parquet at {parquet_path}")
        return None
    df = pd.read_parquet(parquet_path)
    # Normalise: strip 'chr' prefix so parquet IDs match gt_index keys
    df["variant_id"] = df["variant_id"].str.replace(r"^chr", "", regex=True)
    vidx = df["variant_id"].map(gt_index)
    keep = vidx.notna()
    if not keep.any():
        return None
    df = df.loc[keep].copy()
    df["vidx"] = vidx.loc[keep].astype(np.int32)

    result = {}
    for col in FM_SCORE_COLS:
        if col not in df.columns:
            continue
        arr = np.full(n_variants, np.nan, dtype=np.float64)
        arr[df["vidx"].values] = df[col].values.astype(np.float64)
        result[col] = arr
    return result


# =========================================================================
# Analysis Q2.3:  Cross-cell-type rank consistency
# =========================================================================

def run_cross_celltype(gt_all, scores_all, out_dir, no_plots):
    """Compare model scores and MPRA log2FC between cell types sharing variants.

    For each pair of cell types with >=100 shared variants:
      1. MPRA ground-truth correlation (log2FC ct1 vs ct2)
      2. Model score correlation between cell types
         - For seq2func: baseline_mean_sad, filtered_mean_sad, filtered_mean_abs_sad
         - For foundation models: ND_influence_score, ND_mean, ND_mean_ABS
      3. Differential significance analysis:
         - Variants significant in ct1 but not ct2 (and vice versa)
         - Do model scores distinguish differential variants?
      4. Scatter + residual analysis
    """
    print("\n" + "=" * 70)
    print("Q2.3: Cross-cell-type rank consistency")
    print("=" * 70)

    cts = list(gt_all.keys())
    rows = []

    for i in range(len(cts)):
        for j in range(i + 1, len(cts)):
            ct1, ct2 = cts[i], cts[j]
            lab1 = EXPERIMENTS[ct1]["label"]
            lab2 = EXPERIMENTS[ct2]["label"]

            gt1 = gt_all[ct1].set_index("variant_id")
            gt2 = gt_all[ct2].set_index("variant_id")
            shared = gt1.index.intersection(gt2.index)

            if len(shared) < 100:
                print(f"\n  {lab1} vs {lab2}: only {len(shared)} shared variants, skipping")
                continue

            print(f"\n  {lab1} vs {lab2}: {len(shared):,} shared variants")

            lfc1 = gt1.loc[shared, "log2FC"].values.astype(np.float64)
            lfc2 = gt2.loc[shared, "log2FC"].values.astype(np.float64)
            sig1 = gt1.loc[shared, "sig"].values.astype(bool)
            sig2 = gt2.loc[shared, "sig"].values.astype(bool)

            # Ground-truth MPRA correlation
            sp_gt, _ = spearmanr(lfc1, lfc2)
            print(f"    MPRA log2FC Spearman: {sp_gt:.4f}")

            # Differential significance categories
            both_sig = sig1 & sig2
            only1 = sig1 & ~sig2
            only2 = ~sig1 & sig2
            neither = ~sig1 & ~sig2
            print(f"    Sig both: {both_sig.sum():,}  only {lab1}: {only1.sum():,}  "
                  f"only {lab2}: {only2.sum():,}  neither: {neither.sum():,}")

            rows.append({
                "ct1": ct1, "ct2": ct2, "model": "MPRA_ground_truth",
                "score_type": "log2FC",
                "n_shared": int(len(shared)),
                "spearman": float(sp_gt),
                "n_both_sig": int(both_sig.sum()),
                "n_only_ct1_sig": int(only1.sum()),
                "n_only_ct2_sig": int(only2.sum()),
            })

            if not no_plots:
                _plot_cross_ct_scatter(
                    lfc1, lfc2, sig1, sig2,
                    f"MPRA log2FC: {lab1} vs {lab2}\nSpearman={sp_gt:.4f}",
                    f"{lab1} log2FC", f"{lab2} log2FC",
                    out_dir / f"cross_ct_{ct1}_{ct2}_mpra_log2fc.png",
                )

            # --- Model score comparison ---
            # Build index maps once per pair (O(n) dict build, O(1) lookup per variant)
            _pos1 = {v: i for i, v in enumerate(gt_all[ct1]["variant_id"].values)}
            _pos2 = {v: i for i, v in enumerate(gt_all[ct2]["variant_id"].values)}
            idx1 = np.array([_pos1[v] for v in shared], dtype=np.int32)
            idx2 = np.array([_pos2[v] for v in shared], dtype=np.int32)

            all_models = set()
            for ct in [ct1, ct2]:
                if ct in scores_all:
                    all_models.update(scores_all[ct].keys())

            for model in sorted(all_models):
                sc1 = scores_all.get(ct1, {}).get(model)
                sc2 = scores_all.get(ct2, {}).get(model)
                if sc1 is None or sc2 is None:
                    continue

                # Determine which score columns to compare
                if model in SEQ2FUNC_MODELS:
                    score_types = ["baseline_mean_sad", "filtered_mean_sad", "filtered_mean_abs_sad"]
                else:
                    score_types = list(sc1.keys())

                for st in score_types:
                    if st not in sc1 or st not in sc2:
                        continue
                    s1 = sc1[st][idx1]
                    s2 = sc2[st][idx2]
                    valid = np.isfinite(s1) & np.isfinite(s2)
                    if valid.sum() < 10:
                        continue

                    sp_model, _ = spearmanr(s1[valid], s2[valid])
                    print(f"    {model} | {st}: Spearman(ct1 vs ct2)={sp_model:.4f}")

                    rows.append({
                        "ct1": ct1, "ct2": ct2, "model": model,
                        "score_type": st,
                        "n_shared": int(valid.sum()),
                        "spearman": float(sp_model),
                        "n_both_sig": int(both_sig.sum()),
                        "n_only_ct1_sig": int(only1.sum()),
                        "n_only_ct2_sig": int(only2.sum()),
                    })

                    if not no_plots and st in ("filtered_mean_sad", "ND_mean"):
                        _plot_cross_ct_scatter(
                            s1[valid], s2[valid],
                            sig1[valid], sig2[valid],
                            f"{model} | {st}: {lab1} vs {lab2}\nSpearman={sp_model:.4f}",
                            f"{lab1} score", f"{lab2} score",
                            out_dir / f"cross_ct_{ct1}_{ct2}_{model}_{st}.png",
                        )

                # --- Differential analysis ---
                # For the primary score, check if model captures cell-type differential
                primary_st = "filtered_mean_sad" if model in SEQ2FUNC_MODELS else (
                    "ND_mean" if "ND_mean" in sc1 else list(sc1.keys())[0]
                )
                if primary_st in sc1 and primary_st in sc2:
                    s1p = sc1[primary_st][idx1]
                    s2p = sc2[primary_st][idx2]
                    delta_model = s1p - s2p   # positive = higher score in ct1
                    delta_mpra = lfc1 - lfc2  # positive = higher log2FC in ct1
                    valid_d = np.isfinite(delta_model) & np.isfinite(delta_mpra)
                    if valid_d.sum() >= 10:
                        sp_diff, _ = spearmanr(delta_mpra[valid_d], delta_model[valid_d])
                        print(f"    {model} | differential Δscore vs Δlog2FC: Spearman={sp_diff:.4f}")
                        rows.append({
                            "ct1": ct1, "ct2": ct2, "model": model,
                            "score_type": f"differential_{primary_st}",
                            "n_shared": int(valid_d.sum()),
                            "spearman": float(sp_diff),
                            "n_both_sig": int(both_sig.sum()),
                            "n_only_ct1_sig": int(only1.sum()),
                            "n_only_ct2_sig": int(only2.sum()),
                        })

                        if not no_plots:
                            _plot_cross_ct_scatter(
                                delta_mpra[valid_d], delta_model[valid_d],
                                (sig1 | sig2)[valid_d], (sig1 & sig2)[valid_d],
                                (f"{model} | Δ{primary_st} vs Δlog2FC\n"
                                 f"{lab1}−{lab2}, Spearman={sp_diff:.4f}"),
                                f"Δlog2FC ({lab1}−{lab2})",
                                f"Δscore ({lab1}−{lab2})",
                                out_dir / f"cross_ct_{ct1}_{ct2}_{model}_differential.png",
                            )

    if rows:
        df = pd.DataFrame(rows)
        out = out_dir / "cross_celltype_metrics.csv"
        df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")
    return rows


def _plot_cross_ct_scatter(x, y, sig_group1, sig_group2, title, xlabel, ylabel, out_path):
    """Scatter with color coding for significance categories."""
    fig, ax = plt.subplots(figsize=(6, 6))
    both = sig_group1 & sig_group2
    only1 = sig_group1 & ~sig_group2
    only2 = ~sig_group1 & sig_group2
    neither = ~sig_group1 & ~sig_group2

    for mask, color, label, alpha, zorder in [
        (neither, "grey", "neither sig", 0.15, 1),
        (only1, "#E65100", "sig ct1 only", 0.5, 2),
        (only2, "#1565C0", "sig ct2 only", 0.5, 2),
        (both, "#2E7D32", "both sig", 0.6, 3),
    ]:
        if mask.sum() > 0:
            ax.scatter(x[mask], y[mask], c=color, s=6, alpha=alpha, label=label, zorder=zorder, linewidths=0)

    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")
    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k-", lw=0.5, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        xlabel: x, ylabel: y,
        "sig_group1": sig_group1.astype(int),
        "sig_group2": sig_group2.astype(int),
    }).to_csv(plot_data_path(out_path), index=False)


# =========================================================================
# Analysis Q1.2 / Q1.3:  Linear model integration of assay scores
# =========================================================================

def run_linear_model(gt_all, scores_all, out_dir, no_plots):
    """For each seq2func model × cell type, build a Ridge/Lasso regression
    from per-assay SAD scores to predict MPRA log2FC.

    This answers:
      Q1.2: Does a linear combination of assays outperform single-track mean?
      Q1.3: Which assays are most informative?

    Approach:
      1. Build feature matrix X (n_variants × n_assays) from cell-type-matched
         per-assay mean_sad scores.
      2. Target y = log2FC.
      3. Ridge regression with 5-fold CV → R² and Spearman of held-out predictions.
      4. Lasso for automatic feature selection → which assays survive?
      5. Compare to single-score baseline (filtered_mean_sad Spearman).
    """
    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("\n" + "=" * 70)
    print("Q1.2 / Q1.3: Linear integration of assay scores")
    print("=" * 70)

    rows = []

    for ct in gt_all:
        gt = gt_all[ct]
        log2fc = gt["log2FC"].values.astype(np.float64)
        sig = gt["sig"].values.astype(bool)
        label = EXPERIMENTS[ct]["label"]

        for model in SEQ2FUNC_MODELS:
            sc = scores_all.get(ct, {}).get(model)
            if sc is None or "assay_scores" not in sc:
                continue
            assay_scores = sc["assay_scores"]
            if len(assay_scores) < 2:
                print(f"\n  {label}/{model}: only {len(assay_scores)} assay types, skipping regression")
                continue

            print(f"\n  {label} / {model}: {len(assay_scores)} assay types")

            # Build feature matrix for mean_sad
            for score_type in ["mean_sad", "mean_abs_sad"]:
                assay_names = sorted(assay_scores.keys())
                X = np.column_stack([assay_scores[a][score_type] for a in assay_names])
                y = log2fc.copy()

                # Keep only variants where ALL assays and log2FC are valid
                valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
                nv = valid.sum()
                if nv < 50:
                    print(f"    {score_type}: only {nv} variants with all assays, skipping")
                    continue

                Xv, yv = X[valid], y[valid]
                sv = sig[valid]
                print(f"    {score_type}: {nv:,} complete variants, {int(sv.sum()):,} significant")

                # --- Individual assay performance ---
                indiv_rows = []
                for ai, aname in enumerate(assay_names):
                    sp_all, _ = spearmanr(yv, Xv[:, ai])
                    sp_sig = np.nan
                    if sv.sum() >= 10:
                        sp_sig, _ = spearmanr(yv[sv], Xv[:, ai][sv])
                    indiv_rows.append({
                        "cell_type": ct, "model": model, "score_type": score_type,
                        "assay": aname, "method": "individual",
                        "spearman_all": float(sp_all), "spearman_sig": float(sp_sig),
                        "n": int(nv), "n_sig": int(sv.sum()),
                    })
                indiv_df = pd.DataFrame(indiv_rows).sort_values("spearman_all", ascending=False)
                print(f"    Top assay (individual): {indiv_df.iloc[0]['assay']} "
                      f"Spearman={indiv_df.iloc[0]['spearman_all']:.4f}")

                # --- Baseline: single-score filtered_mean_sad ---
                filt_score = sc["filtered_mean_sad" if score_type == "mean_sad"
                                else "filtered_mean_abs_sad"]
                filt_valid = np.isfinite(filt_score) & np.isfinite(y)
                if filt_valid.sum() >= 10:
                    sp_baseline, _ = spearmanr(y[filt_valid], filt_score[filt_valid])
                    sp_baseline_sig = np.nan
                    if sig[filt_valid].sum() >= 10:
                        sp_baseline_sig, _ = spearmanr(
                            y[filt_valid & sig], filt_score[filt_valid & sig]
                        )
                else:
                    sp_baseline = sp_baseline_sig = np.nan
                print(f"    Baseline filtered {score_type}: Spearman_all={sp_baseline:.4f}, "
                      f"Spearman_sig={sp_baseline_sig:.4f}")

                # --- Ridge regression (CV) ---
                ridge_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)),
                ])
                y_pred_ridge = cross_val_predict(ridge_pipe, Xv, yv, cv=5)
                sp_ridge, _ = spearmanr(yv, y_pred_ridge)
                sp_ridge_sig = np.nan
                if sv.sum() >= 10:
                    sp_ridge_sig, _ = spearmanr(yv[sv], y_pred_ridge[sv])

                # Fit final model for coefficients
                ridge_pipe.fit(Xv, yv)
                ridge_coefs = ridge_pipe.named_steps["ridge"].coef_
                ridge_alpha = ridge_pipe.named_steps["ridge"].alpha_

                print(f"    Ridge CV: Spearman_all={sp_ridge:.4f}, "
                      f"Spearman_sig={sp_ridge_sig:.4f}, alpha={ridge_alpha:.2e}")

                # --- Lasso regression for feature selection ---
                lasso_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lasso", LassoCV(alphas=np.logspace(-5, 1, 50), cv=5, max_iter=10000)),
                ])
                y_pred_lasso = cross_val_predict(lasso_pipe, Xv, yv, cv=5)
                sp_lasso, _ = spearmanr(yv, y_pred_lasso)
                sp_lasso_sig = np.nan
                if sv.sum() >= 10:
                    sp_lasso_sig, _ = spearmanr(yv[sv], y_pred_lasso[sv])

                lasso_pipe.fit(Xv, yv)
                lasso_coefs = lasso_pipe.named_steps["lasso"].coef_
                lasso_alpha = lasso_pipe.named_steps["lasso"].alpha_
                n_selected = int(np.sum(np.abs(lasso_coefs) > 1e-10))

                print(f"    Lasso CV: Spearman_all={sp_lasso:.4f}, "
                      f"Spearman_sig={sp_lasso_sig:.4f}, {n_selected}/{len(assay_names)} assays selected")

                # --- Mean of top-K assays (simpler integration) ---
                top_k_rows = []
                ranked_assays = indiv_df["assay"].values
                for k in range(1, len(ranked_assays) + 1):
                    top_cols = [assay_names.index(a) for a in ranked_assays[:k]]
                    mean_top_k = Xv[:, top_cols].mean(axis=1)
                    sp_topk, _ = spearmanr(yv, mean_top_k)
                    top_k_rows.append({"k": k, "spearman_all": float(sp_topk)})
                top_k_df = pd.DataFrame(top_k_rows)
                best_k = top_k_df.loc[top_k_df["spearman_all"].abs().idxmax()]
                print(f"    Best mean-of-top-K: K={int(best_k['k'])}, Spearman={best_k['spearman_all']:.4f}")

                # Collect summary rows
                rows.append({
                    "cell_type": ct, "model": model, "score_type": score_type,
                    "method": "baseline_filtered", "spearman_all": float(sp_baseline),
                    "spearman_sig": float(sp_baseline_sig),
                })
                rows.append({
                    "cell_type": ct, "model": model, "score_type": score_type,
                    "method": "ridge_cv", "spearman_all": float(sp_ridge),
                    "spearman_sig": float(sp_ridge_sig),
                })
                rows.append({
                    "cell_type": ct, "model": model, "score_type": score_type,
                    "method": "lasso_cv", "spearman_all": float(sp_lasso),
                    "spearman_sig": float(sp_lasso_sig),
                    "n_assays_selected": n_selected,
                })
                rows.append({
                    "cell_type": ct, "model": model, "score_type": score_type,
                    "method": f"mean_top_{int(best_k['k'])}",
                    "spearman_all": float(best_k["spearman_all"]),
                })

                # --- Plots ---
                if not no_plots:
                    # (a) Ridge coefficient bar plot
                    coef_df = pd.DataFrame({
                        "assay": assay_names,
                        "ridge_coef": ridge_coefs,
                        "lasso_coef": lasso_coefs,
                    }).sort_values("ridge_coef", ascending=True)

                    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(assay_names) * 0.5)))
                    ax1, ax2 = axes

                    ax1.barh(coef_df["assay"], coef_df["ridge_coef"], color="steelblue")
                    ax1.axvline(0, color="k", lw=0.5)
                    ax1.set_xlabel("Standardized Ridge coefficient")
                    ax1.set_title(f"{label} | {model} | {score_type}\nRidge (α={ridge_alpha:.1e})", fontsize=9)

                    lasso_sorted = coef_df.sort_values("lasso_coef", ascending=True)
                    colors = ["#E65100" if abs(c) > 1e-10 else "lightgrey" for c in lasso_sorted["lasso_coef"]]
                    ax2.barh(lasso_sorted["assay"], lasso_sorted["lasso_coef"], color=colors)
                    ax2.axvline(0, color="k", lw=0.5)
                    ax2.set_xlabel("Standardized Lasso coefficient")
                    ax2.set_title(f"Lasso (α={lasso_alpha:.1e}, {n_selected} selected)", fontsize=9)

                    plt.tight_layout()
                    out = out_dir / f"linear_model_{ct}_{model}_{score_type}_coefficients.png"
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    coef_df.to_csv(plot_data_path(out), index=False)

                    # (b) Comparison bar: baseline vs ridge vs lasso vs top-K
                    fig, ax = plt.subplots(figsize=(7, 4))
                    methods = ["baseline_filtered", "ridge_cv", "lasso_cv",
                               f"mean_top_{int(best_k['k'])}"]
                    vals = [sp_baseline, sp_ridge, sp_lasso, best_k["spearman_all"]]
                    colors_bar = ["grey", "#1565C0", "#E65100", "#2E7D32"]
                    ax.bar(methods, vals, color=colors_bar)
                    ax.set_ylabel("Spearman (all variants)")
                    ax.set_title(f"{label} | {model} | {score_type}\nIntegration method comparison", fontsize=9)
                    for xi, vi in enumerate(vals):
                        ax.text(xi, vi + 0.002, f"{vi:.4f}", ha="center", fontsize=8)
                    plt.xticks(rotation=20, ha="right")
                    plt.tight_layout()
                    out = out_dir / f"linear_model_{ct}_{model}_{score_type}_comparison.png"
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    # (c) Mean-of-top-K curve
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(top_k_df["k"], top_k_df["spearman_all"], marker="o", markersize=3, color="steelblue")
                    ax.axhline(sp_baseline, color="grey", ls="--", lw=1, label=f"baseline filtered ({sp_baseline:.4f})")
                    ax.axhline(sp_ridge, color="#1565C0", ls="--", lw=1, label=f"ridge ({sp_ridge:.4f})")
                    ax.set_xlabel("K (number of top assays)")
                    ax.set_ylabel("Spearman (all)")
                    ax.set_title(f"{label} | {model} | {score_type}\nMean of top-K assays", fontsize=9)
                    ax.legend(fontsize=7)
                    plt.tight_layout()
                    out = out_dir / f"linear_model_{ct}_{model}_{score_type}_topK.png"
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    top_k_df.to_csv(plot_data_path(out), index=False)

                # Save individual assay performance
                indiv_out = out_dir / f"linear_model_{ct}_{model}_{score_type}_individual_assays.csv"
                indiv_df.to_csv(indiv_out, index=False)

    if rows:
        df = pd.DataFrame(rows)
        out = out_dir / "linear_model_metrics.csv"
        df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")
    return rows


# =========================================================================
# Analysis Q3.1:  Performance by genomic region (promoter vs enhancer)
# =========================================================================

def _load_ccre(ccre_path):
    """Load ENCODE cCRE BED file.

    Expected format (tab-delimited, V3):
      chrom  start  end  accession1  accession2  classification

    Classification examples: "dELS", "PLS,CTCF-bound", "pELS,CTCF-bound"
    We take the PRIMARY type (before comma): PLS, pELS, dELS, CTCF-only, DNase-H3K4me3.

    Returns a dict: chrom -> sorted list of (start, end, primary_cCRE_type)
    """
    regions = {}
    with open(ccre_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 6:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            # Classification is in column 6 (index 5); take primary type before comma
            ccre_type = parts[5].split(",")[0].strip()
            if chrom not in regions:
                regions[chrom] = []
            regions[chrom].append((start, end, ccre_type))
    # Sort by start position for binary search
    for chrom in regions:
        regions[chrom].sort()
    return regions


def _annotate_variant_region(variant_id, ccre_regions):
    """Annotate a variant_id (chr:pos:ref>alt) with cCRE type using binary search."""
    parts = variant_id.split(":")
    chrom = parts[0]
    pos = int(parts[1]) - 1  # VCF is 1-based, BED is 0-based

    if chrom not in ccre_regions:
        return "intergenic"

    intervals = ccre_regions[chrom]
    # Binary search for overlapping interval
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, _ = intervals[mid]
        if pos < start:
            hi = mid - 1
        elif pos >= end:
            lo = mid + 1
        else:
            return intervals[mid][2]  # found overlap

    # Check neighboring intervals (binary search may miss due to overlapping regions)
    for check in range(max(0, lo - 2), min(len(intervals), lo + 3)):
        s, e, t = intervals[check]
        if s <= pos < e:
            return t

    return "intergenic"


def run_genomic_regions(gt_all, scores_all, out_dir, no_plots, ccre_path):
    """Compute metrics stratified by genomic region (cCRE classification).

    For each variant, determine if it falls in:
      - PLS (promoter-like signature)
      - pELS (proximal enhancer-like)
      - dELS (distal enhancer-like)
      - CTCF-bound
      - DNase-H3K4me3
      - intergenic (no cCRE overlap)
    Then compute Spearman, AUROC, AUPRC per region per model.
    """
    print("\n" + "=" * 70)
    print("Q3.1: Performance by genomic region")
    print("=" * 70)

    if ccre_path is None:
        ccre_path = str(BASE / "Data/GRCh38-cCREs.bed")
    # Search candidate locations in priority order
    candidates = [
        Path(ccre_path),
        BASE / "Data/GRCh38-cCREs.bed",
        Path("/tmp/GRCh38-cCREs.bed"),
    ]
    resolved = next((str(p) for p in candidates if p.exists()), None)
    if resolved is None:
        print(f"\n  ERROR: cCRE BED file not found. Searched:")
        for p in candidates:
            print(f"    {p}")
        print("  Download it with:")
        print(f"    wget -O Data/GRCh38-cCREs.bed https://downloads.wenglab.org/V3/GRCh38-cCREs.bed")
        return []

    ccre_path = resolved
    print(f"  Loading cCRE annotations from {ccre_path} ...", flush=True)
    ccre = _load_ccre(ccre_path)
    n_regions = sum(len(v) for v in ccre.values())
    print(f"  Loaded {n_regions:,} cCRE regions across {len(ccre)} chromosomes")

    rows = []

    for ct in gt_all:
        gt = gt_all[ct]
        label = EXPERIMENTS[ct]["label"]
        log2fc = gt["log2FC"].values.astype(np.float64)
        sig = gt["sig"].values.astype(bool)

        # Annotate variants
        print(f"\n  Annotating {len(gt):,} variants for {label} ...", flush=True)
        region_labels = np.array([
            _annotate_variant_region(vid, ccre) for vid in gt["variant_id"].values
        ])

        region_counts = pd.Series(region_labels).value_counts()
        print(f"  Region distribution:")
        for reg, cnt in region_counts.items():
            print(f"    {reg}: {cnt:,} ({100*cnt/len(gt):.1f}%)")

        for model in sorted(scores_all.get(ct, {}).keys()):
            sc = scores_all[ct][model]

            # Determine score columns to evaluate
            if model in SEQ2FUNC_MODELS:
                score_cols = {"filtered_mean_sad": sc.get("filtered_mean_sad"),
                              "filtered_mean_abs_sad": sc.get("filtered_mean_abs_sad"),
                              "baseline_mean_sad": sc.get("baseline_mean_sad")}
            else:
                score_cols = {k: v for k, v in sc.items()}

            for st_name, score_arr in score_cols.items():
                if score_arr is None:
                    continue

                for region in sorted(set(region_labels)):
                    mask = region_labels == region
                    if mask.sum() < 20:
                        continue
                    m = compute_metrics(log2fc[mask], sig[mask], score_arr[mask])
                    if m is None:
                        continue
                    m.update({
                        "cell_type": ct, "model": model, "score_type": st_name,
                        "region": region,
                    })
                    rows.append(m)

            # Overall metrics for reference
            for st_name, score_arr in score_cols.items():
                if score_arr is None:
                    continue
                m = compute_metrics(log2fc, sig, score_arr)
                if m is not None:
                    m.update({
                        "cell_type": ct, "model": model, "score_type": st_name,
                        "region": "ALL",
                    })
                    rows.append(m)

        # --- Plots ---
        if not no_plots:
            for model in sorted(scores_all.get(ct, {}).keys()):
                sc = scores_all[ct][model]
                primary_st = "filtered_mean_sad" if model in SEQ2FUNC_MODELS else (
                    "ND_mean" if "ND_mean" in sc else list(sc.keys())[0]
                )
                score_arr = sc.get(primary_st)
                if score_arr is None:
                    continue

                region_order = ["PLS", "pELS", "dELS", "CTCF-bound", "DNase-H3K4me3", "intergenic"]
                plot_regions = [r for r in region_order if r in set(region_labels)]

                sp_vals = []
                auroc_vals = []
                n_vals = []
                for region in plot_regions:
                    mask = region_labels == region
                    m = compute_metrics(log2fc[mask], sig[mask], score_arr[mask])
                    if m:
                        sp_vals.append(m["spearman_r"])
                        auroc_vals.append(m["auroc"])
                        n_vals.append(m["n"])
                    else:
                        sp_vals.append(np.nan)
                        auroc_vals.append(np.nan)
                        n_vals.append(0)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                colors = plt.cm.Set2(np.linspace(0, 1, len(plot_regions)))
                bars1 = ax1.bar(plot_regions, sp_vals, color=colors)
                ax1.axhline(0, color="k", lw=0.5)
                ax1.set_ylabel("Spearman r (all variants)")
                ax1.set_title(f"{label} | {model} | {primary_st}\nSpearman by genomic region", fontsize=9)
                for bi, (bar, n) in enumerate(zip(bars1, n_vals)):
                    ax1.text(bi, bar.get_height() + 0.003, f"n={n:,}", ha="center", fontsize=7, rotation=0)
                ax1.tick_params(axis="x", rotation=30)

                bars2 = ax2.bar(plot_regions, auroc_vals, color=colors)
                ax2.axhline(0.5, color="k", lw=0.5, ls="--")
                ax2.set_ylabel("AUROC")
                ax2.set_title(f"AUROC by genomic region", fontsize=9)
                ax2.tick_params(axis="x", rotation=30)

                plt.tight_layout()
                out = out_dir / f"genomic_region_{ct}_{model}_{primary_st}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)

                pd.DataFrame({
                    "region": plot_regions,
                    "spearman_r": sp_vals,
                    "auroc": auroc_vals,
                    "n": n_vals,
                }).to_csv(plot_data_path(out), index=False)

    if rows:
        df = pd.DataFrame(rows)
        out = out_dir / "genomic_region_metrics.csv"
        df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")
    return rows


# =========================================================================
# Analysis Q2.2:  Cell-type prediction from model scores
# =========================================================================

def run_celltype_prediction(gt_all, scores_all, out_dir, no_plots):
    """For shared variants between cell types, predict differential cell-type
    activity from model scores.

    Method:
      - Take variants shared between two cell types (HepG2 ↔ HEK293T).
      - Define differential labels based on MPRA significance:
        * "ct1_specific": sig in ct1 but not ct2
        * "ct2_specific": sig in ct2 but not ct1
        * "both" / "neither" for reference
      - Feature matrix: For seq2func models, per-assay-type scores as features.
        For foundation models, ND score columns as features.
      - Logistic regression to classify ct1_specific vs ct2_specific.
      - Which features (tracks/assays) best discriminate cell-type-specific activity?

    Alternative framing (also implemented):
      - Continuous target: Δlog2FC = log2FC_ct1 − log2FC_ct2
      - Ridge regression from model features → predict Δlog2FC direction/magnitude
    """
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score, classification_report

    print("\n" + "=" * 70)
    print("Q2.2: Cell-type prediction from model scores")
    print("=" * 70)

    cts = list(gt_all.keys())
    rows = []

    for i in range(len(cts)):
        for j in range(i + 1, len(cts)):
            ct1, ct2 = cts[i], cts[j]
            lab1 = EXPERIMENTS[ct1]["label"]
            lab2 = EXPERIMENTS[ct2]["label"]

            gt1 = gt_all[ct1].set_index("variant_id")
            gt2 = gt_all[ct2].set_index("variant_id")
            shared = gt1.index.intersection(gt2.index)

            if len(shared) < 200:
                print(f"\n  {lab1} vs {lab2}: only {len(shared)} shared variants, skipping")
                continue

            print(f"\n  {lab1} vs {lab2}: {len(shared):,} shared variants")

            sig1 = gt1.loc[shared, "sig"].values.astype(bool)
            sig2 = gt2.loc[shared, "sig"].values.astype(bool)
            lfc1 = gt1.loc[shared, "log2FC"].values.astype(np.float64)
            lfc2 = gt2.loc[shared, "log2FC"].values.astype(np.float64)

            # Differential labels
            only1 = sig1 & ~sig2  # active in ct1, not ct2
            only2 = ~sig1 & sig2  # active in ct2, not ct1
            diff_mask = only1 | only2
            n_diff = diff_mask.sum()
            print(f"    Differential variants: {n_diff:,} "
                  f"({only1.sum():,} {lab1}-specific, {only2.sum():,} {lab2}-specific)")

            if n_diff < 50:
                print(f"    Too few differential variants, skipping classification")
                continue

            # Labels: 1 = ct1-specific, 0 = ct2-specific
            y_class = only1[diff_mask].astype(int)

            # Build feature index maps for shared variants — dict lookup is O(1) per variant
            _pos1 = {v: i for i, v in enumerate(gt_all[ct1]["variant_id"].values)}
            _pos2 = {v: i for i, v in enumerate(gt_all[ct2]["variant_id"].values)}
            idx1 = np.array([_pos1[v] for v in shared], dtype=np.int32)
            idx2 = np.array([_pos2[v] for v in shared], dtype=np.int32)

            all_models = set()
            for ct in [ct1, ct2]:
                if ct in scores_all:
                    all_models.update(scores_all[ct].keys())

            for model in sorted(all_models):
                sc1 = scores_all.get(ct1, {}).get(model)
                sc2 = scores_all.get(ct2, {}).get(model)
                if sc1 is None and sc2 is None:
                    continue

                print(f"\n    Model: {model}")

                # --- Approach 1: Classification from assay features ---
                # Build feature matrix: for seq2func models, use per-assay scores
                # from EITHER cell type context (we use ct1 scores since they're
                # the same DNA sequence — the assay predictions don't depend on
                # MPRA cell type).
                feat_names = []
                feat_cols = []

                if model in SEQ2FUNC_MODELS and sc1 is not None and "assay_scores" in sc1:
                    for assay in sorted(sc1["assay_scores"].keys()):
                        arr = sc1["assay_scores"][assay]["mean_sad"][idx1]
                        feat_names.append(f"{assay}_mean_sad")
                        feat_cols.append(arr)
                        arr_abs = sc1["assay_scores"][assay]["mean_abs_sad"][idx1]
                        feat_names.append(f"{assay}_mean_abs_sad")
                        feat_cols.append(arr_abs)
                elif model in FOUNDATION_MODELS:
                    sc_use = sc1 if sc1 is not None else sc2
                    idx_use = idx1 if sc1 is not None else idx2
                    for col_name in sorted(sc_use.keys()):
                        arr = sc_use[col_name][idx_use]
                        feat_names.append(col_name)
                        feat_cols.append(arr)

                if len(feat_cols) < 2:
                    print(f"      Not enough features ({len(feat_cols)}), skipping")
                    continue

                X = np.column_stack(feat_cols)
                valid = np.all(np.isfinite(X), axis=1)
                X_diff = X[diff_mask]
                valid_diff = np.all(np.isfinite(X_diff), axis=1)

                if valid_diff.sum() < 30:
                    print(f"      Only {valid_diff.sum()} valid differential variants, skipping")
                    continue

                X_clf = X_diff[valid_diff]
                y_clf = y_class[valid_diff]

                print(f"      Classification: {valid_diff.sum()} variants, "
                      f"{len(feat_names)} features")

                # Logistic Regression with CV
                lr_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegressionCV(cv=5, max_iter=5000, penalty="l2",
                                                scoring="roc_auc")),
                ])
                y_pred_proba = cross_val_predict(lr_pipe, X_clf, y_clf, cv=5, method="predict_proba")[:, 1]
                try:
                    auc = roc_auc_score(y_clf, y_pred_proba)
                except ValueError:
                    auc = np.nan

                # Fit final model for coefficients
                lr_pipe.fit(X_clf, y_clf)
                lr_coefs = lr_pipe.named_steps["lr"].coef_.ravel()

                print(f"      Classification AUROC: {auc:.4f}")

                rows.append({
                    "ct1": ct1, "ct2": ct2, "model": model,
                    "task": "classification",
                    "n_diff": int(valid_diff.sum()),
                    "n_ct1_specific": int(y_clf.sum()),
                    "n_ct2_specific": int(len(y_clf) - y_clf.sum()),
                    "auroc": float(auc),
                })

                if not no_plots:
                    # Feature importance bar plot
                    feat_imp = pd.DataFrame({
                        "feature": feat_names,
                        "lr_coef": lr_coefs,
                    }).sort_values("lr_coef", ascending=True)

                    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_names) * 0.35)))
                    colors = ["#E65100" if c > 0 else "#1565C0" for c in feat_imp["lr_coef"]]
                    ax.barh(feat_imp["feature"], feat_imp["lr_coef"], color=colors)
                    ax.axvline(0, color="k", lw=0.5)
                    ax.set_xlabel("Logistic regression coefficient (standardized)")
                    ax.set_title(
                        f"{model}: predicting {lab1}-specific vs {lab2}-specific\n"
                        f"AUROC={auc:.4f}", fontsize=9,
                    )
                    plt.tight_layout()
                    out = out_dir / f"celltype_pred_{ct1}_{ct2}_{model}_coefficients.png"
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    feat_imp.to_csv(plot_data_path(out), index=False)

                # --- Approach 2: Regression on Δlog2FC ---
                # Can model features predict the continuous differential effect?
                delta_lfc = lfc1 - lfc2
                X_all = X[valid]
                delta_valid = delta_lfc[valid]
                finite_y = np.isfinite(delta_valid)
                X_reg = X_all[finite_y]
                y_reg = delta_valid[finite_y]

                if len(y_reg) >= 50:
                    ridge_pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("ridge", RidgeCV(alphas=np.logspace(-3, 5, 30), cv=5)),
                    ])
                    y_pred_reg = cross_val_predict(ridge_pipe, X_reg, y_reg, cv=5)
                    sp_reg, _ = spearmanr(y_reg, y_pred_reg)
                    print(f"      Regression Δlog2FC: Spearman={sp_reg:.4f}")

                    rows.append({
                        "ct1": ct1, "ct2": ct2, "model": model,
                        "task": "regression_delta_log2fc",
                        "n_diff": int(len(y_reg)),
                        "spearman": float(sp_reg),
                    })

                    if not no_plots:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.scatter(y_reg, y_pred_reg, s=3, alpha=0.15, color="steelblue", linewidths=0)
                        ax.axhline(0, color="k", lw=0.4, ls="--")
                        ax.axvline(0, color="k", lw=0.4, ls="--")
                        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                                max(ax.get_xlim()[1], ax.get_ylim()[1])]
                        ax.plot(lims, lims, "k-", lw=0.5, alpha=0.3)
                        ax.set_xlabel(f"Δlog2FC ({lab1}−{lab2})")
                        ax.set_ylabel("Predicted Δlog2FC (CV)")
                        ax.set_title(
                            f"{model}: predicting Δlog2FC\nSpearman={sp_reg:.4f}", fontsize=9
                        )
                        plt.tight_layout()
                        out = out_dir / f"celltype_pred_{ct1}_{ct2}_{model}_delta_regression.png"
                        fig.savefig(out, dpi=150, bbox_inches="tight")
                        plt.close(fig)

    if rows:
        df = pd.DataFrame(rows)
        out = out_dir / "celltype_prediction_metrics.csv"
        df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")
    return rows


# =========================================================================
# Main orchestration
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended MPRA benchmark analyses (Q2.3, Q1.2/Q1.3, Q3.1, Q2.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cell-types", nargs="+",
        default=["hepg2", "hek293t", "ngn2"],
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument(
        "--analyses", nargs="+",
        default=["cross_celltype", "linear_model", "genomic_regions", "celltype_prediction"],
        choices=["cross_celltype", "linear_model", "genomic_regions", "celltype_prediction"],
    )
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--ccre-bed", type=str,
        default=str(BASE / "Data/GRCh38-cCREs.bed"),
        help="Path to ENCODE cCRE BED file for genomic region analysis",
    )
    parser.add_argument(
        "--exclude-biosample-regex", type=str,
        default=DEFAULT_EXCLUDE_RE,
        help="Regex for biosamples to exclude (genetically modified etc.)",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore cached scores and recompute from parquet",
    )
    parser.add_argument(
        "--sig-only",
        action="store_true",
        default=False,
        help="Restrict all analyses to only significant variants (padj < alpha=0.1). "
             "Outputs go to results/extended_analyses_sigonly/.",
    )
    parser.add_argument(
        "--parquet-dir", type=str, default="",
        help="Override root dir for model parquets. Expects {parquet-dir}/{ct}/{model}/{model}_scores.parquet.",
    )
    return parser.parse_args()


def main():
    global PARQUET_ROOT
    args = parse_args()
    if args.parquet_dir:
        PARQUET_ROOT = Path(args.parquet_dir)
        print(f"Parquet root overridden: {PARQUET_ROOT}")

    _out_subdir = "extended_analyses_sigonly" if args.sig_only else "extended_analyses"
    out_dir = BASE / "results" / _out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ground truth for all requested cell types ---
    gt_all = {}
    for ct in args.cell_types:
        cfg = EXPERIMENTS[ct]
        gt = load_gt(cfg)
        if args.sig_only:
            n_before = len(gt)
            gt = gt[gt["sig"]].reset_index(drop=True)
            print(f"  --sig-only [{cfg['label']}]: {n_before:,} → {len(gt):,} significant variants")
        gt_all[ct] = gt
        print(f"Loaded {cfg['label']}: {len(gt):,} variants, {int(gt['sig'].sum()):,} significant")

    # --- Load / compute model scores ---
    print("\n--- Loading model scores ---")
    scores_all = {}  # scores_all[ct][model] = dict of score arrays

    for ct in args.cell_types:
        scores_all[ct] = {}
        gt = gt_all[ct]
        gt_index = {v: i for i, v in enumerate(gt["variant_id"].values)}
        n_var = len(gt)

        # Seq2func models
        for model in SEQ2FUNC_MODELS:
            if args.force_recompute:
                # Remove caches
                for kind in ["baseline", "filtered"]:
                    p = _cache_path(ct, model, kind)
                    if p.exists():
                        p.unlink()
                p = TMP_ROOT / ct / f"{model}_assay_cache.npz"
                if p.exists():
                    p.unlink()

            result = get_seq2func_variant_scores(
                ct, model, gt_index, n_var, args.batch_size,
                args.exclude_biosample_regex,
            )
            if result is not None:
                scores_all[ct][model] = result

        # Foundation models
        for model in FOUNDATION_MODELS:
            result = get_foundation_model_scores(ct, model, gt_index, n_var)
            if result is not None:
                scores_all[ct][model] = result

    # --- Run requested analyses ---
    if "cross_celltype" in args.analyses:
        run_cross_celltype(gt_all, scores_all, out_dir, args.no_plots)

    if "linear_model" in args.analyses:
        run_linear_model(gt_all, scores_all, out_dir, args.no_plots)

    if "genomic_regions" in args.analyses:
        run_genomic_regions(gt_all, scores_all, out_dir, args.no_plots, args.ccre_bed)

    if "celltype_prediction" in args.analyses:
        run_celltype_prediction(gt_all, scores_all, out_dir, args.no_plots)

    print("\n" + "=" * 70)
    print("All requested analyses complete.")
    print(f"Results in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
