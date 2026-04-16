#!/usr/bin/env python3
"""
Detailed cell-type filtered biosample/assay analysis.

What this does:
1) Filter parquet rows to biosamples matching each cell type.
2) Compute per-group variant scores for:
   - biosample
   - assay_type
   - biosample x assay_type
3) Compare each group against MPRA ground truth with Spearman/AUROC/AUPRC.
4) Plot:
   - biosample vs baseline (all-tracks mean and ct-filtered mean)
   - biosample-assay heatmap (within/between biosample comparison)

Outputs per cell type (results/<ct>/analysis/):
  <ct>_filtered_biosample_metrics.csv
  <ct>_filtered_assay_metrics.csv
  <ct>_filtered_biosample_assay_metrics.csv
  <ct>_<model>_biosample_vs_mean_spearman_sig.png
  <ct>_<model>_biosample_assay_spearman_sig_heatmap.png
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
import matplotlib.cm as cm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        "qval_thr": 1.3,
        "label": "HEK293T",
    },
    "hepg2": {
        "vcf": VCF_DIR / "IGVFFI4378PZYI.vcf",
        "has_qval": True,
        "qval_thr": 1.3,
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

_LOG2FC_RE = re.compile(r"LOG2FC=([^;]+)")
_QVAL_RE = re.compile(r"QVAL=([^;]+)")


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


def load_gt(cfg):
    records = []
    with open(cfg["vcf"]) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
            info = cols[7] if len(cols) > 7 else ""
            vid = f"{chrom}:{pos}:{ref}>{alt}"
            m_lfc = _LOG2FC_RE.search(info)
            if m_lfc is None:
                continue
            lfc = float(m_lfc.group(1))
            qval = np.nan
            if cfg["has_qval"]:
                m_q = _QVAL_RE.search(info)
                if m_q:
                    qval = float(m_q.group(1))
            records.append({"variant_id": vid, "log2FC": lfc, "QVAL": qval})

    gt = pd.DataFrame(records).drop_duplicates(subset="variant_id")
    if cfg["has_qval"]:
        gt["sig"] = gt["QVAL"] > cfg.get("qval_thr", 1.3)
    else:
        gt["sig"] = gt["log2FC"].abs() > cfg.get("sig_lfc_thr", 0.5)
    return gt


def compute_metrics_vectors(log2fc, sig, score):
    mask = np.isfinite(score) & np.isfinite(log2fc)
    x = log2fc[mask]
    y = score[mask]
    s = sig[mask]

    if len(x) < 10:
        return None

    sp_all, _ = spearmanr(x, y)
    sp_sig = np.nan
    if s.sum() >= 10:
        sp_sig, _ = spearmanr(x[s], y[s])

    return {
        "n": int(len(x)),
        "n_sig": int(s.sum()),
        "spearman_r": float(sp_all),
        "spearman_sig": float(sp_sig),
        "auroc": float(_roc_auc(s.astype(float), np.abs(y))),
        "auprc": float(_avg_precision(s.astype(float), np.abs(y))),
    }


def _new_group_arrays(n_variants):
    return np.zeros(n_variants, dtype=np.float32), np.zeros(n_variants, dtype=np.uint16)


def aggregate_biosample_assay(parquet_path, gt_index, n_variants, bio_pattern, batch_size=1_000_000):
    """Accumulate sum/count per (biosample, assay, variant_index)."""
    pf = pq.ParquetFile(parquet_path)
    pair_acc = {}
    matched_biosamples = set()
    t0 = time.time()

    for n_batches_, batch in enumerate(
        pf.iter_batches(
            batch_size=batch_size,
            columns=["variant_id", "biosample_name", "assay_type", "raw_score"],
        )
    ):
        df = batch.to_pandas()

        bio_mask = df["biosample_name"].str.contains(
            bio_pattern, case=False, na=False, regex=True
        )
        if not bio_mask.any():
            continue

        df = df.loc[bio_mask, ["variant_id", "biosample_name", "assay_type", "raw_score"]]
        if df.empty:
            continue

        matched_biosamples.update(df["biosample_name"].dropna().unique())

        vidx = df["variant_id"].map(gt_index)
        keep = vidx.notna()
        if not keep.any():
            continue

        df = df.loc[keep].copy()
        df["variant_idx"] = vidx.loc[keep].astype(np.int32)
        df["assay_type"] = df["assay_type"].fillna("UNKNOWN")

        for (bio, assay), grp in df.groupby(["biosample_name", "assay_type"], sort=False):
            key = (str(bio), str(assay))
            if key not in pair_acc:
                pair_acc[key] = _new_group_arrays(n_variants)
            sum_arr, cnt_arr = pair_acc[key]

            idx = grp["variant_idx"].to_numpy(dtype=np.int32, copy=False)
            score = grp["raw_score"].to_numpy(dtype=np.float32, copy=False)
            np.add.at(sum_arr, idx, score)
            np.add.at(cnt_arr, idx, 1)

        n_batches = n_batches_ + 1
        if n_batches % 20 == 0:
            print(f"    batch {n_batches}: {time.time()-t0:.0f}s elapsed", flush=True)

    return pair_acc, matched_biosamples


def metrics_from_sum_count(sum_arr, cnt_arr, gt_log2fc, gt_sig):
    valid = cnt_arr > 0
    if valid.sum() < 10:
        return None
    score = np.full_like(gt_log2fc, np.nan, dtype=np.float32)
    score[valid] = sum_arr[valid] / cnt_arr[valid]
    return compute_metrics_vectors(gt_log2fc, gt_sig, score)


def build_pair_metrics(pair_acc, gt_log2fc, gt_sig, model):
    rows = []
    for (biosample, assay_type), (sum_arr, cnt_arr) in pair_acc.items():
        m = metrics_from_sum_count(sum_arr, cnt_arr, gt_log2fc, gt_sig)
        if m is None:
            continue
        m.update(
            {
                "model": model,
                "biosample_name": biosample,
                "assay_type": assay_type,
            }
        )
        rows.append(m)
    return pd.DataFrame(rows)


def build_group_metrics_from_pairs(pair_acc, gt_log2fc, gt_sig, model, group_name):
    group_pos = 0 if group_name == "biosample_name" else 1
    group_vals = sorted({k[group_pos] for k in pair_acc.keys()})

    rows = []
    n_variants = len(gt_log2fc)
    for gv in group_vals:
        sum_acc = np.zeros(n_variants, dtype=np.float32)
        cnt_acc = np.zeros(n_variants, dtype=np.uint16)

        for key, (sum_arr, cnt_arr) in pair_acc.items():
            if key[group_pos] == gv:
                sum_acc += sum_arr
                cnt_acc += cnt_arr

        m = metrics_from_sum_count(sum_acc, cnt_acc, gt_log2fc, gt_sig)
        if m is None:
            continue
        m.update({"model": model, group_name: gv})
        rows.append(m)

    return pd.DataFrame(rows)


def read_baseline_metric(ct, model, metric_col="spearman_sig"):
    all_csv = BASE / f"results/{ct}/analysis/{ct}_metrics.csv"
    filt_csv = BASE / f"results/{ct}/analysis/{ct}_filtered_metrics.csv"

    all_val = np.nan
    filt_val = np.nan

    if all_csv.exists():
        adf = pd.read_csv(all_csv)
        sub = adf[(adf["model"] == model) & (adf["score_type"] == "mean_sad")]
        if not sub.empty and metric_col in sub.columns:
            all_val = float(sub.iloc[0][metric_col])

    if filt_csv.exists():
        fdf = pd.read_csv(filt_csv)
        sub = fdf[(fdf["model"] == model) & (fdf["score_type"] == "mean_sad")]
        if not sub.empty and metric_col in sub.columns:
            filt_val = float(sub.iloc[0][metric_col])

    return all_val, filt_val


def plot_biosample_vs_baseline(bio_df, ct, model, out_dir):
    if bio_df.empty:
        return

    bio_df = bio_df.sort_values("spearman_sig", ascending=False).copy()
    top_n = 30
    show_df = bio_df.head(top_n)

    all_val, filt_val = read_baseline_metric(ct, model, metric_col="spearman_sig")

    fig_h = max(6, 0.28 * len(show_df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(show_df))
    ax.barh(y, show_df["spearman_sig"].values, color=MODEL_COLORS.get(model, "#607D8B"), alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(show_df["biosample_name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman (significant variants)")
    ax.set_title(f"{EXPERIMENTS[ct]['label']} | {model} | Biosample-level vs mean-track baselines")

    if np.isfinite(all_val):
        ax.axvline(all_val, color="#546E7A", lw=1.5, ls="--", label=f"All-tracks mean: {all_val:.3f}")
    if np.isfinite(filt_val):
        ax.axvline(filt_val, color="#E65100", lw=1.5, ls="-.", label=f"CT-filtered mean: {filt_val:.3f}")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = out_dir / f"{ct}_{model}_biosample_vs_mean_spearman_sig.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {out}")


def plot_biosample_assay_heatmap(pair_df, ct, model, out_dir):
    if pair_df.empty:
        return

    piv = pair_df.pivot_table(
        index="biosample_name",
        columns="assay_type",
        values="spearman_sig",
        aggfunc="mean",
    )
    if piv.empty:
        return

    row_rank = piv.max(axis=1, skipna=True).fillna(-np.inf)
    piv = piv.loc[row_rank.sort_values(ascending=False).index]

    max_rows = 60
    if len(piv) > max_rows:
        piv = piv.iloc[:max_rows]

    data = piv.to_numpy(dtype=float)

    fig_w = max(8, 1.5 + 0.9 * len(piv.columns))
    fig_h = max(6, 2.0 + 0.22 * len(piv.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = cm.RdBu_r.copy()
    cmap.set_bad("#E0E0E0")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.4, vmax=0.8)

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index.tolist(), fontsize=7)

    ax.set_title(
        f"{EXPERIMENTS[ct]['label']} | {model} | Spearman(sig) by biosample x assay\n"
        f"(top {len(piv)} biosamples by best assay)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman(sig)")

    plt.tight_layout()
    out = out_dir / f"{ct}_{model}_biosample_assay_spearman_sig_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {out}")


def run_celltype(ct, cfg):
    out_dir = BASE / f"results/{ct}/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = TMP_ROOT / ct
    if ct == "ngn2" and not tmp_dir.exists():
        legacy = Path("/tmp/ngn2_data")
        if legacy.exists():
            tmp_dir = legacy

    if not tmp_dir.exists():
        print(f"WARNING: {tmp_dir} not found; skipping {ct}")
        return

    print("\n" + "=" * 70)
    print(f"{cfg['label']} | detailed biosample/assay analysis")
    print("=" * 70)

    gt = load_gt(cfg)
    gt = gt.drop_duplicates(subset="variant_id").reset_index(drop=True)
    gt_index = {v: i for i, v in enumerate(gt["variant_id"].values)}
    gt_log2fc = gt["log2FC"].to_numpy(dtype=np.float32)
    gt_sig = gt["sig"].to_numpy(dtype=bool)

    print(f"Ground truth variants: {len(gt):,} | significant: {int(gt_sig.sum()):,}")

    biosample_frames = []
    assay_frames = []
    pair_frames = []

    for model in TRACK_MODELS:
        parquet = tmp_dir / f"{model}_scores.parquet"
        if not parquet.exists():
            print(f"\n  SKIP {model}: missing {parquet}")
            continue

        print(f"\n  Processing {model} ...", flush=True)
        bio_pattern = BIOSAMPLE_FILTERS[ct][model]

        pair_acc, matched = aggregate_biosample_assay(
            parquet, gt_index, len(gt), bio_pattern, batch_size=1_000_000
        )

        if not pair_acc:
            print("    No matching biosample-assay groups after filtering")
            continue

        print(f"    Matched biosamples: {len(matched)}")
        print(f"    Matched biosample-assay groups: {len(pair_acc)}")

        pair_df = build_pair_metrics(pair_acc, gt_log2fc, gt_sig, model)
        bio_df = build_group_metrics_from_pairs(pair_acc, gt_log2fc, gt_sig, model, "biosample_name")
        assay_df = build_group_metrics_from_pairs(pair_acc, gt_log2fc, gt_sig, model, "assay_type")

        if not pair_df.empty:
            pair_frames.append(pair_df)
        if not bio_df.empty:
            biosample_frames.append(bio_df)
        if not assay_df.empty:
            assay_frames.append(assay_df)

        if not bio_df.empty:
            plot_biosample_vs_baseline(bio_df, ct, model, out_dir)
        if not pair_df.empty:
            plot_biosample_assay_heatmap(pair_df, ct, model, out_dir)

    if biosample_frames:
        bios = pd.concat(biosample_frames, ignore_index=True)
        bios = bios.sort_values(["model", "spearman_sig"], ascending=[True, False])
        bios_out = out_dir / f"{ct}_filtered_biosample_metrics.csv"
        bios.to_csv(bios_out, index=False)
        print(f"\nSaved: {bios_out}")

    if assay_frames:
        ass = pd.concat(assay_frames, ignore_index=True)
        ass = ass.sort_values(["model", "spearman_sig"], ascending=[True, False])
        ass_out = out_dir / f"{ct}_filtered_assay_metrics.csv"
        ass.to_csv(ass_out, index=False)
        print(f"Saved: {ass_out}")

    if pair_frames:
        pairs = pd.concat(pair_frames, ignore_index=True)
        pairs = pairs.sort_values(["model", "spearman_sig"], ascending=[True, False])
        pair_out = out_dir / f"{ct}_filtered_biosample_assay_metrics.csv"
        pairs.to_csv(pair_out, index=False)
        print(f"Saved: {pair_out}")


def main():
    parser = argparse.ArgumentParser(description="Detailed biosample/assay analysis for filtered tracks")
    parser.add_argument(
        "--cell-types",
        nargs="+",
        default=["hepg2", "hek293t", "ngn2"],
        choices=list(EXPERIMENTS.keys()),
    )
    args = parser.parse_args()

    for ct in args.cell_types:
        run_celltype(ct, EXPERIMENTS[ct])

    print("\nDone.")


if __name__ == "__main__":
    main()
