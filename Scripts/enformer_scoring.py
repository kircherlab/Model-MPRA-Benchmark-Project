#!/usr/bin/env python3
"""
Enformer SAD Scoring — Tidy Format

Computes Sequence Activity Difference (SAD) for genetic variants using
the Enformer model. Outputs one row per variant × track in tidy/long
format — the same structure used by AlphaGenome's tidy_scores().

Output (Parquet, snappy-compressed):
    variant_id      chr1:10588:G>A
    assay_type      DNASE, CAGE, CHIP, ...
    biosample_name  biosample description (e.g. HepG2, cerebellum male adult)
    raw_score       SAD = mean(alt_pred - ref_pred) over spatial bins  [float32]
"""

import argparse
import gzip
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
import tensorflow_hub as hub
import kipoiseq
from kipoiseq import Interval
import pyfaidx


# ── Configuration ────────────────────────────────────────────────────────────

ENFORMER_URL = "https://tfhub.dev/deepmind/enformer/1"
SEQUENCE_LENGTH = 393_216
WRITE_CHUNK = 100       # flush every N variants (100 × ~6000 tracks ≈ 50 MB)
VALID_BASES = {"A", "C", "G", "T"}
NON_ACGTN = re.compile(r"[^ACGTN]")


# ── Logging ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Target tracks ────────────────────────────────────────────────────────────

def load_targets(targets_path):
    """Load track descriptions, parse output_type and biosample_name.

    Returns:
        descriptions   list[str]  full track descriptions
        output_types   list[str]  e.g. DNASE, CAGE, CHIP
        biosample_names list[str] e.g. HepG2, cerebellum male adult
    """
    df = pd.read_csv(targets_path, sep="\t")
    if "description" not in df.columns:
        raise ValueError(
            f"'description' column missing in {targets_path}. "
            f"Available: {list(df.columns)}"
        )

    descriptions = df["description"].astype(str).tolist()
    output_types = []
    biosample_names = []

    for desc in descriptions:
        if ":" in desc:
            parts = desc.split(":", 1)
            output_types.append(parts[0].strip())
            biosample_names.append(parts[1].strip())
        else:
            output_types.append("UNKNOWN")
            biosample_names.append(desc)

    log(f"Loaded {len(descriptions)} track descriptions from {targets_path}")
    return descriptions, output_types, biosample_names


# ── VCF parsing ──────────────────────────────────────────────────────────────

def parse_vcf(vcf_path, max_variants=None):
    """Parse VCF into list of SNV dicts, sorted by |LOG2FC| descending.

    Returns list of dicts with keys: chrom, pos, id, ref, alt, log2fc
    """
    variants = []
    is_gz = vcf_path.endswith(".gz")
    opener = gzip.open if is_gz else open

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, vid, ref, alts, _, _, info = fields[:8]
            ref = ref.upper()

            # Extract LOG2FC from INFO
            log2fc = 0.0
            for item in info.split(";"):
                if item.startswith("LOG2FC="):
                    try:
                        log2fc = float(item.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                    break

            for alt in alts.split(","):
                alt = alt.upper()
                if (len(ref) == 1 and len(alt) == 1
                        and ref in VALID_BASES and alt in VALID_BASES):
                    variants.append({
                        "chrom": chrom,
                        "pos": int(pos),
                        "id": vid,
                        "ref": ref,
                        "alt": alt,
                        "log2fc": log2fc,
                    })

    # Sort by effect size (strongest first)
    variants.sort(key=lambda v: abs(v["log2fc"]), reverse=True)

    if max_variants is not None and len(variants) > max_variants:
        variants = variants[:max_variants]
        log(f"Selected top {max_variants} variants by |LOG2FC|")

    log(f"Parsed {len(variants)} SNVs from VCF")
    return variants


# ── FASTA handling ───────────────────────────────────────────────────────────

class FastaExtractor:
    """Extract sequences from a FASTA file, handles chr-prefix mismatches."""

    def __init__(self, fasta_path):
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.chrom_sizes = {name: len(seq) for name, seq in self.fasta.items()}
        self.has_chr = any(n.startswith("chr") for n in self.chrom_sizes)

    def norm_chrom(self, chrom):
        """Normalize chromosome name to match FASTA conventions."""
        if self.has_chr and not chrom.startswith("chr"):
            candidate = "chr" + chrom
            if candidate in self.chrom_sizes:
                return candidate
        elif not self.has_chr and chrom.startswith("chr"):
            candidate = chrom[3:]
            if candidate in self.chrom_sizes:
                return candidate
        return chrom

    def extract(self, interval):
        """Extract sequence for a kipoiseq Interval, N-pad at boundaries."""
        chrom = self.norm_chrom(interval.chrom)

        if chrom not in self.chrom_sizes:
            return "N" * (interval.end - interval.start)

        chrom_len = self.chrom_sizes[chrom]
        safe_start = max(interval.start, 0)
        safe_end = min(interval.end, chrom_len)

        seq = str(self.fasta.get_seq(chrom, safe_start + 1, safe_end).seq)

        pad_left = "N" * max(-interval.start, 0)
        pad_right = "N" * max(interval.end - chrom_len, 0)

        return NON_ACGTN.sub("N", (pad_left + seq + pad_right).upper())

    def close(self):
        self.fasta.close()


# ── Enformer model ───────────────────────────────────────────────────────────

def load_enformer():
    """Load Enformer from TF Hub."""
    log("Loading Enformer model from TF Hub...")
    model = hub.load(ENFORMER_URL).model
    log("Model loaded")
    return model


def one_hot(seq):
    """One‐hot encode a DNA sequence (N → [0,0,0,0])."""
    return kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32)


def predict_sad(model, ref_seq, alt_seq):
    """Compute SAD (alt − ref) averaged over spatial bins.

    Returns:
        np.ndarray of shape (n_tracks,)
    """
    ref_pred = model.predict_on_batch(one_hot(ref_seq)[np.newaxis])["human"].numpy()
    alt_pred = model.predict_on_batch(one_hot(alt_seq)[np.newaxis])["human"].numpy()
    # Shapes: (1, n_bins, n_tracks) → mean over bins → (1, n_tracks) → [0]
    return (alt_pred - ref_pred).mean(axis=1)[0]


# ── Scoring loop ─────────────────────────────────────────────────────────────

def score_variant(model, variant, fasta):
    """Score a single variant and return SAD array (n_tracks,).

    Uses kipoiseq.extractors.VariantSeqExtractor for proper variant handling
    that accounts for indel positions / anchoring.
    """
    chrom = fasta.norm_chrom(variant["chrom"])
    pos = variant["pos"]

    # Centre an interval of SEQUENCE_LENGTH around the variant
    interval = Interval(chrom, pos, pos).resize(SEQUENCE_LENGTH)
    center_offset = interval.center() - interval.start

    # Extract ref and alt sequences using kipoiseq's VariantSeqExtractor
    vseq = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta)
    var_obj = kipoiseq.dataclasses.Variant(
        chrom=chrom, pos=pos, ref=variant["ref"], alt=variant["alt"], id=variant["id"]
    )

    ref_seq = NON_ACGTN.sub("N", vseq.extract(interval, [], anchor=center_offset).upper())
    alt_seq = NON_ACGTN.sub("N", vseq.extract(interval, [var_obj], anchor=center_offset).upper())

    return predict_sad(model, ref_seq, alt_seq)


def build_tidy_chunk(variants_scores, descriptions, output_types, biosample_names):
    """Convert a batch of (variant, scores) pairs into a tidy DataFrame.

    Args:
        variants_scores: list of (variant_dict, scores_array) tuples
        descriptions:    list of track descriptions (length = n_tracks)
        output_types:    list of output types (same length)
        biosample_names: list of biosample names (same length)

    Returns:
        pd.DataFrame with columns: variant_id, assay_type, biosample_name, raw_score
    """
    n_tracks = len(descriptions)
    n_variants = len(variants_scores)

    # Pre-allocate arrays for efficiency
    all_variant_ids = []
    all_scores = []

    for variant, scores in variants_scores:
        vid = f"{variant['chrom']}:{variant['pos']}:{variant['ref']}>{variant['alt']}"
        all_variant_ids.extend([vid] * n_tracks)
        all_scores.append(scores)

    return pd.DataFrame({
        "variant_id": all_variant_ids,
        "assay_type": output_types * n_variants,
        "biosample_name": biosample_names * n_variants,
        "raw_score": np.concatenate(all_scores),
    })


# Parquet schema — string columns get dictionary-encoded automatically by PyArrow
# (repeat strings stored once per row group → large space saving in tidy format)
PARQUET_SCHEMA = pa.schema([
    ("variant_id",    pa.string()),
    ("assay_type",    pa.string()),
    ("biosample_name",pa.string()),
    ("raw_score",     pa.float32()),
])


def open_parquet_writer(path):
    """Open a PyArrow ParquetWriter for incremental chunk writing."""
    return pq.ParquetWriter(path, PARQUET_SCHEMA, compression="snappy", use_dictionary=True)


def write_parquet_chunk(writer, df):
    """Write one DataFrame chunk as a Parquet row group."""
    df = df.astype({"raw_score": "float32"})
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    writer.write_table(table)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enformer SAD scoring — tidy/long-format output"
    )
    parser.add_argument("--vcf", required=True, help="Input VCF (gzipped)")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA")
    parser.add_argument("--targets", required=True, help="Targets TSV (needs 'description' column)")
    parser.add_argument("--out", required=True, help="Output Parquet path (.parquet)")
    parser.add_argument("--max_variants", default="all", help="Integer N or 'all'")
    parser.add_argument("--print_every", type=int, default=50, help="Log every N variants")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    log("==== Enformer SAD Scoring (Tidy Format) ====")
    log(f"VCF:     {args.vcf}")
    log(f"Targets: {args.targets}")
    log(f"Output:  {args.out}")

    # ── Validate inputs ──────────────────────────────────────────────────
    for path, label in [(args.vcf, "VCF"), (args.fasta, "FASTA"), (args.targets, "Targets")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    # ── Load metadata ────────────────────────────────────────────────────
    descriptions, output_types, biosample_names = load_targets(args.targets)
    n_tracks = len(descriptions)

    max_variants = None
    mv = str(args.max_variants).strip().lower()
    if mv not in ("all", ""):
        max_variants = int(mv)

    variants = parse_vcf(args.vcf, max_variants)
    if not variants:
        log("No variants found — writing empty output")
        empty = pa.table({"variant_id": pa.array([], type=pa.string()),
                          "assay_type": pa.array([], type=pa.string()),
                          "biosample_name": pa.array([], type=pa.string()),
                          "raw_score": pa.array([], type=pa.float32())})
        pq.write_table(empty, args.out, compression="snappy")
        return

    log(f"Max variants: {max_variants or 'all'}")

    # ── Checkpoint directory ──────────────────────────────────────────────
    checkpoint_dir = args.out + ".ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    n_chunks = (len(variants) + WRITE_CHUNK - 1) // WRITE_CHUNK
    log(f"Checkpoint dir: {checkpoint_dir}  ({n_chunks} chunks of {WRITE_CHUNK})")

    # Determine which chunks are already done
    done_chunks = set()
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("chunk_") and fname.endswith(".parquet"):
            try:
                pq.read_metadata(os.path.join(checkpoint_dir, fname))
                idx = int(fname.split("_")[1].split(".")[0])
                done_chunks.add(idx)
            except Exception:
                os.remove(os.path.join(checkpoint_dir, fname))

    if done_chunks:
        log(f"Resuming: {len(done_chunks)}/{n_chunks} chunks already complete")

    # ── Load model ───────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    log(f"GPUs available: {len(gpus)}")

    model = load_enformer()
    fasta = FastaExtractor(args.fasta)

    # ── Score variants in chunks ─────────────────────────────────────────
    log(f"Scoring {len(variants)} variants × {n_tracks} tracks ...")
    n_scored = 0
    n_skipped = 0
    tick = time.time()

    try:
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * WRITE_CHUNK
            chunk_end = min(chunk_start + WRITE_CHUNK, len(variants))
            ckpt_path = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:05d}.parquet")

            if chunk_idx in done_chunks:
                n_skipped += (chunk_end - chunk_start)
                n_scored += (chunk_end - chunk_start)
                continue

            buffer = []
            for variant in variants[chunk_start:chunk_end]:
                try:
                    scores = score_variant(model, variant, fasta)
                except Exception as e:
                    log(f"ERROR scoring {variant['id']} ({variant['chrom']}:{variant['pos']}): {e}")
                    continue

                if scores.shape[-1] != n_tracks:
                    raise RuntimeError(
                        f"Score dimension {scores.shape[-1]} ≠ targets dimension {n_tracks}"
                    )

                buffer.append((variant, scores))
                n_scored += 1

                # Progress
                if n_scored % args.print_every == 0:
                    dt = time.time() - tick
                    tick = time.time()
                    log(f"  scored {n_scored} variants  (Δ{dt:.1f}s)")

            # Write chunk checkpoint
            if buffer:
                df = build_tidy_chunk(buffer, descriptions, output_types, biosample_names)
                df = df.astype({"raw_score": "float32"})
                table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
                pq.write_table(table, ckpt_path, compression="snappy", use_dictionary=True)
    finally:
        fasta.close()

    # ── Merge checkpoints into final output ──────────────────────────────
    ckpt_files = sorted(
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("chunk_") and f.endswith(".parquet")
    )
    if ckpt_files:
        log(f"Merging {len(ckpt_files)} checkpoint files into {args.out} ...")
        writer = open_parquet_writer(args.out)
        try:
            for cf in ckpt_files:
                tbl = pq.read_table(cf, schema=PARQUET_SCHEMA)
                writer.write_table(tbl)
        finally:
            writer.close()
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        log("Checkpoint directory removed.")

    elapsed = time.time() - start_time
    total_rows = n_scored * n_tracks
    log(f"Done — {n_scored} variants × {n_tracks} tracks = "
        f"{total_rows:,} rows → {args.out}")
    if n_skipped:
        log(f"  ({n_skipped} variants restored from checkpoints)")
    log(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
