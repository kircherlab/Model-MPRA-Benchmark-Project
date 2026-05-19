#!/usr/bin/env python3
"""
Basenji SAD Scoring — Tidy Format

Wraps the external basenji_sad.py script that computes SAD scores,
then converts the HDF5 output to tidy/long-format TSV matching the
Enformer and AlphaGenome output structure.

Output (Parquet, snappy-compressed):
    variant_id      chr1:10588:G>A
    assay_type      DNASE, CAGE, CHIP, ...
    biosample_name  biosample description (e.g. HepG2, cerebellum male adult)
    raw_score       SAD score (sum over spatial bins, done inside basenji_sad.py)  [float32]
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ── Configuration ────────────────────────────────────────────────────────────

WRITE_CHUNK = 500   # variants per checkpoint file


# ── Logging ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Target track metadata ───────────────────────────────────────────────────

def load_targets(targets_path):
    """Load track descriptions, parse output_type and biosample_name.

    Returns:
        descriptions    list[str]  full track descriptions
        output_types    list[str]  e.g. DNASE, CAGE, CHIP
        biosample_names list[str]  e.g. HepG2, cerebellum male adult
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


# ── Run basenji_sad.py ───────────────────────────────────────────────────────

def filter_snps_only(vcf_path):
    """Write a temp gzipped VCF containing only SNPs (len(REF)==1, len(ALT)==1).

    Basenji SAD crashes on indels (empty tensor in reverse-complement layer).
    Returns (tmp_vcf_path, n_kept, n_skipped).
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix="_snps_only.vcf.gz", delete=False
    )
    n_kept = n_skipped = 0
    with gzip.open(vcf_path, "rt") as fin, gzip.open(tmp.name, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            cols = line.split("\t")
            ref, alt = cols[3], cols[4]
            if len(ref) == 1 and len(alt) == 1:
                fout.write(line)
                n_kept += 1
            else:
                n_skipped += 1
                log(f"Skipping non-SNP variant (indel): "
                    f"{cols[0]}:{cols[1]} {ref}>{alt}")
    return tmp.name, n_kept, n_skipped


def run_basenji_sad(vcf, fasta, params, model, basenji_script, out_dir,
                    rc=False, shifts="0"):
    """Run the external basenji_sad.py and return path to the output HDF5.

    basenji_sad.py computes SAD = alt − ref with spatial aggregation (sum)
    applied internally on GPU. The output HDF5 contains SAD scores of shape
    (n_variants, n_tracks).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Set BASENJIDIR so the basenji module can be imported
    basenji_dir = os.path.dirname(os.path.dirname(basenji_script))
    os.environ["BASENJIDIR"] = basenji_dir
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{basenji_dir}:{os.environ['PYTHONPATH']}"
    else:
        os.environ["PYTHONPATH"] = basenji_dir

    cmd = [
        "python", basenji_script,
        "-f", fasta,
        "-o", out_dir,
        "--shifts", shifts,
    ]
    if rc:
        cmd.append("--rc")
    cmd.extend([params, model, vcf])

    h5_path = os.path.join(out_dir, "sad.h5")

    # ── Resume checkpoint: skip GPU run if HDF5 already exists and is valid ──
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, "r") as _:
                pass
            log(f"Found existing HDF5 output, skipping basenji_sad.py run: {h5_path}")
            return h5_path
        except Exception:
            log("Existing HDF5 appears corrupt, re-running basenji_sad.py ...")
            os.remove(h5_path)

    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Expected HDF5 output not found: {h5_path}")

    return h5_path


# ── HDF5 → tidy TSV conversion ──────────────────────────────────────────────

def convert_h5_to_parquet(h5_path, out_parquet, descriptions, output_types,
                           biosample_names):
    """Convert Basenji HDF5 output to tidy-format Parquet.

    Reads the SAD matrix and writes one row per variant × track,
    processing in chunks to keep memory bounded.

    Args:
        h5_path:         path to sad.h5 from basenji_sad.py
        out_parquet:     output Parquet file path
        descriptions:    track descriptions (length = n_tracks)
        output_types:    parsed output types
        biosample_names: parsed biosample names
    """
    n_tracks = len(descriptions)

    parquet_schema = pa.schema([
        ("variant_id",    pa.string()),
        ("assay_type",    pa.string()),
        ("biosample_name",pa.string()),
        ("raw_score",     pa.float32()),
    ])

    # ── Checkpoint directory ──────────────────────────────────────────────
    checkpoint_dir = out_parquet + ".ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        # Read variant metadata
        chroms = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in f["chr"][:]]
        positions = f["pos"][:]
        refs = [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["ref_allele"][:]]
        alts = [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["alt_allele"][:]]

        n_variants = len(chroms)

        # SAD shape: (n_variants, n_tracks) or sometimes (n_variants, n_pos, n_tracks)
        sad_dataset = f["SAD"]
        sad_shape = sad_dataset.shape
        log(f"HDF5 SAD shape: {sad_shape}")

        if sad_shape[-1] != n_tracks:
            log(f"WARNING: SAD has {sad_shape[-1]} tracks but targets has {n_tracks}. "
                f"Using min({sad_shape[-1]}, {n_tracks}) tracks.")

        n_chunks = (n_variants + WRITE_CHUNK - 1) // WRITE_CHUNK
        log(f"Converting {n_variants} variants × {n_tracks} tracks to tidy Parquet ...")
        log(f"Checkpoint dir: {checkpoint_dir}  ({n_chunks} chunks of {WRITE_CHUNK})")

        # Detect already-completed chunks
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

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * WRITE_CHUNK
            chunk_end = min(chunk_start + WRITE_CHUNK, n_variants)
            ckpt_path = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:05d}.parquet")

            if chunk_idx in done_chunks:
                continue

            n_chunk = chunk_end - chunk_start

            # Read SAD scores for this chunk
            sad_chunk = sad_dataset[chunk_start:chunk_end]

            # If 3D (n_variants, n_positions, n_tracks), sum over spatial dim
            if sad_chunk.ndim == 3:
                sad_chunk = sad_chunk.sum(axis=1)

            # Truncate to n_tracks if dimensions don't match
            actual_tracks = min(sad_chunk.shape[1], n_tracks)

            # Build variant_id for each row
            variant_ids = []
            for i in range(n_chunk):
                idx = chunk_start + i
                vid = f"{chroms[idx]}:{positions[idx]}:{refs[idx]}>{alts[idx]}"
                variant_ids.extend([vid] * actual_tracks)

            df = pd.DataFrame({
                "variant_id": variant_ids,
                "assay_type": output_types[:actual_tracks] * n_chunk,
                "biosample_name": biosample_names[:actual_tracks] * n_chunk,
                "raw_score": sad_chunk[:, :actual_tracks].flatten().astype("float32"),
            })
            table = pa.Table.from_pandas(df, schema=parquet_schema, preserve_index=False)
            pq.write_table(table, ckpt_path, compression="snappy", use_dictionary=True)

            log(f"  written chunk {chunk_idx} (variants {chunk_start + 1}–{chunk_end})")

    # ── Merge checkpoints into final output ──────────────────────────────
    ckpt_files = sorted(
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("chunk_") and f.endswith(".parquet")
    )
    if ckpt_files:
        log(f"Merging {len(ckpt_files)} checkpoint files into {out_parquet} ...")
        writer = pq.ParquetWriter(out_parquet, parquet_schema, compression="snappy", use_dictionary=True)
        try:
            for cf in ckpt_files:
                tbl = pq.read_table(cf, schema=parquet_schema)
                writer.write_table(tbl)
        finally:
            writer.close()
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        log("Checkpoint directory removed.")

    log(f"Output: {n_variants * n_tracks:,} rows → {out_parquet}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Basenji SAD scoring — tidy/long-format output"
    )
    parser.add_argument("--vcf", required=True, help="Input VCF (gzipped)")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA")
    parser.add_argument("--params", required=True, help="Model parameters JSON")
    parser.add_argument("--model", required=True, help="Model weights HDF5")
    parser.add_argument("--targets", required=True, help="Targets TSV file")
    parser.add_argument("--out_dir", default="basenji_sad_output",
                        help="Temp dir for basenji_sad.py output")
    parser.add_argument("--out", required=True, help="Output Parquet path (.parquet)")
    parser.add_argument("--basenji_script", required=True,
                        help="Path to basenji_sad.py")
    parser.add_argument("--rc", action="store_true",
                        help="Average forward and reverse complement")
    parser.add_argument("--shifts", default="0", help="Ensemble prediction shifts")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    log("==== Basenji SAD Scoring (Tidy Format) ====")
    log(f"VCF:     {args.vcf}")
    log(f"Targets: {args.targets}")
    log(f"Output:  {args.out}")

    # ── Validate inputs ──────────────────────────────────────────────────
    for path, label in [(args.vcf, "VCF"), (args.fasta, "FASTA"),
                        (args.params, "Params"), (args.model, "Model"),
                        (args.targets, "Targets")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # ── Load track metadata ──────────────────────────────────────────────
    descriptions, output_types, biosample_names = load_targets(args.targets)

    # ── Step 1a: Filter to SNPs only (indels crash basenji's RC layer) ────
    snp_vcf, n_kept, n_skipped = filter_snps_only(args.vcf)
    if n_skipped:
        log(f"Filtered {n_skipped} non-SNP variant(s) — Basenji supports SNPs only. "
            f"{n_kept} SNPs passed.")

    # ── Step 1b: Run basenji_sad.py → HDF5 ───────────────────────────────
    h5_path = run_basenji_sad(
        vcf=snp_vcf,
        fasta=args.fasta,
        params=args.params,
        model=args.model,
        basenji_script=args.basenji_script,
        out_dir=args.out_dir,
        rc=args.rc,
        shifts=args.shifts,
    )

    # ── Step 2: Convert HDF5 → tidy TSV ──────────────────────────────────
    convert_h5_to_parquet(h5_path, args.out, descriptions, output_types,
                           biosample_names)

    # ── Step 3: Cleanup temp files ────────────────────────────────────────
    try:
        os.remove(h5_path)
        os.rmdir(args.out_dir)
        log("Cleaned up temporary HDF5 files")
    except Exception as e:
        log(f"Warning: cleanup failed: {e}")

    try:
        os.remove(snp_vcf)
    except Exception:
        pass

    elapsed = time.time() - start_time
    log(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
