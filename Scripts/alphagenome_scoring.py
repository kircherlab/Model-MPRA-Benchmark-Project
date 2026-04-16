#!/usr/bin/env python3
"""
AlphaGenome Variant Scoring Script
===================================
Scores SNP variants from a VCF using the AlphaGenome API (google-deepmind).

AlphaGenome uses CenterMaskScorer for variant-centric scoring, which computes
aggregated ALT vs REF differences within a spatial mask around the variant.

Output format: Standard AlphaGenome tidy/long TSV produced by
``variant_scorers.tidy_scores()`` with columns:
  variant_id, scored_interval, gene_id, gene_name, gene_type, gene_strand,
  junction_Start, junction_End, output_type, variant_scorer, track_name,
  track_strand, Assay title, ontology_curie, biosample_name, biosample_type,
  transcription_factor, histone_mark, gtex_tissue, raw_score, quantile_score

Requires:
  - API key via env var ALPHAGENOME_API_KEY or --api_key flag
  - pip install alphagenome (from github: google-deepmind/alphagenome)
"""

import argparse
import gzip
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

# ============================================================================
# CLI ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(
    description="AlphaGenome variant effect scoring via API"
)
parser.add_argument("--vcf", required=True, help="Input VCF file (optionally gzipped)")
parser.add_argument("--out", required=True, help="Output CSV path")
parser.add_argument(
    "--api_key",
    default=None,
    help="AlphaGenome API key. Falls back to ALPHAGENOME_API_KEY env var.",
)
parser.add_argument(
    "--window_size",
    type=int,
    default=16384,
    help="Input sequence window size centered on variant (default: 16384 = 16kb)",
)
parser.add_argument(
    "--max_variants",
    default="all",
    help="Max variants to score (integer or 'all')",
)
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=500,
    help="Variants per API batch. Smaller = more frequent checkpoints (default: 500)",
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=4,
    help="Number of parallel API workers (default: 4)",
)
parser.add_argument(
    "--scorers",
    nargs="+",
    default=["ATAC", "DNASE", "CAGE", "CHIP_HISTONE", "RNA_SEQ"],
    help="Which recommended variant scorers to use (default: ATAC DNASE CAGE CHIP_HISTONE RNA_SEQ)",
)

parser.add_argument(
    "--ontology_terms",
    nargs="*",
    default=None,
    help="Ontology terms for cell-type/tissue filtering (e.g., UBERON:0002107 for liver). Leave empty for all cell-types.",
)
args = parser.parse_args()

# ============================================================================
# SETUP
# ============================================================================
DNA_BASES = set("ACGT")


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _parse_max(x):
    if x is None:
        return None
    x = str(x).strip()
    if x.lower() == "all" or x == "":
        return None
    return int(x)


MAX_VARIANTS = _parse_max(args.max_variants)

# Resolve API key (priority: --api_key CLI flag > ALPHAGENOME_API_KEY env var)
API_KEY = args.api_key or os.environ.get("ALPHAGENOME_API_KEY")
if not API_KEY:
    _log("ERROR: No API key provided. Set 'api_key' in config.yaml, "
         "ALPHAGENOME_API_KEY env var, or use --api_key")
    sys.exit(1)

# ============================================================================
# VCF PARSING (consistent with other pipeline scripts)
# ============================================================================
def parse_vcf_variants(vcf_path, max_variants=None):
    """Parse VCF, return list of variant dicts sorted by |LOG2FC|."""
    variants = []
    opener = gzip.open if vcf_path.endswith(".gz") else open

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, vid, ref, alts, _, _, info_field = fields[:8]
            pos = int(pos)
            ref = ref.upper()

            # Extract LOG2FC for sorting by effect size
            log2fc = 0.0
            for item in info_field.split(";"):
                if item.startswith("LOG2FC="):
                    try:
                        log2fc = float(item.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                    break

            for alt in alts.split(","):
                alt = alt.upper()
                # Only SNVs
                if (
                    len(ref) == 1
                    and len(alt) == 1
                    and ref in DNA_BASES
                    and alt in DNA_BASES
                ):
                    variants.append(
                        {
                            "chrom": chrom,
                            "pos": pos,
                            "id": vid,
                            "ref": ref,
                            "alt": alt,
                            "log2fc": log2fc,
                        }
                    )

    # Sort by absolute effect size (strongest first)
    variants.sort(key=lambda x: abs(x["log2fc"]), reverse=True)
    if max_variants and len(variants) > max_variants:
        variants = variants[:max_variants]
        _log(f"Selected top {max_variants} variants by |LOG2FC|")

    return variants


# ============================================================================
# HELPERS
# ============================================================================
def _postprocess_df(df):
    """Normalize column names and variant_id format to pipeline standard."""
    for col in ("variant_id", "scored_interval"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "output_type" in df.columns:
        df = df.rename(columns={"output_type": "assay_type"})
    if "variant_id" in df.columns:
        df["variant_id"] = df["variant_id"].str.replace(
            r'^(chr\w+):(\d+):(\w+):(\w+)$', r'\1:\2:\3>\4', regex=True
        )
    return df


# ============================================================================
# ALPHAGENOME SCORING
# ============================================================================
def build_variant_scorers(scorer_names):
    """Build the list of CenterMaskScorer objects from names.
    
    Uses AlphaGenome's recommended scorer configurations which define
    the appropriate width and aggregation type per output modality.
    """
    available = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    selected = []
    for name in scorer_names:
        name_upper = name.upper()
        if name_upper in available:
            selected.append(available[name_upper])
        else:
            _log(f"WARNING: Scorer '{name}' not in RECOMMENDED_VARIANT_SCORERS. "
                 f"Available: {list(available.keys())}")
    if not selected:
        _log("ERROR: No valid scorers selected. Using all recommended scorers.")
        selected = list(available.values())
    return selected


def score_variants_batch(model, variants, window_size, scorers, max_workers,
                         ontology_terms=None, chunk_size=500, checkpoint_dir=None):
    """Score all variants using AlphaGenome's batch scoring API in chunks.

    Each completed chunk is saved to checkpoint_dir/chunk_NNNNN.parquet
    immediately after scoring. On restart, existing checkpoints are skipped.

    Args:
        ontology_terms: List of ontology term strings for cell-type filtering.
        chunk_size:     Variants per API batch (default: 500).
        checkpoint_dir: Directory to store per-chunk checkpoint parquets.

    Returns list of checkpoint file paths (all chunks, existing + new).
    """
    # Build genome objects
    ag_intervals = []
    ag_variants = []
    valid_variants = []

    for var in variants:
        chrom = var["chrom"]
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom

        # AlphaGenome uses 0-based coordinates for intervals; 
        # VCF pos is 1-based; genome.Variant uses 1-based position
        center = var["pos"]  # 1-based
        half = window_size // 2
        start = max(0, center - half)
        end = center + half

        interval = genome.Interval(chromosome=chrom, start=start, end=end)
        variant = genome.Variant(
            chromosome=chrom,
            position=center,
            reference_bases=var["ref"],
            alternate_bases=var["alt"],
        )

        ag_intervals.append(interval)
        ag_variants.append(variant)
        valid_variants.append(var)

    total = len(ag_variants)
    n_chunks = (total + chunk_size - 1) // chunk_size
    _log(f"Scoring {total} variants with {len(scorers)} scorers in {n_chunks} chunks of {chunk_size}...")
    if ontology_terms:
        _log(f"  Cell-type filtering requested: {ontology_terms}")
    if checkpoint_dir:
        _log(f"  Checkpoint dir: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    ontology_terms_supported = True
    ckpt_files = []
    scored_total = 0

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, total)
        chunk_intervals = ag_intervals[c_start:c_end]
        chunk_variants = ag_variants[c_start:c_end]
        chunk_valid = valid_variants[c_start:c_end]
        ckpt_file = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:05d}.parquet") if checkpoint_dir else None

        # ── Resume: skip already-completed chunks ──────────────────────────
        if ckpt_file and os.path.exists(ckpt_file):
            try:
                existing = pd.read_parquet(ckpt_file, columns=["variant_id"])
                n_in = existing["variant_id"].nunique()
                _log(f"  Chunk {chunk_idx + 1}/{n_chunks}: SKIP (checkpoint has {n_in} variants)")
                ckpt_files.append(ckpt_file)
                scored_total += len(chunk_valid)
                continue
            except Exception:
                _log(f"  Chunk {chunk_idx + 1}/{n_chunks}: checkpoint corrupt — re-scoring")
                os.remove(ckpt_file)

        _log(f"  Chunk {chunk_idx + 1}/{n_chunks}: variants {c_start}–{c_end - 1}")

        def _call_chunk(use_ontology):
            kwargs = dict(
                intervals=chunk_intervals,
                variants=chunk_variants,
                variant_scorers=scorers,
                max_workers=max_workers,
                progress_bar=True,
            )
            if use_ontology and ontology_terms:
                kwargs["ontology_terms"] = ontology_terms
            for attempt in range(1, 8):
                try:
                    return model.score_variants(**kwargs)
                except Exception as exc:
                    msg = str(exc)
                    if "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg or "quota" in msg.lower():
                        wait = min(60 * (2 ** (attempt - 1)), 600)
                        _log(f"    Quota exceeded (attempt {attempt}/7). Waiting {wait}s before retry...")
                        time.sleep(wait)
                    elif any(code in msg for code in ("INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED")):
                        wait = min(30 * (2 ** (attempt - 1)), 300)
                        _log(f"    Transient server error (attempt {attempt}/7). Waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError("API quota retry limit exceeded after 7 attempts.")

        def _call_single(interval, variant, use_ontology):
            kwargs = dict(interval=interval, variant=variant, variant_scorers=scorers)
            if use_ontology and ontology_terms:
                kwargs["ontology_terms"] = ontology_terms
            for attempt in range(1, 8):
                try:
                    return model.score_variant(**kwargs)
                except Exception as exc:
                    msg = str(exc)
                    if "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg or "quota" in msg.lower():
                        wait = min(60 * (2 ** (attempt - 1)), 600)
                        _log(f"    Quota exceeded (attempt {attempt}/7). Waiting {wait}s before retry...")
                        time.sleep(wait)
                    elif any(code in msg for code in ("INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED")):
                        wait = min(30 * (2 ** (attempt - 1)), 300)
                        _log(f"    Transient server error (attempt {attempt}/7). Waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError("API quota retry limit exceeded after 7 attempts.")

        # Score this chunk
        chunk_scores = None
        try:
            if ontology_terms_supported and ontology_terms:
                try:
                    chunk_scores = _call_chunk(use_ontology=True)
                except TypeError as e:
                    if "ontology_terms" in str(e):
                        _log("  WARNING: ontology_terms not supported. Scoring all cell-types.")
                        ontology_terms_supported = False
                        chunk_scores = _call_chunk(use_ontology=False)
                    else:
                        raise
            else:
                chunk_scores = _call_chunk(use_ontology=False)
        except Exception as e:
            _log(f"  Batch scoring failed for chunk {chunk_idx + 1}: {e}")
            _log(f"  Falling back to single-variant scoring for this chunk...")
            chunk_scores = []
            for i, (interval, variant) in enumerate(zip(chunk_intervals, chunk_variants)):
                try:
                    scores = _call_single(interval, variant, use_ontology=ontology_terms_supported)
                    chunk_scores.append(scores)
                except Exception as e2:
                    _log(f"    Error scoring variant {c_start + i} ({chunk_valid[i]['id']}): {e2}")
                    chunk_scores.append(None)

        # Convert chunk to tidy DataFrame and save checkpoint
        if chunk_scores:
            valid_scores = [s for s in chunk_scores if s is not None]
            n_failed = len(chunk_scores) - len(valid_scores)
            if n_failed:
                _log(f"    {n_failed} variant(s) failed in this chunk")
            if valid_scores:
                chunk_df = variant_scorers.tidy_scores(valid_scores)
                if chunk_df is not None and not chunk_df.empty:
                    chunk_df = _postprocess_df(chunk_df)
                    if ckpt_file:
                        chunk_df.to_parquet(ckpt_file, index=False, compression="snappy")
                        ckpt_files.append(ckpt_file)
                    scored_total += len(valid_scores)

        _log(f"  Chunk {chunk_idx + 1}/{n_chunks} done. Total scored: {scored_total}/{total}")

    if not ckpt_files:
        _log("WARNING: All variants failed scoring — no checkpoints written.")
        return []

    return ckpt_files


# ============================================================================
# MAIN
# ============================================================================
def main():
    start_time = time.time()
    _log("==== AlphaGenome Variant Scoring ====")
    _log(f"VCF: {args.vcf}")
    _log(f"Output: {args.out}")
    _log(f"Window: {args.window_size}bp | Max variants: {MAX_VARIANTS or 'all'}")
    _log(f"Scorers: {args.scorers}")
    _log(f"Output format: Parquet (snappy-compressed)")

    # Parse VCF
    variants = parse_vcf_variants(args.vcf, MAX_VARIANTS)
    if not variants:
        _log("No SNV variants found in VCF. Exiting.")
        sys.exit(1)
    _log(f"Parsed {len(variants)} SNV variants from VCF")

    # Initialize AlphaGenome client
    _log("Connecting to AlphaGenome API...")
    model = dna_client.create(API_KEY)
    _log("Connected successfully")

    # Build scorers
    scorers = build_variant_scorers(args.scorers)
    _log(f"Using {len(scorers)} variant scorers")

    # Checkpoint directory sits next to the output file
    checkpoint_dir = args.out + ".ckpt"

    # Score variants — checkpoints written per chunk
    ckpt_files = score_variants_batch(
        model, variants, args.window_size, scorers,
        args.max_workers, args.ontology_terms,
        chunk_size=args.chunk_size, checkpoint_dir=checkpoint_dir
    )

    # Merge all checkpoint parquets into the final output
    if ckpt_files:
        _log(f"Merging {len(ckpt_files)} checkpoint files into {args.out} ...")
        all_dfs = [pd.read_parquet(f) for f in sorted(ckpt_files)]
        tidy_df = pd.concat(all_dfs, ignore_index=True)
        tidy_df.to_parquet(args.out, index=False, compression="snappy")
        n_variants = tidy_df["variant_id"].nunique() if "variant_id" in tidy_df.columns else "?"
        _log(f"Saved {len(tidy_df)} rows ({n_variants} unique variants) to {args.out}")
        # Clean up checkpoints only after successful write
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        _log("Checkpoint directory removed.")
    else:
        _log("WARNING: No results produced. Creating empty output.")
        import pyarrow as pa
        import pyarrow.parquet as pq
        empty = pa.table({
            col: pa.array([], type=pa.string())
            for col in ["variant_id", "scored_interval", "gene_id", "gene_name",
                        "gene_type", "gene_strand", "assay_type", "variant_scorer",
                        "track_name", "track_strand", "raw_score", "quantile_score"]
        })
        pq.write_table(empty, args.out, compression="snappy")

    _log(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
