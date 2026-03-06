#!/usr/bin/env python3
"""
AlphaGenome integration script for the benchmark pipeline.

Design goals:
1) Generate raw AlphaGenome predictions in tidy long format (chunked TSVs).
2) Finalize raw predictions into the benchmark-wide model score contract:
   columns: chrom,pos,id,ref,alt,<track columns>

The finalized CSV can be consumed directly by Scripts/merge_data.py.
"""

import argparse
import gzip
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SUPPORTED_SCORER_KEYS = [
    "rna_seq",
    "cage",
    "procap",
    "atac",
    "dnase",
    "chip_histone",
    "chip_tf",
    "polyadenylation",
    "splice_sites",
    "splice_site_usage",
    "splice_junctions",
]


@dataclass(frozen=True)
class VariantRecord:
    chrom: str
    pos: int
    vid: str
    ref: str
    alt: str


def _log(msg: str) -> None:
    print(msg, flush=True)


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def _clean_token(value: str) -> str:
    # Keep labels parseable for downstream assay/biosample splitting.
    return " ".join(str(value).strip().replace("\t", " ").split())


def _as_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_vcf_variants(vcf_path: str) -> List[VariantRecord]:
    records: List[VariantRecord] = []
    with _open_text(vcf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            chrom, pos, vid, ref, alts = cols[:5]
            ref = ref.upper()
            for alt in alts.split(","):
                alt = alt.upper()
                if len(ref) == 1 and len(alt) == 1:
                    records.append(
                        VariantRecord(
                            chrom=_normalize_chrom(chrom),
                            pos=int(pos),
                            vid=vid,
                            ref=ref,
                            alt=alt,
                        )
                    )
    return records


def _parse_variant_id(variant_id: str) -> Tuple[str, int, str, str]:
    """
    Parse common AlphaGenome variant_id styles:
    - chr1:12345:A>G
    - chr1:12345:A:G
    """
    text = str(variant_id)
    parts = text.split(":")
    if len(parts) == 3 and ">" in parts[2]:
        chrom, pos, ref_alt = parts
        ref, alt = ref_alt.split(">", 1)
        return _normalize_chrom(chrom), int(pos), ref.upper(), alt.upper()

    if len(parts) == 4:
        chrom, pos, ref, alt = parts
        return _normalize_chrom(chrom), int(pos), ref.upper(), alt.upper()

    raise ValueError(f"Unsupported variant_id format: {variant_id}")


def _build_scorers_from_flags(args: argparse.Namespace):
    """Resolve scorer list from --scorers and per-scorer boolean flags."""
    chosen: List[str] = []

    if args.scorers:
        chosen.extend([s.strip().lower() for s in args.scorers.split(",") if s.strip()])

    for key in SUPPORTED_SCORER_KEYS:
        if _as_bool(getattr(args, f"score_{key}", None)):
            chosen.append(key)

    # Deduplicate while preserving order.
    dedup: List[str] = []
    seen = set()
    for key in chosen:
        if key not in seen:
            dedup.append(key)
            seen.add(key)

    # If nothing was selected, use all recommended scorers.
    return dedup


def _resolve_sequence_length(requested: str, supported: Dict[str, object]):
    """Resolve user-facing sequence length token against SDK supported keys."""
    token = str(requested).strip().upper().replace(" ", "")
    candidates = [
        token,
        f"SEQUENCE_LENGTH_{token}",
        token.replace("KIB", "KB").replace("MIB", "MB"),
    ]

    for key in candidates:
        if key in supported:
            return supported[key], key

    return None, token


def run_score(args: argparse.Namespace) -> None:
    """Score variants with AlphaGenome API and write raw tidy chunk outputs."""
    # Lazy imports so 'finalize' works without AlphaGenome package installed.
    from alphagenome.data import genome  # type: ignore
    from alphagenome.models import dna_client, variant_scorers  # type: ignore

    vcf_path = args.vcf
    out_dir = Path(args.output_raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    variants = read_vcf_variants(vcf_path)
    if not variants:
        raise ValueError("No SNV variants found in input VCF.")

    organism_map = {
        "human": dna_client.Organism.HOMO_SAPIENS,
        "mouse": dna_client.Organism.MUS_MUSCULUS,
    }
    organism = organism_map[args.organism.lower()]

    sequence_length, resolved_key = _resolve_sequence_length(
        args.sequence_length,
        dna_client.SUPPORTED_SEQUENCE_LENGTHS,
    )
    if sequence_length is None:
        allowed_tokens = [
            k.replace("SEQUENCE_LENGTH_", "") for k in dna_client.SUPPORTED_SEQUENCE_LENGTHS
        ]
        allowed = ", ".join(allowed_tokens)
        extra = ""
        if str(args.sequence_length).strip().upper() == "2KB" and "16KB" in allowed_tokens:
            extra = (
                " Installed alphagenome SDK does not expose 2KB for this API. "
                "This is a known docs/SDK mismatch in some versions. "
                "Use 16KB or pin/install a version that provides 2KB."
            )
        raise ValueError(
            f"Unsupported sequence_length '{args.sequence_length}'. Use one of: {allowed}.{extra}"
        )

    _log(f"Using sequence_length={resolved_key}")

    all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    chosen_keys = _build_scorers_from_flags(args)

    if chosen_keys:
        missing = [k for k in chosen_keys if k not in all_scorers]
        if missing:
            raise ValueError(f"Unknown scorer keys: {missing}. Supported keys: {sorted(all_scorers.keys())}")
        selected_scorers = [all_scorers[k] for k in chosen_keys]
    else:
        selected_scorers = list(all_scorers.values())

    unsupported = [
        s
        for s in selected_scorers
        if (
            organism.value not in variant_scorers.SUPPORTED_ORGANISMS[s.base_variant_scorer]
            or (
                s.requested_output == dna_client.OutputType.PROCAP
                and organism == dna_client.Organism.MUS_MUSCULUS
            )
        )
    ]
    if unsupported:
        _log(f"Excluding {len(unsupported)} unsupported scorers for organism={organism.value}")
        selected_scorers = [s for s in selected_scorers if s not in unsupported]

    if not selected_scorers:
        raise ValueError("No compatible scorers left after filtering.")

    _log("Creating AlphaGenome client...")
    model = dna_client.create(args.api_key)

    chunk_size = int(args.chunk_size)
    all_rows = []
    chunk_results = []
    chunk_idx = 0

    _log(f"Scoring {len(variants)} SNVs with {len(selected_scorers)} scorer(s)")

    for idx, rec in enumerate(variants, start=1):
        variant = genome.Variant(
            chromosome=rec.chrom,
            position=rec.pos,
            reference_bases=rec.ref,
            alternate_bases=rec.alt,
            name=rec.vid,
        )
        interval = variant.reference_interval.resize(sequence_length)

        scored = model.score_variant(
            interval=interval,
            variant=variant,
            variant_scorers=selected_scorers,
            organism=organism,
        )
        chunk_results.append(scored)

        if idx % chunk_size == 0:
            df_chunk = variant_scorers.tidy_scores(chunk_results)
            all_rows.append(df_chunk)
            chunk_path = out_dir / f"chunk_{chunk_idx:05d}_{len(chunk_results)}_alphagenome_raw.tsv.gz"
            df_chunk.to_csv(chunk_path, sep="\t", index=False)
            _log(f"Saved {len(chunk_results)} variants to {chunk_path}")
            chunk_results = []
            chunk_idx += 1

    if chunk_results:
        df_chunk = variant_scorers.tidy_scores(chunk_results)
        all_rows.append(df_chunk)
        chunk_path = out_dir / f"chunk_{chunk_idx:05d}_{len(chunk_results)}_alphagenome_raw.tsv.gz"
        df_chunk.to_csv(chunk_path, sep="\t", index=False)
        _log(f"Saved {len(chunk_results)} variants to {chunk_path}")

    if args.raw_combined:
        combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        combined.to_csv(args.raw_combined, sep="\t", index=False)
        _log(f"Saved combined raw AlphaGenome table: {args.raw_combined}")


def _discover_raw_files(raw_input: str, pattern: str) -> List[Path]:
    p = Path(raw_input)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob(pattern))
        if not files:
            raise ValueError(f"No files matched pattern '{pattern}' in directory: {raw_input}")
        return files
    raise FileNotFoundError(f"raw_input does not exist: {raw_input}")


def _pick_score_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    fallback_order = ["raw_score", "quantile_score", "score"]
    for col in fallback_order:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find score column '{requested}' and no fallback in {fallback_order}. "
        f"Available columns (sample): {list(df.columns)[:20]}"
    )


def _variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{_normalize_chrom(chrom)}:{int(pos)}:{ref.upper()}:{alt.upper()}"


def _vcf_id_map(vcf_path: str) -> Dict[str, str]:
    id_map: Dict[str, str] = {}
    for rec in read_vcf_variants(vcf_path):
        id_map[_variant_key(rec.chrom, rec.pos, rec.ref, rec.alt)] = rec.vid
    return id_map


def _extract_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer explicit columns if present in raw output.
    has_explicit = {"chrom", "pos", "ref", "alt"}.issubset(set(df.columns))
    if has_explicit:
        out = pd.DataFrame(
            {
                "chrom": df["chrom"].astype(str).map(_normalize_chrom),
                "pos": df["pos"].astype(int),
                "ref": df["ref"].astype(str).str.upper(),
                "alt": df["alt"].astype(str).str.upper(),
            }
        )
        return out

    if "variant_id" not in df.columns:
        raise ValueError("Raw AlphaGenome table needs either (chrom,pos,ref,alt) or variant_id column.")

    parsed = df["variant_id"].astype(str).map(_parse_variant_id)
    out = pd.DataFrame(parsed.tolist(), columns=["chrom", "pos", "ref", "alt"])
    return out


def _build_track_name(output_type: str, biosample: Optional[str], include_biosample: bool) -> str:
    assay = _clean_token(output_type)
    if include_biosample:
        bs = _clean_token(biosample if biosample else "NA")
        return f"{assay}:{bs}"
    return assay


def run_finalize(args: argparse.Namespace) -> None:
    """
    Convert raw AlphaGenome tidy output to benchmark model-score CSV contract:
      chrom,pos,id,ref,alt,<track columns>
    """
    raw_files = _discover_raw_files(args.raw_input, args.pattern)
    _log(f"Loading {len(raw_files)} raw AlphaGenome file(s)")

    frames = [pd.read_csv(p, sep="\t", low_memory=False) for p in raw_files]
    raw_df = pd.concat(frames, ignore_index=True)
    if raw_df.empty:
        raise ValueError("No rows found in raw AlphaGenome input.")

    score_col = _pick_score_column(raw_df, args.score_column)
    if "output_type" not in raw_df.columns:
        raise ValueError("Raw AlphaGenome table must contain 'output_type'.")

    variant_df = _extract_variant_columns(raw_df)
    raw_df = raw_df.copy()
    raw_df["chrom"] = variant_df["chrom"]
    raw_df["pos"] = variant_df["pos"]
    raw_df["ref"] = variant_df["ref"]
    raw_df["alt"] = variant_df["alt"]

    if args.variant_id_column and args.variant_id_column in raw_df.columns:
        raw_df["id"] = raw_df[args.variant_id_column].astype(str)
    else:
        raw_df["id"] = [
            f"{c}:{p}:{r}:{a}" for c, p, r, a in zip(raw_df["chrom"], raw_df["pos"], raw_df["ref"], raw_df["alt"])
        ]

    if args.vcf_for_ids:
        id_map = _vcf_id_map(args.vcf_for_ids)
        keys = [
            _variant_key(c, p, r, a)
            for c, p, r, a in zip(raw_df["chrom"], raw_df["pos"], raw_df["ref"], raw_df["alt"])
        ]
        raw_df["id"] = [id_map.get(k, prev) for k, prev in zip(keys, raw_df["id"])]

    if args.use_absolute:
        raw_df["_score"] = pd.to_numeric(raw_df[score_col], errors="coerce").abs()
    else:
        raw_df["_score"] = pd.to_numeric(raw_df[score_col], errors="coerce")

    include_biosample = not args.no_biosample and "biosample_name" in raw_df.columns
    if include_biosample:
        raw_df["track"] = [
            _build_track_name(a, b, True)
            for a, b in zip(raw_df["output_type"], raw_df["biosample_name"])
        ]
    else:
        raw_df["track"] = [_build_track_name(a, None, False) for a in raw_df["output_type"]]

    group_cols = ["chrom", "pos", "id", "ref", "alt", "track"]

    agg_mode = args.aggregation.lower()
    if agg_mode == "mean":
        agg_series = raw_df.groupby(group_cols)["_score"].mean()
    elif agg_mode == "median":
        agg_series = raw_df.groupby(group_cols)["_score"].median()
    elif agg_mode == "max":
        agg_series = raw_df.groupby(group_cols)["_score"].max()
    elif agg_mode == "sum":
        agg_series = raw_df.groupby(group_cols)["_score"].sum()
    else:
        raise ValueError(f"Unsupported aggregation mode: {args.aggregation}")

    long_df = agg_series.reset_index(name="value")

    wide = long_df.pivot_table(
        index=["chrom", "pos", "id", "ref", "alt"],
        columns="track",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Keep consistent metadata order and deterministic track order.
    meta = ["chrom", "pos", "id", "ref", "alt"]
    tracks = sorted([c for c in wide.columns if c not in meta])
    final_df = wide[meta + tracks]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    _log(f"Wrote finalized AlphaGenome model scores: {out_path}")
    _log(f"Variants: {len(final_df)} | Track columns: {len(tracks)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaGenome scorer + formatter for benchmark pipeline integration"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Run AlphaGenome API scoring and write tidy raw outputs")
    score.add_argument("--vcf", required=True, help="Input VCF (plain/gz)")
    score.add_argument("--api_key", required=True, help="AlphaGenome API key")
    score.add_argument("--organism", required=True, choices=["human", "mouse"])
    score.add_argument(
        "--sequence_length",
        required=True,
        help="One of AlphaGenome sequence keys: 2KB,16KB,100KB,500KB,1MB",
    )
    score.add_argument("--chunk_size", type=int, default=100)
    score.add_argument("--output_raw_dir", required=True, help="Output directory for raw chunk TSVs")
    score.add_argument(
        "--raw_combined",
        default=None,
        help="Optional path for one combined raw TSV after chunking",
    )
    score.add_argument(
        "--scorers",
        default=None,
        help=(
            "Optional comma-separated scorer keys. "
            "Example: cage,dnase,atac. If omitted, uses all recommended scorers."
        ),
    )
    for key in SUPPORTED_SCORER_KEYS:
        score.add_argument(
            f"--score_{key}",
            default=None,
            help=f"Optional boolean flag for scorer '{key}' (true/false)",
        )

    finalize = sub.add_parser(
        "finalize",
        help="Convert raw AlphaGenome tidy output to benchmark model score CSV",
    )
    finalize.add_argument(
        "--raw_input",
        required=True,
        help="Raw AlphaGenome input path (single TSV/TSV.GZ file or directory)",
    )
    finalize.add_argument(
        "--pattern",
        default="*_alphagenome_raw.tsv.gz",
        help="Glob pattern when raw_input is a directory",
    )
    finalize.add_argument("--out_csv", required=True, help="Output CSV path for merge_data.py")
    finalize.add_argument(
        "--score_column",
        default="quantile_score",
        help="Preferred score column in raw table (fallback: raw_score/quantile_score/score)",
    )
    finalize.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "median", "max", "sum"],
        help="How to collapse duplicate rows per variant+track",
    )
    finalize.add_argument(
        "--use_absolute",
        action="store_true",
        help="Use absolute score values before aggregation",
    )
    finalize.add_argument(
        "--no_biosample",
        action="store_true",
        help="If set, use output_type only as track name (no assay:biosample split)",
    )
    finalize.add_argument(
        "--variant_id_column",
        default=None,
        help="Optional raw column to map to output 'id' before VCF ID override",
    )
    finalize.add_argument(
        "--vcf_for_ids",
        default=None,
        help="Optional VCF to restore original ID values by chrom:pos:ref:alt",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "score":
        run_score(args)
    elif args.command == "finalize":
        run_finalize(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
