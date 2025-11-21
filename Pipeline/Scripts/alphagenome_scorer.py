#!/usr/bin/env python3
"""
AlphaGenome Variant Scorer
Wrapper script for AlphaGenome API-based variant scoring
"""

import pandas as pd
import numpy as np
from alphagenome.models import dna_client, variant_scorers
from alphagenome.data import genome
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Score variants using AlphaGenome API')
    parser.add_argument('--vcf', required=True, help='Input VCF file (gzipped)')
    parser.add_argument('--fasta', required=True, help='Reference genome FASTA')
    parser.add_argument('--api_key', required=True, help='AlphaGenome API key')
    parser.add_argument('--organism', default='human', choices=['human', 'mouse'], help='Organism')
    parser.add_argument('--sequence_length', default='1MB', help='Sequence length around variants')
    parser.add_argument('--scorers', nargs='+', default=['atac', 'dnase', 'chip_histone', 'chip_tf'],
                       help='Scorers to use')
    parser.add_argument('--out_csv', required=True, help='Output CSV file')

    args = parser.parse_args()

    # Validate API key
    if not args.api_key or args.api_key == "":
        print("ERROR: AlphaGenome API key is required. Set it in config.yaml or pass --api_key", file=sys.stderr)
        sys.exit(1)

    # Create DNA model client
    try:
        dna_model = dna_client.create(args.api_key)
    except Exception as e:
        print(f"ERROR: Failed to create AlphaGenome client: {e}", file=sys.stderr)
        print("Make sure your API key is valid and you have internet connection", file=sys.stderr)
        sys.exit(1)

    # Parse organism
    organism_map = {
        'human': dna_client.Organism.HOMO_SAPIENS,
        'mouse': dna_client.Organism.MUS_MUSCULUS,
    }
    organism = organism_map[args.organism]

    # Parse sequence length
    seq_length_map = {
        '2KB': dna_client.SUPPORTED_SEQUENCE_LENGTHS.SEQUENCE_LENGTH_2KB,
        '16KB': dna_client.SUPPORTED_SEQUENCE_LENGTHS.SEQUENCE_LENGTH_16KB,
        '100KB': dna_client.SUPPORTED_SEQUENCE_LENGTHS.SEQUENCE_LENGTH_100KB,
        '500KB': dna_client.SUPPORTED_SEQUENCE_LENGTHS.SEQUENCE_LENGTH_500KB,
        '1MB': dna_client.SUPPORTED_SEQUENCE_LENGTHS.SEQUENCE_LENGTH_1MB,
    }
    sequence_length = seq_length_map.get(args.sequence_length.upper())
    if sequence_length is None:
        print(f"ERROR: Invalid sequence length: {args.sequence_length}", file=sys.stderr)
        sys.exit(1)

    # Load VCF and filter to simple variants (no complex indels)
    print(f"Loading VCF: {args.vcf}")
    if args.vcf.endswith('.gz'):
        vcf_df = pd.read_csv(args.vcf, sep='\t', comment='#', header=None,
                           names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])
    else:
        # Read VCF properly handling header
        with open(args.vcf, 'r') as f:
            lines = f.readlines()
        header_lines = [line for line in lines if line.startswith('#')]
        data_lines = [line for line in lines if not line.startswith('#')]

        # Parse data lines
        vcf_data = []
        for line in data_lines:
            fields = line.strip().split('\t')
            if len(fields) >= 5:
                vcf_data.append({
                    'CHROM': fields[0],
                    'POS': int(fields[1]),
                    'ID': fields[2] if fields[2] != '.' else f"{fields[0]}:{fields[1]}:{fields[3]}:{fields[4]}",
                    'REF': fields[3],
                    'ALT': fields[4]
                })
        vcf_df = pd.DataFrame(vcf_data)

    # Filter to simple SNPs (no indels or complex variants)
    simple_variants = vcf_df[
        (vcf_df['REF'].str.len() == 1) &
        (vcf_df['ALT'].str.len() == 1) &
        (~vcf_df['REF'].str.contains(',')) &  # No multi-allelic
        (~vcf_df['ALT'].str.contains(','))
    ].copy()

    print(f"Loaded {len(vcf_df)} total variants, {len(simple_variants)} simple SNPs")

    if len(simple_variants) == 0:
        print("ERROR: No simple SNPs found in VCF", file=sys.stderr)
        sys.exit(1)

    # Select scorers
    all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    selected_scorers = []

    scorer_map = {
        'rna_seq': 'rna_seq',
        'cage': 'cage',
        'procap': 'procap',
        'atac': 'atac',
        'dnase': 'dnase',
        'chip_histone': 'chip_histone',
        'chip_tf': 'chip_tf',
        'polyadenylation': 'polyadenylation',
        'splice_sites': 'splice_sites',
        'splice_site_usage': 'splice_site_usage',
        'splice_junctions': 'splice_junctions',
    }

    for scorer_name in args.scorers:
        scorer_key = scorer_map.get(scorer_name.lower())
        if scorer_key and scorer_key in all_scorers:
            scorer = all_scorers[scorer_key]
            # Check if supported for organism
            if organism.value in variant_scorers.SUPPORTED_ORGANISMS[scorer.base_variant_scorer]:
                selected_scorers.append(scorer)
                print(f"Selected scorer: {scorer_name}")
            else:
                print(f"Warning: {scorer_name} not supported for {args.organism}, skipping")

    if not selected_scorers:
        print("ERROR: No valid scorers selected", file=sys.stderr)
        sys.exit(1)

    # Score variants
    results = []
    print(f"Scoring {len(simple_variants)} variants with {len(selected_scorers)} scorers...")

    for i, (_, vcf_row) in enumerate(simple_variants.iterrows()):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(simple_variants)} variants")

        try:
            variant = genome.Variant(
                chromosome=str(vcf_row.CHROM),
                position=int(vcf_row.POS),
                reference_bases=vcf_row.REF,
                alternate_bases=vcf_row.ALT,
                name=vcf_row.ID,
            )

            interval = variant.reference_interval.resize(sequence_length)

            variant_scores = dna_model.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=selected_scorers,
                organism=organism,
            )
            results.append(variant_scores)

        except Exception as e:
            print(f"Warning: Failed to score variant {vcf_row.ID}: {e}", file=sys.stderr)
            continue

    if not results:
        print("ERROR: No variants were successfully scored", file=sys.stderr)
        sys.exit(1)

    # Convert to tidy DataFrame
    df_scores = variant_scorers.tidy_scores(results)

    # Convert to format similar to other models (one row per variant per assay)
    output_rows = []

    for _, row in df_scores.iterrows():
        variant_id = row['variant_name']
        assay_type = row['modality']
        tissue = row['tissue']
        score = row['score']

        # Create assay name similar to other models
        if assay_type == 'ATAC':
            assay_name = 'ATAC'
        elif assay_type == 'DNASE':
            assay_name = 'DNASE'
        elif assay_type == 'CHIP_HISTONE':
            assay_name = f"CHIP_HISTONE_{row.get('antibody', 'UNKNOWN')}"
        elif assay_type == 'CHIP_TF':
            assay_name = f"CHIP_TF_{row.get('antibody', 'UNKNOWN')}"
        elif assay_type == 'RNA_SEQ':
            assay_name = 'RNA_SEQ'
        elif assay_type == 'CAGE':
            assay_name = 'CAGE'
        elif assay_type == 'PROCAP':
            assay_name = 'PROCAP'
        else:
            assay_name = assay_type

        output_rows.append({
            'variant_id': variant_id,
            'chrom': row['variant_name'].split(':')[0],
            'pos': int(row['variant_name'].split(':')[1].split(':')[0]),
            'ref': row['variant_name'].split(':')[2],
            'alt': row['variant_name'].split(':')[3],
            'assay': assay_name,
            'tissue': tissue,
            'score': score
        })

    output_df = pd.DataFrame(output_rows)

    # Save output
    output_df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(output_df)} scores to {args.out_csv}")

    # Summary
    print(f"Scored {len(simple_variants)} variants across {len(selected_scorers)} modalities")
    print(f"Total scores: {len(output_df)}")

if __name__ == '__main__':
    main()