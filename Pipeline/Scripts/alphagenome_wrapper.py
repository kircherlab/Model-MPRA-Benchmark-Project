#!/usr/bin/env python3
"""
AlphaGenome Variant Scoring Wrapper
Adapted from the AlphaGenome documentation and user-provided script
"""

import pandas as pd
from alphagenome import dna_client, variant_scorers
from alphagenome.data import genome
import os
import sys
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Score variants using AlphaGenome')
    parser.add_argument('--vcf', required=True, help='Path to VCF file (tab-separated with variant_id, CHROM, POS, REF, ALT)')
    parser.add_argument('--api_key', required=True, help='AlphaGenome API key')
    parser.add_argument('--organism', default='human', choices=['human', 'mouse'], help='Organism')
    parser.add_argument('--sequence_length', default='1MB', choices=['2KB', '16KB', '100KB', '500KB', '1MB'], help='Sequence length around variants')
    parser.add_argument('--out_csv', required=True, help='Output CSV file path')
    parser.add_argument('--modalities', nargs='+', default=['atac', 'cage', 'dnase', 'rna_seq'],
                       help='Modalities to score (default: atac cage dnase rna_seq)')

    args = parser.parse_args()

    # Check if VCF file exists
    if not os.path.exists(args.vcf):
        raise FileNotFoundError(f"VCF file not found: {args.vcf}")

    # Load VCF file
    print(f"Loading VCF file: {args.vcf}")
    vcf = pd.read_csv(args.vcf, sep='\t')

    required_columns = ['variant_id', 'CHROM', 'POS', 'REF', 'ALT']
    for column in required_columns:
        if column not in vcf.columns:
            raise ValueError(f'VCF file missing required column: {column}')

    print(f"Loaded {len(vcf)} variants")

    # Initialize AlphaGenome client
    print("Initializing AlphaGenome client...")
    dna_model = dna_client.create(args.api_key)

    # Parse organism
    organism_map = {
        'human': dna_client.Organism.HOMO_SAPIENS,
        'mouse': dna_client.Organism.MUS_MUSCULUS,
    }
    organism = organism_map[args.organism]

    # Parse sequence length
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f'SEQUENCE_LENGTH_{args.sequence_length}'
    ]

    # Configure modalities to score
    scorer_selections = {mod: mod in args.modalities for mod in [
        'rna_seq', 'cage', 'procap', 'atac', 'dnase',
        'chip_histone', 'chip_tf', 'polyadenylation',
        'splice_sites', 'splice_site_usage', 'splice_junctions'
    ]}

    all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    selected_scorers = [
        all_scorers[key]
        for key in all_scorers
        if scorer_selections.get(key.lower(), False)
    ]

    # Filter unsupported scorers for the chosen organism
    unsupported_scorers = [
        scorer
        for scorer in selected_scorers
        if (
            organism.value
            not in variant_scorers.SUPPORTED_ORGANISMS[scorer.base_variant_scorer]
        ) | (
            (scorer.requested_output == dna_client.OutputType.PROCAP)
            & (organism == dna_client.Organism.MUS_MUSCULUS)
        )
    ]

    if unsupported_scorers:
        print(f'Excluding {len(unsupported_scorers)} unsupported scorers for {args.organism}')
        for unsupported_scorer in unsupported_scorers:
            selected_scorers.remove(unsupported_scorer)

    print(f"Selected {len(selected_scorers)} scorers: {[s.base_variant_scorer for s in selected_scorers]}")

    # Score variants
    results = []
    print("Scoring variants...")

    for i, vcf_row in tqdm(vcf.iterrows(), total=len(vcf)):
        try:
            variant = genome.Variant(
                chromosome=str(vcf_row.CHROM),
                position=int(vcf_row.POS),
                reference_bases=vcf_row.REF,
                alternate_bases=vcf_row.ALT,
                name=vcf_row.variant_id,
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
            print(f"Error scoring variant {vcf_row.variant_id}: {e}")
            continue

    # Convert results to tidy format
    if results:
        print("Converting results to tidy format...")
        df_results = variant_scorers.tidy_scores(results)

        # Save results
        df_results.to_csv(args.out_csv, index=False)
        print(f"Saved {len(df_results)} scored variants to {args.out_csv}")
        print(f"Columns: {list(df_results.columns)}")
    else:
        print("No variants were successfully scored")
        # Create empty output file to avoid pipeline errors
        pd.DataFrame().to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()