#!/usr/bin/env python
"""
Basenji SAD Wrapper Script
Computes SNP Activity Difference (SAD) scores for variants in a VCF file.
Wraps the original basenji_sad.py with simplified interface.
"""

import os
import sys
import argparse
import subprocess
import h5py
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Basenji SAD score computation wrapper')
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--fasta', required=True, help='Reference genome FASTA')
    parser.add_argument('--params', required=True, help='Model parameters JSON file')
    parser.add_argument('--model', required=True, help='Model weights file (.h5)')
    parser.add_argument('--targets', default=None, help='Targets file (optional)')
    parser.add_argument('--out_dir', default='basenji_sad_output', help='Output directory')
    parser.add_argument('--out_csv', required=True, help='Output CSV file for scores')
    parser.add_argument('--basenji_script', required=True, help='Path to basenji_sad.py')
    parser.add_argument('--rc', action='store_true', help='Average forward and reverse complement')
    parser.add_argument('--shifts', default='0', help='Ensemble prediction shifts')
    parser.add_argument('--aggregation', choices=['mean', 'sum'], default='sum',
                        help='Method to collapse spatial dimension: mean or sum (default: sum)')
    
    args = parser.parse_args()
    
    # Set BASENJIDIR environment variable and add to PYTHONPATH
    basenji_dir = os.path.dirname(os.path.dirname(args.basenji_script))
    os.environ['BASENJIDIR'] = basenji_dir
    
    # Add basenji directory to PYTHONPATH so the basenji module can be imported
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = f"{basenji_dir}:{os.environ['PYTHONPATH']}"
    else:
        os.environ['PYTHONPATH'] = basenji_dir
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Build basenji_sad.py command
    # NOTE: We do NOT pass --targets to basenji_sad.py because it expects
    # a different format with clip_soft and other columns. We only use
    # the targets file later for labeling the output CSV columns.
    cmd = [
        'python', args.basenji_script,
        '-f', args.fasta,
        '-o', args.out_dir,
        '--shifts', args.shifts,
    ]
    
    if args.rc:
        cmd.append('--rc')
    
    # Add positional arguments
    cmd.extend([args.params, args.model, args.vcf])
    
    print(f"Running Basenji SAD: {' '.join(cmd)}", flush=True)
    print(f"Using aggregation method: {args.aggregation}", flush=True)
    
    # Run basenji_sad.py
    result = subprocess.run(cmd, check=True)
    
    # Convert HDF5 output to CSV with proper target labels
    print("Converting HDF5 output to CSV...", flush=True)
    h5_file = os.path.join(args.out_dir, 'sad.h5')
    
    if os.path.exists(h5_file):
        convert_h5_to_csv(h5_file, args.out_csv, args.targets, args.aggregation)
        print(f"SAD scores saved to {args.out_csv}", flush=True)
    else:
        print(f"ERROR: Expected output file {h5_file} not found!", file=sys.stderr)
        sys.exit(1)

def convert_h5_to_csv(h5_file, out_csv, targets_file=None, aggregation='sum'):
    """
    Convert Basenji HDF5 output to CSV format matching Enformer's output structure.
    Applies aggregation transformation if needed to match desired method.
    """
    with h5py.File(h5_file, 'r') as f:
        # Extract variant information
        chrom = [x.decode() if isinstance(x, bytes) else str(x) for x in f['chr'][:]]
        pos = f['pos'][:]
        ref = [x.decode() if isinstance(x, bytes) else str(x) for x in f['ref_allele'][:]]
        alt = [x.decode() if isinstance(x, bytes) else str(x) for x in f['alt_allele'][:]]
        
        # Create variant IDs (using SNP IDs from basenji if available)
        if 'snp' in f:
            variant_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['snp'][:]]
        else:
            variant_ids = [f"{c}:{p}:{r}:{a}" for c, p, r, a in zip(chrom, pos, ref, alt)]
        
        # Extract SAD scores
        sad_scores = f['SAD'][:]  # Shape: (n_variants, n_targets)
        
        # Basenji's SAD is computed as sum by default
        # If mean is requested, divide by sequence length (896 bins for Basenji)
        if aggregation == 'mean':
            # Basenji has 896 prediction bins
            seq_length = 896
            sad_scores = sad_scores / seq_length
            print(f"Converting SAD from sum to mean (dividing by {seq_length})", flush=True)
    
    # Load target descriptions from targets file if provided
    if targets_file and os.path.exists(targets_file):
        targets_df = pd.read_csv(targets_file, sep='\t')
        if 'description' in targets_df.columns:
            # Use descriptions as column names (same format as Enformer)
            target_columns = targets_df['description'].astype(str).tolist()
        else:
            target_columns = [f"t{i}" for i in range(sad_scores.shape[1])]
    else:
        # Fallback to simple t0, t1, t2 naming
        target_columns = [f"t{i}" for i in range(sad_scores.shape[1])]
    
    # Create DataFrame with SAD scores per target
    df = pd.DataFrame(sad_scores, columns=target_columns)
    
    # Add variant metadata columns at the front (same order as Enformer)
    df.insert(0, 'chrom', chrom)
    df.insert(1, 'pos', pos)
    df.insert(2, 'id', variant_ids)
    df.insert(3, 'ref', ref)
    df.insert(4, 'alt', alt)
    
    # Save to CSV
    df.to_csv(out_csv, index=False)
    
    return df

if __name__ == '__main__':
    main()
