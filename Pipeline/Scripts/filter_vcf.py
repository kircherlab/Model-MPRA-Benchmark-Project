#!/usr/bin/env python3
"""Filter VCF to top N variants (or copy all if max_variants='all')"""

import sys
import gzip
import shutil

def filter_vcf(input_vcf, output_vcf, max_variants, log_file):
    """Filter VCF file to first N variants or copy all"""
    
    # If no filtering, just copy
    if max_variants == 'all':
        shutil.copy(input_vcf, output_vcf)
        with open(log_file, 'w') as log:
            log.write("No filtering applied (max_variants=all)\n")
        return
    
    # Convert max_variants to int
    max_variants = int(max_variants)
    
    # Filter VCF
    variant_count = 0
    opener = gzip.open if input_vcf.endswith('.gz') else open
    writer = gzip.open if output_vcf.endswith('.gz') else open
    
    with opener(input_vcf, 'rt') as infile, writer(output_vcf, 'wt') as outfile:
        for line in infile:
            # Always write header lines
            if line.startswith('#'):
                outfile.write(line)
                continue
            
            # Write variant lines until limit
            if variant_count < max_variants:
                outfile.write(line)
                variant_count += 1
            else:
                break
    
    # Log results
    with open(log_file, 'w') as log:
        log.write(f"Filtered VCF to first {max_variants} variants\n")
        log.write(f"Total variants written: {variant_count}\n")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: filter_vcf.py <input_vcf> <output_vcf> <max_variants> <log_file>")
        sys.exit(1)
    
    filter_vcf(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
