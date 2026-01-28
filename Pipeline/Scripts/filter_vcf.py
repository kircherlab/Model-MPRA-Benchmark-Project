#!/usr/bin/env python3
"""Filter VCF to top N variants sorted by abs(LOG2FC)"""

import sys
import gzip
import shutil
import re

def extract_log2fc(vcf_line):
    """Extract LOG2FC value from INFO field, return 0.0 if not found"""
    match = re.search(r'LOG2FC=([-+]?[0-9]*\.?[0-9]+)', vcf_line)
    if match:
        return float(match.group(1))
    return 0.0

def filter_vcf(input_vcf, output_vcf, max_variants, log_file):
    """Filter VCF file to top N variants sorted by abs(LOG2FC)"""
    
    # If no filtering, just copy
    if max_variants == 'all':
        shutil.copy(input_vcf, output_vcf)
        with open(log_file, 'w') as log:
            log.write("No filtering applied (max_variants=all)\n")
        return
    
    # Convert max_variants to int
    max_variants = int(max_variants)
    
    # Read VCF
    opener = gzip.open if input_vcf.endswith('.gz') else open
    header_lines = []
    variant_lines = []
    
    with opener(input_vcf, 'rt') as infile:
        for line in infile:
            if line.startswith('#'):
                header_lines.append(line)
            else:
                log2fc = extract_log2fc(line)
                variant_lines.append((abs(log2fc), line))
    
    # Sort by abs(LOG2FC) descending
    variant_lines.sort(key=lambda x: x[0], reverse=True)
    
    # Write filtered VCF
    writer = gzip.open if output_vcf.endswith('.gz') else open
    with writer(output_vcf, 'wt') as outfile:
        # Write headers
        for line in header_lines:
            outfile.write(line)
        
        # Write top N variants
        variant_count = 0
        for abs_log2fc, line in variant_lines[:max_variants]:
            outfile.write(line)
            variant_count += 1
    
    # Log results
    with open(log_file, 'w') as log:
        log.write(f"Sorted variants by abs(LOG2FC) and took top {max_variants}\n")
        log.write(f"Total variants written: {variant_count}\n")
        if variant_count > 0:
            log.write(f"Top abs(LOG2FC): {variant_lines[0][0]:.4f}\n")
            if variant_count == max_variants and len(variant_lines) > max_variants:
                log.write(f"Last included abs(LOG2FC): {variant_lines[max_variants-1][0]:.4f}\n")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: filter_vcf.py <input_vcf> <output_vcf> <max_variants> <log_file>")
        sys.exit(1)
    
    filter_vcf(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
