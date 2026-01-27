import pandas as pd
import sys
import gzip
import argparse
from collections import OrderedDict
import numpy as np
import re


# This script now expects to be run from Snakemake where the following are provided.
# Use a globals() check so static import-time analysis won't fail when `snakemake` is absent.
if 'snakemake' in globals():
    _sn = globals().get('snakemake')
    model_scores_file = _sn.input.model_scores # type: ignore
    vcf_file = _sn.input.vcf # type: ignore
    output_file = _sn.output[0] # type: ignore
    params = getattr(_sn, 'params', {})
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_scores', required=True, help='Model scores CSV file (Enformer, Basenji, etc.)')
    parser.add_argument('--vcf', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--agg_global', action='store_true')
    parser.add_argument('--agg_global_max', action='store_true')
    parser.add_argument('--agg_global_min', action='store_true')
    parser.add_argument('--agg_per_biosample', action='store_true')
    parser.add_argument('--agg_per_biosample_max', action='store_true')
    parser.add_argument('--agg_per_biosample_min', action='store_true')
    parser.add_argument('--agg_per_assay', action='store_true')
    parser.add_argument('--agg_per_assay_max', action='store_true')
    parser.add_argument('--agg_per_assay_min', action='store_true')
    a = parser.parse_args()
    model_scores_file = a.model_scores
    vcf_file = a.vcf
    output_file = a.out
    params = {
        'agg_global': a.agg_global,
        'agg_global_max': a.agg_global_max,
        'agg_global_min': a.agg_global_min,
        'agg_per_biosample': a.agg_per_biosample,
        'agg_per_biosample_max': a.agg_per_biosample_max,
        'agg_per_biosample_min': a.agg_per_biosample_min,
        'agg_per_assay': a.agg_per_assay,
        'agg_per_assay_max': a.agg_per_assay_max,
        'agg_per_assay_min': a.agg_per_assay_min,
    }


def read_model_scores(scores_path):
    print(f"Loading model scores from: {scores_path}")
    df = pd.read_csv(scores_path)
    return df


def parse_info_field(info_str):
    """Parse VCF INFO string into a dict. Handles key=value;key2=value2 or flag keys."""
    info = {}
    if info_str == '.' or info_str is None:
        return info
    for entry in info_str.split(';'):
        if '=' in entry:
            k, v = entry.split('=', 1)
            info[k] = v
        else:
            info[entry] = True
    return info


def read_vcf_mpra(vcf_path):
    """Read VCF and extract INFO fields into a list of dicts keyed by chrom:pos:ref:alt"""
    print(f"Reading VCF for MPRA INFO: {vcf_path}")
    opener = gzip.open if vcf_path.endswith('.gz') else open
    records = OrderedDict()
    with opener(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'): continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 8: continue
            chrom, pos, vid, ref, alts, qual, filt, info = cols[:8]
            # handle multi-allelic by creating one record per alt allele
            for alt in alts.split(','):
                key = f"{chrom}:{pos}:{ref.upper()}:{alt.upper()}"
                info_dict = parse_info_field(info)
                # store INFO as-is (string values). Later we may expand specific MPRA keys.
                records[key] = {'chrom': chrom, 'pos': int(pos), 'id': vid, 'ref': ref.upper(), 'alt': alt.upper(), 'info': info_dict}
    print(f"Parsed {len(records)} VCF variant entries")
    return records


def expand_info_columns(records):
    """From the VCF INFO dicts, determine all INFO keys and produce a DataFrame where keys become columns."""
    # collect keys
    keys = set()
    for v in records.values():
        keys.update(v['info'].keys())
    keys = sorted(keys)
    rows = []
    for key, v in records.items():
        base = {'chrom': v['chrom'], 'pos': v['pos'], 'id': v['id'], 'ref': v['ref'], 'alt': v['alt']}
        for k in keys:
            val = v['info'].get(k, None)
            # try numeric conversion when possible
            if isinstance(val, str) and re.match(r'^-?\d+(?:\.\d+)?(?:,.*)?$', val):
                # keep as string if comma-separated list; else convert to float
                if ',' in val:
                    base[k] = val
                else:
                    try:
                        base[k] = float(val)
                    except ValueError:
                        base[k] = val
            else:
                base[k] = val
        rows.append(base)
    df = pd.DataFrame(rows)
    cols = ['chrom', 'pos', 'id', 'ref', 'alt'] + [k for k in keys]
    df = df[cols]
    return df


def compute_aggregations(enf_df, descriptions, params):
    """Compute aggregations across the track columns. Returns a dict of agg_name -> value for each row."""
    # build assay/biosample mappings from descriptions
    assays = []
    biosamples = []
    for d in descriptions:
        if ':' in d:
            a, b = d.split(':', 1)
            assays.append(a.strip())
            biosamples.append(re.sub(r'\s+', ' ', re.sub(r'(?i)\b(male|female)\b', '', b)).strip())
        else:
            assays.append(d.strip())
            biosamples.append('NA')

    assay_to_idx = {}
    biosample_to_idx = {}
    for i, (a, b) in enumerate(zip(assays, biosamples)):
        assay_to_idx.setdefault(a, []).append(i)
        biosample_to_idx.setdefault(b, []).append(i)

    # For each row in enf_df, compute aggs based on params
    use_abs = params.get('use_absolute', False)
    aggs_rows = []
    track_cols = descriptions
    for _, row in enf_df.iterrows():
        scores = row[track_cols].values.astype(float)
        row_aggs = {}
        if params.get('agg_global'):
            row_aggs['SAD_GLOBAL_MEAN'] = np.nanmean(scores)
            if use_abs:
                row_aggs['SAD_GLOBAL_MEAN_ABS'] = np.nanmean(np.abs(scores))
        if params.get('agg_global_max'):
            row_aggs['SAD_GLOBAL_MAX'] = np.nanmax(scores)
            if use_abs:
                row_aggs['SAD_GLOBAL_MAX_ABS'] = np.nanmax(np.abs(scores))
        if params.get('agg_global_min'):
            row_aggs['SAD_GLOBAL_MIN'] = np.nanmin(scores)
            if use_abs:
                row_aggs['SAD_GLOBAL_MIN_ABS'] = np.nanmin(np.abs(scores))

        if params.get('agg_per_biosample') or params.get('agg_per_biosample_max') or params.get('agg_per_biosample_min'):
            for b in sorted(biosample_to_idx.keys()):
                idxs = biosample_to_idx[b]
                if not idxs: continue
                if params.get('agg_per_biosample'):
                    row_aggs[f'SAD_BIOSAMPLE_{b}'] = np.nanmean(scores[idxs])
                    if use_abs:
                        row_aggs[f'SAD_BIOSAMPLE_{b}_ABS'] = np.nanmean(np.abs(scores[idxs]))
                if params.get('agg_per_biosample_max'):
                    row_aggs[f'SAD_BIOSAMPLE_MAX_{b}'] = np.nanmax(scores[idxs])
                    if use_abs:
                        row_aggs[f'SAD_BIOSAMPLE_MAX_{b}_ABS'] = np.nanmax(np.abs(scores[idxs]))
                if params.get('agg_per_biosample_min'):
                    row_aggs[f'SAD_BIOSAMPLE_MIN_{b}'] = np.nanmin(scores[idxs])
                    if use_abs:
                        row_aggs[f'SAD_BIOSAMPLE_MIN_{b}_ABS'] = np.nanmin(np.abs(scores[idxs]))

        if params.get('agg_per_assay') or params.get('agg_per_assay_max') or params.get('agg_per_assay_min'):
            for a in sorted(assay_to_idx.keys()):
                idxs = assay_to_idx[a]
                if not idxs: continue
                if params.get('agg_per_assay'):
                    row_aggs[f'SAD_ASSAY_{a}'] = np.nanmean(scores[idxs])
                    if use_abs:
                        row_aggs[f'SAD_ASSAY_{a}_ABS'] = np.nanmean(np.abs(scores[idxs]))
                if params.get('agg_per_assay_max'):
                    row_aggs[f'SAD_ASSAY_MAX_{a}'] = np.nanmax(scores[idxs])
                    if use_abs:
                        row_aggs[f'SAD_ASSAY_MAX_{a}_ABS'] = np.nanmax(np.abs(scores[idxs]))
                if params.get('agg_per_assay_min'):
                    row_aggs[f'SAD_ASSAY_MIN_{a}'] = np.nanmin(scores[idxs])
                    if use_abs:
                        row_aggs[f'SAD_ASSAY_MIN_{a}_ABS'] = np.nanmin(np.abs(scores[idxs]))

        aggs_rows.append(row_aggs)
    aggs_df = pd.DataFrame(aggs_rows)
    return aggs_df



# --- Chromosome normalization helper ---
def norm_chrom(chrom):
    """Normalize chromosome name to 'chrN' format."""
    chrom = str(chrom)
    if chrom.startswith('chr'):
        return chrom
    # handle numeric chromosomes and X/Y
    if chrom in ['X', 'Y']:
        return f'chr{chrom}'
    if chrom.isdigit():
        return f'chr{chrom}'
    return chrom

def make_join_key(row):
    chrom = norm_chrom(row['chrom'])
    pos = int(row['pos'])
    ref = str(row['ref']).upper()
    alt = str(row['alt']).upper()
    return f"{chrom}:{pos}:{ref}:{alt}"

def main():
    model_df = read_model_scores(model_scores_file)
    meta_cols = ['chrom', 'pos', 'id', 'ref', 'alt', 'variant_id']
    # Get all columns that are not metadata (these are the SAD scores)
    track_cols = [c for c in model_df.columns if c not in meta_cols and not c.startswith('SAD_GLOBAL') and not c.startswith('SAD_ASSAY') and not c.startswith('SAD_BIOSAMPLE')]
    descriptions = track_cols.copy()

    vcf_records = read_vcf_mpra(vcf_file)
    mpra_df = expand_info_columns(vcf_records)

    # Normalize join keys for both dataframes
    model_df['join_key'] = model_df.apply(make_join_key, axis=1)
    mpra_df['join_key'] = mpra_df.apply(make_join_key, axis=1)

    merged = pd.merge(model_df, mpra_df, on='join_key', how='inner', suffixes=('', '_mpra'))
    if merged.empty:
        print('No overlap between model output and VCF MPRA entries after normalization. Exiting.')
        # Optionally, print a few sample keys from each side for debugging
        print('Sample model keys:', model_df['join_key'].head(5).tolist())
        print('Sample MPRA keys:', mpra_df['join_key'].head(5).tolist())
        sys.exit(1)

    aggs_df = compute_aggregations(merged, descriptions, params)

    mpra_info_cols = [c for c in mpra_df.columns if c not in ['chrom', 'pos', 'id', 'ref', 'alt', 'join_key']]
    agg_cols = list(aggs_df.columns)

    # Determine which track columns to keep
    # Keep tracks if: (1) there are very few (<20), or (2) they start with ND_ (HyenaDNA scores)
    keep_tracks = len(descriptions) < 20 or any(d.startswith('ND_') for d in descriptions)
    
    # Build final output: metadata + MPRA INFO + aggregations + (optionally) track scores
    out_meta = ['chrom', 'pos', 'id', 'ref', 'alt']
    
    if keep_tracks:
        # Keep track scores (e.g., HyenaDNA ND_* scores, or models with few tracks)
        final = pd.concat([
            merged[out_meta].reset_index(drop=True),
            merged[mpra_info_cols].reset_index(drop=True),
            aggs_df.reset_index(drop=True),
            merged[descriptions].reset_index(drop=True)
        ], axis=1)
        print(f'Wrote merged MPRA data + aggregations + track scores to: {output_file}')
        print(f'Columns: {len(out_meta)} metadata + {len(mpra_info_cols)} MPRA + {len(agg_cols)} aggregations + {len(descriptions)} tracks')
    else:
        # Don't keep track scores (e.g., Enformer/Basenji with thousands of tracks)
        final = pd.concat([
            merged[out_meta].reset_index(drop=True),
            merged[mpra_info_cols].reset_index(drop=True),
            aggs_df.reset_index(drop=True)
        ], axis=1)
        print(f'Wrote merged MPRA data + aggregations to: {output_file}')
        print(f'Columns: {len(out_meta)} metadata + {len(mpra_info_cols)} MPRA + {len(agg_cols)} aggregations')
        print(f'Note: {len(descriptions)} track scores excluded to avoid data duplication (available in model output files)')

    final.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
