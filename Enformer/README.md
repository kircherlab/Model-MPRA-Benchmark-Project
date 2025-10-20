# Enformer SAD Scoring — Quick Reference 
## Usage
Compute raw SAD scores (ALT − REF) per variant from Enformer, with optional aggregations.

Load the environment file `env/environment.yml` using this command:

`conda env create -f environment.yml`
`conda activate enformer`

And then start. 

`````
python enformer_SAD.py [--max_variants N|all]
--vcf `in.vcf.gz` --fasta `hg38.fa` --targets `targets_human.txt`
[--out `variant_SAD_raw.csv`]
[--agg_global] [--agg_global_max] [--agg_global_min]
[--agg_per_biosample] [--agg_per_biosample_max] [--agg_per_biosample_min]
[--agg_per_assay] [--agg_per_assay_max] [--agg_per_assay_min]
`````

## Arguments

`--vcf` : Input VCF (gz ok)

`--fasta` : Reference FASTA

`--targets` : Borzoi Targets TSV

`--out` : Output CSV (default: `variant_SAD_raw.csv`)

`--max_variants` : Integer N or `all` (default `all`)

## Aggregation Flags

Global (over all tracks):

`--agg_global` → `SAD_GLOBAL_MEAN`

`--agg_global_max` → `SAD_GLOBAL_MAX`

`--agg_global_min` → `SAD_GLOBAL_MIN`

Per-biosample (over that biosample’s tracks):

`--agg_per_biosample` → `SAD_BIOSAMPLE_<biosample>` (mean)

`--agg_per_biosample_max` → `SAD_BIOSAMPLE_MAX_<biosample>` (max)

`--agg_per_biosample_min` → `SAD_BIOSAMPLE_MIN_<biosample>` (min)

Per-assay (over all biosamples of that assay):

`--agg_per_assay` → `SAD_ASSAY_<assay>` (mean)

`--agg_per_assay_max` → `SAD_ASSAY_MAX_<assay>` (max)

`--agg_per_assay_min` → `SAD_ASSAY_MIN_<assay>` (min)

## Output Columns (order)
`````
chrom, pos, id, ref, alt → [global mean/max/min] → [per-biosample mean/max/min] → [per-assay mean/max/min] → per-track SADs
`````

## Examples

First 10 variants with global + per-biosample mean, and per-assay max/min:
`````
python enformer_SAD.py --max_variants 10 --agg_global --agg_per_biosample --agg_per_assay_max --agg_per_assay_min
`````

All variants with global max/min only:
`````
python enformer_SAD.py --agg_global_max --agg_global_min
`````
# Model-MPRA-Benchmark-Project
Benchmarking DNA sequence models against multiple MPRA datasets with standardized pipelines
