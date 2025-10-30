# Quick Start Guide

## Setup (First Time Only)

```bash
# Navigate to pipeline directory
cd /data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/Pipeline

# Check that everything looks good
ls -l Data/VCF/
ls -l Data/MPRA/
ls -l Scripts/
```

## Test Run (Dry Run)

```bash
# See what will be executed without actually running
snakemake -n --use-conda
```

## Run the Pipeline

### Option 1: Run Both Models
```bash
snakemake --cores 8 --use-conda
```

### Option 2: Run Only Enformer
```bash
# First disable Basenji in config
# Edit Configs/config.yaml and set:
#   models:
#     enformer: true
#     basenji: false

snakemake results/enformer/plots/correlation.png --cores 4 --use-conda
```

### Option 3: Run Only Basenji  
```bash
# First disable Enformer in config
# Edit Configs/config.yaml and set:
#   models:
#     enformer: false
#     basenji: true

snakemake results/basenji/plots/correlation.png --cores 4 --use-conda
```

## Check Results

```bash
# Enformer results
ls -lh results/enformer/
cat results/enformer/enformer_scores.csv | head
cat results/enformer/merged_data.tsv | head

# Basenji results
ls -lh results/basenji/
cat results/basenji/basenji_scores.csv | head
cat results/basenji/merged_data.tsv | head

# View plots
# Copy to your local machine or use image viewer
```

## Common Tasks

### Change VCF Input
Edit `Configs/config.yaml`:
```yaml
vcf: "Data/VCF/IGVFFI4378PZYI.vcf.gz"  # Use the other VCF
```

### Change Cell Line
Edit `Configs/config.yaml`:
```yaml
mpra_data: "Data/MPRA/HEPG2_reporter_variants.tsv.gz"
cell_line: "HepG2"
```

### Process All Variants (Not Just 1000)
Edit `Configs/config.yaml`:
```yaml
enformer:
  max_variants: "all"  # Change from "1000" to "all"
```

### Clean Up and Rerun
```bash
# Remove all results
rm -rf results/* logs/*

# Rerun
snakemake --cores 8 --use-conda
```

### View Log Files
```bash
# Enformer logs
tail -f logs/enformer/enformer.log

# Basenji logs  
tail -f logs/basenji/basenji.log
```

## Troubleshooting

### Pipeline fails immediately
```bash
# Check conda environments are set up
conda env list

# Manually create if needed
conda env create -f Configs/enformer-env.yaml
conda env create -f Configs/basenji-env.yaml
conda env create -f Configs/analysis-env.yaml
```

### GPU not found
```bash
# Check GPU visibility
nvidia-smi

# Check TensorFlow can see GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of memory
- Reduce number of variants in config (use max_variants: "100" for testing)
- Request more memory from cluster
- Process VCFs in batches

## Expected Runtime

- **Enformer** (1000 variants): ~30-60 minutes (GPU), several hours (CPU)
- **Basenji** (1000 variants): ~1-2 hours (GPU), very slow on CPU
- **Merge + Plots**: <5 minutes each

## Next Steps

1. Run pipeline on test data (1000 variants)
2. Verify outputs look correct
3. Scale up to all variants
4. Add more models (AlphaGenome, Borzoi)
5. Compare model performances
