# Model-MPRA Benchmark Project

Unified Snakemake pipeline for benchmarking DNA sequence models (Enformer, Basenji, AlphaGenome) against MPRA experimental data.

## Directory Structure

```
Model_MPRA_Benchmark/
├── Pipeline/
│   ├── Snakefile
│   ├── Configs/
│   ├── Scripts/
│   ├── Data/
│   ├── Models/
│   ├── results/
│   └── logs/
├── Visualizations/
└── Archive/
```

## Quick Start

```bash
# Navigate to the pipeline directory
cd Pipeline

# Test run with 10 variants (fast)
snakemake --cores 8 --use-conda

# Full analysis with all variants
# First edit Configs/config.yaml: set max_variants: "all"
snakemake --cores 8 --use-conda

# Clean up outputs and restart
snakemake --delete-all-output
```

## Configuration

Edit `Pipeline/Configs/config.yaml`:

```yaml
# Enable/disable models
models:
  enformer: true
  basenji: false
  alphagenome: false

# Limit variants for testing (or "all" for full run)
max_variants: "10"

# Input files
vcf: "Data/VCF/IGVFFI4134MFLL.vcf"
fasta: "Data/Genome/hg38.fa"
targets: "Data/Targets/targets_human.txt"

# Aggregation options
agg_global: true           # Compute mean across all tracks
agg_per_biosample: true    # Compute mean per biosample
agg_per_assay: true        # Compute mean per assay type
```

## Models Supported

- **Enformer**: TensorFlow Hub model (5,313 tracks, auto-downloaded)
- **Basenji**: Local TensorFlow model (requires manual setup)
- **AlphaGenome**: In development

## Output

Results in `Pipeline/results/{model}/`:
- `{model}_scores.csv` - Raw prediction scores per track
- `merged_data.tsv` - Predictions merged with MPRA data
- `plots/` - Correlation and ROC curves

## Visualization

Generate publication plots:

```bash
cd Visualizations
python publication_plots.py --merged_data ../Pipeline/results/enformer/merged_data.tsv
```

## Requirements

- Snakemake
- Conda/Mamba
- Python 3.9+
- TensorFlow 2.x

All dependencies managed through conda environments in `Pipeline/Configs/`.
