# Model-MPRA Benchmark Pipeline

A unified pipeline for benchmarking DNA sequence models (Enformer, Basenji, etc.) against MPRA experimental data.

## Overview

This pipeline:
1. Computes variant effect predictions using multiple DNA sequence models
2. Merges predictions with experimental MPRA measurements
3. Generates correlation plots and evaluation metrics

## Supported Models

- **Enformer**: Deep learning model predicting regulatory activity from DNA sequence
- **Basenji2**: Multi-task CNN predicting epigenomic profiles
- More models coming soon (AlphaGenome, Borzoi, etc.)

## Directory Structure

```
Pipeline/
├── Snakefile               # Main workflow definition
├── Configs/
│   ├── config.yaml         # Main configuration file
│   ├── enformer-env.yaml   # Enformer conda environment
│   ├── basenji-env.yaml    # Basenji conda environment
│   └── analysis-env.yaml   # Analysis/plotting environment
├── Scripts/
│   ├── enformer_SAD_v2.py  # Enformer variant scoring
│   ├── basenji_sad_wrapper.py  # Basenji variant scoring wrapper
│   ├── merge_data.py       # Merge model scores with MPRA data
│   └── generate_plots.py   # Generate evaluation plots
├── Data/
│   ├── VCF/                # Variant call format files
│   ├── Genome/             # Reference genome (hg38.fa)
│   ├── MPRA/               # MPRA experimental data
│   └── targets_human.txt   # Enformer targets
├── logs/                   # Log files from pipeline runs
└── results/                # Output results per model
    ├── enformer/
    │   ├── enformer_scores.csv
    │   ├── merged_data.tsv
    │   └── plots/
    └── basenji/
        ├── basenji_scores.csv
        ├── merged_data.tsv
        └── plots/
```

## Setup

### 1. Install Snakemake

```bash
conda install -c conda-forge -c bioconda snakemake
```

### 2. Configure the Pipeline

Edit `Configs/config.yaml` to:
- Enable/disable models
- Set input VCF and MPRA data paths
- Configure aggregation options
- Set model-specific parameters

Example configuration:
```yaml
models:
  enformer: true
  basenji: true

vcf: "Data/VCF/IGVFFI4134MFLL.vcf.gz"
mpra_data: "Data/MPRA/HEK293T_reporter_variants.tsv.gz"
```

## Usage

### Run the Complete Pipeline

```bash
cd Pipeline
snakemake --cores 8 --use-conda
```

This will:
- Run all enabled models on the configured VCF file
- Merge predictions with MPRA data
- Generate correlation plots

### Run Specific Models

**Enformer only:**
```bash
snakemake results/enformer/plots/correlation.png --cores 4 --use-conda
```

**Basenji only:**
```bash
snakemake results/basenji/plots/correlation.png --cores 4 --use-conda
```

### Dry Run (see what will be executed)

```bash
snakemake -n
```

### Generate DAG visualization

```bash
snakemake --dag | dot -Tpdf > pipeline_dag.pdf
```

## Configuration Options

### Model Selection

In `config.yaml`, toggle models on/off:
```yaml
models:
  enformer: true  # Run Enformer
  basenji: false  # Skip Basenji
```

### Aggregation Options

Control how SAD scores are aggregated:
```yaml
agg_global: true              # Global mean across all tracks
agg_global_max: false         # Global max
agg_per_assay: true          # Per-assay (CAGE, DNase, etc.)
agg_per_biosample: false     # Per-biosample (cell type)
```

### Model-Specific Parameters

**Enformer:**
```yaml
enformer:
  targets: "Data/targets_human.txt"
  max_variants: "1000"  # Process first 1000 variants (use "all" for all)
```

**Basenji:**
```yaml
basenji:
  params: "/path/to/params_human.json"
  model: "/path/to/model_human.h5"
  rc: true      # Average forward and reverse complement
  shifts: "0"   # Ensemble shifts
```

## Output Files

### Per Model Results

Each model produces:

1. **`{model}_scores.csv`**: Raw SAD scores for all variants
   - Columns: chrom, pos, ref, alt, variant_id, SAD scores per track
   
2. **`merged_data.tsv`**: Merged model predictions with MPRA data
   - Includes: variant info, MPRA measurements, SAD scores, aggregations
   
3. **`plots/correlation.png`**: Correlation plots
   - Per-assay scatter plots of model predictions vs MPRA log2FC
   - Histograms of SAD score distributions

### Logs

- `logs/{model}/{model}.log`: Detailed execution logs for each model

## Adding New Models

To add a new model to the pipeline:

1. Create a scoring script in `Scripts/`
2. Add conda environment in `Configs/`
3. Add model config section to `config.yaml`
4. Add Snakemake rules to `Snakefile`:
   - `run_{model}`: Score variants
   - `merge_data_{model}`: Merge with MPRA
   - `generate_plots_{model}`: Create plots
5. Update `models` section in config

## Troubleshooting

### Environment Issues

If conda environments fail to resolve:
```bash
# Manually create environments
conda env create -f Configs/enformer-env.yaml
conda env create -f Configs/basenji-env.yaml
conda env create -f Configs/analysis-env.yaml
```

### Memory Issues

For large VCF files, increase memory allocation or process variants in batches.

### GPU Not Detected

Ensure CUDA is properly installed and visible:
```bash
nvidia-smi  # Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Citation

If you use this pipeline, please cite:
- Enformer: Avsec et al., Nature Methods 2021
- Basenji2: Kelley et al., Genome Research 2018
- MPRA data: [Your MPRA paper]

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
