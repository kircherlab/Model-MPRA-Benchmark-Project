# Model-MPRA Benchmark Project

Unified pipeline for benchmarking DNA sequence models (Enformer, Basenji) against MPRA experimental data.

## Directory Structure

```
Model_MPRA_Benchmark/
├── Pipeline/           # Main unified pipeline (USE THIS)
│   ├── Snakefile      # Workflow orchestration
│   ├── Configs/       # Configuration files
│   ├── Scripts/       # Python scripts for analysis
│   ├── Data/          # Input data (VCF, genome, MPRA)
│   ├── Models/        # Model files (Basenji)
│   ├── results/       # Pipeline outputs (gitignored)
│   └── logs/          # Execution logs (gitignored)
│
└── Archive/           # Old project folders (for review/deletion)
    ├── Arjun/        # Original Basenji setup
    ├── Enformer/     # Old Enformer-specific pipeline
    ├── Basenji/      # Old Basenji-specific pipeline
    └── Presentations/
```

## Quick Start

```bash
cd Pipeline

# Test with 10 variants
snakemake --cores 8 --use-conda

# Run full analysis (set max_variants: "all" in Configs/config.yaml)
snakemake --cores 8 --use-conda
```

## Configuration

Edit `Pipeline/Configs/config.yaml` to:
- Enable/disable models: `models: {enformer: true, basenji: true}`
- Set max variants: `max_variants: "10"` or `"all"`
- Choose VCF and MPRA data files

## Models Supported

- **Enformer**: TensorFlow Hub model (automatically downloaded)
- **Basenji**: Local model files in `Pipeline/Models/Basenji/`

## Output

Results are saved in `Pipeline/results/{model}/`:
- `{model}_scores.csv` - Raw prediction scores
- `merged_data.tsv` - Predictions + MPRA data
- `plots/correlation.png` - Correlation plots

## Archive Folder

The `Archive/` folder contains old project structures that can be reviewed and deleted if not needed.
