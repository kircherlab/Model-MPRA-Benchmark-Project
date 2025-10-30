# Pipeline Reorganization Summary

## What Was Done

Successfully reorganized the Model-MPRA Benchmark project from model-specific folders into a unified pipeline structure that supports multiple DNA sequence models.

## Changes Made

### 1. **Created Main Pipeline Directory**
   - Location: `/data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/Pipeline/`
   - Unified structure for all models (Enformer, Basenji, and future models)

### 2. **Moved Files from Enformer Folder**
   **Data Files:**
   - `Data/Genome/` ← Enformer/Pipeline/Data/Genome/*
   - `Data/VCF/` ← Enformer/Pipeline/Data/VCF/*
   - `Data/MPRA/` ← Enformer/Pipeline/Data/MPRA/*
   - `Data/targets_human.txt` ← Enformer/Pipeline/Data/Targets/targets_human.txt

   **Scripts:**
   - `Scripts/enformer_SAD_v2.py` (copied from Enformer/Pipeline/Scripts/)
   - `Scripts/merge_data.py` (copied and updated to work with all models)
   - `Scripts/generate_plots.py` (copied from Enformer/Pipeline/Scripts/)
   - `Scripts/basenji_sad_wrapper.py` (from Basenji implementation)

   **Configs:**
   - `Configs/enformer-env.yaml`
   - `Configs/analysis-env.yaml`
   - `Configs/basenji-env.yaml`
   - `Configs/config.yaml` (unified configuration)

### 3. **Updated Scripts**

   **merge_data.py:**
   - Changed from `enformer_file` → `model_scores_file` (generic)
   - Function `read_enformer()` → `read_model_scores()` (works with any model)
   - Updated to handle both Enformer and Basenji output formats
   - Support for `variant_id` column from Basenji

   **Paths Updated:**
   - All scripts now use relative paths from Pipeline/ directory
   - No hardcoded absolute paths in main scripts

### 4. **Created Unified Snakefile**
   
   **Features:**
   - Dynamically runs only enabled models (configured in config.yaml)
   - Separate rule sets for each model:
     - `run_{model}` - Score variants
     - `merge_data_{model}` - Merge with MPRA data  
     - `generate_plots_{model}` - Generate visualizations
   - Organized outputs: `results/{model}/` per model
   - Organized logs: `logs/{model}/` per model

### 5. **Created Unified Config**

   **Structure:**
   ```yaml
   models:
     enformer: true
     basenji: true
   
   # Shared paths
   vcf: "Data/VCF/..."
   fasta: "Data/Genome/hg38.fa"
   mpra_data: "Data/MPRA/..."
   
   # Model-specific sections
   enformer:
     targets: ...
     max_variants: ...
   
   basenji:
     params: ...
     model: ...
     script: ...
   ```

## Directory Structure

```
Model_MPRA_Benchmark/
├── Pipeline/                    # ← NEW: Unified pipeline
│   ├── Snakefile               # Unified workflow
│   ├── README.md               # Documentation
│   ├── Configs/
│   │   ├── config.yaml         # Unified config
│   │   ├── enformer-env.yaml
│   │   ├── basenji-env.yaml
│   │   └── analysis-env.yaml
│   ├── Scripts/
│   │   ├── enformer_SAD_v2.py
│   │   ├── basenji_sad_wrapper.py
│   │   ├── merge_data.py       # Updated to work with all models
│   │   └── generate_plots.py
│   ├── Data/                   # ← Moved from Enformer/
│   │   ├── Genome/
│   │   ├── VCF/
│   │   ├── MPRA/
│   │   └── targets_human.txt
│   ├── logs/
│   └── results/
│       ├── enformer/
│       │   ├── enformer_scores.csv
│       │   ├── merged_data.tsv
│       │   └── plots/
│       └── basenji/
│           ├── basenji_scores.csv
│           ├── merged_data.tsv
│           └── plots/
├── Enformer/                   # ← Original (can be archived)
├── Basenji/                    # ← Temporary (can be removed)
└── Arjun/                      # Arjun's work (kept separate)
```

## How to Use

### Run Both Models
```bash
cd Pipeline
snakemake --cores 8 --use-conda
```

### Run Only Enformer
Edit `Configs/config.yaml`:
```yaml
models:
  enformer: true
  basenji: false
```

Then:
```bash
snakemake --cores 4 --use-conda
```

### Run Only Basenji
Edit `Configs/config.yaml`:
```yaml
models:
  enformer: false
  basenji: true
```

## Benefits of New Structure

1. **Single Source of Truth**: All data files in one place
2. **Easy Model Addition**: Add new models by adding rules to Snakefile
3. **Consistent Outputs**: All models output to same structure
4. **No Code Duplication**: Shared scripts (merge_data.py, generate_plots.py)
5. **Flexible**: Enable/disable models via config
6. **Organized**: Results separated by model but comparable

## Next Steps

1. **Test the pipeline**:
   ```bash
   cd Pipeline
   snakemake -n  # Dry run to check
   ```

2. **Run on test data** (currently configured for 1000 variants)

3. **Add more models**: AlphaGenome, Borzoi, etc.

4. **Archive old folders** once confirmed working:
   - Can keep `Enformer/Script/` for notebooks
   - Remove `Basenji/Pipeline/` (no longer needed)
   - Keep `Arjun/` for reference

## Files That Can Be Cleaned Up Later

- `Enformer/Pipeline/` - mostly empty now, data moved
- `Basenji/Pipeline/` - was temporary, scripts now in main Pipeline
- Old result files in `Enformer/Pipeline/results/` if any

## Important Notes

- **Basenji paths**: Still points to Arjun's directory for model files (in config.yaml)
- **VCF files**: Available from both Arjun's directory and main Pipeline/Data/VCF/
- **No symlinks**: All files were moved (not linked) as requested
