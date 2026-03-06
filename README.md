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

## Data Requirements

### Reference Genome

The pipeline requires the human reference genome (hg38) and its index file:

**Required files:**
- `Pipeline/Data/Genome/hg38.fa` - Reference genome FASTA
- `Pipeline/Data/Genome/hg38.fa.fai` - FASTA index (created automatically)

**Download and setup:**
```bash
# Navigate to the data directory
cd Pipeline/Data/Genome

# Download hg38 reference genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz

# Create FASTA index (required for fast random access)
samtools faidx hg38.fa
```

If you already have the hg38 fasta and index file (they need to be in the same directory), update the `fasta` path in `Configs/config.yaml`.

```bash
# For AlphaGenome: make sure that your API key can be accessed by the pipeline:
export ALPHAGENOME_API_KEY="..."
```

You can also set the key directly in `Pipeline/Configs/config.yaml` under:

```yaml
alphagenome:
  api_key: "..."
```

If `alphagenome.api_key` is empty, the pipeline falls back to `alphagenome.api_key_env`.

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

## Data Merging Process

The pipeline merges model predictions with MPRA experimental data using variant coordinates.

### Input 1: Model Scores CSV

Each model outputs predictions in a standardized format:

```
┌────────────────────────────────────────────────────────────────┐
│ chrom │ pos    │ id  │ ref │ alt │ CAGE:liver │ DNase:heart │..│
├────────────────────────────────────────────────────────────────┤
│ chr1  │ 100000 │ rs1 │ A   │ G   │ 0.12345    │ -0.02341    │..│
│ chr1  │ 200000 │ rs2 │ C   │ T   │ -0.00123   │ 0.03456     │..│
└────────────────────────────────────────────────────────────────┘
```

- **Metadata columns**: `chrom`, `pos`, `id`, `ref`, `alt`, `variant_id`
- **Score columns**: One per track/tissue (5,313 for Enformer, varies for Basenji)
- **Values**: SAD (Sequence Activity Difference) scores
- **Assays and Samples**: One Column per Biosample in ```Pipeline/Data/targets_human.txt```

### Input 2: VCF with MPRA Data

Experimental MPRA results stored in VCF INFO field. This was done by Arjun:

```
#CHROM  POS     ID   REF  ALT  QUAL  FILTER  INFO
chr1    100000  rs1  A    G    .     .       mpra_log2FC=1.23;mpra_pvalue=0.001
chr1    200000  rs2  C    T    .     .       mpra_log2FC=-0.45;mpra_pvalue=0.05
```

### Merging Process

Variants are matched using `chrom:pos:ref:alt` keys:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Model Predictions (enformer_scores.csv)                 │
├─────────────────────────────────────────────────────────────────┤
│ chr1:100000:A:G  →  CAGE:liver=0.123, DNase:heart=-0.023, ...   │
│ chr1:200000:C:T  →  CAGE:liver=-0.001, DNase:heart=0.034, ...   │
└─────────────────────────────────────────────────────────────────┘
                                +
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: MPRA Experimental Data (VCF INFO field)                 │
├─────────────────────────────────────────────────────────────────┤
│ chr1:100000:A:G  →  mpra_log2FC=1.23, mpra_pvalue=0.001         │
│ chr1:200000:C:T  →  mpra_log2FC=-0.45, mpra_pvalue=0.05         │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Join on Variant Key & Compute Aggregations              │
├─────────────────────────────────────────────────────────────────┤
│ chr1:100000:A:G  →  mpra_log2FC=1.23                            │
│                  →  SAD_GLOBAL_MEAN=0.050 (mean of all tracks)  │
│                  →  SAD_BIOSAMPLE_liver=0.123                   │
│                  →  SAD_ASSAY_CAGE=0.123, SAD_ASSAY_DNase=-0.02 │
│                  →  Individual tracks: CAGE:liver, DNase:heart..│
└─────────────────────────────────────────────────────────────────┘
```

### Score Aggregation

After merging, aggregated scores are computed from individual track predictions:

**SAD_GLOBAL_MEAN/MAX/MIN**
- Computed across **all tracks** (e.g., all 5,313 Enformer tracks)
- `SAD_GLOBAL_MEAN` = mean of all track SAD scores for a variant
- `SAD_GLOBAL_MAX` = maximum SAD score across all tracks
- `SAD_GLOBAL_MIN` = minimum SAD score across all tracks

**SAD_BIOSAMPLE_***
- Grouped by **biosample/tissue** (extracted from track descriptions)
- Example: `SAD_BIOSAMPLE_liver` = mean of all liver-related tracks
- Track descriptions parsed as `ASSAY:biosample` (e.g., `CAGE:liver male`)
- Gender terms removed for grouping

**SAD_ASSAY_***
- Grouped by **assay type** (CAGE, DNase, ATAC-seq, etc.)
- Example: `SAD_ASSAY_CAGE` = mean of all CAGE tracks
- Assay type extracted from first part of track name before `:`

**Calculation example for one variant**:
```
Tracks:  CAGE:liver    DNase:liver    CAGE:heart    DNase:heart
Values:  0.5           0.3            -0.2          0.1

SAD_GLOBAL_MEAN      = mean(0.5, 0.3, -0.2, 0.1) = 0.175
SAD_BIOSAMPLE_liver  = mean(0.5, 0.3) = 0.4
SAD_BIOSAMPLE_heart  = mean(-0.2, 0.1) = -0.05
SAD_ASSAY_CAGE       = mean(0.5, -0.2) = 0.15
SAD_ASSAY_DNase      = mean(0.3, 0.1) = 0.2
```

Enable/disable aggregations in `Configs/config.yaml` using the `agg_*` options.

### Output: Merged Data TSV

Combined dataset with predictions and experimental results:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ chrom│pos   │ref│alt│mpra_log2FC│mpra_pval│SAD_GLOBAL_MEAN│CAGE:liver│...│
├──────────────────────────────────────────────────────────────────────────┤
│ chr1 │100000│ A │ G │ 1.23      │ 0.001   │ 0.12345       │ 0.12345  │...│
│ chr1 │200000│ C │ T │ -0.45     │ 0.05    │ -0.00123      │ -0.00123 │...│
└──────────────────────────────────────────────────────────────────────────┘
```

**Column structure**:
1. Variant metadata (`chrom`, `pos`, `ref`, `alt`)
2. MPRA experimental data (from VCF INFO field)
3. Aggregated scores (global/biosample/assay means)
4. Individual track scores (all model predictions)

This format is consistent across Enformer, Basenji, (and AlphaGenome) outputs.

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
- samtools (for FASTA indexing)

All dependencies managed through conda environments in `Pipeline/Configs/`.
