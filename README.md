# Model–MPRA Benchmark Pipeline (v2)

Benchmarking DNA sequence-to-function models against MPRA variant-effect measurements across multiple cell types.

---

## Overview

The pipeline has two stages:

1. **Scoring** (`Snakefile`) — runs each model on a VCF and produces a tidy Parquet of per-variant, per-track scores.
2. **Analysis** (`Scripts/unified_celltype_analysis.py`) — loads the Parquets and computes correlation metrics against MPRA ground truth, producing figures and CSVs.

### Models supported

| Model | Type | Output |
|---|---|---|
| Enformer | Track-based (SAD) | Parquet |
| Basenji2 | Track-based (SAD) | Parquet |
| AlphaGenome | Track-based (SAD) | Parquet |
| HyenaDNA | Embedding | Parquet |
| DNABERT-2 | Embedding | Parquet |

### Cell types / MPRA datasets

| ID | Cell type | VCF |
|---|---|---|
| `hek293t` | HEK293T | `IGVFFI4134MFLL.vcf` |
| `hepg2` | HepG2 | `IGVFFI4378PZYI.vcf` |
| `ngn2` | NGN2 neurons | `80k_normalized.vcf` |

---

## Requirements

- Snakemake ≥ 7
- Conda (for per-model environments)
- NVIDIA GPU (required for all scoring rules)
- Reference genome: `Data/Genome/hg38.fa` + `.fai`
- Enformer target file: `Data/targets_human.txt`
- Basenji model weights: `Models/Basenji/basenji/manuscripts/cross2020/`

---

## Stage 1 — Scoring

Each cell type has its own config. Run one cell type at a time.

```bash
snakemake --configfile Configs/config_hek293t.yaml \
  --use-conda --cores 1 \
  --resources nvidia_gpu=1
```

Replace `config_hek293t.yaml` with `config_hepg2.yaml` or `config_ngn2.yaml` for other cell types.

Available configs:

| Config | Cell type |
|---|---|
| `Configs/config_hek293t.yaml` | HEK293T |
| `Configs/config_hepg2.yaml` | HepG2 |
| `Configs/config_ngn2.yaml` | NGN2 |

**Outputs** go into `results/<experiment_name>/<model>/<model>_scores.parquet`.

### AlphaGenome note

AlphaGenome uses the Google DeepMind public API and requires an API key set in the config:
```yaml
models:
  alphagenome:
    api_key: "YOUR_KEY_HERE"
```
It is run separately from other models to avoid quota contention (`enabled: false` in default configs).

### Optional: filter VCF before scoring

```yaml
filter_vcf: true
max_variants: 5000
```

---

## Stage 2 — Analysis

After scoring, run the unified analysis script.  
Parquets are expected at `{parquet-dir}/{cell_type}/{model}/{model}_scores.parquet`.

```bash
python3 -u Scripts/unified_celltype_analysis.py \
  --cell-types hepg2 hek293t ngn2 \
  --analyses baseline filtered detailed \
  --batch-size 2000000 \
  --workers 1 \
  --save-score-caches \
  --parquet-dir /path/to/parquets
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--cell-types` | One or more of `hepg2`, `hek293t`, `ngn2` |
| `--analyses` | `baseline` (all tracks), `filtered` (cell-type tracks only), `detailed` (per-assay/biosample) |
| `--parquet-dir` | Root directory containing `{ct}/{model}/` subfolders |
| `--batch-size` | Rows per Parquet batch (default 5 000 000; use 2 000 000 on limited RAM) |
| `--workers` | Parallel cell types — keep at `1` to avoid OOM |
| `--save-score-caches` | Write `.npz` caches to `/tmp/mpra_analysis/` for downstream scripts |
| `--sig-only` | Restrict all metrics to MPRA-significant variants only |

**Outputs** per cell type go into `results/{ct}/analysis/`:
- `*_metrics_unified.csv` — baseline Spearman ρ per model × score type
- `*_filtered_metrics_unified.csv` — CT-filtered Spearman ρ
- `*_alltrack_assay_barchart_*.png` — per-assay performance across all tracks
- `*_assay_global_vs_filtered_*.png` — global vs CT-filtered comparison per assay
- Scatter grids, biosample heatmaps, bin-correlation plots

---

## Repository structure

```
Snakefile                    # Scoring pipeline
Configs/
  config_hek293t.yaml        # Per-cell-type scoring configs
  config_hepg2.yaml
  config_ngn2.yaml
  *-env.yaml                 # Conda environments per model
Data/
  VCF/                       # Input variant files
  Genome/                    # hg38 reference (not tracked)
  MPRA/                      # Ground-truth MPRA tables
  targets_human.txt          # Enformer track list
Models/
  Basenji/                   # Basenji2 model weights & scripts
Scripts/
  unified_celltype_analysis.py   # Main analysis script
  enformer_scoring.py
  basenji_scoring.py
  alphagenome_scoring.py
  hyenadna_scoring.py
  dnabert2_scoring.py
  filter_vcf.py
  gen_alltrack_figures.py    # Lightweight standalone figure regenerator
results/                     # Generated outputs (not tracked)
```
