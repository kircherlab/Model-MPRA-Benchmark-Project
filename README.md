# Model–MPRA Benchmark Pipeline (v2)

Benchmarking DNA sequence-to-function models against MPRA variant-effect measurements across multiple cell types.

---

## Overview

The pipeline scores genetic variants from a VCF through multiple sequence-to-function models and produces tidy Parquet files of per-variant, per-track scores.

### Models

| Model | Type | Output |
|---|---|---|
| Enformer | Track-based (SAD) | Parquet |
| Basenji2 | Track-based (SAD) | Parquet |
| AlphaGenome | API-based (SAD) | Parquet |
| HyenaDNA | Embedding | Parquet |
| DNABERT-2 | Embedding | Parquet |

### Cell types / MPRA datasets

| Config | Cell type | VCF |
|---|---|---|
| `config_hek293t.yaml` | HEK293T | `IGVFFI4134MFLL.vcf` |
| `config_hepg2.yaml` | HepG2 | `IGVFFI4378PZYI.vcf` |
| `config_ngn2.yaml` | NGN2 neurons | `80k_normalized.vcf` |

---

## Requirements

- Snakemake ≥ 7
- Conda (manages per-model environments automatically)
- NVIDIA GPU (required for all scoring models)
- Reference genome: `Data/Genome/hg38.fa` + `.fai` (not tracked — obtain separately)
- Enformer target file: `Data/targets_human.txt`
- Basenji model weights: `Models/Basenji/basenji/manuscripts/cross2020/`

---

## Quick start

```bash
# Dry run first — see which jobs will execute
snakemake -n --configfile Configs/config_hek293t.yaml --rerun-triggers mtime

# Run on a GPU node (recommended via srun on HPC)
srun --cpus-per-task=4 --mem=32G --gres=gpu:1 --time=72:00:00 \
  snakemake -j 1 --configfile Configs/config_hek293t.yaml \
  --rerun-triggers mtime --use-conda
```

Replace `config_hek293t.yaml` with `config_hepg2.yaml` or `config_ngn2.yaml` for other cell types.

---

## Configuration

Each cell type has a dedicated config file in `Configs/`. The most important options:

```yaml
experiment_name: "hek293t"       # used for output directory naming
vcf: "Data/VCF/IGVFFI4134MFLL.vcf.gz"
fasta: "Data/Genome/hg38.fa"
filter_vcf: false                 # set true + max_variants to subsample
max_variants: 5000

models:
  enformer:
    enabled: true
  hyenadna:
    enabled: true
    model_name: "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    sequence_length: 1024
  alphagenome:
    enabled: false                # run separately, see below
    api_key: "YOUR_KEY_HERE"
  dnabert2:
    enabled: true
```

### Test run

`Configs/config_test.yaml` filters down to 10 variants and runs all non-API models — useful for verifying the environment before a full run:

```bash
# Dry run
snakemake -n --configfile Configs/config_test.yaml --rerun-triggers mtime

# Full test run on a GPU node
srun --cpus-per-task=4 --mem=32G --gres=gpu:1 --time=4:00:00 \
  snakemake -j 1 --configfile Configs/config_test.yaml \
  --rerun-triggers mtime --use-conda
```

---

## Output

Results are written to `results/<run_id>/` where `run_id` is a timestamp (`YYYYMMDD_HHMMSS`) generated at run start.

```
results/<run_id>/
  run_info.yaml                  # config snapshot for reproducibility
  enformer/enformer_scores.parquet
  basenji/basenji_scores.parquet
  hyenadna/hyenadna_scores.parquet
  alphagenome/alphagenome_scores.parquet
  dnabert2/dnabert2_scores.parquet
  logs/                          # per-model logs
```

All scoring scripts support **crash-safe checkpointing**: interrupted runs resume from the last completed chunk automatically.

---

## Running scoring scripts standalone

All scoring scripts have a full CLI and can be run outside Snakemake. This is useful for resuming crashed runs or running models on separate GPUs in parallel.

To get the Snakemake-managed conda environment paths:
```bash
ls .snakemake/conda/
```
Or create environments once with `--create-envs-only`, then activate by path.

**Enformer**
```bash
conda activate .snakemake/conda/<enformer-hash>
python Scripts/enformer_scoring.py \
  --vcf Data/VCF/IGVFFI4134MFLL.vcf.gz \
  --fasta Data/Genome/hg38.fa \
  --targets Data/targets_human.txt \
  --out results/<run_id>/enformer/enformer_scores.parquet \
  --print_every 50
```

**Basenji2**
```bash
conda activate .snakemake/conda/<basenji-hash>
python Scripts/basenji_scoring.py \
  --vcf Data/VCF/IGVFFI4134MFLL.vcf.gz \
  --fasta Data/Genome/hg38.fa \
  --params Models/Basenji/basenji/manuscripts/cross2020/params_human.json \
  --model Models/Basenji/basenji/manuscripts/cross2020/model_human.h5 \
  --basenji_script Models/Basenji/basenji/bin/basenji_sad.py \
  --targets Models/Basenji/basenji/manuscripts/cross2020/targets_human.txt \
  --out_dir results/<run_id>/basenji/basenji_sad_output \
  --out results/<run_id>/basenji/basenji_scores.parquet \
  --rc --shifts 0
```

**HyenaDNA**
```bash
conda activate .snakemake/conda/<hyenadna-hash>
python Scripts/hyenadna_scoring.py \
  --vcf Data/VCF/IGVFFI4134MFLL.vcf.gz \
  --fasta Data/Genome/hg38.fa \
  --out results/<run_id>/hyenadna/hyenadna_scores.parquet \
  --model "LongSafari/hyenadna-tiny-1k-seqlen-hf" \
  --sequence_length 1024 --context_window 200 --print_every 50
```

**DNABERT-2**
```bash
conda activate .snakemake/conda/<dnabert2-hash>
pip uninstall -y triton 2>&1 | head -5 || true   # avoid triton conflicts
python Scripts/dnabert2_scoring.py \
  --vcf Data/VCF/IGVFFI4134MFLL.vcf.gz \
  --fasta Data/Genome/hg38.fa \
  --out results/<run_id>/dnabert2/dnabert2_scores.parquet \
  --model "zhihan1996/DNABERT-2-117M" \
  --sequence_length 1024 --context_window 200 --print_every 10
```

**AlphaGenome** (API-based — requires Google DeepMind API key)

AlphaGenome queries the public API and is disabled by default to avoid quota issues when running other models in parallel. Set your key in the config:

```yaml
models:
  alphagenome:
    enabled: true
    api_key: "YOUR_KEY_HERE"
```

Or run directly:
```bash
conda activate .snakemake/conda/<alphagenome-hash>
python Scripts/alphagenome_scoring.py \
  --vcf Data/VCF/IGVFFI4134MFLL.vcf.gz \
  --out results/<run_id>/alphagenome/alphagenome_scores.parquet \
  --api_key YOUR_KEY_HERE
```

---

## Repository structure

```
Snakefile                    # Pipeline definition
Configs/
  config_hek293t.yaml        # Per-cell-type run configs
  config_hepg2.yaml
  config_ngn2.yaml
  config_test.yaml           # Quick 10-variant test
  *-env.yaml                 # Conda environments per model
Scripts/                     # Scoring scripts (called by Snakefile)
  enformer_scoring.py
  basenji_scoring.py
  alphagenome_scoring.py
  hyenadna_scoring.py
  dnabert2_scoring.py
  filter_vcf.py
Data/
  VCF/                       # Input variant files
  Genome/                    # hg38 reference (not tracked)
  MPRA/                      # Ground-truth MPRA effect tables
  targets_human.txt          # Enformer track list
Models/
  Basenji/                   # Basenji2 model weights & scripts
results/                     # Generated outputs (not tracked)
```
