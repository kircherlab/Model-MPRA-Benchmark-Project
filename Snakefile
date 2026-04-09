# =============================================================================
# Pipeline v2 — Raw Scoring Only
# =============================================================================
# Generates variant effect scores for each model.
# Results go into results/<timestamp>/ with a run_info.yaml log.
#
# Track-based models (Enformer, Basenji, AlphaGenome) → tidy Parquet.
# Embedding models (HyenaDNA, DNABERT-2) → Parquet.
# =============================================================================

import datetime

configfile: "Configs/config.yaml"

# ── Configuration ────────────────────────────────────────────────────────────

EXPERIMENT   = config.get("experiment_name", "default")
VCF_RAW      = config["vcf"]
DO_FILTER    = config.get("filter_vcf", False)
MAX_VARIANTS = config.get("max_variants", "all")
VCF_FILTERED = f"Data/VCF/{EXPERIMENT}_filtered.vcf.gz"

# Timestamped run directory: results/YYYYMMDD_HHMMSS/
# Note: YAML 1.1 treats _ in numbers as a thousands separator, so
# --config run_id=20260324_151601 arrives as int 20260324151601.
# We convert back to the YYYYMMDD_HHMMSS string format.
RUN_ID = str(config.get("run_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
if RUN_ID.isdigit() and len(RUN_ID) == 14:
    RUN_ID = f"{RUN_ID[:8]}_{RUN_ID[8:]}"
RUN_DIR = f"results/{RUN_ID}"

def vcf_for_scoring():
    """Return the VCF path that scoring rules should consume."""
    return VCF_FILTERED if DO_FILTER else VCF_RAW

# Build list of enabled models
MODELS = []
for name, settings in config.get("models", {}).items():
    if isinstance(settings, dict) and settings.get("enabled", False):
        MODELS.append(name)

# ── Output file paths ────────────────────────────────────────────────────────

def scoring_outputs():
    """Return list of all expected output files."""
    outputs = [f"{RUN_DIR}/run_info.yaml"]
    for model in MODELS:
        outputs.append(f"{RUN_DIR}/{model}/{model}_scores.parquet")
    return outputs

rule all:
    input:
        scoring_outputs()


# =============================================================================
# RUN INFO — Log config & parameters for reproducibility
# =============================================================================

rule write_run_info:
    output:
        f"{RUN_DIR}/run_info.yaml"
    params:
        run_id     = RUN_ID,
        experiment = EXPERIMENT,
        vcf_raw    = VCF_RAW,
        do_filter  = DO_FILTER,
        max_vars   = MAX_VARIANTS,
        vcf_used   = vcf_for_scoring(),
        fasta      = config["fasta"],
        targets    = config["targets"],
        models     = ",".join(MODELS),
    shell:
        """
        mkdir -p $(dirname {output})
        python3 -c "
import json, socket, datetime
info = {{
    'run_id': '{params.run_id}',
    'experiment_name': '{params.experiment}',
    'timestamp': datetime.datetime.now().isoformat(),
    'hostname': socket.gethostname(),
    'vcf_raw': '{params.vcf_raw}',
    'filter_vcf': {params.do_filter},
    'max_variants': '{params.max_vars}',
    'vcf_used': '{params.vcf_used}',
    'fasta': '{params.fasta}',
    'targets': '{params.targets}',
    'models_enabled': '{params.models}'.split(','),
}}
with open('{output}', 'w') as f:
    json.dump(info, f, indent=2)
"
        """


# =============================================================================
# PREPROCESSING — Filter VCF  (only runs when filter_vcf: true in config)
# =============================================================================

rule filter_vcf:
    input:
        script = "Scripts/filter_vcf.py",
        vcf    = VCF_RAW
    output:
        VCF_FILTERED
    params:
        max_variants = MAX_VARIANTS
    log:
        f"{RUN_DIR}/logs/filter_vcf.log"
    shell:
        """
        mkdir -p Data/VCF $(dirname {log})
        python {input.script} {input.vcf} {output} {params.max_variants} {log}
        """


# =============================================================================
# ENFORMER — Track-based SAD scoring (tidy output)
# =============================================================================

rule enformer:
    input:
        script   = "Scripts/enformer_scoring.py",
        vcf      = vcf_for_scoring(),
        fasta    = config["fasta"],
        targets  = config["targets"],
        run_info = f"{RUN_DIR}/run_info.yaml"
    output:
        f"{RUN_DIR}/enformer/enformer_scores.parquet"
    log:
        f"{RUN_DIR}/logs/enformer.log"
    resources:
        nvidia_gpu = 1
    conda:
        "Configs/enformer-env.yaml"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
        python {input.script} \
            --vcf {input.vcf} \
            --fasta {input.fasta} \
            --targets {input.targets} \
            --out {output} \
            --print_every 50 \
            2>&1 | tee {log}
        """


# =============================================================================
# BASENJI — Track-based SAD scoring (tidy output)
# =============================================================================

def basenji_inputs(wildcards):
    inputs = {
        "script":         "Scripts/basenji_scoring.py",
        "vcf":            vcf_for_scoring(),
        "fasta":          config["fasta"],
        "params_json":    config["basenji"]["params"],
        "model_h5":       config["basenji"]["model"],
        "basenji_script": config["basenji"]["script"],
        "run_info":       f"{RUN_DIR}/run_info.yaml",
    }
    if "enformer" in MODELS:
        inputs["enformer_done"] = f"{RUN_DIR}/enformer/enformer_scores.parquet"
    return inputs

rule basenji:
    input:
        unpack(basenji_inputs)
    output:
        f"{RUN_DIR}/basenji/basenji_scores.parquet"
    params:
        targets = config["basenji"]["targets"],
        rc      = "--rc" if config["basenji"].get("rc", False) else "",
        shifts  = config["basenji"].get("shifts", "0"),
    log:
        f"{RUN_DIR}/logs/basenji.log"
    resources:
        nvidia_gpu = 1
    conda:
        "Configs/basenji-env.yaml"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
        python {input.script} \
            --vcf {input.vcf} \
            --fasta {input.fasta} \
            --params {input.params_json} \
            --model {input.model_h5} \
            --basenji_script {input.basenji_script} \
            --targets {params.targets} \
            --out_dir $(dirname {output})/basenji_sad_output \
            --out {output} \
            --shifts {params.shifts} \
            {params.rc} \
            2>&1 | tee {log}
        """


# =============================================================================
# HYENADNA — Nucleotide dependency scoring
# =============================================================================

rule hyenadna:
    input:
        script   = "Scripts/hyenadna_scoring.py",
        vcf      = vcf_for_scoring(),
        fasta    = config["fasta"],
        run_info = f"{RUN_DIR}/run_info.yaml"
    output:
        f"{RUN_DIR}/hyenadna/hyenadna_scores.parquet"
    params:
        model   = config["models"]["hyenadna"].get("model_name", "LongSafari/hyenadna-tiny-1k-seqlen-hf"),
        seq_len = config["models"]["hyenadna"].get("sequence_length", 1024),
        context = config["models"]["hyenadna"].get("context_window", 200),
    log:
        f"{RUN_DIR}/logs/hyenadna.log"
    resources:
        nvidia_gpu = 1
    conda:
        "Configs/hyenadna-env.yaml"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        PYTHONUNBUFFERED=1 python {input.script} \
            --vcf {input.vcf} \
            --fasta {input.fasta} \
            --out {output} \
            --model "{params.model}" \
            --sequence_length {params.seq_len} \
            --context_window {params.context} \
            --print_every 50 \
            2>&1 | tee {log}
        """


# =============================================================================
# ALPHAGENOME — API-based scoring (tidy output via tidy_scores())
# =============================================================================

def alphagenome_inputs(wildcards):
    inputs = {
        "script":   "Scripts/alphagenome_scoring.py",
        "vcf":      vcf_for_scoring(),
        "run_info": f"{RUN_DIR}/run_info.yaml",
    }
    for dep in ["enformer", "basenji", "hyenadna"]:
        if dep in MODELS:
            inputs[f"{dep}_done"] = f"{RUN_DIR}/{dep}/{dep}_scores.parquet"
    return inputs

rule alphagenome:
    input:
        unpack(alphagenome_inputs)
    output:
        f"{RUN_DIR}/alphagenome/alphagenome_scores.parquet"
    params:
        window_size = config["models"]["alphagenome"].get("window_size", 16384),
        max_workers = config["models"]["alphagenome"].get("max_workers", 4),
        scorers     = " ".join(config["models"]["alphagenome"].get("scorers",
                        ["ATAC", "DNASE", "CAGE", "CHIP_HISTONE", "RNA_SEQ"])),
        api_key_flag = (
            "--api_key " + config["models"]["alphagenome"]["api_key"]
            if config["models"]["alphagenome"].get("api_key", "")
            else ""
        ),
        ontology_terms = (
            " ".join(config["models"]["alphagenome"].get("ontology_terms", []))
            if config["models"]["alphagenome"].get("ontology_terms")
            else ""
        ),
    log:
        f"{RUN_DIR}/logs/alphagenome.log"
    resources:
        nvidia_gpu = 1
    conda:
        "Configs/alphagenome-env.yaml"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        ONTOLOGY_ARG=""
        if [ -n "{params.ontology_terms}" ]; then
            ONTOLOGY_ARG="--ontology_terms {params.ontology_terms}"
        fi
        python {input.script} \
            --vcf {input.vcf} \
            --out {output} \
            --window_size {params.window_size} \
            --max_workers {params.max_workers} \
            --scorers {params.scorers} \
            {params.api_key_flag} \
            $ONTOLOGY_ARG \
            --print_every 10 \
            2>&1 | tee {log}
        """


# =============================================================================
# DNABERT-2 — Nucleotide dependency scoring
# =============================================================================

def dnabert2_inputs(wildcards):
    inputs = {
        "script":   "Scripts/dnabert2_scoring.py",
        "vcf":      vcf_for_scoring(),
        "fasta":    config["fasta"],
        "run_info": f"{RUN_DIR}/run_info.yaml",
    }
    for dep in ["enformer", "basenji", "hyenadna"]:
        if dep in MODELS:
            inputs[f"{dep}_done"] = f"{RUN_DIR}/{dep}/{dep}_scores.parquet"
    return inputs

rule dnabert2:
    input:
        unpack(dnabert2_inputs)
    output:
        f"{RUN_DIR}/dnabert2/dnabert2_scores.parquet"
    params:
        model   = config["models"]["dnabert2"].get("model_name", "zhihan1996/DNABERT-2-117M"),
        seq_len = config["models"]["dnabert2"].get("sequence_length", 512),
        context = config["models"]["dnabert2"].get("context_window", 200),
    log:
        f"{RUN_DIR}/logs/dnabert2.log"
    resources:
        nvidia_gpu = 1
    conda:
        "Configs/dnabert2-env.yaml"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        pip uninstall -y triton 2>&1 | head -5 || true
        PYTHONUNBUFFERED=1 python {input.script} \
            --vcf {input.vcf} \
            --fasta {input.fasta} \
            --out {output} \
            --model "{params.model}" \
            --sequence_length {params.seq_len} \
            --context_window {params.context} \
            --print_every 10 \
            2>&1 | tee {log}
        """
