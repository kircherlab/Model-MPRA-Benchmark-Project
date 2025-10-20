import os, gzip, re, argparse, time, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import kipoiseq
from kipoiseq import Interval
import pyfaidx


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(add_help=True)

# --- MODIFIED: Removed hardcoded defaults and made paths required ---
parser.add_argument('--vcf', help='Input VCF (gzipped)', required=True)
parser.add_argument('--fasta', help='Reference FASTA', required=True)
parser.add_argument('--targets', help='Targets TSV with a "description" column', required=True)
parser.add_argument('--out', help='Output CSV path', required=True)

# --- MODIFIED: Changed default to 'all' which is more sensible for a pipeline ---
parser.add_argument('--max_variants', help="Integer N or 'all' (default: all)", required=True)
parser.add_argument('--print_every', type=int, default=50, help='Progress print frequency (variants)')



# Print flag help
parser.add_argument('--help_flags', action='store_true', help='Explain flags and exit')

args = parser.parse_args()

def _log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def print_flags_help():
    print(r"""
Aggregation/Control Flags
  --max_variants N | all
      Limit how many variants are scored. Integer N (e.g., 10) or 'all' (default).
# ... (rest of help text is unchanged) ...
""")
    raise SystemExit(0)

if args.help_flags:
    print_flags_help()

def _parse_max(x):
    if x is None: return None
    x = str(x).strip()
    if x.lower() == 'all' or x == '': return None
    return int(x)

start_time = time.time()
MAX_VARIANTS   = _parse_max(args.max_variants)
MODEL_PATH     = 'https://tfhub.dev/deepmind/enformer/1'
FASTA_FILE     = args.fasta
VCF_FILE       = args.vcf
TARGETS_TXT    = args.targets
OUT_CSV        = args.out
SEQUENCE_LENGTH = 393216

# Startup summary
gpu_devices = tf.config.list_physical_devices('GPU')
_log('==== Enformer SAD runner ====')
_log(f'VCF: {VCF_FILE}')
_log(f'FASTA: {FASTA_FILE}')
_log(f'Targets: {TARGETS_TXT}')
_log(f'Output: {OUT_CSV}')
_log(f'max_variants: {MAX_VARIANTS if MAX_VARIANTS is not None else "all"} | print_every: {args.print_every}')
_log(f'GPU visible: {len(gpu_devices)} -> {gpu_devices}')

# ---------------- utils ----------------
VALID_A   = set("ACGT")
DNA_UPPER = re.compile(r'[^ACGTN]')
CLEAN_SEX = re.compile(r'\b(male|female)\b', flags=re.IGNORECASE)
MULTISPACE= re.compile(r'\s+')

def sanitize_seq(s: str) -> str:
    return DNA_UPPER.sub('N', s.upper())

def clean_biosample_name(s: str) -> str:
    return MULTISPACE.sub(' ', CLEAN_SEX.sub('', s)).strip()

def one_hot_encode(seq: str) -> np.ndarray:
    return kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32) # type: ignore

# ---------------- FASTA ----------------
class FastaStringExtractor:
    def __init__(self, fasta_path):
        _log('Loading FASTA...')
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.names = list(self.fasta.keys())
        self.has_chr = any(n.startswith("chr") for n in self.names)
        self.chrom_sizes = {k: len(v) for k, v in self.fasta.items()}
        _log(f'FASTA loaded. has_chr={self.has_chr} | chroms={len(self.chrom_sizes)}')

    def norm_chrom(self, chrom_from_vcf: str) -> str:
        if self.has_chr and not chrom_from_vcf.startswith("chr"):
            cand = "chr" + chrom_from_vcf
            return cand if cand in self.chrom_sizes else chrom_from_vcf
        if not self.has_chr and chrom_from_vcf.startswith("chr"):
            cand = chrom_from_vcf[3:]
            return cand if cand in self.chrom_sizes else chrom_from_vcf
        return chrom_from_vcf

    def extract(self, interval: Interval) -> str:
        chrom = self.norm_chrom(interval.chrom)
        if chrom not in self.chrom_sizes:
            return "N" * (interval.end - interval.start)
        clen = self.chrom_sizes[chrom]
        trimmed = Interval(chrom, max(interval.start, 0), min(interval.end, clen))
        seq = str(self.fasta.get_seq(trimmed.chrom, trimmed.start + 1, trimmed.stop).seq)
        pad_up = 'N' * max(-interval.start, 0)
        pad_dn = 'N' * max(interval.end - clen, 0)
        return sanitize_seq(pad_up + seq + pad_dn)

    def close(self): 
        self.fasta.close()
        _log('FASTA closed.')

# ---------------- VCF (SNVs only) ----------------
def variant_generator_only_snvs(vcf_path, gzipped=True):
    _log(f'Reading VCF (SNVs only): {vcf_path}')
    _open = (lambda p: gzip.open(p, 'rt')) if gzipped else (lambda p: open(p))
    with _open(vcf_path) as f:
        for line in f:
            if line.startswith('#'): continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 5: continue
            chrom, pos, vid, ref, alts = fields[:5]
            ref = ref.strip().upper()
            for alt in alts.strip().split(','):
                alt = alt.upper()
                if len(ref)==1 and len(alt)==1 and (ref in VALID_A) and (alt in VALID_A):
                    yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos, ref=ref, alt=alt, id=vid)

def variant_centered_inputs(vcf_path, seq_len, fasta_extractor, gzipped=True):
    vseq = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
    for var in variant_generator_only_snvs(vcf_path, gzipped=gzipped):
        chrom = fasta_extractor.norm_chrom(var.chrom)
        iv = Interval(chrom, var.pos, var.pos).resize(seq_len)
        center = iv.center() - iv.start
        ref_seq = sanitize_seq(vseq.extract(iv, [], anchor=center))
        alt_seq = sanitize_seq(vseq.extract(iv, [var], anchor=center))
        yield {'inputs': {'ref': one_hot_encode(ref_seq), 'alt': one_hot_encode(alt_seq)},
               'meta':   {'chrom': chrom, 'pos': var.pos, 'id': var.id, 'ref': var.ref, 'alt': var.alt}}

# ---------------- Enformer ----------------
class Enformer:
    def __init__(self, tfhub_url):
        _log('Loading Enformer TF-Hub model (this may take a bit)...')
        self._model = hub.load(tfhub_url).model
        _log('Enformer model loaded.')

    def predict_on_batch(self, x):
        out = self._model.predict_on_batch(x)
        return {k: v.numpy() for k, v in out.items()}

class EnformerScoreVariantsRaw:
    def __init__(self, tfhub_url, organism='human'):
        self._m = Enformer(tfhub_url); self._org = organism
    def predict_on_batch(self, inputs):
        ref = self._m.predict_on_batch(inputs['ref'])[self._org]  # [B, L, T]
        alt = self._m.predict_on_batch(inputs['alt'])[self._org]  # [B, L, T]
        return alt.mean(axis=1) - ref.mean(axis=1)                # [B, T]

# ---------------- Targets & groupings ----------------
_log('Loading targets...')
targets_df = pd.read_csv(TARGETS_TXT, sep='\t')
if 'description' not in targets_df.columns:
    raise ValueError(f"'description' column not found in {TARGETS_TXT}. Columns present: {list(targets_df.columns)}")
descriptions = targets_df['description'].astype(str).tolist()
T = len(descriptions)
_log(f'Targets loaded: {T} tracks.')
if T > 0:
    preview = ', '.join(descriptions[:3]) + (' ...' if T > 3 else '')
    _log(f'First descriptions: {preview}')

assays, biosamples = [], []
for d in descriptions:
    if ':' in d:
        a, b = d.split(':', 1)
        assays.append(a.strip())
        biosamples.append(clean_biosample_name(b))
    else:
        assays.append(d.strip())
        biosamples.append('NA')
assays = np.array(assays, dtype=object)
biosamples = np.array(biosamples, dtype=object)

assay_to_idx = {}
biosample_to_idx = {}
for i, (a, b) in enumerate(zip(assays, biosamples)):
    assay_to_idx.setdefault(a, []).append(i)
    biosample_to_idx.setdefault(b, []).append(i)

_log(f'Unique assays: {len(assay_to_idx)} | Unique biosamples: {len(biosample_to_idx)}')

# ---------------- Run scoring ----------------
model = EnformerScoreVariantsRaw(MODEL_PATH, organism='human')
fasta = FastaStringExtractor(FASTA_FILE)

rows, n_done = [], 0
last_tick = time.time()
_log('Scoring variants...')
for ex in variant_centered_inputs(VCF_FILE, SEQUENCE_LENGTH, fasta, gzipped=True):
    if MAX_VARIANTS is not None and n_done >= MAX_VARIANTS: 
        break
    try:
        scores = model.predict_on_batch({k: v[tf.newaxis] for k, v in ex['inputs'].items()})[0]  # [T]
    except Exception as e:
        _log(f'Predict error at variant {ex["meta"]}: {e}')
        continue

    if scores.shape[-1] != T:
        raise RuntimeError(f"Score dim {scores.shape[-1]} != targets dim {T}")

    meta = dict(ex['meta'])

    # Round track scores to 5 decimal places to limit file size/precision
    tracks = {descriptions[i]: round(float(scores[i]), 5) for i in range(T)}
    row = {**meta, **tracks}
    rows.append(row)
    n_done += 1

    if n_done % max(1, args.print_every) == 0:
        dt = time.time() - last_tick
        last_tick = time.time()
        _log(f'Processed {n_done} variants. Last meta: {meta} | Δt ~ {dt:.1f}s')

fasta.close()

df = pd.DataFrame(rows)

# ----- Column ordering -----

# ----- Column ordering -----
meta_cols = ['chrom', 'pos', 'id', 'ref', 'alt']
track_cols = [d for d in descriptions if d in df.columns]
ordered_cols = [c for c in meta_cols if c in df.columns] + track_cols
df = df[ordered_cols]

df.to_csv(OUT_CSV, index=False)
_log(f'Wrote {OUT_CSV} with {df.shape[0]} variants (limit={args.max_variants}).')
_log(f'Columns: meta({len(meta_cols)}) + tracks({len(track_cols)}).')
_log(f'Total elapsed: {(time.time()-start_time):.1f}s')
