import os, gzip, re, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import kipoiseq
from kipoiseq import Interval
import pyfaidx

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--max_variants', default='all', help="Integer N or 'all' (default: all)")
parser.add_argument('--vcf', default='/data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/01_Data/IGVFFI4378PZYI.vcf.gz', help='Input VCF (gzipped)')
parser.add_argument('--fasta', default='/data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/01_Data/hg38.fa', help='Reference FASTA')
parser.add_argument('--targets', default='/data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/01_Data/targets_human.txt', help='Targets TSV with a "description" column')
parser.add_argument('--out', default='variant_SAD_raw_PZYI.csv', help='Output CSV')

# Means
parser.add_argument('--agg_global', action='store_true', help='Add SAD_GLOBAL_MEAN across all tracks')
parser.add_argument('--agg_per_biosample', action='store_true', help='Add SAD_BIOSAMPLE_<biosample> mean across all tracks for that biosample')
parser.add_argument('--agg_per_assay', action='store_true', help='Add SAD_ASSAY_<assay> mean across all biosamples for that assay')

# Max
parser.add_argument('--agg_global_max', action='store_true', help='Add SAD_GLOBAL_MAX = max over all tracks')
parser.add_argument('--agg_per_biosample_max', action='store_true', help='Add SAD_BIOSAMPLE_MAX_<biosample> = max over tracks in that biosample')
parser.add_argument('--agg_per_assay_max', action='store_true', help='Add SAD_ASSAY_MAX_<assay> = max over biosamples for that assay')

# Min
parser.add_argument('--agg_global_min', action='store_true', help='Add SAD_GLOBAL_MIN = min over all tracks')
parser.add_argument('--agg_per_biosample_min', action='store_true', help='Add SAD_BIOSAMPLE_MIN_<biosample> = min over tracks in that biosample')
parser.add_argument('--agg_per_assay_min', action='store_true', help='Add SAD_ASSAY_MIN_<assay> = min over biosamples for that assay')

# Print flag help
parser.add_argument('--help_flags', action='store_true', help='Explain flags and exit')

args = parser.parse_args()

def print_flags_help():
    print(r"""
Aggregation/Control Flags
  --max_variants N | all
      Limit how many variants are scored. Integer N (e.g., 10) or 'all' (default).

MEAN aggregations
  --agg_global
      SAD_GLOBAL_MEAN = mean over all tracks (assay × biosample).
  --agg_per_biosample
      SAD_BIOSAMPLE_<biosample> = mean across all tracks belonging to that biosample
      (e.g. all DNase/ChIP/CAGE tracks for HepG2).
  --agg_per_assay
      SAD_ASSAY_<assay> = mean across all biosamples for that assay
      (e.g. all DNase tracks across tissues/cell lines).

MAX aggregations
  --agg_global_max
      SAD_GLOBAL_MAX = max over all tracks.
  --agg_per_biosample_max
      SAD_BIOSAMPLE_MAX_<biosample> = max over all tracks for that biosample.
  --agg_per_assay_max
      SAD_ASSAY_MAX_<assay> = max over all biosamples for that assay.

MIN aggregations
  --agg_global_min
      SAD_GLOBAL_MIN = min over all tracks.
  --agg_per_biosample_min
      SAD_BIOSAMPLE_MIN_<biosample> = min over all tracks for that biosample.
  --agg_per_assay_min
      SAD_ASSAY_MIN_<assay> = min over all biosamples for that assay.

Output column order
  chrom, pos, id, ref, alt  →  [global mean/max/min]  →  [per-biosample mean/max/min]  →  [per-assay mean/max/min]  →  per-track SADs.
""")
    raise SystemExit(0)

if args.help_flags:
    print_flags_help()

def _parse_max(x):
    if x is None: return None
    x = str(x).strip()
    if x.lower() == 'all' or x == '': return None
    return int(x)

MAX_VARIANTS   = _parse_max(args.max_variants)
MODEL_PATH     = 'https://tfhub.dev/deepmind/enformer/1'
FASTA_FILE     = args.fasta
VCF_FILE       = args.vcf
TARGETS_TXT    = args.targets
OUT_CSV        = args.out
SEQUENCE_LENGTH = 393216

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
    return kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32)

# ---------------- FASTA ----------------
class FastaStringExtractor:
    def __init__(self, fasta_path):
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.names = list(self.fasta.keys())
        self.has_chr = any(n.startswith("chr") for n in self.names)
        self.chrom_sizes = {k: len(v) for k, v in self.fasta.items()}
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
    def close(self): self.fasta.close()

# ---------------- VCF (SNVs only) ----------------
def variant_generator_only_snvs(vcf_path, gzipped=True):
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
        self._model = hub.load(tfhub_url).model
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
targets_df = pd.read_csv(TARGETS_TXT, sep='\t')
if 'description' not in targets_df.columns:
    raise ValueError(f"'description' column not found in {TARGETS_TXT}. Columns present: {list(targets_df.columns)}")
descriptions = targets_df['description'].astype(str).tolist()
T = len(descriptions)

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

# ---------------- Run scoring ----------------
model = EnformerScoreVariantsRaw(MODEL_PATH, organism='human')
fasta = FastaStringExtractor(FASTA_FILE)

rows, n_done = [], 0
for ex in variant_centered_inputs(VCF_FILE, SEQUENCE_LENGTH, fasta, gzipped=True):
    if MAX_VARIANTS is not None and n_done >= MAX_VARIANTS: break
    scores = model.predict_on_batch({k: v[tf.newaxis] for k, v in ex['inputs'].items()})[0]  # [T]
    if scores.shape[-1] != T:
        raise RuntimeError(f"Score dim {scores.shape[-1]} != targets dim {T}")

    meta = dict(ex['meta'])
    aggs = {}

    # ---- Global aggregations
    if args.agg_global:
        aggs['SAD_GLOBAL_MEAN'] = float(np.nanmean(scores))
    if args.agg_global_max:
        aggs['SAD_GLOBAL_MAX'] = float(np.nanmax(scores))
    if args.agg_global_min:
        aggs['SAD_GLOBAL_MIN'] = float(np.nanmin(scores))

    # ---- Per-biosample aggregations
    if args.agg_per_biosample or args.agg_per_biosample_max or args.agg_per_biosample_min:
        for b in sorted(biosample_to_idx.keys()):
            idxs = biosample_to_idx[b]
            if not idxs: 
                continue
            if args.agg_per_biosample:
                aggs[f'SAD_BIOSAMPLE_{b}'] = float(np.nanmean(scores[idxs]))
            if args.agg_per_biosample_max:
                aggs[f'SAD_BIOSAMPLE_MAX_{b}'] = float(np.nanmax(scores[idxs]))
            if args.agg_per_biosample_min:
                aggs[f'SAD_BIOSAMPLE_MIN_{b}'] = float(np.nanmin(scores[idxs]))

    # ---- Per-assay aggregations
    if args.agg_per_assay or args.agg_per_assay_max or args.agg_per_assay_min:
        for a in sorted(assay_to_idx.keys()):
            idxs = assay_to_idx[a]
            if not idxs:
                continue
            if args.agg_per_assay:
                aggs[f'SAD_ASSAY_{a}'] = float(np.nanmean(scores[idxs]))
            if args.agg_per_assay_max:
                aggs[f'SAD_ASSAY_MAX_{a}'] = float(np.nanmax(scores[idxs]))
            if args.agg_per_assay_min:
                aggs[f'SAD_ASSAY_MIN_{a}'] = float(np.nanmin(scores[idxs]))

    # Per-track raw SAD
    tracks = {descriptions[i]: float(scores[i]) for i in range(T)}

    # enforce column order: meta -> aggs -> tracks
    row = {**meta, **aggs, **tracks}
    rows.append(row)
    n_done += 1

fasta.close()

df = pd.DataFrame(rows)

# ----- Column ordering -----
meta_cols = ['chrom', 'pos', 'id', 'ref', 'alt']
agg_cols = []

# Global in fixed order
if args.agg_global:     agg_cols.append('SAD_GLOBAL_MEAN')
if args.agg_global_max: agg_cols.append('SAD_GLOBAL_MAX')
if args.agg_global_min: agg_cols.append('SAD_GLOBAL_MIN')

# Per-biosample (alphabetical biosample names) in order: mean, max, min
if args.agg_per_biosample:
    agg_cols.extend([f'SAD_BIOSAMPLE_{b}' for b in sorted(biosample_to_idx.keys()) if f'SAD_BIOSAMPLE_{b}' in df.columns])
if args.agg_per_biosample_max:
    agg_cols.extend([f'SAD_BIOSAMPLE_MAX_{b}' for b in sorted(biosample_to_idx.keys()) if f'SAD_BIOSAMPLE_MAX_{b}' in df.columns])
if args.agg_per_biosample_min:
    agg_cols.extend([f'SAD_BIOSAMPLE_MIN_{b}' for b in sorted(biosample_to_idx.keys()) if f'SAD_BIOSAMPLE_MIN_{b}' in df.columns])

# Per-assay (alphabetical assay names) in order: mean, max, min
if args.agg_per_assay:
    agg_cols.extend([f'SAD_ASSAY_{a}' for a in sorted(assay_to_idx.keys()) if f'SAD_ASSAY_{a}' in df.columns])
if args.agg_per_assay_max:
    agg_cols.extend([f'SAD_ASSAY_MAX_{a}' for a in sorted(assay_to_idx.keys()) if f'SAD_ASSAY_MAX_{a}' in df.columns])
if args.agg_per_assay_min:
    agg_cols.extend([f'SAD_ASSAY_MIN_{a}' for a in sorted(assay_to_idx.keys()) if f'SAD_ASSAY_MIN_{a}' in df.columns])

track_cols = [d for d in descriptions if d in df.columns]
ordered_cols = [c for c in meta_cols if c in df.columns] + agg_cols + track_cols
df = df[ordered_cols]

df.to_csv(OUT_CSV, index=False)
print(f'Wrote {OUT_CSV} with {df.shape[0]} variants (limit={args.max_variants}).')
print(f'Columns: meta({len(meta_cols)}) + aggs({len(agg_cols)}) + tracks({len(track_cols)}).')
