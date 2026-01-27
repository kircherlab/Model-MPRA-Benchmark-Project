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
parser.add_argument('--max_variants', help="Integer N or 'all' (default: all)", default='all')
parser.add_argument('--print_every', type=int, default=50, help='Progress print frequency (variants)')
parser.add_argument('--aggregation', choices=['mean', 'sum'], default='mean', 
                    help='Method to collapse spatial dimension: mean or sum (default: mean)')
parser.add_argument('--use_absolute', action='store_true', 
                    help='Use absolute SAD values before aggregation: abs(alt - ref) instead of (alt - ref)')

# Print flag help
parser.add_argument('--help_flags', action='store_true', help='Explain flags and exit')

args = parser.parse_args()

# ---------------- Helper Functions ----------------
def _log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def _parse_max(x):
    if x is None: return None
    x = str(x).strip()
    if x.lower() == 'all' or x == '': return None
    return int(x)

def sanitize_seq(s: str) -> str:
    return DNA_UPPER.sub('N', s.upper())

def clean_biosample_name(s: str) -> str:
    return MULTISPACE.sub(' ', CLEAN_SEX.sub('', s)).strip()

def one_hot_encode(seq: str) -> np.ndarray:
    return kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32) # type: ignore

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

# ---------------- Constants & Configuration ----------------
VALID_A = set("ACGT")
DNA_UPPER = re.compile(r'[^ACGTN]')
CLEAN_SEX = re.compile(r'\b(male|female)\b', flags=re.IGNORECASE)
MULTISPACE = re.compile(r'\s+')

start_time = time.time()
MAX_VARIANTS = _parse_max(args.max_variants)
MODEL_PATH = 'https://tfhub.dev/deepmind/enformer/1'
FASTA_FILE = args.fasta
VCF_FILE = args.vcf
TARGETS_TXT = args.targets
OUT_CSV = args.out
SEQUENCE_LENGTH = 393216

# ---------------- Startup & Validation ----------------
gpu_devices = tf.config.list_physical_devices('GPU')
_log('==== Enformer SAD runner ====')
_log(f'VCF: {VCF_FILE}')
_log(f'Output: {OUT_CSV}')
_log(f'Max variants: {MAX_VARIANTS or "all"} | Aggregation: {args.aggregation}')
_log(f'GPUs available: {len(gpu_devices)}')

# Validate all inputs before loading heavy models
_log('Validating inputs...')

if not os.path.exists(VCF_FILE):
    raise FileNotFoundError(f"VCF file not found: {VCF_FILE}")

if not os.path.exists(FASTA_FILE):
    raise FileNotFoundError(f"FASTA file not found: {FASTA_FILE}")

if not os.path.exists(TARGETS_TXT):
    raise FileNotFoundError(f"Targets file not found: {TARGETS_TXT}")

# Check targets file has required column
targets_df = pd.read_csv(TARGETS_TXT, sep='\t')
if 'description' not in targets_df.columns:
    raise ValueError(f"'description' column not found in {TARGETS_TXT}. Available: {list(targets_df.columns)}")

descriptions = targets_df['description'].astype(str).tolist()
T = len(descriptions)
if T == 0:
    raise ValueError(f"No target tracks found in {TARGETS_TXT}")

# Check output directory is writable
out_dir = os.path.dirname(OUT_CSV)
if out_dir and not os.path.exists(out_dir):
    raise FileNotFoundError(f"Output directory does not exist: {out_dir}")

_log(f'✓ All inputs valid | {T} target tracks')

# ---------------- Class Definitions ----------------
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

    def close(self): 
        self.fasta.close()

class Enformer:
    def __init__(self, tfhub_url):
        _log('Loading Enformer model...')
        self._model = hub.load(tfhub_url).model
        _log('Model loaded')

    def predict_on_batch(self, x):
        out = self._model.predict_on_batch(x)
        return {k: v.numpy() for k, v in out.items()}

class EnformerScoreVariantsRaw:
    def __init__(self, tfhub_url, organism='human', aggregation='mean', use_absolute=False):
        self._m = Enformer(tfhub_url)
        self._org = organism
        self._aggregation = aggregation
        self._use_absolute = use_absolute
    
    def predict_on_batch(self, inputs):
        ref = self._m.predict_on_batch(inputs['ref'])[self._org]  # [B, L, T]
        alt = self._m.predict_on_batch(inputs['alt'])[self._org]  # [B, L, T]
        
        # Calculate SAD (Sequence Activity Difference)
        sad = alt - ref  # [B, L, T]
        
        # Apply absolute value if requested
        if self._use_absolute:
            sad = np.abs(sad)
        
        # Aggregate over spatial dimension (L)
        if self._aggregation == 'mean':
            return sad.mean(axis=1)  # [B, T]
        elif self._aggregation == 'sum':
            return sad.sum(axis=1)   # [B, T]
        else:
            raise ValueError(f"Unknown aggregation method: {self._aggregation}")

# ---------------- VCF Processing Functions ----------------
def variant_generator_only_snvs(vcf_path, gzipped=True, max_variants=None):
    _log(f'Reading VCF (SNVs only): {vcf_path}')
    
    # Collect all variants
    variants = []
    open_func = gzip.open if gzipped else open
    mode = 'rt' if gzipped else 'r'
    
    with open_func(vcf_path, mode) as f:
        for line in f:
            if line.startswith('#'): 
                continue
            
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 8: 
                continue
                
            chrom, pos, vid, ref, alts = fields[:5]
            info_field = fields[7]
            ref = ref.strip().upper()
            
            # Extract LOG2FC from INFO field
            log2fc = 0.0
            for item in info_field.split(';'):
                if item.startswith('LOG2FC='):
                    try:
                        log2fc = float(item.split('=')[1])
                    except (ValueError, IndexError):
                        pass
                    break
            
            # Process each alternate allele
            for alt in alts.strip().split(','):
                alt = alt.upper()
                if len(ref) == 1 and len(alt) == 1 and ref in VALID_A and alt in VALID_A:
                    variants.append((abs(log2fc), chrom, pos, vid, ref, alt))
    
    variants.sort(key=lambda x: x[0], reverse=True)
    
    if max_variants and len(variants) > max_variants:
        variants = variants[:max_variants]
        _log(f'Selected top {max_variants} variants by absolute LOG2FC')
    
    _log(f'Processing {len(variants)} variants (sorted by |LOG2FC|)')
    
    for _, chrom, pos, vid, ref, alt in variants:
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos, ref=ref, alt=alt, id=vid)

def variant_centered_inputs(vcf_path, seq_len, fasta_extractor, gzipped=True, max_variants=None):
    """
    Extrahiert für jede Variante zwei Sequenzen (Referenz und Alt) der Länge seq_len,
    zentriert um die Varianten-Position. One-hot encodiert für das Modell.
    """
    vseq = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
    
    for var in variant_generator_only_snvs(vcf_path, gzipped=gzipped, max_variants=max_variants):
        # Erstelle Interval um Variante herum (z.B. 393kb für Enformer)
        chrom = fasta_extractor.norm_chrom(var.chrom)
        interval = Interval(chrom, var.pos, var.pos).resize(seq_len)
        center_offset = interval.center() - interval.start
        
        # Extrahiere Referenz- und Alt-Sequenz
        ref_seq = sanitize_seq(vseq.extract(interval, [], anchor=center_offset))
        alt_seq = sanitize_seq(vseq.extract(interval, [var], anchor=center_offset))
        
        # Encodiere für Modell (one-hot) und packe Metadaten dazu
        result = {
            'inputs': {
                'ref': one_hot_encode(ref_seq),
                'alt': one_hot_encode(alt_seq)
            },
            'meta': {
                'chrom': chrom,
                'pos': var.pos,
                'id': var.id,
                'ref': var.ref,
                'alt': var.alt
            }
        }
        yield result

# ---------------- Main Execution ----------------
def run_scoring():
    """Führt das komplette Scoring durch: Model laden, Varianten scoren, Ergebnisse speichern."""
    
    # Initialize model and FASTA
    model = EnformerScoreVariantsRaw(MODEL_PATH, organism='human', 
                                     aggregation=args.aggregation,
                                     use_absolute=args.use_absolute)
    fasta = FastaStringExtractor(FASTA_FILE)
    
    # Score all variants
    rows = []
    n_done = 0
    last_tick = time.time()
    _log('Scoring variants...')
    
    for ex in variant_centered_inputs(VCF_FILE, SEQUENCE_LENGTH, fasta, gzipped=True, max_variants=MAX_VARIANTS):
        try:
            scores = model.predict_on_batch({k: v[tf.newaxis] for k, v in ex['inputs'].items()})[0]
        except Exception as e:
            _log(f'Predict error at variant {ex["meta"]}: {e}')
            continue
        
        if scores.shape[-1] != T:
            raise RuntimeError(f"Score dim {scores.shape[-1]} != targets dim {T}")
        
        # Build result row: variant metadata + track scores
        row = ex['meta'].copy()
        for i in range(T):
            row[descriptions[i]] = round(float(scores[i]), 5)
        
        rows.append(row)
        n_done += 1
        
        if n_done % args.print_every == 0:
            dt = time.time() - last_tick
            last_tick = time.time()
            _log(f'Processed {n_done} variants | Δt ~ {dt:.1f}s')
    
    fasta.close()
    
    # Save results
    df = pd.DataFrame(rows)
    meta_cols = ['chrom', 'pos', 'id', 'ref', 'alt']
    track_cols = [d for d in descriptions if d in df.columns]
    df = df[meta_cols + track_cols]
    
    df.to_csv(OUT_CSV, index=False)
    _log(f'Saved {len(df)} variants to {OUT_CSV}')
    _log(f'Total time: {(time.time()-start_time):.1f}s')

if __name__ == '__main__':
    run_scoring()
