#!/usr/bin/env python3
"""HyenaDNA ND Scoring - computes nucleotide dependency scores for variants"""

import argparse, os, shutil, time, gzip
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch._dynamo
import torch.nn.functional as F
import pyfaidx
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CLI ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--vcf', required=True)
parser.add_argument('--fasta', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--model', default="LongSafari/hyenadna-tiny-1k-seqlen-hf")
parser.add_argument('--sequence_length', type=int, default=1024)
parser.add_argument('--context_window', type=int, default=200)
parser.add_argument('--max_variants', default='all')
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--device', default='auto')
# NOTE: use_absolute removed from scoring. ND scores always include both signed
# and absolute statistics. Absolute values are always computed for completeness.
# Post-processing can choose which to use.
args = parser.parse_args()

MAX_VARIANTS = None if args.max_variants.lower() == 'all' else int(args.max_variants)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
DNA_BASES = set('ACGT')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def cosine_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute 1 - cosine similarity between embeddings."""
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return 1 - (a_n * b_n).sum(dim=-1)

# ============================================================================
# ND SCORE COMPUTATION (Attribution-based approach)
# ============================================================================
def compute_nd_score(model, tokenizer, ref_seq, alt_seq, variant_pos, context_window, batch_size=32):
    """
    Calculate nucleotide dependency scores using leave-one-out masking approach.
    Measures how much each position contributes to the ref/alt embedding difference.

    Uses batched forward passes for significant speedup over sequential per-position calls.

    Always returns both signed and absolute statistics for maximum flexibility.
    Post-processing (merge_data.py) decides which to use.

    Returns dictionary with signed impacts, absolute impacts, and RMS influence score.
    """
    # Define window around variant
    seq_len = len(ref_seq)
    window_start = max(0, variant_pos - context_window)
    window_end = min(seq_len, variant_pos + context_window + 1)
    positions = [p for p in range(window_start, window_end) if p != variant_pos]

    # Build all sequences at once: [ref, alt] + all masked variants
    ref_masked_seqs = [ref_seq[:p] + 'N' + ref_seq[p + 1:] for p in positions]
    alt_masked_seqs = [alt_seq[:p] + 'N' + alt_seq[p + 1:] for p in positions]
    all_seqs = [ref_seq, alt_seq] + ref_masked_seqs + alt_masked_seqs

    # Batch-embed all sequences
    all_embs = model.mean_embed(all_seqs)  # [2 + 2*N, hidden_dim]

    emb_ref_full = all_embs[0:1]
    emb_alt_full = all_embs[1:2]
    base_diff = cosine_diff(emb_ref_full, emb_alt_full).item()

    n = len(positions)
    emb_ref_masked = all_embs[2 : 2 + n]          # [N, hidden_dim]
    emb_alt_masked = all_embs[2 + n : 2 + 2 * n]  # [N, hidden_dim]

    # Compute all impacts in one vectorised step
    masked_diffs = cosine_diff(emb_ref_masked, emb_alt_masked)  # [N]
    impacts_tensor = masked_diffs - base_diff                    # [N]

    if n == 0:
        result = {
            'ND_influence_score': 0.0,
            'ND_mean': 0.0,
            'ND_max': 0.0,
            'ND_min': 0.0,
            'ND_std': 0.0,
            'ND_positions': 0,
            'ND_mean_ABS': 0.0,
            'ND_max_ABS': 0.0,
            'ND_min_ABS': 0.0,
            'ND_std_ABS': 0.0
        }
        return result

    impacts = impacts_tensor.cpu().numpy()

    # RMS is always calculated from raw (signed) impacts - always positive
    nd_influence_score = float(np.sqrt(np.mean(impacts**2)))

    # Always compute BOTH signed and absolute statistics
    impacts_abs = np.abs(impacts)
    result = {
        'ND_influence_score': nd_influence_score,
        'ND_mean': float(np.mean(impacts)),
        'ND_max': float(np.max(impacts)),
        'ND_min': float(np.min(impacts)),
        'ND_std': float(np.std(impacts)),
        'ND_positions': len(impacts),
        'ND_mean_ABS': float(np.mean(impacts_abs)),
        'ND_max_ABS': float(np.max(impacts_abs)),
        'ND_min_ABS': float(np.min(impacts_abs)),
        'ND_std_ABS': float(np.std(impacts_abs))
    }

    return result

# ============================================================================
# FASTA EXTRACTION
# ============================================================================
class FastaExtractor:
    def __init__(self, fasta_path):
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.chroms = {k: len(v) for k, v in self.fasta.items()}
        self.has_chr = any(name.startswith("chr") for name in self.chroms.keys())
    
    def normalize_chrom(self, chrom):
        if self.has_chr and not chrom.startswith("chr"):
            return "chr" + chrom if "chr" + chrom in self.chroms else chrom
        if not self.has_chr and chrom.startswith("chr"):
            return chrom[3:] if chrom[3:] in self.chroms else chrom
        return chrom
    
    def extract(self, chrom, start, end):
        chrom = self.normalize_chrom(chrom)
        if chrom not in self.chroms:
            return 'N' * (end - start)
        
        chrom_len = self.chroms[chrom]
        actual_start, actual_end = max(0, start), min(end, chrom_len)
        
        try:
            seq = str(self.fasta[chrom][actual_start:actual_end])
        except:
            return 'N' * (end - start)
        
        pad_left = 'N' * max(0, -start)
        pad_right = 'N' * max(0, end - chrom_len)
        return (pad_left + seq + pad_right).upper()
    
    def close(self):
        self.fasta.close()

# ============================================================================
# MODEL WRAPPER
# ============================================================================
class HyenaDNAModel:
    def __init__(self, model_name, device):
        self.device = device
        # Use bf16 on CUDA (native A100 format, ~2x throughput vs fp32)
        self.dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        print(f"Loading HyenaDNA model: {model_name} on {device} ({self.dtype})", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype
        ).to(device)
        self.model.eval()

        # torch.compile gives ~15-30% additional speedup on A100
        if device.type == 'cuda' and hasattr(torch, 'compile'):
            try:
                torch._dynamo.config.suppress_errors = True
                # Use 'default' mode to avoid unbounded CUDA Graph private-pool
                # memory accumulation across variants (see dnabert2 for details)
                self.model = torch.compile(self.model, mode='default')
                print("torch.compile enabled for HyenaDNA (mode=default, no CUDA Graphs)", flush=True)
            except Exception as e:
                print(f"torch.compile skipped: {e}", flush=True)
        print(f"Model loaded successfully on {device}", flush=True)

    @torch.no_grad()
    def mean_embed(self, sequences, batch_size=128):
        """Return mean-pooled embedding for a list of sequences.

        Pre-tokenizes all sequences on CPU, then processes GPU batches of
        batch_size=128 (tuned for A100 80GB with bf16).
        Returns tensor of shape [N, hidden_dim].
        """
        # Pre-tokenize all sequences on CPU
        all_encodings = [
            self.tokenizer(seq, return_tensors='pt', padding=False, truncation=True)
            for seq in sequences
        ]

        all_pooled = []
        for i in range(0, len(sequences), batch_size):
            batch_enc = all_encodings[i : i + batch_size]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [e['input_ids'].squeeze(0) for e in batch_enc],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id or 0,
            ).to(self.device)
            att_mask = (input_ids != (self.tokenizer.pad_token_id or 0)).long()

            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()  # cast to fp32 for numerics [B, L, H]

            mask = att_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_pooled.append(pooled.cpu())

        return torch.cat(all_pooled, dim=0)  # [N, hidden_dim]

# ============================================================================
# VCF PARSING
# ============================================================================
def parse_vcf_variants(vcf_path, max_variants):
    variants = []
    opener = gzip.open if vcf_path.endswith('.gz') else open
    
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            
            chrom, pos, vid, ref, alts, _, _, info_field = fields[:8]
            pos = int(pos)
            ref = ref.upper()
            
            log2fc = 0.0
            for item in info_field.split(';'):
                if item.startswith('LOG2FC='):
                    try:
                        log2fc = float(item.split('=')[1])
                    except:
                        pass
                    break
            
            for alt in alts.split(','):
                alt = alt.upper()
                if len(ref) == 1 and len(alt) == 1 and ref in DNA_BASES and alt in DNA_BASES:
                    variants.append({
                        'chrom': chrom, 'pos': pos, 'id': vid,
                        'ref': ref, 'alt': alt, 'log2fc': log2fc
                    })
    
    variants.sort(key=lambda x: abs(x['log2fc']), reverse=True)
    if max_variants and len(variants) > max_variants:
        variants = variants[:max_variants]
    
    return variants

# ============================================================================
# MAIN
# ============================================================================
WRITE_CHUNK = 500  # variants per checkpoint file


def main():
    start = time.time()
    print(f"Processing {args.vcf} -> {args.out}", flush=True)
    
    fasta = FastaExtractor(args.fasta)
    model = HyenaDNAModel(args.model, DEVICE)
    variants = parse_vcf_variants(args.vcf, MAX_VARIANTS)
    
    if not variants:
        print("No variants found", flush=True)
        return
    
    # Checkpoint directory
    checkpoint_dir = args.out + ".ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    n_chunks = (len(variants) + WRITE_CHUNK - 1) // WRITE_CHUNK
    print(f"Checkpoint dir: {checkpoint_dir}  ({n_chunks} chunks of {WRITE_CHUNK})", flush=True)

    # Detect already-completed chunks
    done_chunks = set()
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("chunk_") and fname.endswith(".parquet"):
            try:
                pq.read_metadata(os.path.join(checkpoint_dir, fname))
                idx = int(fname.split("_")[1].split(".")[0])
                done_chunks.add(idx)
            except Exception:
                os.remove(os.path.join(checkpoint_dir, fname))

    if done_chunks:
        print(f"Resuming: {len(done_chunks)}/{n_chunks} chunks already complete", flush=True)

    print(f"Scoring {len(variants)} variants...", flush=True)
    n_scored = 0
    n_skipped = 0
    
    try:
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * WRITE_CHUNK
            chunk_end = min(chunk_start + WRITE_CHUNK, len(variants))
            ckpt_path = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:05d}.parquet")

            if chunk_idx in done_chunks:
                n_skipped += (chunk_end - chunk_start)
                n_scored += (chunk_end - chunk_start)
                continue

            chunk_results = []
            for i in range(chunk_start, chunk_end):
                var = variants[i]
                n_scored += 1

                if n_scored % args.print_every == 0:
                    elapsed = time.time() - start
                    rate = n_scored / elapsed if elapsed > 0 else 0
                    eta = (len(variants) - n_scored) / rate if rate > 0 else float('inf')
                    print(f"  {n_scored}/{len(variants)} | {rate:.2f} var/s | ETA {eta/3600:.1f}h", flush=True)

                try:
                    center_idx = var['pos'] - 1
                    start_pos = center_idx - args.sequence_length // 2
                    end_pos = center_idx + args.sequence_length // 2

                    ref_seq = fasta.extract(var['chrom'], start_pos, end_pos)
                    variant_offset = center_idx - start_pos

                    if variant_offset < 0 or variant_offset >= len(ref_seq):
                        continue

                    alt_seq = ref_seq[:variant_offset] + var['alt'] + ref_seq[variant_offset + 1:]

                    scores = compute_nd_score(model, model.tokenizer, ref_seq, alt_seq,
                                             variant_offset, args.context_window)

                    chunk_results.append({
                        'variant_id': f"{var['chrom']}:{var['pos']}:{var['ref']}>{var['alt']}",
                        'chrom': var['chrom'], 'pos': var['pos'], 'id': var['id'],
                        'ref': var['ref'], 'alt': var['alt'], **scores
                    })
                except Exception as e:
                    print(f"Error at variant {i}: {e}", flush=True)
                    continue
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Save chunk checkpoint
            if chunk_results:
                pd.DataFrame(chunk_results).to_parquet(ckpt_path, index=False, compression='snappy')
    finally:
        fasta.close()

    # Merge all checkpoints into final output
    ckpt_files = sorted(
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("chunk_") and f.endswith(".parquet")
    )
    if ckpt_files:
        print(f"Merging {len(ckpt_files)} checkpoint files into {args.out} ...", flush=True)
        all_dfs = [pd.read_parquet(f) for f in ckpt_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_parquet(args.out, index=False, compression='snappy')
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print(f"Saved {len(df)} results in {time.time()-start:.1f}s", flush=True)
        if n_skipped:
            print(f"  ({n_skipped} variants restored from checkpoints)", flush=True)
    else:
        print("No results", flush=True)

if __name__ == '__main__':
    main()
