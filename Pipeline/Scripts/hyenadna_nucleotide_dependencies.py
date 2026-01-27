#!/usr/bin/env python3
"""HyenaDNA ND Scoring - computes nucleotide dependency scores for variants"""

import argparse, time, gzip
import numpy as np
import pandas as pd
import torch
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
parser.add_argument('--use_absolute', action='store_true',
                    help='Use absolute impact values before computing ND statistics (abs(impact) instead of impact)')
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
def compute_nd_score(model, tokenizer, ref_seq, alt_seq, variant_pos, context_window, use_absolute=False):
    """
    Calculate nucleotide dependency scores using leave-one-out masking approach.
    Measures how much each position contributes to the ref/alt embedding difference.
    
    Args:
        use_absolute: If True, use abs(impacts) for statistics (except ND_influence_score which is always RMS)
    
    Returns dictionary with signed or absolute impacts depending on use_absolute.
    """
    # Compute base difference between ref and alt
    emb_ref_full = model.mean_embed([ref_seq])
    emb_alt_full = model.mean_embed([alt_seq])
    base_diff = cosine_diff(emb_ref_full, emb_alt_full).item()
    
    # Define window around variant
    seq_len = len(ref_seq)
    window_start = max(0, variant_pos - context_window)
    window_end = min(seq_len, variant_pos + context_window + 1)
    
    # Compute impact for each position by masking with 'N'
    impacts = []
    for pos in range(window_start, window_end):
        if pos == variant_pos:
            continue
        
        # Mask position with 'N' in both sequences
        ref_masked = ref_seq[:pos] + 'N' + ref_seq[pos + 1:]
        alt_masked = alt_seq[:pos] + 'N' + alt_seq[pos + 1:]
        
        emb_ref_masked = model.mean_embed([ref_masked])
        emb_alt_masked = model.mean_embed([alt_masked])
        masked_diff = cosine_diff(emb_ref_masked, emb_alt_masked).item()
        
        # Impact = how much the difference changes when this position is masked
        # Keep sign: positive = masking reduces difference, negative = masking increases difference
        impacts.append(masked_diff - base_diff)
    
    if len(impacts) == 0:
        return {'ND_influence_score': 0.0, 'ND_mean': 0.0, 'ND_max': 0.0, 
                'ND_min': 0.0, 'ND_std': 0.0, 'ND_positions': 0}
    
    impacts = np.array(impacts)
    
    # RMS is always calculated from raw (signed) impacts
    nd_influence_score = float(np.sqrt(np.mean(impacts**2)))
    
    # Apply absolute value if requested for other statistics
    impacts_for_stats = np.abs(impacts) if use_absolute else impacts
    
    return {
        'ND_influence_score': nd_influence_score,  # RMS (always positive)
        'ND_mean': float(np.mean(impacts_for_stats)),
        'ND_max': float(np.max(impacts_for_stats)),
        'ND_min': float(np.min(impacts_for_stats)),
        'ND_std': float(np.std(impacts_for_stats)),
        'ND_positions': len(impacts)
    }

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def mean_embed(self, sequences):
        """Return mean-pooled embedding for each sequence."""
        enc = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc['input_ids'].to(self.device)
        att_mask = enc.get('attention_mask', None)
        if att_mask is not None:
            att_mask = att_mask.to(self.device)
        
        # Get hidden states from model (HyenaDNA doesn't use attention_mask in forward)
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # Last layer
        
        # Mean pooling with attention mask
        if att_mask is None:
            return hidden.mean(dim=1)
        
        mask = att_mask.unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

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
def main():
    start = time.time()
    print(f"Processing {args.vcf} -> {args.out}")
    
    fasta = FastaExtractor(args.fasta)
    model = HyenaDNAModel(args.model, DEVICE)
    variants = parse_vcf_variants(args.vcf, MAX_VARIANTS)
    
    if not variants:
        print("No variants found")
        return
    
    print(f"Scoring {len(variants)} variants...")
    results = []
    
    for i, var in enumerate(variants):
        if (i + 1) % args.print_every == 0:
            print(f"{i+1}/{len(variants)}")
        
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
                                     variant_offset, args.context_window, 
                                     use_absolute=args.use_absolute)
            
            results.append({
                'chrom': var['chrom'], 'pos': var['pos'], 'id': var['id'],
                'ref': var['ref'], 'alt': var['alt'], **scores
            })
        except Exception as e:
            print(f"Error at variant {i}: {e}")
            continue
    
    if results:
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"Saved {len(results)} results in {time.time()-start:.1f}s")
    else:
        print("No results")
    
    fasta.close()

if __name__ == '__main__':
    main()
