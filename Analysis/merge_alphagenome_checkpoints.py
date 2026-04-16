"""Stream-merge AlphaGenome checkpoint parquets one chunk at a time.

Uses PyArrow ParquetWriter so only one chunk is in memory at a time.
Safe to run with low RAM.

Usage:
    python Scripts/merge_alphagenome_checkpoints.py \
        results/20260324_151527/alphagenome/alphagenome_scores.parquet
"""
import sys
import glob
import os
import pyarrow.parquet as pq


def merge(out_path):
    ckpt_dir = out_path + ".ckpt"
    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: checkpoint dir not found: {ckpt_dir}")
        sys.exit(1)

    if os.path.exists(out_path):
        print(f"SKIP: output already exists: {out_path}")
        return

    chunks = sorted(glob.glob(os.path.join(ckpt_dir, "chunk_*.parquet")))
    if not chunks:
        print(f"ERROR: no chunks found in {ckpt_dir}")
        sys.exit(1)

    print(f"Merging {len(chunks)} chunks -> {out_path}")

    writer = None
    for i, chunk_path in enumerate(chunks):
        table = pq.read_table(chunk_path)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
            print(f"  {i+1}/{len(chunks)} chunks written")

    if writer:
        writer.close()

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Done: {out_path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_alphagenome_checkpoints.py <out.parquet> [out2.parquet ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        merge(path)
