# Genome Data Directory

This directory should contain the reference genome files required by the pipeline.

## Required Files

- `hg38.fa` - Human reference genome (GRCh38/hg38)
- `hg38.fa.fai` - FASTA index file (created automatically)

## Download Instructions

If the genome files are not present, download them using:

```bash
# Download hg38 reference genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz

# Create FASTA index (required for fast random access)
samtools faidx hg38.fa
```

## Notes

- This README ensures the directory structure is preserved in the repository
