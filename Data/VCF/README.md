# VCF Data Directory

This directory contains the input variant files for the pipeline.

## Required Files

| File | Cell type | Source |
|---|---|---|
| `IGVFFI4134MFLL.vcf.gz` | HEK293T | ENCODE MPRA |
| `IGVFFI4378PZYI.vcf.gz` | HepG2 | ENCODE MPRA |
| `80k_normalized.vcf.gz` | NGN2 neurons | Schraivogel et al. |

VCF files are not tracked in this repository due to size. Obtain them from the respective sources and place them here before running the pipeline.

## Index files

All VCF files must be bgzipped and indexed with tabix:

```bash
bgzip yourfile.vcf
tabix -p vcf yourfile.vcf.gz
```
