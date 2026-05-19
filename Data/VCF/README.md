# VCF Data Directory

This directory contains the input variant files for the pipeline.

## Required Files

| File | Cell type | Source |
|---|---|---|
| `hek293t.vcf.gz` | HEK293T | IGVF MPRA (IGVFFI4378PZYI) |
| `hepg2.vcf.gz` | HepG2 | IGVF MPRA (IGVFFI4134MFLL) |
| `ngn2.vcf.gz` | NGN2 neurons (WTC-11 → NGN2, 21d) | IGVF MPRA (IGVFFI7899LTDI) |

VCF files are not tracked in this repository due to size. Obtain them from the respective sources and place them here before running the pipeline.

## Index files

All VCF files must be bgzipped and indexed with tabix:

```bash
bgzip yourfile.vcf
tabix -p vcf yourfile.vcf.gz
```
