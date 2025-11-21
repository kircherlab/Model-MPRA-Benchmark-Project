# Quick Reference: Publication Plots

## 🚀 Quick Start

### Option 1: Single Command
```bash
cd Visualizations
python publication_plots.py \
    --data ../Pipeline/results/enformer/merged_data.tsv \
    --output figures/enformer \
    --model Enformer \
    --cell_line HEK293T
```

### Option 2: Batch All Results
```bash
cd Visualizations
./generate_all_visualizations.sh
```

### Option 3: Interactive Examples
```bash
cd Visualizations
python example_usage.py
# Follow the interactive menu
```

---

## 📊 What You Get

Each run generates **6 types of plots**:

| File | Description | Use Case |
|------|-------------|----------|
| `*_correlation_grid.png` | Individual scatter plots per assay | Detailed per-assay analysis |
| `*_correlation_combined.png` | All assays overlaid | Compare correlations across assays |
| `*_roc_pr_curves.png` | ROC & Precision-Recall curves | Classification performance |
| `*_performance_summary.png` | Bar charts of AUC/AP scores | Quick performance overview |
| `*_distribution_violin.png` | SAD distributions (functional vs not) | Effect size visualization |
| `*_summary_table.png` | Statistics table | Comprehensive metrics |
| `*_performance_summary.csv` | Same as table (data format) | Import to papers/slides |

---

## 📁 Directory Structure

```
Visualizations/
├── publication_plots.py          # Main plotting script
├── generate_all_visualizations.sh # Batch processor
├── example_usage.py              # Interactive examples
├── README.md                      # Full documentation
├── QUICK_REFERENCE.md            # This file
└── figures/                       # Output directory
    ├── enformer/
    │   ├── enformer_correlation_grid.png
    │   ├── enformer_correlation_combined.png
    │   ├── enformer_roc_pr_curves.png
    │   ├── enformer_performance_summary.png
    │   ├── enformer_distribution_violin.png
    │   ├── enformer_summary_table.png
    │   └── enformer_performance_summary.csv
    └── basenji/
        └── ... (same structure)
```

---

## 🎨 Features

✅ **300 DPI** - Journal quality  
✅ **Professional colors** - Publication-ready  
✅ **Statistical tests** - Pearson, Spearman, Mann-Whitney  
✅ **Multiple metrics** - Correlation, AUC, AP  
✅ **Clean design** - Minimal, focused  
✅ **Batch processing** - Generate all at once  

---

## 🔧 Common Use Cases

### For Your Presentation
```bash
python publication_plots.py \
    --data ../Pipeline/results/enformer/merged_data.tsv \
    --output ../Presentations/figures \
    --model Enformer
```

### Compare Models
```bash
# Enformer
python publication_plots.py --data ../Pipeline/results/enformer/merged_data.tsv --output figures --model Enformer

# Basenji
python publication_plots.py --data ../Pipeline/results/basenji/merged_data.tsv --output figures --model Basenji

# Now compare figures/enformer_* with figures/basenji_*
```

### Different Cell Lines
```bash
# HEK293T
python publication_plots.py --data results_hek293t.tsv --output figures/hek293t --model Enformer --cell_line HEK293T

# HepG2
python publication_plots.py --data results_hepg2.tsv --output figures/hepg2 --model Enformer --cell_line HepG2
```

---

## 🐛 Troubleshooting

### Script not found
```bash
cd /data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/Visualizations
```

### Permission denied
```bash
chmod +x publication_plots.py
chmod +x generate_all_visualizations.sh
```

### Missing packages
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### No data files found
```bash
# Make sure pipeline has run:
ls ../Pipeline/results/*/merged_data.tsv
```

---

## 📧 Need Help?

1. Check `README.md` for full documentation
2. Run `python example_usage.py` for interactive examples
3. Open an issue on GitHub

---

## 💡 Pro Tips

1. **Always use absolute paths** if running from different directories
2. **Create separate output dirs** for different runs (prevents overwriting)
3. **Check CSV output** for exact numbers to include in papers
4. **Customize colors** in the script for your institution's branding
5. **Save originals** before editing plots in Illustrator/Inkscape

---

## 📝 Citation

```bibtex
@software{model_mpra_benchmark,
  title = {Model-MPRA-Benchmark: Publication Visualization Tools},
  author = {Your Lab},
  year = {2025},
  url = {https://github.com/kircherlab/Model-MPRA-Benchmark-Project}
}
```
