# Visualizations Directory - Index

Welcome to the publication-quality visualization system for the Model-MPRA-Benchmark project!

## 📂 What's Here

```
Visualizations/
├── publication_plots.py               # Main plotting script (the engine)
├── generate_all_visualizations.sh    # Batch processor (run all at once)
├── example_usage.py                  # Interactive tutorial
├── README.md                          # Full documentation
├── QUICK_REFERENCE.md                # Cheat sheet
├── PLOT_GALLERY.md                   # Visual guide to plots
├── INDEX.md                          # This file
└── figures/                          # Output directory (auto-created)
```

## 🚀 Getting Started (Choose Your Path)

### Path 1: "Just show me the plots!" (2 minutes)
```bash
./generate_all_visualizations.sh
# Done! Check figures/ directory
```

### Path 2: "I want to understand first" (10 minutes)
```bash
python example_usage.py
# Interactive menu with examples
```

### Path 3: "Give me precise control" (5 minutes)
```bash
python publication_plots.py --help
# Then run with custom parameters
```

## 📚 Documentation Quick Links

| Document | Purpose | Read When |
|----------|---------|-----------|
| **QUICK_REFERENCE.md** | Commands & troubleshooting | You just want to run it |
| **README.md** | Complete documentation | You need full details |
| **PLOT_GALLERY.md** | Visual examples | You want to see what plots look like |
| **publication_plots.py** | Source code | You want to customize |

## 🎯 Common Tasks

### Generate plots for one model
```bash
python publication_plots.py \
    --data ../Pipeline/results/enformer/merged_data.tsv \
    --output figures/enformer \
    --model Enformer
```

### Generate plots for all models
```bash
./generate_all_visualizations.sh
```

### Try interactive examples
```bash
python example_usage.py
```

## 📊 What You Get

Each run produces **6 publication-quality plots**:

1. ✨ **Correlation Grid** - Detailed scatter plots per assay
2. 🎨 **Combined Correlation** - All assays overlaid  
3. 📈 **ROC & PR Curves** - Classification performance
4. 📊 **Performance Summary** - Bar charts of metrics
5. 🎻 **Distribution Violins** - Functional vs non-functional
6. 📋 **Summary Table** - Complete statistics (PNG + CSV)

All at **300 DPI** for publications!

## 💡 Pro Tips

1. **Start with batch script** - Generates everything at once
2. **Check CSV output** - Copy exact numbers to papers
3. **Use example_usage.py** - Learn by doing
4. **Read PLOT_GALLERY.md** - Understand what each plot shows
5. **Keep originals** - Save before editing in Illustrator

## 🐛 Troubleshooting

### "command not found"
```bash
chmod +x generate_all_visualizations.sh
chmod +x example_usage.py
```

### "No module named X"
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### "No such file or directory"
```bash
# Make sure pipeline has run first:
ls ../Pipeline/results/*/merged_data.tsv
```

## 🎓 Learn More

- **New users**: Start with `example_usage.py`
- **Quick reference**: See `QUICK_REFERENCE.md`
- **Visual learners**: Check `PLOT_GALLERY.md`
- **Full details**: Read `README.md`
- **Customize**: Edit `publication_plots.py`

## 📧 Questions?

1. Check the documentation files
2. Run `python example_usage.py` for interactive help
3. Open an issue on GitHub

## ✅ Checklist Before First Run

- [ ] Pipeline has generated results (`merged_data.tsv` exists)
- [ ] Python packages installed (`pip install -r requirements.txt` if available)
- [ ] Scripts are executable (`chmod +x *.sh *.py`)
- [ ] You know which model to visualize

## 🎉 Ready to Start!

```bash
# Option 1: Batch process everything
./generate_all_visualizations.sh

# Option 2: Interactive tutorial  
python example_usage.py

# Option 3: Single model
python publication_plots.py --data ../Pipeline/results/enformer/merged_data.tsv --output figures --model Enformer
```

**Happy plotting! 📊✨**
