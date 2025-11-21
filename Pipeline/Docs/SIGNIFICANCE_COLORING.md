# Significance-Based Coloring in Scatter Plots

## Overview

The plotting script now supports significance-based coloring of variants in scatter plots, using PVAL and QVAL values from the VCF file.

## What Changed

### 1. Configuration (`config.yaml`)

Added a new `significance` section with the following parameters:

```yaml
significance:
  use_significance_coloring: true  # Enable/disable significance coloring
  
  # Which statistic to use: "PVAL", "QVAL", or "BOTH"
  statistic: "QVAL"
  
  # Threshold in -log10 scale
  # 1.3 = p/q < 0.05
  # 2.0 = p/q < 0.01
  # 3.0 = p/q < 0.001
  threshold: 1.3
  
  # For "BOTH" mode only
  pval_threshold: 1.3
  qval_threshold: 1.3
  
  # Visual styling
  colors:
    significant: "#E74C3C"         # Red for significant
    nonsignificant: "#95A5A6"      # Gray for non-significant
    significant_alpha: 0.7         # Opacity for significant
    nonsignificant_alpha: 0.3      # Opacity for non-significant
```

### 2. Plotting Script (`generate_plots.py`)

**New Features:**
- Reads significance configuration from `config.yaml`
- Determines variant significance based on PVAL/QVAL thresholds
- Plots significant and non-significant variants with different colors and opacity
- Adds legend to distinguish variant types
- Works with all scatter plot types (multi-panel and combined joint plots)

**Color Scheme:**
- **Significant variants**: Red (`#E74C3C`), higher opacity (0.7), dark red edge
- **Non-significant variants**: Gray (`#95A5A6`), lower opacity (0.3)

## How Significance is Defined

### Understanding PVAL and QVAL

In the VCF file, `PVAL` and `QVAL` are stored as **-log10 transformed** values:

- **PVAL**: `-log10(p-value)` from statistical test
- **QVAL**: `-log10(q-value)` from FDR correction (recommended)

**Higher values = more significant**

### Threshold Examples

| Threshold | Original p/q-value | Interpretation |
|-----------|-------------------|----------------|
| 1.3       | < 0.05            | Standard significance |
| 2.0       | < 0.01            | Strong significance |
| 3.0       | < 0.001           | Very strong significance |

### Modes

1. **PVAL mode** (`statistic: "PVAL"`):
   - Variant is significant if: `PVAL >= threshold`
   
2. **QVAL mode** (`statistic: "QVAL"`, recommended):
   - Variant is significant if: `QVAL >= threshold`
   - Accounts for multiple testing correction (FDR)
   
3. **BOTH mode** (`statistic: "BOTH"`):
   - Variant is significant if: `PVAL >= pval_threshold` AND `QVAL >= qval_threshold`
   - Strictest criterion

## Recommended Settings

### For Exploratory Analysis
```yaml
significance:
  use_significance_coloring: true
  statistic: "QVAL"
  threshold: 1.3  # FDR < 0.05
```

### For Publication-Quality Figures
```yaml
significance:
  use_significance_coloring: true
  statistic: "QVAL"
  threshold: 2.0  # FDR < 0.01
```

### For Conservative Analysis
```yaml
significance:
  use_significance_coloring: true
  statistic: "BOTH"
  pval_threshold: 2.0  # p < 0.01
  qval_threshold: 2.0  # FDR < 0.01
```

## Customizing Colors

You can adjust the color scheme to match your preference:

```yaml
significance:
  colors:
    significant: "#FF5733"         # Orange-red
    nonsignificant: "#BDC3C7"      # Light gray
    significant_alpha: 0.8         # More opaque
    nonsignificant_alpha: 0.2      # More transparent
```

### Color Palettes

**Colorblind-friendly options:**
- Significant: `"#D55E00"` (vermillion)
- Non-significant: `"#999999"` (gray)

**High-contrast options:**
- Significant: `"#000000"` (black)
- Non-significant: `"#CCCCCC"` (light gray)

## Usage Examples

### Example 1: Default FDR < 0.05
```yaml
significance:
  use_significance_coloring: true
  statistic: "QVAL"
  threshold: 1.3
```

Result: Variants with `QVAL >= 1.3` (FDR < 0.05) are highlighted in red.

### Example 2: Strict Filtering
```yaml
significance:
  use_significance_coloring: true
  statistic: "QVAL"
  threshold: 3.0  # FDR < 0.001
```

Result: Only highly significant variants (FDR < 0.001) are highlighted.

### Example 3: Combined P-value and FDR
```yaml
significance:
  use_significance_coloring: true
  statistic: "BOTH"
  pval_threshold: 1.3
  qval_threshold: 2.0
```

Result: Variants must pass both p < 0.05 AND FDR < 0.01 to be highlighted.

### Example 4: Disable Significance Coloring
```yaml
significance:
  use_significance_coloring: false
```

Result: All variants plotted with uniform color (original behavior).

## Output Interpretation

### In Scatter Plots

- **Red points**: Statistically significant variants (according to your threshold)
- **Gray points**: Non-significant variants
- **Legend**: Shows counts of each category

### Console Output

When running the pipeline, you'll see:

```
Significance filtering: 45/1000 (4.5%) variants marked as significant
  Using: QVAL >= 1.3
```

This tells you:
- How many variants passed the significance filter
- What criterion was used

## Troubleshooting

### Warning: "PVAL column not found in data"
**Solution**: Your VCF file may not have PVAL values. Switch to `statistic: "QVAL"` or check VCF format.

### Too Many/Too Few Significant Variants
**Solution**: Adjust the `threshold` value:
- Increase threshold → fewer significant variants (stricter)
- Decrease threshold → more significant variants (more permissive)

### Colors Not Visible
**Solution**: Adjust alpha values:
```yaml
significant_alpha: 0.9      # More opaque
nonsignificant_alpha: 0.2   # More transparent
```

## Integration with Pipeline

The significance configuration is automatically passed from `config.yaml` to `generate_plots.py` via Snakemake:

1. Snakemake reads `config.yaml`
2. Passes config to plotting script
3. Script applies significance filtering
4. Plots generated with color-coded variants

No additional steps required!

## Technical Details

### VCF Format Requirements

Your VCF `INFO` field must contain:
```
PVAL=-log10(p-value)  # e.g., PVAL=1.52 means p=0.03
QVAL=-log10(q-value)  # e.g., QVAL=2.30 means FDR=0.005
```

### Significance Determination Logic

```python
if statistic == 'PVAL':
    is_significant = (PVAL >= threshold)
elif statistic == 'QVAL':
    is_significant = (QVAL >= threshold)
elif statistic == 'BOTH':
    is_significant = (PVAL >= pval_threshold) & (QVAL >= qval_threshold)
```

### Plot Layering

1. Non-significant variants plotted first (background layer)
2. Significant variants plotted on top (foreground layer)
3. Edge color added to significant points for emphasis

This ensures significant variants are always visible.

## Future Enhancements

Potential additions:
- [ ] Support for custom significance columns
- [ ] Gradient coloring by significance level
- [ ] Size scaling by significance magnitude
- [ ] Interactive tooltips showing p/q-values
- [ ] Separate plots for significant-only variants

## References

- **FDR correction**: Benjamini-Hochberg procedure
- **Multiple testing**: Accounts for testing many variants simultaneously
- **Recommended**: Use QVAL (FDR) rather than PVAL for most analyses
