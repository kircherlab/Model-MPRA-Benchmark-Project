# Block 1: Load data and setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # type: ignore
import argparse
import yaml
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score # type: ignore

# --- Argument parsing for inputs and optional exclude_range ---
exclude_min, exclude_max = None, None
parser = argparse.ArgumentParser()
parser.add_argument('--exclude_range', nargs=2, type=float, default=None, help='Exclude mean SAD in [min, max]')
parser.add_argument('--merged_data', type=str, default=None, help='Path to merged_data.tsv')
parser.add_argument('--out_corr', type=str, default=None, help='Output path for correlation plot')
parser.add_argument('--config', type=str, default=None, help='Path to config.yaml for significance thresholds')
args, unknown = parser.parse_known_args()
if args.exclude_range:
    exclude_min, exclude_max = args.exclude_range

# --- Determine I/O: prefer snakemake if available, otherwise use CLI args ---
try:
    merged_file = snakemake.input.merged_data # type: ignore
    out_corr = snakemake.output.corr_plot # type: ignore
    config_file = snakemake.config # type: ignore
except NameError:
    if args.merged_data is None or args.out_corr is None:
        raise RuntimeError('When running outside Snakemake you must provide --merged_data and --out_corr')
    merged_file = args.merged_data
    out_corr = args.out_corr
    config_file = args.config

# Load significance configuration
sig_config = {
    'use_significance_coloring': False,
    'statistic': 'QVAL',
    'threshold': 1.3,
    'pval_threshold': 1.3,
    'qval_threshold': 1.3,
    'colors': {
        'significant': '#E74C3C',
        'nonsignificant': '#95A5A6',
        'significant_alpha': 0.7,
        'nonsignificant_alpha': 0.3
    }
}

if config_file:
    if isinstance(config_file, dict):
        # From Snakemake
        if 'significance' in config_file:
            sig_config.update(config_file['significance'])
    elif isinstance(config_file, str):
        # From file path
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                if 'significance' in config_data:
                    sig_config.update(config_data['significance'])
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")

sns.set_theme(style="whitegrid", context="talk")
df = pd.read_csv(merged_file, sep='\t')

# Block 2: Identify assay columns and compute per-assay means
assay_cols = [c for c in df.columns if c.startswith('SAD_ASSAY_')]
if not assay_cols:
    raise ValueError("No SAD_ASSAY_* columns found in merged data. Run merge with agg_per_assay enabled.")


# Block 3: Prepare data for plotting
# Use LOG2FC column (case-insensitive)
log2fc_col = None
for col in df.columns:
    if col.lower() == 'log2fc':
        log2fc_col = col
        break
if log2fc_col is None:
    raise ValueError("LOG2FC column not found in merged data file.")

# Determine variant significance based on PVAL/QVAL
def determine_significance(df, sig_config):
    """
    Determine which variants are significant based on PVAL/QVAL thresholds.
    
    Returns:
        pd.Series: Boolean series indicating significant variants
    """
    if not sig_config['use_significance_coloring']:
        return pd.Series([False] * len(df), index=df.index)
    
    statistic = sig_config['statistic'].upper()
    
    # Check if required columns exist
    has_pval = 'PVAL' in df.columns
    has_qval = 'QVAL' in df.columns
    
    if statistic == 'PVAL' and not has_pval:
        print("Warning: PVAL column not found in data. Disabling significance coloring.")
        return pd.Series([False] * len(df), index=df.index)
    
    if statistic == 'QVAL' and not has_qval:
        print("Warning: QVAL column not found in data. Disabling significance coloring.")
        return pd.Series([False] * len(df), index=df.index)
    
    if statistic == 'BOTH' and (not has_pval or not has_qval):
        print("Warning: PVAL or QVAL column not found in data. Disabling significance coloring.")
        return pd.Series([False] * len(df), index=df.index)
    
    # Determine significance
    if statistic == 'PVAL':
        is_sig = df['PVAL'] >= sig_config['threshold']
    elif statistic == 'QVAL':
        is_sig = df['QVAL'] >= sig_config['threshold']
    elif statistic == 'BOTH':
        pval_sig = df['PVAL'] >= sig_config['pval_threshold']
        qval_sig = df['QVAL'] >= sig_config['qval_threshold']
        is_sig = pval_sig & qval_sig
    else:
        print(f"Warning: Unknown statistic '{statistic}'. Disabling significance coloring.")
        return pd.Series([False] * len(df), index=df.index)
    
    return is_sig.fillna(False)

# Add significance column to dataframe
df['is_significant'] = determine_significance(df, sig_config)

if sig_config['use_significance_coloring']:
    n_sig = df['is_significant'].sum()
    n_total = len(df)
    print(f"Significance filtering: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%) variants marked as significant")
    print(f"  Using: {sig_config['statistic']} >= {sig_config['threshold']}")

plot_data = []
for assay in assay_cols:
    for idx, row in df.iterrows():
        plot_data.append({
            'assay': assay.replace('SAD_ASSAY_', ''),
            'mean_SAD': row[assay],
            'log2FC': row[log2fc_col],
            'is_significant': row['is_significant']
        })

plot_df = pd.DataFrame(plot_data)

# Optionally exclude points in specified mean SAD range
if exclude_min is not None and exclude_max is not None:
    before = len(plot_df)
    plot_df = plot_df[(plot_df['mean_SAD'] < exclude_min) | (plot_df['mean_SAD'] > exclude_max)]
    after = len(plot_df)
    print(f"Excluded {before - after} points with mean SAD in [{exclude_min}, {exclude_max}]")


# Block 4: Multi-panel scatter plot (one subplot per assay) + histogram panel
num_assays = len(assay_cols)
fig, axes = plt.subplots(2, num_assays, figsize=(6*num_assays, 12), sharey='row')
if num_assays == 1:
    axes = np.array([[axes[0]], [axes[1]]])
for i, assay in enumerate(sorted(plot_df['assay'].unique())):
    sub = plot_df[plot_df['assay'] == assay]
    # Scatter plot (top row) with significance coloring
    ax_scatter = axes[0, i]
    
    if sig_config['use_significance_coloring']:
        # Plot non-significant points first (gray, transparent)
        sub_nonsig = sub[~sub['is_significant']]
        if not sub_nonsig.empty:
            ax_scatter.scatter(sub_nonsig['mean_SAD'], sub_nonsig['log2FC'],
                             color=sig_config['colors']['nonsignificant'],
                             alpha=sig_config['colors']['nonsignificant_alpha'],
                             s=30, label='Non-significant')
        
        # Plot significant points on top (red, more opaque)
        sub_sig = sub[sub['is_significant']]
        if not sub_sig.empty:
            ax_scatter.scatter(sub_sig['mean_SAD'], sub_sig['log2FC'],
                             color=sig_config['colors']['significant'],
                             alpha=sig_config['colors']['significant_alpha'],
                             s=30, label='Significant', edgecolors='darkred', linewidths=0.5)
        
        # Add legend
        if not sub_nonsig.empty and not sub_sig.empty:
            ax_scatter.legend(loc='upper right', fontsize=10, framealpha=0.9)
    else:
        # Original plotting without significance coloring
        sns.scatterplot(x='mean_SAD', y='log2FC', data=sub, ax=ax_scatter, alpha=0.5)
    
    ax_scatter.set_title(f'{assay}: mean SAD vs log2FC', fontsize=16)
    ax_scatter.set_xlabel(f'Mean SAD (Enformer, {assay})', fontsize=14)
    ax_scatter.set_ylabel('log2FoldChange (MPRA)', fontsize=14)
    # Set x/y limits dynamically based on data
    xvals = sub['mean_SAD'].dropna()
    yvals = sub['log2FC'].dropna()
    if len(xvals) > 0:
        xmin, xmax = xvals.min(), xvals.max()
        pad = (xmax - xmin) * 0.1 if xmax > xmin else 0.001
        ax_scatter.set_xlim(xmin - pad, xmax + pad)
    if len(yvals) > 0:
        ymin, ymax = yvals.min(), yvals.max()
        pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.01
        ax_scatter.set_ylim(ymin - pad, ymax + pad)
    # Correlation annotation
    if sub['mean_SAD'].notnull().any() and sub['log2FC'].notnull().any():
        corr = sub[['mean_SAD', 'log2FC']].dropna().corr().iloc[0,1]
        ax_scatter.text(0.05, 0.95, f'Pearson r={corr:.2f}', transform=ax_scatter.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    # Histogram (bottom row)
    ax_hist = axes[1, i]
    sns.histplot(sub['mean_SAD'].dropna(), bins=50, ax=ax_hist, color='skyblue')
    ax_hist.set_title(f'{assay}: mean SAD histogram', fontsize=16)
    ax_hist.set_xlabel('Mean SAD (Enformer)', fontsize=14)
    ax_hist.set_ylabel('Count', fontsize=14)
    # Set histogram x-limits dynamically
    if len(xvals) > 0:
        xmin, xmax = xvals.min(), xvals.max()
        pad = (xmax - xmin) * 0.1 if xmax > xmin else 0.001
        ax_hist.set_xlim(xmin - pad, xmax + pad)
plt.tight_layout()
plt.savefig(out_corr, dpi=300)
print(f"Saved multi-panel assay comparison plot and histograms to {out_corr}")

# Block 5: Combined joint scatter + marginal histograms (one column per assay)
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore

out_dir = '/'.join(out_corr.split('/')[:-1]) or '.'
assays = sorted(plot_df['assay'].unique())
if assays:
    fig = plt.figure(figsize=(6*len(assays), 6))
    axs = []
    for i, assay in enumerate(assays):
        # create main axis for scatter
        ax_main = fig.add_subplot(1, len(assays), i+1)
        axs.append(ax_main)
        sub = plot_df[plot_df['assay'] == assay]
        if sub.empty:
            continue
        
        # Scatter with significance coloring
        if sig_config['use_significance_coloring']:
            # Plot non-significant points first (gray, transparent)
            sub_nonsig = sub[~sub['is_significant']]
            if not sub_nonsig.empty:
                ax_main.scatter(sub_nonsig['mean_SAD'], sub_nonsig['log2FC'],
                               color=sig_config['colors']['nonsignificant'],
                               alpha=sig_config['colors']['nonsignificant_alpha'],
                               s=20, label='Non-significant')
            
            # Plot significant points on top (red, more opaque)
            sub_sig = sub[sub['is_significant']]
            if not sub_sig.empty:
                ax_main.scatter(sub_sig['mean_SAD'], sub_sig['log2FC'],
                               color=sig_config['colors']['significant'],
                               alpha=sig_config['colors']['significant_alpha'],
                               s=20, label='Significant', edgecolors='darkred', linewidths=0.5)
            
            # Add legend
            if not sub_nonsig.empty and not sub_sig.empty:
                ax_main.legend(loc='best', fontsize=9, framealpha=0.9)
        else:
            # Original plotting
            sns.scatterplot(x='mean_SAD', y='log2FC', data=sub, ax=ax_main, alpha=0.6)
        
        # Add regression line
        valid_data = sub[['mean_SAD', 'log2FC']].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['mean_SAD'], valid_data['log2FC'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['mean_SAD'].min(), valid_data['mean_SAD'].max(), 100)
            ax_main.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Regression' if sig_config['use_significance_coloring'] else None)
            if sig_config['use_significance_coloring'] and (not sub_nonsig.empty or not sub_sig.empty):
                ax_main.legend(loc='best', fontsize=9, framealpha=0.9)
        
        ax_main.set_title(f'{assay}: mean SAD vs log2FC', fontsize=14)
        ax_main.set_xlabel(f'Mean SAD (Enformer, {assay})', fontsize=12)
        if i == 0:
            ax_main.set_ylabel('log2FoldChange (MPRA)', fontsize=12)
        else:
            ax_main.set_ylabel('')
        # dynamic limits
        xvals = sub['mean_SAD'].dropna()
        yvals = sub['log2FC'].dropna()
        if len(xvals) > 0:
            xmin, xmax = xvals.min(), xvals.max()
            pad = (xmax - xmin) * 0.1 if xmax > xmin else 0.001
            ax_main.set_xlim(xmin - pad, xmax + pad)
        if len(yvals) > 0:
            ymin, ymax = yvals.min(), yvals.max()
            pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.01
            ax_main.set_ylim(ymin - pad, ymax + pad)
        # add marginal axes
        divider = make_axes_locatable(ax_main)
        ax_xhist = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
        ax_yhist = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
        # plot histograms
        sns.histplot(sub['mean_SAD'].dropna(), bins=40, ax=ax_xhist, color='salmon')
        sns.histplot(y=sub['log2FC'].dropna(), bins=40, ax=ax_yhist, color='salmon')
        # clean marginal axes
        ax_xhist.tick_params(labelbottom=False)
        ax_xhist.set_ylabel('Count', fontsize=10)
        ax_yhist.set_xlabel('Count', fontsize=10)
        ax_yhist.tick_params(labelleft=False)
        # annotate correlation
        if sub['mean_SAD'].notnull().any() and sub['log2FC'].notnull().any():
            corr = sub[['mean_SAD', 'log2FC']].dropna().corr().iloc[0,1]
            ax_main.text(0.05, 0.95, f'Pearson r={corr:.2f}', transform=ax_main.transAxes,
                         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    plt.tight_layout()
    combined_out = f"{out_dir}/combined_joint.png"
    fig.savefig(combined_out, dpi=300)
    plt.close(fig)
    print(f"Saved combined joint scatter+hist for all assays to {combined_out}")

# Block 6: ROC Curves and Performance Metrics
def calculate_roc_metrics(y_true, y_scores, assay_name):
    """Calculate ROC curve and performance metrics for a single assay."""
    # Remove NaN values
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_scores))
    y_true_clean = y_true[valid_idx]
    y_scores_clean = y_scores[valid_idx]

    if len(y_true_clean) < 10:  # Need minimum samples
        return None

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_clean, y_scores_clean)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true_clean, y_scores_clean)
    avg_precision = average_precision_score(y_true_clean, y_scores_clean)

    # Calculate additional metrics at optimal threshold
    # Find threshold that maximizes Youden's J statistic (tpr - fpr)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_scores_clean >= optimal_threshold).astype(int)
    tn = np.sum((y_true_clean == 0) & (y_pred_optimal == 0))
    tp = np.sum((y_true_clean == 1) & (y_pred_optimal == 1))
    fn = np.sum((y_true_clean == 1) & (y_pred_optimal == 0))
    fp = np.sum((y_true_clean == 0) & (y_pred_optimal == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0

    return {
        'assay': assay_name,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'avg_precision': avg_precision,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision_opt': precision_opt,
        'recall_opt': recall_opt,
        'f1_opt': f1_opt,
        'n_samples': len(y_true_clean)
    }

# Create binary labels from LOG2FC (functional vs non-functional)
# Use percentile-based approach: top 25% of absolute log2FC values as functional
log2fc_abs = np.abs(df[log2fc_col])
functional_threshold = np.percentile(log2fc_abs, 75)  # Top 25% as functional
binary_labels = (log2fc_abs >= functional_threshold).astype(int)

print(f"Binary classification: {binary_labels.sum()} functional variants out of {len(binary_labels)} total (top 25% by |log2FC|)")
print(f"Functional ratio: {binary_labels.mean():.2%}")
print(f"Functional threshold: |log2FC| >= {functional_threshold:.4f}")

# Calculate ROC metrics for each assay
roc_results = []
for assay_col in assay_cols:
    assay_name = assay_col.replace('SAD_ASSAY_', '')
    sad_scores = df[assay_col].values
    result = calculate_roc_metrics(binary_labels, sad_scores, assay_name)
    if result is not None:
        roc_results.append(result)

if roc_results:
    # Plot ROC Curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curves
    colors = sns.color_palette("husl", len(roc_results))
    for i, result in enumerate(roc_results):
        ax1.plot(result['fpr'], result['tpr'], color=colors[i],
                label=f'{result["assay"]} (AUC = {result["roc_auc"]:.3f})',
                linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves: SAD Score Prediction of Functional Variants')
    ax1.legend(loc="upper right", fontsize='xx-small', framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curves
    for i, result in enumerate(roc_results):
        ax2.plot(result['recall'], result['precision'], color=colors[i],
                label=f'{result["assay"]} (AP = {result["avg_precision"]:.3f})',
                linewidth=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc="upper right", fontsize='xx-small', framealpha=0.8)
    ax2.grid(True, alpha=0.3)

    # Performance Metrics Bar Plot
    assays = [r['assay'] for r in roc_results]
    auc_scores = [r['roc_auc'] for r in roc_results]
    ap_scores = [r['avg_precision'] for r in roc_results]
    accuracies = [r['accuracy'] for r in roc_results]

    x = np.arange(len(assays))
    width = 0.25

    ax3.bar(x - width, auc_scores, width, label='AUC', alpha=0.8, color='skyblue')
    ax3.bar(x, ap_scores, width, label='Avg Precision', alpha=0.8, color='lightcoral')
    ax3.bar(x + width, accuracies, width, label='Accuracy', alpha=0.8, color='lightgreen')

    ax3.set_xlabel('Assay')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics Summary')
    ax3.set_xticks(x)
    ax3.set_xticklabels(assays, rotation=45)
    ax3.legend(fontsize='xx-small', framealpha=0.8)
    ax3.grid(True, alpha=0.3)

    # Detailed Metrics Table with color gradients
    metrics_data = []
    for result in roc_results:
        metrics_data.append({
            'Assay': result['assay'],
            'AUC': result['roc_auc'],
            'Avg Prec': result['avg_precision'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision_opt'],
            'Recall': result['recall_opt'],
            'F1': result['f1_opt'],
            'Opt Thresh': result['optimal_threshold'],
            'N Samples': result['n_samples']
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create formatted display data for the table
    display_data = []
    for result in roc_results:
        display_data.append([
            result['assay'],
            f"{result['roc_auc']:.3f}",
            f"{result['avg_precision']:.3f}",
            f"{result['accuracy']:.3f}",
            f"{result['precision_opt']:.3f}",
            f"{result['recall_opt']:.3f}",
            f"{result['f1_opt']:.3f}",
            f"{result['optimal_threshold']:.3f}",
            str(result['n_samples'])
        ])

    # Create table with color gradients
    ax4.axis('off')
    table = ax4.table(cellText=display_data,
                     colLabels=['Assay', 'AUC', 'Avg Prec', 'Accuracy', 'Precision', 'Recall', 'F1', 'Opt Thresh', 'N Samples'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.05, 0.05, 0.9, 0.85])

    # Set font sizes
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Add color gradients for numeric columns
    for i in range(len(metrics_df)):
        for j in range(1, len(metrics_df.columns)):  # Skip 'Assay' column
            cell = table[i+1, j]  # +1 because row 0 is headers
            value = metrics_df.iloc[i, j]
            if isinstance(value, (int, float)) and not pd.isna(value):
                if metrics_df.columns[j] in ['AUC', 'Avg Prec', 'Accuracy', 'Precision', 'Recall', 'F1']:
                    # Bright nano red to bright nano green gradient for performance metrics (0-1 scale)
                    intensity = min(float(value), 1.0)  # Cap at 1.0
                    cell.set_facecolor((1-intensity, intensity, 0.0, 0.8))  # Bright nano red to green
                elif metrics_df.columns[j] == 'Opt Thresh':
                    # Blue gradient for threshold (assuming 0-1 range)
                    intensity = min(abs(float(value)), 1.0)
                    cell.set_facecolor((0.2, 0.2, intensity, 0.8))
            else:
                cell.set_facecolor((1, 1, 1, 0.8))  # White for non-numeric

    # Style the header row
    for j in range(len(metrics_df.columns)):
        header_cell = table[0, j]
        header_cell.set_facecolor((0.2, 0.2, 0.2, 0.9))
        header_cell.set_text_props(color='white', weight='bold', fontsize=6)

    ax4.set_title('Detailed Performance Metrics', fontsize=12, pad=10)

    plt.tight_layout()
    roc_out = f"{out_dir}/roc_performance_analysis.png"
    fig.savefig(roc_out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save metrics to CSV
    metrics_csv = f"{out_dir}/performance_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"Saved ROC curves and performance analysis to {roc_out}")
    print(f"Saved detailed metrics to {metrics_csv}")

    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    for result in roc_results:
        print(f"{result['assay']}: AUC={result['roc_auc']:.3f}, AP={result['avg_precision']:.3f}, Acc={result['accuracy']:.3f}")
else:
    print("Warning: Not enough data for ROC analysis (need at least 10 samples per assay)")