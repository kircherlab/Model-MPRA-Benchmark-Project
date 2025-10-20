
# Block 1: Load data and setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # type: ignore
import argparse

# --- Argument parsing for inputs and optional exclude_range ---
exclude_min, exclude_max = None, None
parser = argparse.ArgumentParser()
parser.add_argument('--exclude_range', nargs=2, type=float, default=None, help='Exclude mean SAD in [min, max]')
parser.add_argument('--merged_data', type=str, default=None, help='Path to merged_data.tsv')
parser.add_argument('--out_corr', type=str, default=None, help='Output path for correlation plot')
args, unknown = parser.parse_known_args()
if args.exclude_range:
    exclude_min, exclude_max = args.exclude_range

# --- Determine I/O: prefer snakemake if available, otherwise use CLI args ---
try:
    merged_file = snakemake.input.merged_data # type: ignore
    out_corr = snakemake.output.corr_plot # type: ignore
except NameError:
    if args.merged_data is None or args.out_corr is None:
        raise RuntimeError('When running outside Snakemake you must provide --merged_data and --out_corr')
    merged_file = args.merged_data
    out_corr = args.out_corr

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

plot_data = []
for assay in assay_cols:
    for idx, row in df.iterrows():
        plot_data.append({
            'assay': assay.replace('SAD_ASSAY_', ''),
            'mean_SAD': row[assay],
            'log2FC': row[log2fc_col]
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
    # Scatter plot (top row)
    ax_scatter = axes[0, i]
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
        # scatter
        sns.scatterplot(x='mean_SAD', y='log2FC', data=sub, ax=ax_main, alpha=0.6)
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
        sns.histplot(sub['mean_SAD'].dropna(), bins=40, ax=ax_xhist, color='skyblue')
        sns.histplot(sub['log2FC'].dropna(), bins=40, ax=ax_yhist, color='salmon', orientation='vertical')
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