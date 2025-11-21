#!/usr/bin/env python3
"""
Enhanced Publication-Quality Plots for Model-MPRA Benchmark
============================================================

Creates beautiful, publication-ready visualizations for comparing DNA sequence models
against MPRA experimental data.

Usage:
    python publication_plots.py --data Pipeline/results/enformer/merged_data.tsv \
                                --output Visualizations/figures \
                                --model enformer \
                                --cell_line HEK293T

Features:
- High-resolution plots (300 DPI)
- Professional color schemes
- Statistical annotations
- Multiple plot types (scatter, violin, heatmap, ROC, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse
import os
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Professional color palettes
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'warning': '#D62828',      # Red
    'neutral': '#6C757D',      # Gray
    'light_blue': '#89CFF0',
    'light_purple': '#C8A2C8',
}

MODEL_COLORS = {
    'enformer': '#2E86AB',
    'basenji': '#A23B72',
    'borzoi': '#06A77D',
    'deepsea': '#F18F01',
    'alphagenome': '#D62828',
}


def load_data(filepath):
    """Load merged data with error handling"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded {len(df)} variants")
    print(f"Columns: {list(df.columns)}")
    return df


def get_assay_columns(df):
    """Extract SAD_ASSAY columns"""
    assay_cols = [c for c in df.columns if c.startswith('SAD_ASSAY_')]
    if not assay_cols:
        raise ValueError("No SAD_ASSAY_* columns found")
    print(f"Found {len(assay_cols)} assays: {assay_cols}")
    return assay_cols


def get_log2fc_column(df):
    """Find LOG2FC column (case insensitive)"""
    for col in df.columns:
        if col.lower() == 'log2fc':
            return col
    raise ValueError("LOG2FC column not found")


def plot_enhanced_correlation_grid(df, assay_cols, log2fc_col, output_dir, model_name='Model'):
    """
    Create enhanced correlation scatter plots for all assays in a grid layout
    """
    n_assays = len(assay_cols)
    n_cols = min(3, n_assays)
    n_rows = int(np.ceil(n_assays / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_assays == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, assay_col in enumerate(assay_cols):
        ax = axes[idx]
        assay_name = assay_col.replace('SAD_ASSAY_', '')
        
        # Remove NaN values
        mask = ~(df[assay_col].isna() | df[log2fc_col].isna())
        x = df.loc[mask, assay_col].values
        y = df.loc[mask, log2fc_col].values
        
        if len(x) < 3:
            ax.text(0.5, 0.5, f'Insufficient data\n({len(x)} points)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(assay_name)
            continue
        
        # Calculate statistics
        r, p_val = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)
        
        # Create hexbin plot for density
        hb = ax.hexbin(x, y, gridsize=30, cmap='Blues', mincnt=1, alpha=0.7, edgecolors='none')
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS['warning'], linewidth=2.5, 
                label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
        
        # Styling
        ax.set_xlabel(f'{model_name} SAD Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('MPRA log2FC', fontsize=11, fontweight='bold')
        ax.set_title(f'{assay_name}', fontsize=12, fontweight='bold', pad=10)
        
        # Add statistics box
        stats_text = f'r = {r:.3f}\nρ = {rho:.3f}\nn = {len(x)}'
        if p_val < 0.001:
            stats_text += '\np < 0.001'
        else:
            stats_text += f'\np = {p_val:.3f}'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add colorbar
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Variant Density', fontsize=9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_assays, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name.lower()}_correlation_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_correlation(df, assay_cols, log2fc_col, output_dir, model_name='Model'):
    """
    Single combined scatter plot with all assays overlaid
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    palette = sns.color_palette("husl", len(assay_cols))
    
    for idx, assay_col in enumerate(assay_cols):
        assay_name = assay_col.replace('SAD_ASSAY_', '')
        
        mask = ~(df[assay_col].isna() | df[log2fc_col].isna())
        x = df.loc[mask, assay_col].values
        y = df.loc[mask, log2fc_col].values
        
        if len(x) < 3:
            continue
        
        r, _ = stats.pearsonr(x, y)
        
        ax.scatter(x, y, alpha=0.5, s=20, color=palette[idx], 
                  label=f'{assay_name} (r={r:.3f})', edgecolors='none')
    
    ax.set_xlabel(f'{model_name} SAD Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('MPRA log2FC', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name}: Combined Assay Correlations', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name.lower()}_correlation_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_roc_curves_enhanced(df, assay_cols, log2fc_col, output_dir, model_name='Model'):
    """
    Enhanced ROC curves with better styling and statistics
    """
    # Create binary labels (top 25% by |log2FC|)
    log2fc_abs = np.abs(df[log2fc_col])
    threshold = np.percentile(log2fc_abs.dropna(), 75)
    binary_labels = (log2fc_abs >= threshold).astype(int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    palette = sns.color_palette("husl", len(assay_cols))
    results = []
    
    # Calculate ROC for each assay
    for idx, assay_col in enumerate(assay_cols):
        assay_name = assay_col.replace('SAD_ASSAY_', '')
        
        mask = ~(df[assay_col].isna() | binary_labels.isna())
        y_true = binary_labels[mask].values
        y_scores = df.loc[mask, assay_col].values
        
        if len(y_true) < 10:
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        results.append({
            'assay': assay_name,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'avg_precision': avg_precision,
            'color': palette[idx]
        })
    
    # Plot ROC curves
    for result in results:
        ax1.plot(result['fpr'], result['tpr'], 
                color=result['color'], linewidth=2.5, alpha=0.8,
                label=f"{result['assay']} (AUC={result['roc_auc']:.3f})")
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.500)', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_title(f'{model_name}: ROC Curves\n(Functional vs Non-Functional Variants)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # Plot Precision-Recall curves
    for result in results:
        ax2.plot(result['recall'], result['precision'], 
                color=result['color'], linewidth=2.5, alpha=0.8,
                label=f"{result['assay']} (AP={result['avg_precision']:.3f})")
    
    ax2.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax2.set_title(f'{model_name}: Precision-Recall Curves', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name.lower()}_roc_pr_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return results


def plot_performance_summary(roc_results, output_dir, model_name='Model'):
    """
    Create summary bar chart of model performance
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    assays = [r['assay'] for r in roc_results]
    aucs = [r['roc_auc'] for r in roc_results]
    aps = [r['avg_precision'] for r in roc_results]
    
    x = np.arange(len(assays))
    width = 0.35
    
    # AUC bars
    bars1 = ax1.bar(x, aucs, width, label='ROC AUC', 
                    color=COLORS['primary'], edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random Baseline', alpha=0.7)
    ax1.set_xlabel('Assay', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name}: ROC AUC by Assay', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assays, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Average Precision bars
    bars2 = ax2.bar(x, aps, width, label='Average Precision', 
                    color=COLORS['accent'], edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Assay', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Precision Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name}: Average Precision by Assay', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(assays, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0, 1.05])
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name.lower()}_performance_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_distribution_comparison(df, assay_cols, log2fc_col, output_dir, model_name='Model'):
    """
    Compare distributions of SAD scores for functional vs non-functional variants
    """
    # Create binary labels
    log2fc_abs = np.abs(df[log2fc_col])
    threshold = np.percentile(log2fc_abs.dropna(), 75)
    df['Functional'] = (log2fc_abs >= threshold).map({True: 'Functional', False: 'Non-functional'})
    
    n_assays = len(assay_cols)
    n_cols = min(2, n_assays)
    n_rows = int(np.ceil(n_assays / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
    if n_assays == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, assay_col in enumerate(assay_cols):
        ax = axes[idx]
        assay_name = assay_col.replace('SAD_ASSAY_', '')
        
        # Remove NaN
        data = df[[assay_col, 'Functional']].dropna()
        
        # Violin plot
        parts = ax.violinplot([data[data['Functional'] == 'Non-functional'][assay_col].values,
                               data[data['Functional'] == 'Functional'][assay_col].values],
                              positions=[0, 1], showmeans=True, showmedians=True, widths=0.7)
        
        # Color the violins
        colors = [COLORS['neutral'], COLORS['warning']]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Style
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-functional', 'Functional'], fontsize=11)
        ax.set_ylabel('SAD Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{assay_name}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add statistical test
        nonfunc = data[data['Functional'] == 'Non-functional'][assay_col].values
        func = data[data['Functional'] == 'Functional'][assay_col].values
        
        if len(nonfunc) > 0 and len(func) > 0:
            u_stat, p_val = stats.mannwhitneyu(func, nonfunc, alternative='two-sided')
            sig_text = f'Mann-Whitney U\np = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.4f}'
            
            ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
                   fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Hide unused subplots
    for idx in range(n_assays, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{model_name}: SAD Score Distributions\n(Functional vs Non-functional Variants)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name.lower()}_distribution_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(df, assay_cols, log2fc_col, roc_results, output_dir, model_name='Model'):
    """
    Create summary statistics table
    """
    summary_data = []
    
    for idx, assay_col in enumerate(assay_cols):
        assay_name = assay_col.replace('SAD_ASSAY_', '')
        
        mask = ~(df[assay_col].isna() | df[log2fc_col].isna())
        x = df.loc[mask, assay_col].values
        y = df.loc[mask, log2fc_col].values
        
        if len(x) < 3:
            continue
        
        r, p_val = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)
        
        # Find matching ROC result
        roc_result = next((r for r in roc_results if r['assay'] == assay_name), None)
        
        summary_data.append({
            'Assay': assay_name,
            'N_Variants': len(x),
            'Pearson_r': f'{r:.4f}',
            'Spearman_rho': f'{rho:.4f}',
            'P_value': f'{p_val:.2e}' if p_val < 0.001 else f'{p_val:.4f}',
            'ROC_AUC': f"{roc_result['roc_auc']:.4f}" if roc_result else 'N/A',
            'Avg_Precision': f"{roc_result['avg_precision']:.4f}" if roc_result else 'N/A',
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'{model_name.lower()}_performance_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Create formatted table plot
    fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center', 
                    colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('white')
    
    plt.title(f'{model_name}: Performance Summary Table', 
             fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(output_dir, f'{model_name.lower()}_summary_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {table_path}")
    plt.close()
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality plots for Model-MPRA benchmark')
    parser.add_argument('--data', required=True, help='Path to merged_data.tsv')
    parser.add_argument('--output', default='Visualizations/figures', help='Output directory for plots')
    parser.add_argument('--model', default='Model', help='Model name (for labeling)')
    parser.add_argument('--cell_line', default='', help='Cell line name (optional, for titles)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    df = load_data(args.data)
    
    # Get columns
    assay_cols = get_assay_columns(df)
    log2fc_col = get_log2fc_column(df)
    
    # Model name with cell line if provided
    model_name = args.model
    if args.cell_line:
        model_name = f"{args.model} ({args.cell_line})"
    
    print(f"\nGenerating plots for: {model_name}")
    print("=" * 60)
    
    # Generate all plots
    print("\n1. Correlation Grid...")
    plot_enhanced_correlation_grid(df, assay_cols, log2fc_col, output_dir, model_name)
    
    print("\n2. Combined Correlation...")
    plot_combined_correlation(df, assay_cols, log2fc_col, output_dir, model_name)
    
    print("\n3. ROC & PR Curves...")
    roc_results = plot_roc_curves_enhanced(df, assay_cols, log2fc_col, output_dir, model_name)
    
    print("\n4. Performance Summary...")
    plot_performance_summary(roc_results, output_dir, model_name)
    
    print("\n5. Distribution Violin Plots...")
    plot_distribution_comparison(df, assay_cols, log2fc_col, output_dir, model_name)
    
    print("\n6. Summary Table...")
    summary_df = create_summary_table(df, assay_cols, log2fc_col, roc_results, output_dir, model_name)
    
    print("\n" + "=" * 60)
    print("✓ All plots generated successfully!")
    print(f"✓ Files saved to: {output_dir}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    

if __name__ == "__main__":
    main()
