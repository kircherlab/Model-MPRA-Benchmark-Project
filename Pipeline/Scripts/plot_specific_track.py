#!/usr/bin/env python3
"""
Plot specific track SAD values against MPRA log2FC for model comparison.

This script creates scatter plots comparing model predictions (SAD scores)
for a specific experimental track (e.g., DNASE:HepG2) against MPRA log2FC values.
Each enabled model gets its own plot for independent comparison.

Usage:
    python plot_specific_track.py \
        --track "DNASE:HepG2" \
        --output results/track_plots/ \
        --config Configs/config.yaml \
        [--enformer results/enformer/merged_data.tsv] \
        [--basenji results/basenji/merged_data.tsv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import yaml
import os
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def determine_significance(df, sig_config):
    """
    Determine which variants are significant based on PVAL/QVAL thresholds.
    
    Args:
        df: DataFrame with variant data
        sig_config: Significance configuration dict
        
    Returns:
        pd.Series: Boolean series indicating significant variants
    """
    if not sig_config.get('use_significance_coloring', False):
        return pd.Series([False] * len(df), index=df.index)
    
    statistic = sig_config.get('statistic', 'QVAL').upper()
    threshold = sig_config.get('threshold', 1.3)
    
    # Check if required columns exist
    has_pval = 'PVAL' in df.columns
    has_qval = 'QVAL' in df.columns
    
    if statistic == 'PVAL':
        if not has_pval:
            print("Warning: PVAL column not found. Disabling significance coloring.")
            return pd.Series([False] * len(df), index=df.index)
        return df['PVAL'] >= threshold
    
    elif statistic == 'QVAL':
        if not has_qval:
            print("Warning: QVAL column not found. Disabling significance coloring.")
            return pd.Series([False] * len(df), index=df.index)
        return df['QVAL'] >= threshold
    
    elif statistic == 'both':
        pval_thresh = sig_config.get('pval_threshold', 1.3)
        qval_thresh = sig_config.get('qval_threshold', 1.3)
        if not has_pval or not has_qval:
            print("Warning: PVAL or QVAL column missing. Disabling significance coloring.")
            return pd.Series([False] * len(df), index=df.index)
        return (df['PVAL'] >= pval_thresh) & (df['QVAL'] >= qval_thresh)
    
    return pd.Series([False] * len(df), index=df.index)

def plot_track_correlation(df, track_name, model_name, output_path, sig_config):
    """
    Create scatter plot of track SAD values vs MPRA log2FC.
    
    Args:
        df: DataFrame with merged data
        track_name: Name of the track column to plot
        model_name: Name of the model (for title)
        output_path: Where to save the plot
        sig_config: Significance configuration
    """
    # Find LOG2FC column (case-insensitive)
    log2fc_col = None
    for col in df.columns:
        if col.lower() == 'log2fc':
            log2fc_col = col
            break
    
    if log2fc_col is None:
        raise ValueError("LOG2FC column not found in data")
    
    if track_name not in df.columns:
        raise ValueError(f"Track '{track_name}' not found in data. Available tracks: {[c for c in df.columns if ':' in c][:10]}...")
    
    # Prepare data
    plot_df = df[[log2fc_col, track_name]].copy()
    plot_df = plot_df.dropna()
    
    # Determine significance
    is_significant = determine_significance(df, sig_config).loc[plot_df.index]
    
    # Get colors from config
    colors = sig_config.get('colors', {})
    sig_color = colors.get('significant', '#E74C3C')
    nonsig_color = colors.get('nonsignificant', '#95A5A6')
    sig_alpha = colors.get('significant_alpha', 0.7)
    nonsig_alpha = colors.get('nonsignificant_alpha', 0.3)
    
    # Calculate correlation
    correlation = plot_df[log2fc_col].corr(plot_df[track_name])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if sig_config.get('use_significance_coloring', False):
        # Plot non-significant points
        nonsig_data = plot_df[~is_significant]
        if len(nonsig_data) > 0:
            ax.scatter(nonsig_data[log2fc_col], nonsig_data[track_name],
                      alpha=nonsig_alpha, s=30, color=nonsig_color,
                      label=f'Non-significant (n={len(nonsig_data)})')
        
        # Plot significant points
        sig_data = plot_df[is_significant]
        if len(sig_data) > 0:
            ax.scatter(sig_data[log2fc_col], sig_data[track_name],
                      alpha=sig_alpha, s=30, color=sig_color,
                      label=f'Significant (n={len(sig_data)})')
    else:
        # Single color for all points
        ax.scatter(plot_df[log2fc_col], plot_df[track_name],
                  alpha=0.5, s=30, color='#3498DB')
    
    # Add regression line
    z = np.polyfit(plot_df[log2fc_col], plot_df[track_name], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df[log2fc_col].min(), plot_df[log2fc_col].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Regression line')
    
    # Add correlation text
    ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}\nn = {len(plot_df)}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and title
    ax.set_xlabel('MPRA log2FC', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{model_name} SAD ({track_name})', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name}: {track_name} vs MPRA', fontsize=16, fontweight='bold')
    
    if sig_config.get('use_significance_coloring', False):
        ax.legend(loc='lower right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {model_name} plot to {output_path}")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  n = {len(plot_df)}")

def main():
    parser = argparse.ArgumentParser(description='Plot specific track SAD values against MPRA log2FC')
    parser.add_argument('--track', required=True, type=str,
                       help='Track name to plot (e.g., "DNASE:HepG2")')
    parser.add_argument('--output', required=True, type=str,
                       help='Output directory or file path')
    parser.add_argument('--config', required=True, type=str,
                       help='Path to config.yaml')
    parser.add_argument('--enformer', type=str, default=None,
                       help='Path to Enformer merged_data.tsv')
    parser.add_argument('--basenji', type=str, default=None,
                       help='Path to Basenji merged_data.tsv')
    parser.add_argument('--alphagenome', type=str, default=None,
                       help='Path to AlphaGenome merged_data.tsv')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    sig_config = config.get('significance', {})
    
    # Set up output paths
    if args.output.endswith('.png'):
        # Single output file specified - use it directly
        output_path = args.output
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine which model this is for based on which data file is provided
        models = []
        if args.enformer:
            models.append(('Enformer', args.enformer))
        if args.basenji:
            models.append(('Basenji', args.basenji))
        if args.alphagenome:
            models.append(('AlphaGenome', args.alphagenome))
        
        if len(models) == 0:
            print("Error: No model data provided")
            return 1
        
        if len(models) > 1:
            print("Warning: Multiple models provided but single output file specified. Using first model only.")
        
        model_name, data_path = models[0]
        
        print(f"Processing {model_name}...")
        
        if not os.path.exists(data_path):
            print(f"Error: {model_name} data file not found: {data_path}")
            return 1
        
        # Load data
        df = pd.read_csv(data_path, sep='\t')
        
        # Create plot
        try:
            plot_track_correlation(df, args.track, model_name, output_path, sig_config)
            print(f"\n✓ Successfully generated plot: {output_path}")
            return 0
        except Exception as e:
            print(f"Error plotting {model_name}: {e}")
            return 1
    
    else:
        # Output is a directory - generate plots for all provided models
        output_dir = args.output
        base_name = args.track.replace(':', '_').replace(' ', '_')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each model
        models = {
            'Enformer': args.enformer,
            'Basenji': args.basenji,
            'AlphaGenome': args.alphagenome
        }
        
        generated_plots = []
        
        for model_name, data_path in models.items():
            if data_path is None:
                continue
            
            if not os.path.exists(data_path):
                print(f"Warning: {model_name} data file not found: {data_path}")
                continue
            
            print(f"\nProcessing {model_name}...")
            
            # Load data
            df = pd.read_csv(data_path, sep='\t')
            
            # Generate output path
            output_path = os.path.join(output_dir, f"{base_name}_{model_name.lower()}.png")
            
            # Create plot
            try:
                plot_track_correlation(df, args.track, model_name, output_path, sig_config)
                generated_plots.append(output_path)
            except Exception as e:
                print(f"Error plotting {model_name}: {e}")
        
        if generated_plots:
            print(f"\n✓ Successfully generated {len(generated_plots)} plot(s):")
            for plot_path in generated_plots:
                print(f"  - {plot_path}")
        else:
            print("\n✗ No plots were generated. Check your input files and track name.")
            return 1
        
        return 0

if __name__ == '__main__':
    exit(main())
