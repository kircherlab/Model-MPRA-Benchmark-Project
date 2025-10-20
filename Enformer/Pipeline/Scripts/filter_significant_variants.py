import pandas as pd
import numpy as np

input_file = '/data/humangen_kircherlab/Projects/Model_MPRA_Benchmark/02_Analysis/Enformer/Pipeline/Data/MPRA/HEK293T_reporter_variants.tsv' 

# Set the name for the output file.
output_file = 'filtered_significant_variants.tsv'

# Threshold for effect size (a value of 1.0 means a 2-fold change).
log2fc_threshold = 0.585  # Corresponds to a 1.5-fold change.

# Threshold for statistical significance (a q-value of 0.05 is standard).
q_value_threshold = 0.10 


# --- 2. Load and Filter the Data ---

try:
   
    df = pd.read_csv(input_file, sep='\t')

    # Convert the q-value threshold to the -log10 scale to match the data column.
    minus_log10_q_threshold = -np.log10(q_value_threshold)

    significant_variants = df[
        (df['minusLog10QValue'] > minus_log10_q_threshold) &
        (abs(df['log2FoldChange']) > log2fc_threshold)
    ]

    # --- 3. Display and Save Results ---
    print("--- MPRA Filtering Summary ---")
    print(f"Original number of variants: {len(df)}")
    print(f"Variants with q-value < {q_value_threshold}: { (df['minusLog10QValue'] > minus_log10_q_threshold).sum() }")
    print(f"Variants with |log2FC| > {log2fc_threshold}: { (abs(df['log2FoldChange']) > log2fc_threshold).sum() }")
    print(f"Number of significant variants passing both filters: {len(significant_variants)}")

    print(f"\\nSaving {len(significant_variants)} significant variants to '{output_file}'...")
    significant_variants.to_csv(output_file, index=False)
    
    print("\n--- Top 5 Significant Variants ---")
    print(significant_variants.head())

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure the file name and path are correct.")
except KeyError as e:
    print(f"Error: A required column was not found in the file: {e}")
    print("Please ensure your file has the correct headers (e.g., 'minusLog10QValue', 'log2FoldChange').")