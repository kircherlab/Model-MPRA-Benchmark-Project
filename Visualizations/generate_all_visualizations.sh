#!/bin/bash
#
# generate_all_visualizations.sh
# Batch generate publication-quality plots for all models and cell lines
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_DIR="${SCRIPT_DIR}/../Pipeline"
OUTPUT_BASE="${SCRIPT_DIR}/figures"

echo "=========================================="
echo "Model-MPRA Benchmark: Visualization Generator"
echo "=========================================="
echo ""

# Check if results exist
if [ ! -d "${PIPELINE_DIR}/results" ]; then
    echo "ERROR: No results directory found at ${PIPELINE_DIR}/results"
    echo "Please run the pipeline first to generate results."
    exit 1
fi

# Create base output directory
mkdir -p "${OUTPUT_BASE}"

# Counter for generated plots
PLOT_COUNT=0

# Find all merged_data.tsv files
echo "Searching for merged_data.tsv files..."
echo ""

for merged_file in $(find "${PIPELINE_DIR}/results" -name "merged_data.tsv"); do
    # Extract model name and create output directory
    MODEL=$(echo "${merged_file}" | sed 's|.*/results/\([^/]*\)/.*|\1|')
    MODEL_UPPER=$(echo "${MODEL}" | sed 's/.*/\u&/')  # Capitalize first letter
    
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
    
    echo "Found: ${MODEL}"
    echo "  Input: ${merged_file}"
    echo "  Output: ${OUTPUT_DIR}"
    
    # Check if file has data
    LINE_COUNT=$(wc -l < "${merged_file}")
    if [ "${LINE_COUNT}" -lt 2 ]; then
        echo "  WARNING: File appears empty (${LINE_COUNT} lines). Skipping."
        echo ""
        continue
    fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Generate plots
    echo "  Generating plots..."
    python "${SCRIPT_DIR}/publication_plots.py" \
        --data "${merged_file}" \
        --output "${OUTPUT_DIR}" \
        --model "${MODEL_UPPER}"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success!"
        ((PLOT_COUNT++))
    else
        echo "  ✗ Failed to generate plots for ${MODEL}"
    fi
    
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Generated plots for ${PLOT_COUNT} model(s)"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# List generated files
if [ ${PLOT_COUNT} -gt 0 ]; then
    echo "Generated files:"
    find "${OUTPUT_BASE}" -name "*.png" -o -name "*.csv" | sort
    echo ""
    echo "✓ All visualizations complete!"
else
    echo "No plots were generated. Please check that:"
    echo "  1. Pipeline has been run successfully"
    echo "  2. merged_data.tsv files exist in results directories"
    echo "  3. Python environment has required packages (pandas, matplotlib, seaborn, sklearn)"
fi
