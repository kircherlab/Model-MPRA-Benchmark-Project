#!/usr/bin/env python3
"""
Example: Generate Publication Plots
====================================

This script demonstrates how to use the publication plotting system
with example configurations and multiple scenarios.
"""

import os
import subprocess
import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def find_merged_data_files():
    """Find all merged_data.tsv files in the Pipeline results"""
    base_dir = Path(__file__).parent.parent / "Pipeline" / "results"
    
    if not base_dir.exists():
        return []
    
    return list(base_dir.glob("*/merged_data.tsv"))


def run_plotting_script(data_path, output_dir, model_name, cell_line=None):
    """Run the publication plotting script"""
    
    script_path = Path(__file__).parent / "publication_plots.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--data", str(data_path),
        "--output", str(output_dir),
        "--model", model_name
    ]
    
    if cell_line:
        cmd.extend(["--cell_line", cell_line])
    
    print_info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to generate plots: {e}")
        print(e.stderr)
        return False


def example_1_single_model():
    """Example 1: Generate plots for a single model"""
    print_header("Example 1: Single Model Visualization")
    
    # Find Enformer results
    data_path = Path(__file__).parent.parent / "Pipeline" / "results" / "enformer" / "merged_data.tsv"
    
    if not data_path.exists():
        print_warning(f"Enformer results not found at {data_path}")
        print_info("Please run the pipeline first or adjust the path.")
        return False
    
    output_dir = Path(__file__).parent / "examples" / "enformer_single"
    
    print_info(f"Input: {data_path}")
    print_info(f"Output: {output_dir}")
    
    success = run_plotting_script(data_path, output_dir, "Enformer", "HEK293T")
    
    if success:
        print_success("Example 1 complete!")
        print_info(f"View results in: {output_dir}")
    
    return success


def example_2_compare_models():
    """Example 2: Generate plots for multiple models to compare"""
    print_header("Example 2: Multi-Model Comparison")
    
    models = {
        "enformer": "Enformer",
        "basenji": "Basenji",
    }
    
    output_base = Path(__file__).parent / "examples" / "model_comparison"
    
    success_count = 0
    
    for model_dir, model_name in models.items():
        data_path = Path(__file__).parent.parent / "Pipeline" / "results" / model_dir / "merged_data.tsv"
        
        if not data_path.exists():
            print_warning(f"{model_name} results not found. Skipping.")
            continue
        
        output_dir = output_base / model_dir
        
        print_info(f"Processing {model_name}...")
        success = run_plotting_script(data_path, output_dir, model_name)
        
        if success:
            success_count += 1
            print_success(f"{model_name} plots generated!")
    
    print_info(f"Generated plots for {success_count}/{len(models)} models")
    
    return success_count > 0


def example_3_batch_all():
    """Example 3: Batch generate for all available results"""
    print_header("Example 3: Batch Generation (All Results)")
    
    data_files = find_merged_data_files()
    
    if not data_files:
        print_warning("No merged_data.tsv files found in Pipeline/results/")
        print_info("Run the pipeline first to generate results.")
        return False
    
    print_info(f"Found {len(data_files)} result file(s)")
    
    output_base = Path(__file__).parent / "examples" / "batch_all"
    success_count = 0
    
    for data_path in data_files:
        # Extract model name from path
        model_dir = data_path.parent.name
        model_name = model_dir.capitalize()
        
        output_dir = output_base / model_dir
        
        print_info(f"Processing {model_name} ({data_path})...")
        success = run_plotting_script(data_path, output_dir, model_name)
        
        if success:
            success_count += 1
            print_success(f"{model_name} complete!")
    
    print_info(f"Successfully generated plots for {success_count}/{len(data_files)} datasets")
    
    return success_count > 0


def example_4_custom_output():
    """Example 4: Custom output directory (for presentations)"""
    print_header("Example 4: Custom Output for Presentation")
    
    data_path = Path(__file__).parent.parent / "Pipeline" / "results" / "enformer" / "merged_data.tsv"
    
    if not data_path.exists():
        print_warning(f"Enformer results not found. Skipping example 4.")
        return False
    
    # Custom output directory (e.g., for a specific presentation)
    output_dir = Path(__file__).parent.parent / "Presentations" / "figures" / "2025_lab_meeting"
    
    print_info(f"Generating presentation-ready figures...")
    print_info(f"Output: {output_dir}")
    
    success = run_plotting_script(data_path, output_dir, "Enformer", "HEK293T")
    
    if success:
        print_success("Presentation figures ready!")
        print_info(f"Copy from: {output_dir}")
    
    return success


def main():
    print_header("Publication Plots - Examples")
    
    print("This script demonstrates different ways to use the publication plotting system.\n")
    
    print("Available examples:")
    print("  1. Single model visualization")
    print("  2. Compare multiple models")
    print("  3. Batch generate all available results")
    print("  4. Custom output directory (for presentations)")
    print("  5. Run all examples")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == "0":
                print_info("Exiting...")
                break
            elif choice == "1":
                example_1_single_model()
            elif choice == "2":
                example_2_compare_models()
            elif choice == "3":
                example_3_batch_all()
            elif choice == "4":
                example_4_custom_output()
            elif choice == "5":
                print_header("Running All Examples")
                example_1_single_model()
                example_2_compare_models()
                example_3_batch_all()
                example_4_custom_output()
                print_success("All examples complete!")
                break
            else:
                print_error("Invalid choice. Please enter 0-5.")
        
        except KeyboardInterrupt:
            print("\n" + Colors.WARNING + "Interrupted by user." + Colors.ENDC)
            break
        except Exception as e:
            print_error(f"An error occurred: {e}")
    
    print("\n" + Colors.OKGREEN + "Thank you for using the publication plotting system!" + Colors.ENDC)


if __name__ == "__main__":
    main()
