#!/bin/bash
# run_plot.sh - Batch script to run all plotting scripts and regenerate figures
# This script runs all plotting scripts sequentially to regenerate figures with English text

echo "=============================================="
echo "Starting batch plotting script"
echo "Date: $(date)"
echo "=============================================="

# Change to the script directory (assumed to be the same as this script)
cd "$(dirname "$0")" || exit 1

# Function to run a Python script with error handling
run_python_script() {
    local script_name=$1
    local description=$2
    
    echo ""
    echo "Running: $description"
    echo "Script: $script_name"
    echo "----------------------------------------------"
    
    if [ ! -f "$script_name" ]; then
        echo "WARNING: Script $script_name not found, skipping..."
        return 1
    fi
    
    # Run the script
    python3 "$script_name"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $script_name completed"
    else
        echo "❌ FAILED: $script_name exited with code $exit_code"
        return 1
    fi
    
    return 0
}

# Create necessary directories
mkdir -p figures
mkdir -p images
mkdir -p engineering_charts

# List of plotting scripts to run (in logical order)
echo "Running plotting scripts in sequence..."

# 1. Basic performance comparison
run_python_script "plot_results.py" "Performance comparison charts"

# 2. Dispatch comparison charts
run_python_script "plot_dispatch_comparison.py" "Dispatch comparison charts"

# 3. Engineering charts (comprehensive)
run_python_script "engineering_charts.py" "Engineering comparison charts"

# 4. 3D scatter plot
run_python_script "plot_3d_scatter.py" "3D performance scatter plot"

# 5. Architecture diagram
run_python_script "plot_arch.py" "Model architecture diagram"

# 6. Paper figures
run_python_script "plot_paper_figs.py" "Paper figures for publication"

# 7. Efficiency and expert analysis
run_python_script "test_efficiency.py" "Efficiency boundary comparison"
run_python_script "plot_expert_heatmap.py" "Expert activation heatmap"
run_python_script "plot_expert_heatmap_enhanced.py" "Enhanced expert comparison heatmap"

# 8. Battle and robustness figures
run_python_script "battle_dlinear.py" "DLinear robustness battle"
run_python_script "compare_robustness_real.py" "Transformer robustness comparison"
run_python_script "run_gbdt_robustness.py" "LightGBM robustness battle"
run_python_script "battle_economics.py" "Economic impact analysis"
run_python_script "plot_dlinear_victory.py" "DLinear catastrophic failure visualization"
run_python_script "real_robustness.py" "Real data robustness analysis"
run_python_script "run_universal_robustness.py" "Universal robustness across domains"

# 9. Multi-domain and advanced figures
run_python_script "plot_multi_domain.py" "Multi-domain performance comparison"
run_python_script "generate_figures.py" "Applied Energy style figures"
run_python_script "assemble_figures.py" "Assemble final paper figures"

# 10. Run all figures (if exists)
if [ -f "run_all_figures.py" ]; then
    run_python_script "run_all_figures.py" "All figure generation pipeline"
fi

echo ""
echo "=============================================="
echo "Plotting script execution completed"
echo "=============================================="
echo ""
echo "Generated figures summary:"
echo "--------------------------"
ls -la figures/*.png 2>/dev/null | wc -l | xargs echo "Number of figures in 'figures/' directory:"
ls -la *.png 2>/dev/null | wc -l | xargs echo "Number of figures in current directory:"
ls -la images/*.png 2>/dev/null | wc -l | xargs echo "Number of figures in 'images/' directory:"
ls -la engineering_charts/*.png 2>/dev/null | wc -l | xargs echo "Number of figures in 'engineering_charts/' directory:"

echo ""
echo "Important files generated:"
if [ -f "figures/performance_comparison.png" ]; then
    echo "  ✓ figures/performance_comparison.png"
fi
if [ -f "dispatch_comparison_charts.png" ]; then
    echo "  ✓ dispatch_comparison_charts.png"
fi
if [ -f "dispatch_detailed_analysis.png" ]; then
    echo "  ✓ dispatch_detailed_analysis.png"
fi
if [ -f "engineering_charts/overall_performance.png" ]; then
    echo "  ✓ engineering_charts/overall_performance.png"
fi
if [ -f "3d_performance_scatter.png" ]; then
    echo "  ✓ 3d_performance_scatter.png"
fi
if [ -f "fig_architecture.png" ]; then
    echo "  ✓ fig_architecture.png"
fi
if [ -f "fig_comparison.png" ]; then
    echo "  ✓ fig_comparison.png"
fi

echo ""
echo "Note: All Chinese text in figures has been replaced with English text."
echo "To verify, open any generated PNG file and check for proper English labels."
echo "=============================================="