"""
3D Scatter Plot: Computational Cost (FLOPs/W) vs Prediction Accuracy (R²) vs Annual Penalty Reduction
Based on battle_economics.py data
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# Set English fonts to avoid garbled characters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_economics_data():
    """Load economic data"""
    # Extract data from battle_economics.py results
    # Here we use simulated data, should be loaded from files in practice
    models = ['Ultra-LSNT', 'DLinear', 'Transformer', 'Informer', 'Mamba_S6']
    
    # Simulated data (should be loaded from experimental results files)
    data = {
        'Ultra-LSNT': {
            'R2': 0.8728,
            'FLOPs': 1.2e9,  # Estimated value
            'Power_W': 3.8,
            'Annual_Penalty_Reduction': 150000,  # Annual penalty reduction (USD)
            'Cost_Reduction_Percent': 12.8
        },
        'DLinear': {
            'R2': 0.9400,
            'FLOPs': 0.8e9,
            'Power_W': 2.1,
            'Annual_Penalty_Reduction': 0,  # Baseline
            'Cost_Reduction_Percent': 0
        },
        'Transformer': {
            'R2': 0.7439,
            'FLOPs': 5.4e9,
            'Power_W': 8.6,
            'Annual_Penalty_Reduction': 80000,
            'Cost_Reduction_Percent': 6.5
        },
        'Informer': {
            'R2': 0.8167,
            'FLOPs': 4.2e9,
            'Power_W': 7.8,
            'Annual_Penalty_Reduction': 90000,
            'Cost_Reduction_Percent': 7.2
        },
        'Mamba_S6': {
            'R2': 0.8500,  # Estimated value
            'FLOPs': 1.5e9,
            'Power_W': 4.2,
            'Annual_Penalty_Reduction': 120000,
            'Cost_Reduction_Percent': 9.5
        }
    }
    
    # 尝试从实际文件加载
    try:
        with open('battle_economics_results.json', 'r') as f:
            real_data = json.load(f)
            # 合并数据
            for model in real_data:
                if model in data:
                    data[model].update(real_data[model])
    except FileNotFoundError:
        print("Warning: battle_economics_results.json not found, using simulated data")
    
    return models, data

def plot_3d_scatter():
    """Plot 3D scatter plot"""
    models, model_data = load_economics_data()
    
    # Prepare data
    x_flops = []
    y_r2 = []
    z_penalty = []
    sizes = []
    colors = []
    
    color_map = {
        'Ultra-LSNT': 'red',
        'DLinear': 'blue',
        'Transformer': 'green',
        'Informer': 'orange',
        'Mamba_S6': 'purple'
    }
    
    for model in models:
        if model in model_data:
            data = model_data[model]
            # X: Computational efficiency (FLOPs/W, lower is better)
            efficiency = data['FLOPs'] / max(data['Power_W'], 0.1)
            x_flops.append(efficiency / 1e8)  # Scale for display
            
            # Y: Prediction accuracy (R²)
            y_r2.append(data['R2'])
            
            # Z: Annual penalty reduction (thousand USD)
            z_penalty.append(data['Annual_Penalty_Reduction'] / 1000)
            
            # Point size: cost reduction percentage
            sizes.append(50 + data['Cost_Reduction_Percent'] * 10)
            
            colors.append(color_map.get(model, 'gray'))
    
    # 创建三维图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点
    scatter = ax.scatter(x_flops, y_r2, z_penalty, 
                        s=sizes, c=colors, alpha=0.8, depthshade=True)
    
    # 添加标签（英文以避免字体警告）
    ax.set_xlabel('Computational Cost (FLOPs/W ×10⁸) ↓', fontsize=12, labelpad=15)
    ax.set_ylabel('Prediction Accuracy (R²) ↑', fontsize=12, labelpad=15)
    ax.set_zlabel('Annual Penalty Reduction (kUSD) ↑', fontsize=12, labelpad=15)
    
    ax.set_title('3D Performance Analysis: Cost vs Accuracy vs Economic Benefit', fontsize=14, pad=20)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = []
    for model, color in color_map.items():
        if model in models:
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=model))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # 添加网格和视角
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.view_init(elev=25, azim=45)  # 调整视角
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('3d_performance_scatter.png', dpi=300, bbox_inches='tight')
    print("3D scatter plot saved as 3d_performance_scatter.png")
    
    # 创建二维投影图
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: FLOPs/W vs R²
    ax1 = axes[0, 0]
    for i, model in enumerate(models):
        if model in model_data:
            ax1.scatter(x_flops[i], y_r2[i], s=sizes[i], c=colors[i], label=model, alpha=0.7)
    ax1.set_xlabel('Computational Cost (FLOPs/W ×10⁸) ↓')
    ax1.set_ylabel('Prediction Accuracy (R²) ↑')
    ax1.set_title('(a) Computational Efficiency vs Prediction Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: R² vs Penalty Reduction
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        if model in model_data:
            ax2.scatter(y_r2[i], z_penalty[i], s=sizes[i], c=colors[i], label=model, alpha=0.7)
    ax2.set_xlabel('Prediction Accuracy (R²) ↑')
    ax2.set_ylabel('Annual Penalty Reduction (kUSD) ↑')
    ax2.set_title('(b) Prediction Accuracy vs Economic Benefit')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: FLOPs/W vs Penalty Reduction
    ax3 = axes[1, 0]
    for i, model in enumerate(models):
        if model in model_data:
            ax3.scatter(x_flops[i], z_penalty[i], s=sizes[i], c=colors[i], label=model, alpha=0.7)
    ax3.set_xlabel('Computational Cost (FLOPs/W ×10⁸) ↓')
    ax3.set_ylabel('Annual Penalty Reduction (kUSD) ↑')
    ax3.set_title('(c) Computational Efficiency vs Economic Benefit')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: Pareto Frontier
    ax4 = axes[1, 1]
    # Calculate comprehensive score (higher is better)
    for i, model in enumerate(models):
        if model in model_data:
            # Normalize scores
            norm_r2 = (y_r2[i] - min(y_r2)) / (max(y_r2) - min(y_r2) + 1e-8)
            norm_penalty = (z_penalty[i] - min(z_penalty)) / (max(z_penalty) - min(z_penalty) + 1e-8)
            norm_efficiency = 1 - (x_flops[i] - min(x_flops)) / (max(x_flops) - min(x_flops) + 1e-8)
            
            total_score = norm_r2 * 0.4 + norm_penalty * 0.4 + norm_efficiency * 0.2
            ax4.scatter(i, total_score, s=150, c=colors[i], label=model, alpha=0.8)
            ax4.text(i, total_score + 0.02, f'{total_score:.2f}', ha='center', fontsize=9)
    
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylabel('Comprehensive Performance Score ↑')
    ax4.set_title('(d) Overall Performance Ranking')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis_grid.png', dpi=300, bbox_inches='tight')
    print("Performance analysis grid saved as performance_analysis_grid.png")
    
    # 保存数据到JSON
    output_data = {}
    for i, model in enumerate(models):
        if model in model_data:
            output_data[model] = {
                'R2': y_r2[i],
                'FLOPs_per_Watt': x_flops[i] * 1e8,
                'Annual_Penalty_Reduction_USD': z_penalty[i] * 1000,
                'Cost_Reduction_Percent': model_data[model]['Cost_Reduction_Percent']
            }
    
    with open('performance_metrics_3d.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Performance metrics saved as performance_metrics_3d.json")

if __name__ == '__main__':
    plot_3d_scatter()
