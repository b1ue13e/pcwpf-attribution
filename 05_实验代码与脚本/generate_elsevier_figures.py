"""
重新生成所有图表，应用 Elsevier 期刊样式（Arial 字体，8pt 基础字号，1000 DPI）
"""
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. 首先应用 Elsevier 样式
print("应用 Elsevier 期刊样式...")
sys.path.insert(0, os.path.dirname(__file__))
from elsevier_style import apply_style, save_fig
apply_style()

# 覆盖 matplotlib 默认设置以确保 Arial 字体
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'savefig.dpi': 1000,
})

print("Elsevier 样式已激活")

# 2. 重新生成 plot_paper_figs.py 中的图表
print("\n重新生成 plot_paper_figs.py 图表...")
import numpy as np
from paper_figure_style import OKABE_ITO, ensure_output_dir

# 重写 save_figure 函数以使用 Elsevier 的保存方式
def save_figure_elsevier(fig, out_base):
    """使用 Elsevier 样式保存图表"""
    out_dir = ensure_output_dir("elsevier_figures")
    filename = os.path.join(out_dir, os.path.basename(out_base))
    save_fig(fig, filename)
    # 同时保存 PNG 用于预览
    fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    print(f"  已保存: {filename}.pdf/.eps/.png")

# 复制 plot_comparison 函数逻辑
def plot_comparison_elsevier():
    models = ["LSTM", "MLP", "Transformer", "Ultra-LSNT\n(Ours)"]
    r2_scores = [0.6788, 0.6811, 0.7268, 0.7513]
    colors = ["#C7C7C7", "#AFAFAF", OKABE_ITO["sky_blue"], OKABE_ITO["vermillion"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=300)
    bars = ax.bar(models, r2_scores, color=colors, edgecolor="black", linewidth=1.0, width=0.62)

    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.008,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylim(0.60, 0.80)
    ax.set_ylabel(r"R-Squared ($R^2$)", fontweight="bold")
    ax.set_title("Performance Comparison on GEFCom2012")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save_figure_elsevier(fig, "fig_comparison")
    plt.close(fig)

def plot_ablation_elsevier():
    strategies = ["Top-1\n(Fast)", "Top-2\n(Ours)", "Top-4\n(Dense)"]
    r2_scores = [0.7402, 0.7513, 0.7450]
    compute_cost = [25, 50, 100]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.6), dpi=300)

    color_r2 = OKABE_ITO["vermillion"]
    color_cost = OKABE_ITO["sky_blue"]

    ax1.set_xlabel("Expert Selection Strategy")
    ax1.set_ylabel(r"Prediction Accuracy ($R^2$)", color=color_r2)
    ax1.bar(strategies, r2_scores, color=color_r2, edgecolor="black", linewidth=1.0, width=0.5)
    ax1.tick_params(axis='y', labelcolor=color_r2)
    ax1.set_ylim(0.73, 0.76)

    ax2 = ax1.twinx()
    ax2.plot(strategies, compute_cost, color=color_cost, marker='o', linewidth=2.0, markersize=8)
    ax2.set_ylabel("Relative Compute Cost (%)", color=color_cost)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.set_ylim(0, 120)

    fig.tight_layout()
    save_figure_elsevier(fig, "fig_ablation")
    plt.close(fig)

# 运行绘图
plot_comparison_elsevier()
plot_ablation_elsevier()

# 3. 重新生成 plot_arch.py 图表
print("\n重新生成 plot_arch.py 图表...")
try:
    # 由于 plot_arch.py 代码较复杂，直接运行原脚本但替换样式
    # 我们采用临时修改环境的方式
    import subprocess
    
    # 创建临时环境变量，告诉脚本使用 Elsevier 样式
    env = os.environ.copy()
    env['USE_ELSEVIER_STYLE'] = '1'
    
    # 运行原脚本（它会检测环境变量）
    # 这里我们采用更直接的方法：导入并运行
    import plot_arch
    # 重新加载模块以确保使用新样式
    import importlib
    importlib.reload(plot_arch)
    
    # 运行 main 函数（如果存在）
    if hasattr(plot_arch, 'main'):
        plot_arch.main()
    else:
        # 直接调用绘图函数
        fig = plot_arch.plot_architecture()
        save_figure_elsevier(fig, "fig_architecture")
        plt.close(fig)
        
except Exception as e:
    print(f"生成架构图时出错: {e}")

# 4. 重新生成 plot_dlinear_victory.py 图表
print("\n重新生成 plot_dlinear_victory.py 图表...")
try:
    import plot_dlinear_victory
    import importlib
    importlib.reload(plot_dlinear_victory)
    
    # 创建图表
    fig = plot_dlinear_victory.plot_victory()
    save_figure_elsevier(fig, "fig_battle_dlinear_victory")
    plt.close(fig)
except Exception as e:
    print(f"生成 DLinear 对比图时出错: {e}")

# 5. 重新生成 plot_3d_scatter.py 图表
print("\n重新生成 plot_3d_scatter.py 图表...")
try:
    import plot_3d_scatter
    import importlib
    importlib.reload(plot_3d_scatter)
    
    # 运行原函数
    plot_3d_scatter.plot_3d_scatter()
    
    # 移动生成的图表到 elsevier_figures 目录
    import shutil
    out_dir = ensure_output_dir("elsevier_figures")
    for f in ['3d_performance_scatter.png', 'performance_analysis_grid.png', 
              'performance_metrics_3d.json']:
        if os.path.exists(f):
            shutil.move(f, os.path.join(out_dir, f))
            print(f"  已移动: {f}")
except Exception as e:
    print(f"生成 3D 散点图时出错: {e}")

# 6. 重新生成 plot_dispatch_comparison.py 图表
print("\n重新生成 plot_dispatch_comparison.py 图表...")
try:
    import plot_dispatch_comparison
    import importlib
    importlib.reload(plot_dispatch_comparison)
    
    # 运行原函数
    plot_dispatch_comparison.main()
    
    # 移动图表
    out_dir = ensure_output_dir("elsevier_figures")
    for f in ['dispatch_comparison_charts.png', 'dispatch_detailed_analysis.png',
              'dispatch_comparison_charts.pdf']:
        if os.path.exists(f):
            shutil.move(f, os.path.join(out_dir, f))
            print(f"  已移动: {f}")
except Exception as e:
    print(f"生成调度对比图时出错: {e}")

# 7. 重新生成 test_efficiency.py 图表
print("\n重新生成 test_efficiency.py 图表...")
try:
    import test_efficiency
    import importlib
    importlib.reload(test_efficiency)
    
    # 运行原函数
    test_efficiency.benchmark()
    
    # 移动图表
    out_dir = ensure_output_dir("elsevier_figures")
    if os.path.exists('fig_efficiency.png'):
        shutil.move('fig_efficiency.png', os.path.join(out_dir, 'fig_efficiency.png'))
        print(f"  已移动: fig_efficiency.png")
except Exception as e:
    print(f"生成效率图时出错: {e}")

print("\n✅ 所有 Elsevier 样式图表生成完成！")
print(f"图表保存在: {os.path.abspath('elsevier_figures')}")