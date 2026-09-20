"""
Elsevier 期刊样式配置
适用于 Applied Energy 等 Elsevier 期刊的图表规范
"""
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局样式，符合 Elsevier 标准
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],   # 指定 Arial 字体
    'font.size': 8,                 # 基础字号 8pt
    'axes.labelsize': 9,            # 轴标签 9pt
    'axes.titlesize': 10,           # 标题 10pt
    'legend.fontsize': 7,           # 图例 7pt
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,              # 预览 DPI
    'savefig.dpi': 1000,            # 导出 DPI (线图)
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

# 定义色板 (Colorblind Safe)
palette = {
    'Ultra-LSNT': '#0072B2',  # 深蓝
    'LightGBM': '#E69F00',    # 橙色
    'DLinear': '#009E73',     # 蓝绿
    'Transformer': '#999999',  # 灰色
    'PatchTST': '#CC79A7',    # 粉色
    'Informer': '#56B4E9',    # 浅蓝
    'Autoformer': '#F0E442',  # 黄色
    'TimeMixer': '#D55E00',   # 红棕色
    'Mamba': '#0072B2',       # 同 Ultra-LSNT 色系
}

# 导出函数
def save_fig(fig, filename):
    """保存图表为 PDF 和 EPS 格式，符合期刊要求"""
    fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    # 同时导出 EPS 用于排版
    fig.savefig(f'{filename}.eps', format='eps', bbox_inches='tight')
    print(f"图表已保存: {filename}.pdf, {filename}.eps")

def apply_style():
    """应用样式并返回当前配置"""
    print("Elsevier 样式已应用: Arial 字体, 8pt 基础字号, 1000 DPI 导出")
    return plt.rcParams

if __name__ == '__main__':
    apply_style()
    print("样式配置测试完成")