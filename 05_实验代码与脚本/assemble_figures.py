import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys


# 确保能运行你的脚本
def run_script(script_name):
    print(f"🚀 Running {script_name}...")
    os.system(f"python {script_name}")


def assemble_figure_2():
    print("\n🧩 Assembling Figure 2 (Performance & Efficiency)...")

    # 1. 运行子脚本生成素材
    run_script("plot_paper_figs.py")  # 生成 fig_comparison.png, fig_ablation.png
    run_script("plot_expert_heatmap.py")  # 生成 fig_real_expert_heatmap.png
    # 注意：效率图 fig_efficiency.png 需要由 test_efficiency.py 生成
    if not os.path.exists("fig_efficiency.png"):
        run_script("test_efficiency.py")

    # 2. 检查文件是否存在
    files = ["fig_comparison.png", "fig_efficiency.png", "fig_ablation.png", "fig_real_expert_heatmap.png"]
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Missing {f}, skipping assembly.")
            return

    # 3. 创建 2x2 画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # 加载图片
    img_a = mpimg.imread("fig_comparison.png")
    img_b = mpimg.imread("fig_efficiency.png")
    img_c = mpimg.imread("fig_ablation.png")
    img_d = mpimg.imread("fig_real_expert_heatmap.png")

    # 填入画布
    # 注意：这里直接显示图片。为了更完美，你可以用 gridspec，但 imshow 最简单
    axes[0, 0].imshow(img_a);
    axes[0, 0].axis('off');
    axes[0, 0].set_title('(a) Prediction Accuracy', y=-0.1, fontsize=16)
    axes[0, 1].imshow(img_b);
    axes[0, 1].axis('off');
    axes[0, 1].set_title('(b) Computational Efficiency', y=-0.1, fontsize=16)
    axes[1, 0].imshow(img_c);
    axes[1, 0].axis('off');
    axes[1, 0].set_title('(c) Ablation Study', y=-0.1, fontsize=16)
    axes[1, 1].imshow(img_d);
    axes[1, 1].axis('off');
    axes[1, 1].set_title('(d) Expert Interpretability', y=-0.1, fontsize=16)

    plt.tight_layout()
    plt.savefig("Figure_2_Final.pdf", dpi=300)  # PDF 格式最适合投稿
    plt.savefig("Figure_2_Final.png", dpi=300)
    print("✅ Figure 2 Assembled: Figure_2_Final.png")


def assemble_figure_3():
    print("\n🧩 Assembling Figure 3 (Robustness & Economics)...")

    # 1. 生成素材
    run_script("plot_dlinear_victory.py")  # 生成 fig_battle_dlinear_victory.png
    run_script("battle_economics.py")  # 生成 fig_economic_impact.png

    # 2. 拼图 (1x2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    if os.path.exists("fig_battle_dlinear_victory.png"):
        img_a = mpimg.imread("fig_battle_dlinear_victory.png")
        axes[0].imshow(img_a);
        axes[0].axis('off');
        axes[0].set_title('(a) Robustness vs DLinear', y=-0.15, fontsize=16)

    if os.path.exists("fig_economic_impact.png"):
        img_b = mpimg.imread("fig_economic_impact.png")
        axes[1].imshow(img_b);
        axes[1].axis('off');
        axes[1].set_title('(b) Economic Value Analysis', y=-0.15, fontsize=16)

    plt.tight_layout()
    plt.savefig("Figure_3_Final.png", dpi=300)
    print("✅ Figure 3 Assembled: Figure_3_Final.png")


if __name__ == "__main__":
    assemble_figure_2()
    assemble_figure_3()