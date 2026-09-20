import subprocess
import os
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    print(f"\n🚀 {description}...")
    print(f"执行命令: {cmd}")

    # --- 【核心修复】设置环境变量 ---
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"  # 强制子进程输出为 UTF-8
    my_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止 OMP 报错

    start_time = time.time()
    try:
        # 传入 env=my_env，并设置 errors='ignore' 兜底
        process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # 明确指定以 utf-8 读取输出
            errors='ignore',  # 忽略无法解码的残留 GBK 字符
            env=my_env
        )
        for line in process.stdout:
            print(f"  {line.strip()}")
        process.wait()
        elapsed = time.time() - start_time
        if process.returncode == 0:
            print(f"✅ 完成！耗时: {elapsed:.1f}s")
            return True
        else:
            print(f"❌ 失败！返回码: {process.returncode}")
            return False
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        return False


def main():
    print("=" * 60)
    print("🌟 Ultra-LSNT 真实实验全量运行流水线 (APEN 投稿专用)")
    print("=" * 60)

    # 1. 环境与数据检查
    if not os.path.exists("wind_final.csv"):
        print("❌ 错误：找不到 wind_final.csv，请先运行 data_preprocess.py 或 force_fix.py")
        return

    # 2. 实验步骤定义
    steps = [
        # 第一阶段：基础基线 (Baselines)
        ("python run_gbdt.py", "运行 LightGBM 基线并生成结果文件 (.npy)"),
        ("python run_dlinear.py", "运行 DLinear 基线对比"),
        ("python run_baselines.py", "运行 LSTM/Transformer/MLP 基线对比"),

        # 第二阶段：主模型深度分析 (Main Model Analysis)
        # 假设 best_model.pth 已存在，直接进行鲁棒性和可解释性分析
        ("python real_robustness.py", "执行真实鲁棒性噪声压力测试"),
        ("python run_expert_physics.py", "执行物理可解释性相关性分析"),
        ("python plot_expert_heatmap.py", "提取真实专家激活热力图"),

        # 第三阶段：性能与效率 (Efficiency)
        ("python profile_energy.py", "估算边缘端部署能耗与 FLOPs"),
        ("python check_storage.py", "对比模型存储空间占用"),

        # 第四阶段：统计学检验 (Statistical Significance)
        ("python run_dm_test.py", "执行 Diebold-Mariano 显著性检验 (需 .npy 文件)"),

        # 第五阶段：可视化组装 (Visualization)
        ("python assemble_figures.py", "一键合成论文 Final Figures (PDF/PNG)")
    ]

    success_count = 0
    for cmd, desc in steps:
        if run_command(cmd, desc):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"📊 实验流水线运行报告: 成功 {success_count}/{len(steps)}")
    print("=" * 60)
    print("📖 下一步：请检查生成的 Figure_2_Final.png 和 Figure_3_Final.png")


if __name__ == "__main__":
    main()