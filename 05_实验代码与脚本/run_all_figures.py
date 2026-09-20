#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all_figures.py - 论文配图一键生成器
=======================================
自动运行所有 Battle 脚本，生成 Applied Energy 所需的全套图表。
"""
import subprocess
import sys
import os


def run_script(script_name, description):
    """运行指定的Python脚本"""
    print(f"\n{'=' * 60}")
    print(f"🎬 {description}")
    print(f"   执行脚本: {script_name}")
    print(f"{'=' * 60}")
    try:
        # 使用当前 python 环境运行
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # 让子脚本的输出直接打印到屏幕，方便看进度
            text=True
        )
        print(f"✅ [成功] {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [失败] {script_name} 退出代码: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ [错误] 运行 {script_name} 时出错: {e}")
        return False


def main():
    print("\n🚀 Ultra-LSNT 论文图表生成流水线启动...\n")

    # 定义执行顺序：逻辑上从架构 -> 效率 -> 鲁棒性 -> 经济性
    scripts = [
        # 1. 架构与原理 (需确保 plot_arch.py 存在，没有的话可以跳过)
        # ("plot_arch.py", "Figure 1: 生成模型架构图"),

        # 2. 效率边界 (Efficiency)
        ("test_efficiency.py", "Figure 2: 生成算力边界对比图 (Efficiency Boundary)"),

        # 3. 鲁棒性大乱斗 (Robustness Battles) - 这一步需要跑挺久
        ("battle_dlinear.py", "Figure 3: 生成 DLinear 鲁棒性对比 (Battle DLinear)"),
        ("compare_robustness_real.py", "Figure 4: 生成 Transformer 鲁棒性对比 (Battle Transformer)"),
        ("run_gbdt_robustness.py", "Figure 5: 生成 LightGBM 鲁棒性对比 (Battle GBDT)"),

        # 4. 经济性分析 (Economics)
        ("battle_economics.py", "Figure 6: 生成经济效益分析图 (Economic Impact)"),
    ]

    success_count = 0
    fail_list = []

    for script, desc in scripts:
        if not os.path.exists(script):
            print(f"⚠️ 跳过: 找不到文件 {script}")
            continue

        if run_script(script, desc):
            success_count += 1
        else:
            fail_list.append(script)

    # 总结
    print(f"\n{'=' * 60}")
    print(f"📊 执行报告: 成功 {success_count} / 总计 {len(scripts)}")
    if fail_list:
        print(f"❌ 失败脚本: {fail_list}")
        print("   建议单独运行失败的脚本以排查错误。")
    else:
        print("🎉 所有图表生成完毕！请查看当前目录下的 .png 文件。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()