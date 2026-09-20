import subprocess
import re
import numpy as np
import time
import sys
import os


def run_experiments():
    # ================= 配置区域 =================
    seeds = [42, 43, 44, 45, 46]
    script_name = "ultra_lsnt_timeseries.py"
    data_path = "wind_final.csv"
    target = "power"

    # 你的主程序命令
    cmd_template = f"python {script_name} --data {data_path} --target {target} --seed {{}}"
    # ===========================================

    results = []

    print(f"🚀 启动统计显著性测试 (5 Runs) - OMP 防弹版...")
    print(f"   目标脚本: {script_name}")
    print(f"   种子列表: {seeds}")
    print("-" * 60)

    # 【核心修复】设置环境变量
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"  # 解决中文乱码
    my_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 OMP Error #15 (最关键的一行！)

    total_start = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n▶️ [Run {i + 1}/{len(seeds)}] 正在运行 Seed = {seed} ... (请耐心等待)")
        run_start = time.time()

        cmd = cmd_template.format(seed)

        # 用于存储完整日志以便最后提取 R2
        full_log_buffer = []

        try:
            # 启动子进程
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 把错误也合并到输出流
                env=my_env  # <--- 把“防弹衣”穿给子进程
            )

            # 实时逐行读取并显示
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    try:
                        decoded_line = line.decode('utf-8', errors='ignore').strip()
                        # 过滤掉一些无关紧要的进度条刷屏，只保留关键信息，不然屏幕闪得太快
                        if "Epoch" in decoded_line or "Loss" in decoded_line or "R²" in decoded_line:
                            print(decoded_line)
                        full_log_buffer.append(decoded_line)
                    except:
                        pass

            # 等待进程完全结束
            rc = process.poll()

            if rc != 0:
                print(f"   ❌ 运行失败 (Seed {seed})！")
                # 如果失败了，打印最后几行看看是不是还有其他错
                print("   最后报错日志:", full_log_buffer[-10:] if full_log_buffer else "无日志")
                continue

            # 拼凑完整日志找 R2
            full_output = "\n".join(full_log_buffer)
            match = re.search(r"R[²2]:\s+([0-9\.]+)", full_output)

            if match:
                r2_score = float(match.group(1))
                results.append(r2_score)
                print(f"   ✅ [Seed {seed}] 完成！ R² = {r2_score:.4f} (耗时 {(time.time() - run_start) / 60:.1f} min)")
            else:
                print("   ⚠️ 警告: 跑完了，但在日志里没找到 R² 分数。")

        except Exception as e:
            print(f"   ❌ 脚本异常: {e}")

        sys.stdout.flush()

    # ================= 最终统计 =================
    print("\n" + "=" * 60)
    print("📊 统计显著性测试报告")
    print("=" * 60)

    if len(results) > 0:
        mean_score = np.mean(results)
        std_score = np.std(results)
        max_score = np.max(results)
        min_score = np.min(results)

        print(f"实验次数: {len(results)}")
        print(f"所有成绩: {results}")
        print("-" * 30)
        print(f"平均值 (Mean): {mean_score:.4f}")
        print(f"标准差 (Std) : {std_score:.4f}")
        print("-" * 30)
        print(f"💡 论文写法: Ultra-LSNT achieved an R² of {mean_score:.4f} ± {std_score:.4f} (over 5 runs).")
    else:
        print("❌ 没有收集到有效数据。")

    print(f"\n总耗时: {(time.time() - total_start) / 3600:.2f} hours")


if __name__ == "__main__":
    run_experiments()