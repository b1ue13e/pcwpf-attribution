#!/bin/bash
# 修复失败和未完成的实验脚本补全脚本
# 在后台一次性运行所有修复后的脚本

set -u -o pipefail

# 设置默认变量以避免未绑定错误
PYTHON_CMD="python3"
GPU_OPTIMIZED=0
export PYTHON_CMD GPU_OPTIMIZED

echo "================================================"
echo "Ultra-LSNT 失败实验修复补全脚本 (run_error.sh)"
echo "开始时间: $(date)"
echo "================================================"

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/run_error_${TIMESTAMP}.log"

# 全局Python命令检测
if python3 -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    echo "检测到Python命令: python3"
elif python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    PYTHON_CMD="python"
    echo "检测到Python命令: python"
else
    PYTHON_CMD="python3"
    echo "警告: 未检测到PyTorch，使用python3作为默认命令"
fi
export PYTHON_CMD

# 硬件检测
detect_hardware() {
    echo "硬件检测..."
    CPU_CORES=$(nproc --all 2>/dev/null || echo 1)
    echo "  CPU核心数: $CPU_CORES"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 2>/dev/null || echo "0")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "Unknown")
        
        if [ "$GPU_MEM" != "0" ] && [ ! -z "$GPU_MEM" ]; then
            echo "  GPU型号: $GPU_NAME"
            echo "  GPU显存: ${GPU_MEM}MB"
            
            if [ "$GPU_MEM" -ge 23000 ]; then
                export GPU_OPTIMIZED=1
                echo "  检测到高性能GPU，启用GPU优化模式"
            else
                export GPU_OPTIMIZED=0
            fi
        fi
    else
        export GPU_OPTIMIZED=0
    fi
    
    echo "  使用Python命令: $PYTHON_CMD"
}

# 运行脚本并记录日志
run_script() {
    local script_name=$1
    local log_file="logs/${script_name%.py}_error_${TIMESTAMP}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行修复脚本: $script_name" | tee -a "$MAIN_LOG"
    
    if [ -f "$script_name" ]; then
        if [[ "$script_name" == *.py ]]; then
            $PYTHON_CMD "$script_name" 2>&1 | tee "$log_file"
            exit_code=${PIPESTATUS[0]}
        else
            echo "错误: $script_name 不是Python脚本" | tee -a "$MAIN_LOG"
            exit_code=1
        fi
    else
        echo "错误: 文件 $script_name 不存在" | tee -a "$MAIN_LOG"
        exit_code=1
    fi
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 成功: $script_name" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 失败: $script_name (退出码: $exit_code)" | tee -a "$MAIN_LOG"
        if [ -f "$log_file" ]; then
            echo "  失败日志片段:" | tee -a "$MAIN_LOG"
            tail -10 "$log_file" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        fi
        return 1
    fi
}

# 执行硬件检测
detect_hardware | tee -a "$MAIN_LOG"

echo "" | tee -a "$MAIN_LOG"
echo "=== 第一步：运行之前失败的脚本（已修复版本）===" | tee -a "$MAIN_LOG"

# 1. 概率预测测试 (已修复 numpy.erf 问题)
echo "1. 运行概率预测测试 (test_probabilistic.py)..."
run_script "test_probabilistic.py"

# 2. Diebold-Mariano检验 (已修复权重不匹配)
echo "2. 运行Diebold-Mariano检验 (run_dm_test.py)..."
run_script "run_dm_test.py"

# 3. GBDT鲁棒性对比 (已修复权重不匹配)
echo "3. 运行GBDT鲁棒性对比 (run_gbdt_robustness.py)..."
run_script "run_gbdt_robustness.py"

# 4. 经典基线模型 (内存优化版本)
echo "4. 运行经典基线模型 (内存优化版本)..."
echo "   注：使用子样本和减少模型复杂度以避免内存超限"
# 创建经典基线的内存优化版本临时脚本
cat > /tmp/run_classical_optimized.py << 'EOF'
"""
经典基线模型内存优化版本
使用数据子样本和简化模型以减少内存使用
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')
from classical_baselines import run_all_baselines

print("加载风电数据（内存优化版本）...")
df = pd.read_csv('wind_final.csv')

# 使用子样本（前50%数据）以减少内存使用
sample_frac = 0.5
df = df.iloc[:int(len(df) * sample_frac)].copy()
print(f"使用子样本: {len(df)} 行")

# 选择关键特征列
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'power' in numeric_cols:
    target_col = 'power'
else:
    target_col = numeric_cols[-1]  # 最后一列作为目标

df = df[numeric_cols].dropna()
print(f"数据形状: {df.shape}")

# 运行基线，减少季节长度和预测步长以降低计算复杂度
try:
    results = run_all_baselines(
        df, 
        target_col=target_col, 
        horizon=12,           # 减少预测步长（原24）
        seasonal_period=48,   # 减少季节长度（原96）
        train_ratio=0.6       # 减少训练比例
    )
    print("\n✅ 经典基线模型评估完成（内存优化版本）!")
    print(results)
    
    # 保存结果
    os.makedirs('baseline_results', exist_ok=True)
    results.to_csv('baseline_results/classical_baselines_optimized.csv')
    print("结果保存至: baseline_results/classical_baselines_optimized.csv")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

$PYTHON_CMD /tmp/run_classical_optimized.py 2>&1 | tee "logs/classical_baselines_error_${TIMESTAMP}.log"
classical_exit_code=${PIPESTATUS[0]}
if [ $classical_exit_code -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 成功: classical_baselines.py (内存优化版本)" | tee -a "$MAIN_LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 失败: classical_baselines.py (退出码: $classical_exit_code)" | tee -a "$MAIN_LOG"
fi

echo "" | tee -a "$MAIN_LOG"
echo "=== 第二步：运行其他未完成的脚本 ===" | tee -a "$MAIN_LOG"

# 检查是否有其他未运行的脚本（根据run_all_fixed.sh）
# 这里可以添加其他可能未完成的脚本，但根据日志，大部分已完成
# 可以运行一些额外的验证脚本

echo "5. 运行验证脚本 (validate_r2_correction.py)..."
if [ -f "validate_r2_correction.py" ]; then
    run_script "validate_r2_correction.py"
else
    echo "  脚本不存在，跳过"
fi

echo "6. 运行R2问题分析 (analyze_r2_issue.py)..."
if [ -f "analyze_r2_issue.py" ]; then
    run_script "analyze_r2_issue.py"
else
    echo "  脚本不存在，跳过"
fi

echo "" | tee -a "$MAIN_LOG"
echo "=== 完成总结 ===" | tee -a "$MAIN_LOG"
echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
echo "主日志文件: $MAIN_LOG" | tee -a "$MAIN_LOG"

# 统计成功/失败
SUCCESS_COUNT=$(grep -c "✅ 成功:" "$MAIN_LOG")
FAIL_COUNT=$(grep -c "❌ 失败:" "$MAIN_LOG")
echo "成功脚本数: $SUCCESS_COUNT"
echo "失败脚本数: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "警告: 有 $FAIL_COUNT 个脚本失败，请检查日志"
    exit 1
else
    echo "所有修复脚本成功完成!"
    exit 0
fi