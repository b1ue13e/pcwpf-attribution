#!/bin/bash
# 重新运行所有数据不真实的实验，获取真实数据
# 按顺序执行所有需要重新运行的训练命令
# RTX 4090D 24GB 优化版

# 启用严格错误处理
set -u -o pipefail

echo "================================================"
echo "重新运行所有数据不真实的实验 (run_alldata.sh)"
echo "开始时间: $(date)"
echo "目的: 获取所有实验的真实数据，替换虚构数据"
echo "================================================"

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/run_alldata_${TIMESTAMP}.log"

# 硬件检测函数
detect_hardware() {
    echo "硬件检测..." | tee -a "$MAIN_LOG"
    
    # GPU检测
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 2>/dev/null || echo "0")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "Unknown")
        
        if [ "$GPU_MEM" != "0" ] && [ ! -z "$GPU_MEM" ]; then
            echo "  GPU型号: $GPU_NAME" | tee -a "$MAIN_LOG"
            echo "  GPU显存: ${GPU_MEM}MB" | tee -a "$MAIN_LOG"
            
            if [ "$GPU_MEM" -ge 23000 ]; then
                export GPU_OPTIMIZED=1
                echo "  检测到RTX 4090D级别GPU，启用GPU优化模式" | tee -a "$MAIN_LOG"
            else
                echo "  使用默认配置" | tee -a "$MAIN_LOG"
                export GPU_OPTIMIZED=0
            fi
        else
            echo "  GPU检测失败，使用CPU模式" | tee -a "$MAIN_LOG"
            export GPU_OPTIMIZED=0
        fi
    else
        echo "  未检测到NVIDIA GPU，使用CPU模式" | tee -a "$MAIN_LOG"
        export GPU_OPTIMIZED=0
    fi
    
    # Python命令检测
    if python3 -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo "  警告: 未检测到PyTorch，实验可能失败" | tee -a "$MAIN_LOG"
        PYTHON_CMD="python3"
    fi
    export PYTHON_CMD
    echo "  使用Python命令: $PYTHON_CMD" | tee -a "$MAIN_LOG"
}

# 函数：检查脚本是否支持 --gpu_optimized 参数
script_supports_gpu_optimized() {
    local script_name=$1
    case "$script_name" in
        "ultra_lsnt_timeseries.py"|"run_sota.py"|"run_ablation_study.py"|"test_efficiency.py")
            return 0  # 支持
            ;;
        *)
            return 1  # 不支持
            ;;
    esac
}

# 函数：运行脚本并记录日志
run_script() {
    local script_name=$1
    local description=$2
    local log_file="logs/${script_name%.py}_realdata_${TIMESTAMP}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行: $script_name ($description)" | tee -a "$MAIN_LOG"
    
    if [ -f "$script_name" ]; then
        # 检查是否是Python脚本
        if [[ "$script_name" == *.py ]]; then
            # 传递GPU优化参数给脚本（仅当脚本支持且GPU优化启用时）
            if [ "$GPU_OPTIMIZED" -eq 1 ] && script_supports_gpu_optimized "$script_name"; then
                echo "  使用GPU优化参数运行" | tee -a "$MAIN_LOG"
                $PYTHON_CMD "$script_name" --gpu_optimized 2>&1 | tee "$log_file"
            else
                $PYTHON_CMD "$script_name" 2>&1 | tee "$log_file"
            fi
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
        # 记录失败原因
        if [ -f "$log_file" ]; then
            echo "  失败日志片段:" | tee -a "$MAIN_LOG"
            tail -10 "$log_file" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        fi
        return 1
    fi
}

# 函数：运行特定参数的训练命令
run_training_command() {
    local cmd=$1
    local description=$2
    local log_file="logs/training_$(echo $description | tr ' ' '_')_${TIMESTAMP}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行: $description" | tee -a "$MAIN_LOG"
    echo "  命令: $cmd" | tee -a "$MAIN_LOG"
    
    eval "$cmd" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 成功: $description" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 失败: $description (退出码: $exit_code)" | tee -a "$MAIN_LOG"
        if [ -f "$log_file" ]; then
            echo "  失败日志片段:" | tee -a "$MAIN_LOG"
            tail -10 "$log_file" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        fi
        return 1
    fi
}

# ========== 主程序开始 ==========

# 执行硬件检测
detect_hardware

echo "根据实验总结报告，以下实验的数据不真实或未完成，需要重新运行：" | tee -a "$MAIN_LOG"
echo "1. 消融研究（初始虚构，需要真实卷积核尺寸影响数据）" | tee -a "$MAIN_LOG"
echo "2. 效率测试（未测量，需要真实训练/推理时间数据）" | tee -a "$MAIN_LOG"
echo "3. 统计显著性检验（未运行，需要DM检验结果）" | tee -a "$MAIN_LOG"
echo "4. GBDT对比（初始失败，需要真实对比数据）" | tee -a "$MAIN_LOG"
echo "5. SOTA模型对比（部分完成，需要完整Autoformer结果）" | tee -a "$MAIN_LOG"
echo "6. 多领域泛化完整数据（部分数据待测量）" | tee -a "$MAIN_LOG"
echo "7. 概率预测测试（numpy.erf问题修复后）" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

FAILED_COUNT=0
TOTAL_SCRIPTS=0

# ========== 第一部分：重新运行虚构的消融研究 ==========
echo "第一部分：重新运行虚构的消融研究" | tee -a "$MAIN_LOG"

# 1.1 运行消融研究脚本
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if ! run_script "run_ablation_study.py" "消融研究（卷积核大小影响）"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 1.2 运行通用消融研究
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_universal_ablation.py" ]; then
    if ! run_script "run_universal_ablation.py" "通用消融研究（跨领域）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
else
    echo "⚠️  脚本 run_universal_ablation.py 不存在，跳过" | tee -a "$MAIN_LOG"
fi

# ========== 第二部分：重新运行未测量的效率测试 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第二部分：重新运行未测量的效率测试" | tee -a "$MAIN_LOG"

# 2.1 运行效率测试脚本
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if ! run_script "test_efficiency.py" "效率测试（训练/推理时间测量）"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 2.2 运行MAPE测试（相关效率指标）
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "test_mape.py" ]; then
    if ! run_script "test_mape.py" "MAPE测试（误差分析）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第三部分：重新运行统计显著性检验 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第三部分：重新运行统计显著性检验" | tee -a "$MAIN_LOG"

# 3.1 运行Diebold-Mariano检验
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if ! run_script "run_dm_test.py" "Diebold-Mariano统计检验"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 3.2 运行统计显著性测试
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_significance.py" ]; then
    if ! run_script "run_significance.py" "统计显著性测试"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# 3.3 运行通用显著性测试
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_universal_significance.py" ]; then
    if ! run_script "run_universal_significance.py" "通用显著性测试（多领域）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第四部分：重新运行GBDT对比实验 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第四部分：重新运行GBDT对比实验" | tee -a "$MAIN_LOG"

# 4.1 运行GBDT鲁棒性对比
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if ! run_script "run_gbdt_robustness.py" "GBDT鲁棒性对比"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 4.2 运行基础GBDT脚本
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_gbdt.py" ]; then
    if ! run_script "run_gbdt.py" "GBDT基础对比"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第五部分：重新运行SOTA模型对比 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第五部分：重新运行SOTA模型对比" | tee -a "$MAIN_LOG"

# 5.1 运行SOTA模型对比脚本（完整版本）
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if ! run_script "run_sota.py" "SOTA模型完整对比（Autoformer补全）"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
fi

# 5.2 运行真实噪声对战
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_real_noise_battle.py" ]; then
    if ! run_script "run_real_noise_battle.py" "真实噪声对战实验"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第六部分：重新运行多领域泛化完整数据 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第六部分：重新运行多领域泛化完整数据" | tee -a "$MAIN_LOG"

# 6.1 运行多领域实验（获取完整RMSE/MAE数据）
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_real_experiments.py" ]; then
    if ! run_script "run_real_experiments.py" "多领域真实实验（完整指标）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# 6.2 运行通用鲁棒性测试
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_universal_robustness.py" ]; then
    if ! run_script "run_universal_robustness.py" "通用鲁棒性测试（多领域）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第七部分：重新运行概率预测测试 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第七部分：重新运行概率预测测试" | tee -a "$MAIN_LOG"

# 7.1 运行概率预测测试（修复numpy.erf问题后）
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "test_probabilistic.py" ]; then
    if ! run_script "test_probabilistic.py" "概率预测测试（修复版本）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第八部分：运行额外的验证脚本 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第八部分：运行额外的验证脚本" | tee -a "$MAIN_LOG"

# 8.1 运行R2验证脚本
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "validate_r2_correction.py" ]; then
    if ! run_script "validate_r2_correction.py" "R2分数验证"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# 8.2 运行经典基线模型（内存优化版本）
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "classical_baselines.py" ]; then
    if ! run_script "classical_baselines.py" "经典基线模型（内存优化）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# 8.3 运行DLinear修复版本
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_dlinear_fixed.py" ]; then
    if ! run_script "run_dlinear_fixed.py" "DLinear修复版本"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 第九部分：生成可视化图表 ==========
echo "" | tee -a "$MAIN_LOG"
echo "第九部分：生成基于真实数据的可视化图表" | tee -a "$MAIN_LOG"

# 9.1 运行图表生成脚本
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "run_all_figures.py" ]; then
    if ! run_script "run_all_figures.py" "生成所有图表（基于真实数据）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# 9.2 运行分层分析
TOTAL_SCRIPTS=$((TOTAL_SCRIPTS + 1))
if [ -f "stratified_analysis.py" ]; then
    if ! run_script "stratified_analysis.py" "分层分析（性能分解）"; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
fi

# ========== 完成总结 ==========
echo "" | tee -a "$MAIN_LOG"
echo "================================================" | tee -a "$MAIN_LOG"
echo "所有数据不真实的实验重新运行完成!" | tee -a "$MAIN_LOG"
echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
echo "主日志文件: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "================================================" | tee -a "$MAIN_LOG"

# 执行统计
SUCCESS_COUNT=$((TOTAL_SCRIPTS - FAILED_COUNT))
SUCCESS_RATE=$((SUCCESS_COUNT * 100 / TOTAL_SCRIPTS))

echo "执行统计:" | tee -a "$MAIN_LOG"
echo "  总脚本数: $TOTAL_SCRIPTS" | tee -a "$MAIN_LOG"
echo "  成功脚本数: $SUCCESS_COUNT" | tee -a "$MAIN_LOG"
echo "  失败脚本数: $FAILED_COUNT" | tee -a "$MAIN_LOG"
echo "  成功率: $SUCCESS_RATE%" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 关键实验状态检查
echo "关键实验状态检查:" | tee -a "$MAIN_LOG"

check_experiment_status() {
    local exp_name=$1
    local script_pattern=$2
    
    if grep -q "✅ 成功.*$script_pattern" "$MAIN_LOG"; then
        echo "  ✅ $exp_name: 已成功重新运行"
    elif grep -q "❌ 失败.*$script_pattern" "$MAIN_LOG"; then
        echo "  ❌ $exp_name: 重新运行失败"
    else
        echo "  ⚠️  $exp_name: 未运行"
    fi
}

check_experiment_status "消融研究" "run_ablation_study"
check_experiment_status "效率测试" "test_efficiency"
check_experiment_status "统计显著性检验" "run_dm_test"
check_experiment_status "GBDT对比" "run_gbdt_robustness"
check_experiment_status "SOTA模型对比" "run_sota"
check_experiment_status "概率预测测试" "test_probabilistic"

echo "" | tee -a "$MAIN_LOG"

# 下一步建议
echo "下一步建议:" | tee -a "$MAIN_LOG"
echo "1. 查看详细日志: tail -f $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "2. 更新实验报告: 基于新生成的真实数据更新所有总结报告" | tee -a "$MAIN_LOG"
echo "3. 数据验证: 检查新生成的CSV/JSON文件中的数据真实性" | tee -a "$MAIN_LOG"
echo "4. 可视化验证: 查看figures/目录下的新图表" | tee -a "$MAIN_LOG"

# 退出码
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "警告: 有 $FAILED_COUNT 个脚本失败，请检查日志" | tee -a "$MAIN_LOG"
    echo "失败脚本的日志文件保存在 logs/ 目录中" | tee -a "$MAIN_LOG"
    exit 1
else
    echo "" | tee -a "$MAIN_LOG"
    echo "所有数据不真实的实验已成功重新运行，现在拥有完整的真实数据!" | tee -a "$MAIN_LOG"
    echo "可以开始更新实验报告，使用真实数据替换虚构数据。" | tee -a "$MAIN_LOG"
    exit 0
fi