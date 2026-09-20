#!/bin/bash
# =================================================================
# Ultra-LSNT 专家切换与线性分支分析脚本 (run_expert.sh)
# 功能：
# 1. 训练带线性分支的Ultra-LSNT模型
# 2. 分析专家切换过程（风速突变事件）
# 3. 可视化专家权重切换和分支贡献
# 4. 生成分析报告
# =================================================================
# 作者: Roo
# 日期: 2026-01-25
# =================================================================

set -u -o pipefail  # 启用严格错误处理（但不退出，允许单个脚本失败后继续）

# 配置变量
LOG_DIR="logs_expert"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_expert_${TIMESTAMP}.log"

# 颜色定义（用于终端输出）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 硬件检测函数
detect_hardware() {
    echo -e "${BLUE}硬件检测...${NC}" | tee -a "$MAIN_LOG"
    
    # CPU检测
    CPU_CORES=$(nproc --all 2>/dev/null || echo 1)
    echo -e "  CPU核心数: $CPU_CORES" | tee -a "$MAIN_LOG"
    
    # 内存检测
    if [ -f /proc/meminfo ]; then
        TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024))
        echo -e "  系统内存: ${TOTAL_MEM_GB}GB" | tee -a "$MAIN_LOG"
    fi
    
    # GPU检测
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 2>/dev/null || echo "0")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "Unknown")
        
        if [ "$GPU_MEM" != "0" ] && [ ! -z "$GPU_MEM" ]; then
            echo -e "  GPU型号: $GPU_NAME" | tee -a "$MAIN_LOG"
            echo -e "  GPU显存: ${GPU_MEM}MB" | tee -a "$MAIN_LOG"
            
            # 根据显存设置优化标志
            if [ "$GPU_MEM" -ge 20000 ]; then  # 20GB以上
                export GPU_OPTIMIZED=1
                echo -e "  ${GREEN}检测到高性能GPU，启用GPU优化模式${NC}" | tee -a "$MAIN_LOG"
            else
                export GPU_OPTIMIZED=0
                echo -e "  ${YELLOW}使用默认GPU配置${NC}" | tee -a "$MAIN_LOG"
            fi
        else
            echo -e "  ${YELLOW}GPU检测失败，使用CPU模式${NC}" | tee -a "$MAIN_LOG"
            export GPU_OPTIMIZED=0
        fi
    else
        echo -e "  ${YELLOW}未检测到NVIDIA GPU，使用CPU模式${NC}" | tee -a "$MAIN_LOG"
        export GPU_OPTIMIZED=0
    fi
}

# 运行脚本函数（支持后台运行）
run_script_background() {
    local script_name=$1
    local script_args=${2:-""}
    local log_file="$LOG_DIR/${script_name%.py}_${TIMESTAMP}.log"
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行: $script_name $script_args${NC}" | tee -a "$MAIN_LOG"
    
    if [ -f "$script_name" ]; then
        # 在后台运行脚本
        nohup python3 "$script_name" $script_args > "$log_file" 2>&1 &
        local pid=$!
        
        echo -e "  进程ID: $pid" | tee -a "$MAIN_LOG"
        echo -e "  日志文件: $log_file" | tee -a "$MAIN_LOG"
        
        # 等待进程完成（或超时）
        local timeout=3600  # 1小时超时
        local start_time=$(date +%s)
        local elapsed=0
        
        while [ $elapsed -lt $timeout ]; do
            if kill -0 $pid 2>/dev/null; then
                # 进程仍在运行
                sleep 10
                elapsed=$(( $(date +%s) - start_time ))
                
                # 每60秒打印一次状态
                if [ $((elapsed % 60)) -eq 0 ]; then
                    echo -e "  ${YELLOW}运行中... ${elapsed}s${NC}" | tee -a "$MAIN_LOG"
                fi
            else
                # 进程已完成
                wait $pid
                local exit_code=$?
                break
            fi
        done
        
        # 检查是否超时
        if [ $elapsed -ge $timeout ]; then
            echo -e "  ${RED}超时: $script_name 运行超过 $timeout 秒${NC}" | tee -a "$MAIN_LOG"
            kill $pid 2>/dev/null
            return 1
        fi
        
        if [ $exit_code -eq 0 ]; then
            echo -e "  ${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 成功: $script_name${NC}" | tee -a "$MAIN_LOG"
            return 0
        else
            echo -e "  ${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 失败: $script_name (退出码: $exit_code)${NC}" | tee -a "$MAIN_LOG"
            
            # 打印错误日志片段
            if [ -f "$log_file" ]; then
                echo -e "  错误日志片段:" | tee -a "$MAIN_LOG"
                tail -20 "$log_file" | sed 's/^/    /' | tee -a "$MAIN_LOG"
            fi
            return 1
        fi
    else
        echo -e "  ${RED}错误: 文件 $script_name 不存在${NC}" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# 运行脚本函数（前台运行）
run_script_foreground() {
    local script_name=$1
    local script_args=${2:-""}
    local log_file="$LOG_DIR/${script_name%.py}_${TIMESTAMP}.log"
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行: $script_name $script_args${NC}" | tee -a "$MAIN_LOG"
    
    if [ -f "$script_name" ]; then
        python3 "$script_name" $script_args 2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            echo -e "  ${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 成功: $script_name${NC}" | tee -a "$MAIN_LOG"
            return 0
        else
            echo -e "  ${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 失败: $script_name (退出码: $exit_code)${NC}" | tee -a "$MAIN_LOG"
            return 1
        fi
    else
        echo -e "  ${RED}错误: 文件 $script_name 不存在${NC}" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# 检查必要文件
check_required_files() {
    echo -e "${BLUE}检查必要文件...${NC}" | tee -a "$MAIN_LOG"
    
    REQUIRED_FILES=(
        "wind_final.csv"
        "ultra_lsnt_timeseries.py"
    )
    
    OPTIONAL_FILES=(
        "ultra_lsnt_linear_branch.py"
        "analyze_expert_switching.py"
        "analyze_expert_switching_standard.py"
        "run_expert_physics.py"
        "plot_expert_heatmap_enhanced.py"
    )
    
    missing_required=0
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "  ${GREEN}✅ $file 存在${NC}" | tee -a "$MAIN_LOG"
        else
            echo -e "  ${RED}❌ $file 不存在${NC}" | tee -a "$MAIN_LOG"
            missing_required=$((missing_required + 1))
        fi
    done
    
    for file in "${OPTIONAL_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "  ${GREEN}✅ $file 存在${NC}" | tee -a "$MAIN_LOG"
        else
            echo -e "  ${YELLOW}⚠️  $file 不存在（可选）${NC}" | tee -a "$MAIN_LOG"
        fi
    done
    
    if [ $missing_required -gt 0 ]; then
        echo -e "${RED}错误: 缺少 $missing_required 个必需文件${NC}" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    return 0
}

# 检查数据质量
check_data_quality() {
    echo -e "${BLUE}检查数据质量...${NC}" | tee -a "$MAIN_LOG"
    
    if [ -f "wind_final.csv" ]; then
        # 使用Python快速检查数据
        python3 -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_csv('wind_final.csv')
    print('数据形状:', df.shape)
    print('数据列名:', list(df.columns))
    
    # 检查风速列
    wind_cols = ['windspeed', 'wind_speed', '风速', 'ws', 'wind speed']
    found = False
    for col in wind_cols:
        if col in df.columns:
            print(f'找到风速列: {col}')
            wind_speed = df[col].values
            print(f'风速范围: [{wind_speed.min():.2f}, {wind_speed.max():.2f}]')
            print(f'风速均值: {wind_speed.mean():.2f}')
            found = True
            break
    
    if not found:
        print('警告: 未找到标准风速列')
        print('使用第一列作为风速参考')
        first_col = df.columns[0]
        wind_speed = df[first_col].values
        print(f'{first_col} 范围: [{wind_speed.min():.2f}, {wind_speed.max():.2f}]')
    
    # 检查缺失值
    missing = df.isnull().sum().sum()
    print(f'缺失值总数: {missing}')
    
    if missing > 0:
        print('警告: 数据中存在缺失值')
    
except Exception as e:
    print(f'数据检查错误: {e}')
" 2>&1 | tee -a "$MAIN_LOG"
    else
        echo -e "  ${RED}错误: wind_final.csv 不存在${NC}" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    return 0
}

# 主程序
main() {
    echo -e "${GREEN}===============================================================${NC}"
    echo -e "${GREEN}Ultra-LSNT 专家切换与线性分支分析脚本 (run_expert.sh)${NC}"
    echo -e "${GREEN}开始时间: $(date)${NC}"
    echo -e "${GREEN}===============================================================${NC}"
    echo -e "${BLUE}日志文件: $MAIN_LOG${NC}"
    echo ""
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 硬件检测
    detect_hardware
    
    # 检查必要文件
    if ! check_required_files; then
        echo -e "${RED}必要文件检查失败，终止执行${NC}" | tee -a "$MAIN_LOG"
        exit 1
    fi
    
    # 检查数据质量
    check_data_quality
    
    echo -e "${BLUE}===============================================================${NC}"
    echo -e "${BLUE}开始执行专家分析流程${NC}"
    echo -e "${BLUE}===============================================================${NC}"
    
    # 第一步：训练带线性分支的模型（如果权重不存在）
    echo -e "${YELLOW}第一步：训练带线性分支的Ultra-LSNT模型${NC}" | tee -a "$MAIN_LOG"
    
    # 检查是否已有预训练权重
    if [ -f "checkpoints_ts/linear_branch/best_model.pth" ] || [ -f "checkpoints_ts/main/best_model.pth" ]; then
        echo -e "  ${GREEN}检测到已有预训练权重，跳过训练${NC}" | tee -a "$MAIN_LOG"
        echo -e "  ${YELLOW}如果要重新训练，请删除 checkpoints_ts/ 目录${NC}" | tee -a "$MAIN_LOG"
    else
        echo -e "  ${YELLOW}未找到预训练权重，开始训练...${NC}" | tee -a "$MAIN_LOG"
        
        # 训练带线性分支的模型
        if [ -f "ultra_lsnt_linear_branch.py" ]; then
            echo -e "  ${BLUE}训练带线性分支的模型...${NC}" | tee -a "$MAIN_LOG"
            run_script_background "ultra_lsnt_linear_branch.py" "--linear_weight 0.5 --experiment_name linear_branch"
            
            if [ $? -ne 0 ]; then
                echo -e "  ${YELLOW}线性分支模型训练失败，尝试训练标准模型...${NC}" | tee -a "$MAIN_LOG"
                run_script_background "ultra_lsnt_timeseries.py" "--data wind_final.csv --target power --gpu_optimized --experiment_name main"
            fi
        else
            echo -e "  ${YELLOW}未找到线性分支训练脚本，训练标准模型...${NC}" | tee -a "$MAIN_LOG"
            run_script_background "ultra_lsnt_timeseries.py" "--data wind_final.csv --target power --gpu_optimized --experiment_name main"
        fi
    fi
    
    # 等待所有后台任务完成
    echo -e "${YELLOW}等待训练任务完成...${NC}" | tee -a "$MAIN_LOG"
    wait
    
    # 第二步：分析专家切换过程
    echo -e "${YELLOW}第二步：分析专家切换过程${NC}" | tee -a "$MAIN_LOG"
    
    # 优先使用标准模型分析脚本
    if [ -f "analyze_expert_switching_standard.py" ]; then
        echo -e "  ${GREEN}使用标准模型分析脚本${NC}" | tee -a "$MAIN_LOG"
        run_script_foreground "analyze_expert_switching_standard.py"
    elif [ -f "analyze_expert_switching.py" ]; then
        echo -e "  ${YELLOW}使用线性分支模型分析脚本${NC}" | tee -a "$MAIN_LOG"
        run_script_foreground "analyze_expert_switching.py"
    else
        echo -e "  ${RED}错误: 未找到专家切换分析脚本${NC}" | tee -a "$MAIN_LOG"
        echo -e "  ${YELLOW}跳过专家切换分析${NC}" | tee -a "$MAIN_LOG"
    fi
    
    # 第三步：运行专家物理可解释性分析
    echo -e "${YELLOW}第三步：运行专家物理可解释性分析${NC}" | tee -a "$MAIN_LOG"
    
    if [ -f "run_expert_physics.py" ]; then
        run_script_foreground "run_expert_physics.py"
    else
        echo -e "  ${YELLOW}警告: run_expert_physics.py 不存在${NC}" | tee -a "$MAIN_LOG"
        echo -e "  ${YELLOW}跳过物理可解释性分析${NC}" | tee -a "$MAIN_LOG"
    fi
    
    # 第四步：生成专家热力图
    echo -e "${YELLOW}第四步：生成专家热力图${NC}" | tee -a "$MAIN_LOG"
    
    if [ -f "plot_expert_heatmap_enhanced.py" ]; then
        run_script_foreground "plot_expert_heatmap_enhanced.py"
    elif [ -f "plot_expert_heatmap.py" ]; then
        run_script_foreground "plot_expert_heatmap.py"
    else
        echo -e "  ${YELLOW}警告: 未找到专家热力图脚本${NC}" | tee -a "$MAIN_LOG"
        echo -e "  ${YELLOW}跳过热力图生成${NC}" | tee -a "$MAIN_LOG"
    fi
    
    # 第五步：生成汇总报告
    echo -e "${YELLOW}第五步：生成分析汇总报告${NC}" | tee -a "$MAIN_LOG"
    
    # 创建汇总报告
    cat > generate_expert_summary.py << 'EOF'
"""
生成专家分析汇总报告（增强版）
包含专家切换频率检查和关键指标分析
"""
import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime

def generate_expert_summary():
    print("生成专家分析汇总报告（增强版）...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Ultra-LSNT 专家分析汇总报告（增强版）")
    report_lines.append("=" * 70)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_lines.append(f"生成时间: {timestamp}")
    report_lines.append("")
    
    # 1. 检查生成的文件
    report_lines.append("1. 生成文件清单")
    report_lines.append("-" * 40)
    
    expert_files = [
        ("expert_switching_analysis_report*.txt", "专家切换分析报告"),
        ("expert_switching_event_*.png", "专家切换可视化图"),
        ("wind_expert_activation*.csv", "风电专家激活数据"),
        ("expert_analysis_summary.txt", "专家分析汇总报告"),
    ]
    
    for pattern, description in expert_files:
        files = glob.glob(pattern)
        if files:
            report_lines.append(f"✓ {description}:")
            for f in files[:3]:  # 最多显示3个文件
                report_lines.append(f"    - {os.path.basename(f)}")
            if len(files) > 3:
                report_lines.append(f"    ... 共 {len(files)} 个文件")
        else:
            report_lines.append(f"✗ {description}: 未找到")
    
    # 2. 检查模型权重
    report_lines.append("\n2. 模型权重状态")
    report_lines.append("-" * 40)
    
    weight_files = [
        "checkpoints_ts/linear_branch/best_model.pth",
        "checkpoints_ts/main/best_model.pth",
    ]
    
    for weight_file in weight_files:
        if os.path.exists(weight_file):
            file_size = os.path.getsize(weight_file) / (1024*1024)  # MB
            report_lines.append(f"✓ {weight_file}: {file_size:.1f} MB")
        else:
            report_lines.append(f"✗ {weight_file}: 不存在")
    
    # 3. 关键分析结果和指标
    report_lines.append("\n3. 关键分析结果和指标")
    report_lines.append("-" * 40)
    
    # 3.1 尝试读取专家切换分析报告
    report_files = glob.glob("expert_switching_analysis_report*.txt")
    if report_files:
        try:
            with open(report_files[0], "r", encoding='utf-8') as f:
                lines = f.readlines()
                # 提取关键信息
                for line in lines:
                    if "最活跃的专家" in line or "风速突变阈值" in line or "分析事件数" in line or "结论:" in line:
                        report_lines.append(f"  {line.strip()}")
        except:
            report_lines.append("  无法读取专家切换分析报告")
    else:
        report_lines.append("  未找到专家切换分析报告")
    
    # 3.2 分析专家切换频率（如果有时序数据）
    report_lines.append("\n  专家切换频率分析:")
    try:
        # 查找专家激活数据文件
        activation_files = glob.glob("wind_expert_activation*.csv")
        if activation_files:
            df = pd.read_csv(activation_files[0])
            print(f"  读取激活数据文件: {activation_files[0]}, 形状: {df.shape}")
            
            # 检查是否有时间序列数据（包含top_expert列）
            if 'top_expert' in df.columns:
                # 计算专家切换率：即 Top-1 专家发生改变的频率
                switches = (df['top_expert'].shift(1) != df['top_expert']).sum()
                total_steps = len(df) - 1  # 排除第一个
                switch_rate = switches / total_steps if total_steps > 0 else 0
                
                report_lines.append(f"    关键指标: 专家切换总次数: {switches} (反映了模型对风速波动的敏感度)")
                report_lines.append(f"    切换频率: {switch_rate:.2%} ({switches}/{total_steps} 个时间步发生切换)")
                
                # 分析专家主导情况
                expert_counts = df['top_expert'].value_counts().sort_index()
                report_lines.append(f"    各专家主导时间步统计:")
                for expert, count in expert_counts.items():
                    proportion = count / len(df)
                    report_lines.append(f"      Expert {expert}: {count} 步 ({proportion:.1%})")
            else:
                # 事件级别的数据（我们生成的格式）
                # 检查是否有专家权重列
                expert_cols = [col for col in df.columns if col.startswith('Expert_') or col.startswith('expert_')]
                if len(expert_cols) >= 2:  # 至少有2个专家列
                    report_lines.append(f"    检测到 {len(expert_cols)} 个专家的激活权重数据")
                    
                    # 计算每个事件中最活跃的专家
                    if len(df) > 0:
                        # 对于每一行（事件），找出权重最高的专家
                        expert_arrays = []
                        for col in expert_cols:
                            expert_arrays.append(df[col].values)
                        
                        # 转换为numpy数组
                        weights_matrix = np.column_stack(expert_arrays)  # (n_events, n_experts)
                        top_experts = np.argmax(weights_matrix, axis=1) + 1  # 1-based
                        
                        # 统计专家主导事件数
                        expert_event_counts = {}
                        for expert in top_experts:
                            expert_event_counts[expert] = expert_event_counts.get(expert, 0) + 1
                        
                        report_lines.append(f"    专家主导事件统计:")
                        for expert, count in sorted(expert_event_counts.items()):
                            proportion = count / len(df)
                            report_lines.append(f"      Expert {expert}: {count} 个事件 ({proportion:.1%})")
                        
                        # 计算事件间专家切换次数
                        if len(top_experts) > 1:
                            event_switches = (top_experts[:-1] != top_experts[1:]).sum()
                            event_switch_rate = event_switches / (len(top_experts) - 1)
                            report_lines.append(f"    事件间专家切换次数: {event_switches} (切换率: {event_switch_rate:.2%})")
                    
                    # 分析专家权重分布
                    report_lines.append(f"    专家权重统计:")
                    for i, col in enumerate(expert_cols, 1):
                        mean_weight = df[col].mean()
                        std_weight = df[col].std()
                        report_lines.append(f"      {col}: 均值={mean_weight:.4f}, 标准差={std_weight:.4f}")
                else:
                    report_lines.append("    警告: 数据格式不包含专家权重列，无法计算切换频率")
        else:
            report_lines.append("    未找到专家激活数据文件")
    except Exception as e:
        report_lines.append(f"    专家切换频率分析错误: {str(e)}")
    
    # 4. 建议和下一步工作
    report_lines.append("\n4. 建议和下一步工作")
    report_lines.append("-" * 40)
    report_lines.append("  • 检查专家切换可视化图，分析风速突变时的专家响应")
    report_lines.append("  • 分析专家分工模式：不同专家是否处理不同的风速模式（如稳定风、突变风）")
    report_lines.append("  • 验证专家激活权重与物理变量的相关性（如风速、湍流强度）")
    report_lines.append("  • 考虑调整线性分支权重以优化模型性能")
    report_lines.append("  • 可以尝试不同的突变检测阈值以获得更多分析案例")
    report_lines.append("  • 建议生成时间序列级别的专家激活数据以进行更精细的切换频率分析")
    
    # 5. 技术备注
    report_lines.append("\n5. 技术备注")
    report_lines.append("-" * 40)
    report_lines.append("  • 当前分析基于标准Ultra-LSNT模型（hidden_dim=128, num_experts=4）")
    report_lines.append("  • 风速突变检测阈值: 5.0 m/s，时间窗口: 12个时间步")
    report_lines.append("  • 专家切换频率反映了模型对输入变化的敏感度和适应能力")
    report_lines.append("  • 高切换频率可能表示模型对不同风速模式有细粒度响应")
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open("expert_analysis_summary_enhanced.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    # 同时更新原有的expert_analysis_summary.txt
    with open("expert_analysis_summary.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n专家分析汇总报告已保存到: expert_analysis_summary.txt 和 expert_analysis_summary_enhanced.txt")

if __name__ == '__main__':
    generate_expert_summary()
EOF
    
    python3 generate_expert_summary.py 2>&1 | tee "$LOG_DIR/expert_summary_${TIMESTAMP}.log"
    
    # 计算总运行时间
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    
    # 完成总结
    echo -e "${GREEN}===============================================================${NC}"
    echo -e "${GREEN}专家分析流程完成!${NC}"
    echo -e "${GREEN}总运行时间: ${TOTAL_TIME} 秒 ($((TOTAL_TIME/60)) 分钟)${NC}"
    echo -e "${GREEN}完成时间: $(date)${NC}"
    echo -e "${GREEN}主日志文件: $MAIN_LOG${NC}"
    echo -e "${GREEN}===============================================================${NC}"
    
    # 显示关键输出文件
    echo -e "${BLUE}关键输出文件:${NC}"
    ls -la *.png *.csv *.txt 2>/dev/null | grep -E "(expert|branch|analysis|report)" | head -10 | tee -a "$MAIN_LOG"
    
    echo -e "\n${YELLOW}专家分析汇总报告摘要:${NC}"
    if [ -f "expert_analysis_summary.txt" ]; then
        tail -20 "expert_analysis_summary.txt" | tee -a "$MAIN_LOG"
    fi
    
    echo -e "\n${GREEN}所有分析日志保存在 $LOG_DIR/ 目录中${NC}"
    echo -e "${GREEN}使用以下命令查看详细日志:${NC}"
    echo -e "  tail -f $MAIN_LOG"
}

# 执行主程序
main "$@"

# 退出码
exit 0