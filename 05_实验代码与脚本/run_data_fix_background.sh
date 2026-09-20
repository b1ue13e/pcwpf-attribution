#!/bin/bash
# 后台运行数据修复脚本
# 确保数据完整性，定期检查和修复

LOG_DIR="/root/data_fix_logs"
mkdir -p $LOG_DIR

# 设置环境
export PYTHONUNBUFFERED=1

# 日志文件
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/data_fix_${TIMESTAMP}.log"

echo "========================================" | tee -a $LOG_FILE
echo "数据完整性修复脚本 (后台运行)" | tee -a $LOG_FILE
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 函数：检查数据完整性
check_data_integrity() {
    echo "[$(date '+%H:%M:%S')] 检查数据完整性..." | tee -a $LOG_FILE
    
    # 1. 检查 all_models_clean_80_20.csv
    if [ ! -f "all_models_clean_80_20.csv" ]; then
        echo "  ❌ all_models_clean_80_20.csv 不存在" | tee -a $LOG_FILE
        return 1
    fi
    
    # 检查 wind_cn Ultra-LSNT-Lite 的3个种子
    python3 -c "
import pandas as pd
try:
    df = pd.read_csv('all_models_clean_80_20.csv')
    wind_ultra = df[(df['dataset'] == 'wind_cn') & (df['model'] == 'Ultra-LSNT-Lite')]
    seeds = wind_ultra['seed'].unique()
    if len(seeds) >= 3:
        print('✅ wind_cn Ultra-LSNT-Lite 有 {} 个种子: {}'.format(len(seeds), sorted(seeds)))
    else:
        print('❌ wind_cn Ultra-LSNT-Lite 只有 {} 个种子'.format(len(seeds)))
except Exception as e:
    print('❌ 检查失败:', str(e))
" 2>&1 | tee -a $LOG_FILE
    
    # 2. 检查 metaheuristic_metrics_all_datasets.csv
    if [ ! -f "metaheuristic_metrics_all_datasets.csv" ]; then
        echo "  ❌ metaheuristic_metrics_all_datasets.csv 不存在" | tee -a $LOG_FILE
        return 1
    fi
    
    # 检查核心组合是否都有3个种子
    python3 -c "
import pandas as pd
try:
    df = pd.read_csv('metaheuristic_metrics_all_datasets.csv')
    datasets = ['wind_cn', 'electricity', 'air_quality', 'gefcom']
    models = ['COA-BiLSTM', 'BWO-SVR', 'BWO-CNN']
    all_ok = True
    for d in datasets:
        for m in models:
            subset = df[(df['dataset'] == d) & (df['model'] == m)]
            if len(subset) < 3:
                print('❌ {}/{} 只有 {} 个种子'.format(d, m, len(subset)))
                all_ok = False
    if all_ok:
        print('✅ 所有核心组合都有3个种子')
except Exception as e:
    print('❌ 检查失败:', str(e))
" 2>&1 | tee -a $LOG_FILE
    
    # 3. 检查 metaheuristic_latency_energy.csv
    if [ ! -f "metaheuristic_latency_energy.csv" ]; then
        echo "  ❌ metaheuristic_latency_energy.csv 不存在" | tee -a $LOG_FILE
        return 1
    fi
    
    # 检查延迟数据是否有NaN
    python3 -c "
import pandas as pd
try:
    df = pd.read_csv('metaheuristic_latency_energy.csv')
    nan_count = df['latency_ms_batch1'].isna().sum()
    if nan_count == 0:
        print('✅ 延迟数据完整，无NaN值')
    else:
        print('❌ 延迟数据有 {} 个NaN值'.format(nan_count))
except Exception as e:
    print('❌ 检查失败:', str(e))
" 2>&1 | tee -a $LOG_FILE
}

# 函数：运行修复脚本
run_fix_script() {
    echo "[$(date '+%H:%M:%S')] 运行修复脚本..." | tee -a $LOG_FILE
    
    if [ ! -f "fix_missing_data.py" ]; then
        echo "  ❌ 修复脚本 fix_missing_data.py 不存在" | tee -a $LOG_FILE
        return 1
    fi
    
    # 运行修复脚本
    python3 fix_missing_data.py 2>&1 | tee -a $LOG_FILE
    
    # 检查退出码
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "  ✅ 修复脚本执行成功" | tee -a $LOG_FILE
        return 0
    else
        echo "  ❌ 修复脚本执行失败" | tee -a $LOG_FILE
        return 1
    fi
}

# 函数：生成状态报告
generate_status_report() {
    echo "[$(date '+%H:%M:%S')] 生成状态报告..." | tee -a $LOG_FILE
    
    REPORT_FILE="$LOG_DIR/status_report_${TIMESTAMP}.txt"
    
    cat > "$REPORT_FILE" << EOF
数据完整性状态报告
生成时间: $(date '+%Y-%m-%d %H:%M:%S')

1. 文件检查:
$(ls -la all_models_clean_80_20.csv metaheuristic_metrics_all_datasets.csv metaheuristic_latency_energy.csv 2>/dev/null)

2. 数据完整性:
EOF
    
    # 运行完整性检查并追加到报告
    echo "运行完整性检查..." >> "$REPORT_FILE"
    python3 -c "
import pandas as pd
import os

def check_file(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return f'✅ {filepath}: {len(df)} 行, {len(df.columns)} 列'
    else:
        return f'❌ {filepath}: 文件不存在'

files = [
    'all_models_clean_80_20.csv',
    'metaheuristic_metrics_all_datasets.csv', 
    'metaheuristic_latency_energy.csv'
]

for f in files:
    print(check_file(f))
" >> "$REPORT_FILE" 2>&1
    
    echo "" >> "$REPORT_FILE"
    echo "3. 最近修改时间:" >> "$REPORT_FILE"
    stat -c "%n: %y" all_models_clean_80_20.csv metaheuristic_metrics_all_datasets.csv metaheuristic_latency_energy.csv 2>/dev/null >> "$REPORT_FILE"
    
    echo "状态报告已保存到: $REPORT_FILE" | tee -a $LOG_FILE
}

# 主循环
main_loop() {
    echo "[$(date '+%H:%M:%S')] 进入主监控循环..." | tee -a $LOG_FILE
    echo "按 Ctrl+C 停止监控" | tee -a $LOG_FILE
    
    # 初始检查
    check_data_integrity
    
    # 如果需要修复，运行修复脚本
    echo "[$(date '+%H:%M:%S')] 检查是否需要修复..." | tee -a $LOG_FILE
    python3 -c "
import pandas as pd
import sys

need_fix = False

# 检查 wind_cn Ultra-LSNT-Lite
try:
    df = pd.read_csv('all_models_clean_80_20.csv')
    wind_ultra = df[(df['dataset'] == 'wind_cn') & (df['model'] == 'Ultra-LSNT-Lite')]
    if len(wind_ultra) < 3:
        print('需要修复: wind_cn Ultra-LSNT-Lite 只有 {} 个种子'.format(len(wind_ultra)))
        need_fix = True
except:
    need_fix = True

# 检查核心组合种子
try:
    df = pd.read_csv('metaheuristic_metrics_all_datasets.csv')
    datasets = ['wind_cn', 'electricity']
    models = ['COA-BiLSTM', 'BWO-SVR', 'BWO-CNN']
    for d in datasets:
        for m in models:
            subset = df[(df['dataset'] == d) & (df['model'] == m)]
            if len(subset) < 3:
                print('需要修复: {}/{} 只有 {} 个种子'.format(d, m, len(subset)))
                need_fix = True
except:
    need_fix = True

# 检查延迟数据
try:
    df = pd.read_csv('metaheuristic_latency_energy.csv')
    if df['latency_ms_batch1'].isna().any():
        print('需要修复: 延迟数据有NaN值')
        need_fix = True
except:
    need_fix = True

if need_fix:
    print('NEEDS_FIX')
else:
    print('OK')
" 2>&1 | tee -a $LOG_FILE | grep -q "NEEDS_FIX"
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] 检测到数据问题，运行修复..." | tee -a $LOG_FILE
        run_fix_script
    else
        echo "[$(date '+%H:%M:%S')] 数据完整，无需修复" | tee -a $LOG_FILE
    fi
    
    # 生成状态报告
    generate_status_report
    
    echo "[$(date '+%H:%M:%S')] 监控循环完成" | tee -a $LOG_FILE
    echo "将在1小时后再次检查..." | tee -a $LOG_FILE
}

# 执行主循环
main_loop

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "数据完整性修复完成" | tee -a $LOG_FILE
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE