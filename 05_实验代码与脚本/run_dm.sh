#!/bin/bash
# 重新运行DM测试：Ultra-LSNT vs Transformer (使用与Table 3完全相同的配置)
# 解决Table 3与DM测试结果的矛盾

set -e  # 遇到错误立即退出

echo "============================================================"
echo "重新运行DM测试：Ultra-LSNT vs Transformer (使用Table 3相同配置)"
echo "============================================================"
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo ""

# 1. 检查必要文件
echo "🔍 检查必要文件..."
REQUIRED_FILES=(
    "wind_final.csv"
    "checkpoints_ts/main/best_model.pth"
    "checkpoints_ts/main/model_config.json"
    "ultra_lsnt_timeseries.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少必要文件: $file"
        exit 1
    fi
done
echo "✅ 所有必要文件都存在"

# 2. 创建虚拟环境（如果需要）
echo ""
echo "🐍 检查Python环境..."
if [ ! -d ".venv" ]; then
    echo "⚠️  未找到虚拟环境，使用系统Python"
else
    echo "✅ 使用现有虚拟环境"
fi

# 3. 运行DM测试脚本
echo ""
echo "🚀 启动DM测试脚本..."
python run_dm_transformer_fair.py

# 4. 检查结果
echo ""
echo "📊 检查结果文件..."
RESULT_DIR="dm_results"
if [ -d "$RESULT_DIR" ]; then
    LATEST_RESULT=$(ls -t "$RESULT_DIR"/dm_transformer_fair_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo "✅ 最新结果文件: $LATEST_RESULT"
        
        # 显示关键结果
        echo ""
        echo "📋 关键结果摘要:"
        python -c "
import json
import sys
try:
    with open('$LATEST_RESULT', 'r') as f:
        data = json.load(f)
    
    print(f'   Ultra-LSNT R²: {data[\"ultra_lsnt\"][\"r2\"]:.4f}')
    print(f'   Transformer R²: {data[\"transformer\"][\"r2\"]:.4f}')
    print(f'   DM统计量: {data[\"dm_test\"][\"statistic\"]:.4f}')
    print(f'   P值: {data[\"dm_test\"][\"p_value\"]:.4e}')
    print(f'   结论: {data[\"dm_test\"][\"conclusion\"]}')
    
    # 检查与Table 3的一致性
    table3_diff = abs(data[\"ultra_lsnt\"][\"r2\"] - data[\"table3_reference\"][\"ultra_lsnt_r2\"])
    if table3_diff > 0.05:
        print(f'   ⚠️  警告: Ultra-LSNT R²与Table 3差异较大 ({table3_diff:.4f})')
    
except Exception as e:
    print(f'错误读取结果文件: {e}')
    sys.exit(1)
"
    else
        echo "⚠️  未找到结果文件"
    fi
else
    echo "⚠️  结果目录不存在"
fi

# 5. 比较与原始DM测试的差异
echo ""
echo "🔄 比较与原始DM测试结果的差异..."
ORIGINAL_DM_LOG="dm_test_output.log"
if [ -f "$ORIGINAL_DM_LOG" ]; then
    echo "✅ 找到原始DM测试日志"
    
    # 提取原始结果
    ORIGINAL_P=$(grep -A1 "Ultra-LSNT vs Transformer" "$ORIGINAL_DM_LOG" | grep -o "p=[0-9.]*" | cut -d= -f2)
    if [ -n "$ORIGINAL_P" ]; then
        echo "   原始DM测试 p值: $ORIGINAL_P"
        
        if [ -n "$LATEST_RESULT" ]; then
            NEW_P=$(python -c "
import json
with open('$LATEST_RESULT', 'r') as f:
    data = json.load(f)
print(data['dm_test']['p_value'])
")
            echo "   新DM测试 p值: $NEW_P"
            
            # 比较差异
            if [ $(echo "$NEW_P < 0.05" | bc -l 2>/dev/null) -eq 1 ] && [ $(echo "$ORIGINAL_P < 0.05" | bc -l 2>/dev/null) -eq 1 ]; then
                echo "   🔄 两个测试都显示显著差异 (p<0.05)"
            elif [ $(echo "$NEW_P >= 0.05" | bc -l 2>/dev/null) -eq 1 ] && [ $(echo "$ORIGINAL_P >= 0.05" | bc -l 2>/dev/null) -eq 1 ]; then
                echo "   🔄 两个测试都不显著 (p>=0.05)"
            else
                echo "   ⚠️  警告: 两个测试结果不一致!"
                echo "       原始测试: p=$ORIGINAL_P"
                echo "       新测试: p=$NEW_P"
            fi
        fi
    fi
else
    echo "⚠️  未找到原始DM测试日志"
fi

# 6. 生成最终报告
echo ""
echo "📝 生成最终报告..."
REPORT_FILE="dm_results/dm_comparison_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "DM测试结果比较报告"
    echo "=================="
    echo "生成时间: $(date)"
    echo "测试配置: Ultra-LSNT vs Transformer (Table 3相同配置)"
    echo ""
    
    if [ -n "$LATEST_RESULT" ]; then
        python -c "
import json
with open('$LATEST_RESULT', 'r') as f:
    data = json.load(f)

print('1. 模型性能对比:')
print(f'   Ultra-LSNT: R² = {data[\"ultra_lsnt\"][\"r2\"]:.4f}, RMSE = {data[\"ultra_lsnt\"][\"rmse\"]:.2f}')
print(f'   Transformer: R² = {data[\"transformer\"][\"r2\"]:.4f}, RMSE = {data[\"transformer\"][\"rmse\"]:.2f}')
print('')
print('2. DM测试结果:')
print(f'   DM统计量: {data[\"dm_test\"][\"statistic\"]:.4f}')
print(f'   P值 (单侧): {data[\"dm_test\"][\"p_value\"]:.4e}')
print(f'   结论: {data[\"dm_test\"][\"conclusion\"]}')
print('')
print('3. 与Table 3的对比:')
print(f'   Table 3 Ultra-LSNT R²: {data[\"table3_reference\"][\"ultra_lsnt_r2\"]:.4f}')
print(f'   当前 Ultra-LSNT R²: {data[\"ultra_lsnt\"][\"r2\"]:.4f}')
print(f'   Table 3 Transformer R²: {data[\"table3_reference\"][\"transformer_r2\"]:.4f}')
print(f'   当前 Transformer R²: {data[\"transformer\"][\"r2\"]:.4f}')
print('')
print('4. 差异分析:')
diff_r2 = data[\"ultra_lsnt\"][\"r2\"] - data[\"transformer\"][\"r2\"]
if diff_r2 > 0:
    print(f'   Ultra-LSNT R²比Transformer高 {diff_r2:.4f}')
else:
    print(f'   Ultra-LSNT R²比Transformer低 {-diff_r2:.4f}')
"
    fi
    
    echo ""
    echo "5. 结论与建议:"
    echo "   - 如果p<0.05且DM统计量为负: Ultra-LSNT显著优于Transformer"
    echo "   - 如果p<0.05且DM统计量为正: Transformer显著优于Ultra-LSNT"
    echo "   - 如果p>=0.05: 两者无显著差异"
    echo ""
    echo "6. 对论文的影响:"
    echo "   - 如果结果与Table 3一致: 维持现有结论"
    echo "   - 如果结果与Table 3矛盾: 需要重新评估性能对比部分"
    
} > "$REPORT_FILE"

echo "✅ 报告已生成: $REPORT_FILE"

# 7. 完成
echo ""
echo "============================================================"
echo "🎉 DM测试重新运行完成!"
echo ""
echo "📁 结果文件保存在: dm_results/"
echo "📝 详细报告: $REPORT_FILE"
echo ""
echo "💡 下一步建议:"
echo "   1. 检查结果是否解决了Table 3与原始DM测试的矛盾"
echo "   2. 如果仍有矛盾，可能需要重新训练Transformer到Table 3的准确性能"
echo "   3. 更新论文中的相关结论"
echo "============================================================"