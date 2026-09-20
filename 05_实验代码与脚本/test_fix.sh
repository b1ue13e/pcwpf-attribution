#!/bin/bash
echo "=== 修复方案验证测试 ==="
echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 检查修复脚本语法
echo "1. 检查修复脚本语法..."
bash -n /root/run_coa_single_thread_fixed.sh
if [ $? -eq 0 ]; then
    echo "   ✅ 脚本语法正确"
else
    echo "   ❌ 脚本语法错误"
fi

# 2. 检查关键修复功能
echo ""
echo "2. 检查关键修复功能..."

# 检查内存监控函数
if grep -q "check_memory()" /root/run_coa_single_thread_fixed.sh; then
    echo "   ✅ 内存监控函数存在"
else
    echo "   ❌ 内存监控函数缺失"
fi

# 检查超时设置
if grep -q "timeout_hours=" /root/run_coa_single_thread_fixed.sh; then
    timeout_hours=$(grep "timeout_hours=" /root/run_coa_single_thread_fixed.sh | head -1 | cut -d'=' -f2)
    echo "   ✅ 超时设置存在: ${timeout_hours}小时"
else
    echo "   ❌ 超时设置缺失"
fi

# 检查GPU内存限制
if grep -q "set_per_process_memory_fraction" /root/run_coa_single_thread_fixed.sh; then
    echo "   ✅ GPU内存限制存在"
else
    echo "   ❌ GPU内存限制缺失"
fi

# 3. 检查当前进程状态
echo ""
echo "3. 当前进程状态:"
ps aux | grep -E "python.*coa|python3.*-u" | grep -v grep | while read line; do
    echo "   $line"
done

# 4. 检查日志文件完整性
echo ""
echo "4. 日志文件检查:"
log_count=$(ls /root/experiment_logs/coa_*.log 2>/dev/null | wc -l)
done_count=$(ls /root/experiment_logs/coa_*_done 2>/dev/null | wc -l)
echo "   COA日志文件数: $log_count"
echo "   完成标记文件数: $done_count"

# 5. 验证修复脚本可以安全运行
echo ""
echo "5. 安全运行验证..."
# 模拟运行第一个未完成的任务（不实际执行）
if [ -f /root/run_coa_single_thread_fixed.sh ]; then
    echo "   修复脚本存在，可以手动运行以下命令继续实验:"
    echo "   cd /root && ./run_coa_single_thread_fixed.sh"
else
    echo "   ❌ 修复脚本不存在"
fi

echo ""
echo "=== 验证完成 ==="
echo "建议操作:"
echo "1. 如需继续实验，运行: ./run_coa_single_thread_fixed.sh"
echo "2. 如需清理卡住进程，运行修复脚本中的清理函数"
echo "3. 监控GPU内存使用: watch -n 10 nvidia-smi"
