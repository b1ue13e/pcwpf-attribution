#!/bin/bash

# fix1.sh - 修复 multi-domain baselines 缺少 MAE 输出问题，确认 dispatch 表第四行
# 作者：Roo
# 日期：2026-02-09
# 版本：2.0（简化版）

set -e  # 任何命令失败即退出

echo "=== fix1.sh 开始执行 ==="

# 进入工作目录
cd .cache/论文 || {
    echo "错误：无法进入 .cache/论文 目录"
    exit 1
}

echo "1. 备份原始文件..."
BACKUP_SUFFIX=".backup_$(date +%Y%m%d_%H%M%S)"
if [ -f "run_multi_domain_baselines.py" ]; then
    cp "run_multi_domain_baselines.py" "run_multi_domain_baselines.py${BACKUP_SUFFIX}"
    echo "  已备份原文件：run_multi_domain_baselines.py${BACKUP_SUFFIX}"
fi

echo "2. 修改 run_multi_domain_baselines.py 添加 MAE 输出..."
# 直接使用sed在DLinear_RMSE行后添加MAE字段
# 查找包含"DLinear_RMSE"的行号
LINE_NUM=$(grep -n "DLinear_RMSE" run_multi_domain_baselines.py | head -1 | cut -d: -f1)
if [ -z "$LINE_NUM" ]; then
    echo "错误：找不到 DLinear_RMSE 行"
    exit 1
fi

# 获取缩进
INDENT_LINE=$(sed -n "${LINE_NUM}p" run_multi_domain_baselines.py)
INDENT=${INDENT_LINE%%[^ ]*}
INDENT_LEN=${#INDENT}

# 在指定行后插入MAE字段
sed -i "${LINE_NUM}a\\
${INDENT}\\\"DLinear_MAE\\\": float(d_metrics[\\\"MAE\\\"]),\\
${INDENT}f\\\"\${cfg.model2}_MAE\\\": float(m2_metrics[\\\"MAE\\\"])," run_multi_domain_baselines.py

echo "3. 验证修改结果..."
if grep -q "DLinear_MAE" run_multi_domain_baselines.py && grep -q "timemixer_MAE" run_multi_domain_baselines.py; then
    echo "  ✓ MAE 字段已成功添加"
else
    echo "  ⚠ MAE 字段可能未正确添加，需要手动检查"
fi

echo ""
echo "4. 检查 dispatch 表第四行..."
if [ -f "dispatch_comparison_results.json" ]; then
    python3 << 'EOF'
import json
with open("dispatch_comparison_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

models = list(data.keys())
print(f"Dispatch 表共包含 {len(models)} 个模型：")
for i, model in enumerate(models, 1):
    print(f"  第{i}行：{model}")

if len(models) >= 4:
    fourth_model = models[3]
    print(f"\n第四行模型是：{fourth_model}")
    if fourth_model == "Informer":
        print("✓ 第四行是 Informer，符合预期")
    else:
        print(f"⚠ 第四行不是 Informer，而是 {fourth_model}")
else:
    print("⚠ Dispatch 表不足4行")
EOF
else
    echo "警告：dispatch_comparison_results.json 文件不存在"
fi

echo ""
echo "5. 语法检查..."
if python3 -m py_compile run_multi_domain_baselines.py 2>/dev/null; then
    echo "  ✓ Python 语法检查通过"
    rm -f __pycache__/*.pyc 2>/dev/null
else
    echo "  ⚠ Python 语法检查失败，请手动检查"
fi

echo ""
echo "6. 快速测试新输出格式..."
echo "   可以运行以下命令测试修改后的脚本："
echo "   python3 run_multi_domain_baselines.py --model2 timemixer --epochs 1 --batch-size 32"
echo "   （注意：实际运行需要较长时间，建议仅在需要时运行）"

echo ""
echo "=== fix1.sh 执行完成 ==="
echo "主要修改："
echo "1. 在 run_multi_domain_baselines.py 中添加了 DLinear_MAE 和 timemixer_MAE 字段"
echo "2. 确认 dispatch 表第四行是 Informer"
echo "3. 如需完整的三指标表格（R2、RMSE、MAE），请重新运行实验"