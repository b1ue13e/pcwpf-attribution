#!/bin/bash

# fix1_fixed.sh - 修复 multi-domain baselines 缺少 MAE 输出问题（简化版）
# 作者：Roo
# 日期：2026-02-09

set -e

echo "=== fix1_fixed.sh 开始执行 ==="

cd .cache/论文 || {
    echo "错误：无法进入 .cache/论文 目录"
    exit 1
}

echo "1. 检查当前文件状态..."
if [ ! -f "run_multi_domain_baselines.py" ]; then
    echo "错误：run_multi_domain_baselines.py 不存在"
    exit 1
fi

echo "2. 备份原始文件..."
BACKUP_FILE="run_multi_domain_baselines.py.backup_$(date +%Y%m%d_%H%M%S)"
cp "run_multi_domain_baselines.py" "$BACKUP_FILE"
echo "  已备份到: $BACKUP_FILE"

echo "3. 使用 Python 进行精确修改..."
python3 << 'EOF'
import sys

filename = "run_multi_domain_baselines.py"
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找返回字典的行
found = False
for i, line in enumerate(lines):
    if 'DLinear_RMSE' in line and 'float(d_metrics["RMSE"])' in line:
        # 检查是否已经包含 MAE 字段
        if i+1 < len(lines) and 'DLinear_MAE' in lines[i+1]:
            print("MAE 字段已存在，无需修改")
            found = True
            break
            
        # 确保下一行是 timemixer_RMSE 或 patchtst_RMSE
        if i+1 < len(lines) and ('timemixer_RMSE' in lines[i+1] or 'patchtst_RMSE' in lines[i+1]):
            # 获取缩进
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # 在第 i 行之后插入 MAE 字段
            insert_lines = [
                f'{indent_str}"DLinear_MAE": float(d_metrics["MAE"]),\n',
                f'{indent_str}f"{{cfg.model2}}_MAE": float(m2_metrics["MAE"]),\n'
            ]
            
            # 插入行
            lines = lines[:i+1] + insert_lines + lines[i+1:]
            found = True
            print("已添加 MAE 字段")
            break

if not found:
    # 尝试另一种查找模式：找到包含 f"{cfg.model2}_RMSE 的行
    for i, line in enumerate(lines):
        if 'f"{cfg.model2}_RMSE"' in line:
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # 在这一行之前插入 MAE 字段
            insert_lines = [
                f'{indent_str}"DLinear_MAE": float(d_metrics["MAE"]),\n',
                f'{indent_str}f"{{cfg.model2}}_MAE": float(m2_metrics["MAE"]),\n'
            ]
            
            lines = lines[:i] + insert_lines + lines[i:]
            print("已添加 MAE 字段（第二种方式）")
            found = True
            break

if not found:
    print("警告：未找到合适的插入位置，请手动检查文件")
    sys.exit(1)

# 写回文件
with open(filename, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("文件修改完成")
EOF

echo "4. 验证修改..."
if grep -q '"DLinear_MAE": float(d_metrics["MAE"])' run_multi_domain_baselines.py && \
   grep -q 'f"{cfg.model2}_MAE": float(m2_metrics["MAE"])' run_multi_domain_baselines.py; then
    echo "  ✓ MAE 字段已成功添加"
else
    echo "  ⚠ MAE 字段可能未正确添加"
    echo "  当前文件内容摘要："
    grep -n "DLinear_MAE\|timemixer_MAE\|patchtst_MAE" run_multi_domain_baselines.py || true
fi

echo ""
echo "5. 检查 dispatch 表第四行..."
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
echo "6. 语法检查..."
if python3 -m py_compile run_multi_domain_baselines.py 2>/dev/null; then
    echo "  ✓ Python 语法检查通过"
    rm -f __pycache__/*.pyc 2>/dev/null
else
    echo "  ⚠ Python 语法检查失败，请手动检查"
fi

echo ""
echo "=== fix1_fixed.sh 执行完成 ==="
echo "总结："
echo "1. 已备份原始文件"
echo "2. 已添加 MAE 输出字段"
echo "3. 确认 dispatch 表第四行是 Informer"
echo "4. 如需三指标表格，请重新运行实验："
echo "   python3 run_multi_domain_baselines.py --model2 timemixer"