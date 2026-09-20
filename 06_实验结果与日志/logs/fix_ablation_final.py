#!/usr/bin/env python3
"""
修复 run_ablation_study.py 中的CUDA/CPU设备不匹配问题 - 最终版本
"""

import os
import sys

def fix_ablation_study():
    filepath = "run_ablation_study.py"
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    
    # 修复 violent_replace_conv_kernel 函数中的设备问题
    for i, line in enumerate(lines):
        if 'new_layer = nn.Conv1d(' in line and i+1 < len(lines):
            next_line = lines[i+1]
            if '.to(' not in next_line:
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                lines.insert(i+1, f'{indent_str}# 确保新层与模型在同一设备上\n')
                lines.insert(i+2, f'{indent_str}new_layer = new_layer.to(next(model.parameters()).device)\n')
                modified = True
                break
    
    # 确保训练循环中的设备一致性
    for i, line in enumerate(lines):
        if 'out, _ = model(bx)' in line and i > 0:
            found = False
            for j in range(max(0, i-3), i):
                if 'bx = bx.to(device)' in lines[j] or 'bx, by = bx.to(device)' in lines[j]:
                    found = True
                    break
            if not found:
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                lines.insert(i, f'{indent_str}# 确保输入在正确设备上\n')
                lines.insert(i+1, f'{indent_str}bx = bx.to(device)\n')
                modified = True
                break
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ 已修复 {filepath} 中的设备不匹配问题")
        
        # 创建设备修复补丁
        patch_content = '''#!/usr/bin/env python3
# 消融实验设备修复补丁
import torch

def apply_ablation_device_fix():
    """
    应用消融实验的设备修复
    用法：在 run_ablation_study.py 的 main 函数开始时调用
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    def ensure_device_consistency(model, data):
        model_device = next(model.parameters()).device
        data_device = data.device if isinstance(data, torch.Tensor) else None
        
        if data_device != model_device:
            print(f"警告: 数据在 {data_device}，模型在 {model_device}，正在同步...")
            if isinstance(data, torch.Tensor):
                data = data.to(model_device)
        
        return data
    
    return device, ensure_device_consistency

if __name__ == "__main__":
    print("消融实验设备修复补丁已加载")
'''
        
        with open("ablation_device_fix_final.py", "w") as f:
            f.write(patch_content)
        print("已创建设备修复补丁: ablation_device_fix_final.py")
        
        return True
    else:
        print(f"ℹ️  {filepath} 无需修改")
        return True

if __name__ == "__main__":
    success = fix_ablation_study()
    sys.exit(0 if success else 1)
