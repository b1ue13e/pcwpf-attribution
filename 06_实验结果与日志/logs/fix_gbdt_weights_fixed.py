#!/usr/bin/env python3
"""
修复 run_gbdt_robustness.py 中的模型权重不匹配问题
主要问题：hidden_dim=128 vs 256 的架构不匹配
"""

import os
import sys
import json

def fix_gbdt_robustness():
    filepath = "run_gbdt_robustness.py"
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 1. 修复权重加载逻辑，使用更智能的匹配
    old_loading = '''    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model_state_dict = model_ours.state_dict()
    
    # 处理权重不匹配
    filtered_state_dict = {}
    for k, v in ckpt_state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"   ⚠️ 跳过权重 {k}：形状不匹配 {v.shape} vs {model_state_dict[k].shape}")
                # 使用模型默认初始化
        else:
            print(f"   ⚠️ 忽略不存在键：{k}")
    
    # 加载过滤后的权重
    model_state_dict.update(filtered_state_dict)
    model_ours.load_state_dict(model_state_dict, strict=False)
    print(f"   ✅ Ultra-LSNT 就位，成功加载 {len(filtered_state_dict)}/{len(model_state_dict)} 个参数")'''
    
    new_loading = '''    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # 智能权重匹配：处理架构变化 (hidden_dim 128->256 或 num_experts 4->8)
    print("   🧠 智能权重匹配中...")
    
    # 获取模型当前状态字典
    model_state_dict = model_ours.state_dict()
    
    # 首先尝试标准加载
    try:
        model_ours.load_state_dict(ckpt_state_dict, strict=False)
        print(f"   ✅ 标准加载成功 (strict=False)")
    except Exception as e1:
        print(f"   ⚠️ 标准加载失败: {e1}")
        print("   尝试维度适配...")
        
        # 手动适配权重
        filtered_state_dict = {}
        skip_count = 0
        adapt_count = 0
        
        for k, v in ckpt_state_dict.items():
            if k in model_state_dict:
                target_shape = model_state_dict[k].shape
                source_shape = v.shape
                
                if target_shape == source_shape:
                    filtered_state_dict[k] = v
                else:
                    # 尝试维度适配 (hidden_dim 128->256)
                    if len(target_shape) == len(source_shape) == 2:
                        if target_shape[0] == 256 and source_shape[0] == 128:
                            # 线性层适配：重复权重
                            import torch
                            repeat_factor = target_shape[0] // source_shape[0]
                            new_weight = v.repeat(repeat_factor, 1)
                            if target_shape[1] == source_shape[1]:
                                filtered_state_dict[k] = new_weight
                                adapt_count += 1
                                print(f"      🔧 适配权重 {k}: {source_shape} -> {target_shape}")
                                continue
                    
                    print(f"      ⚠️ 跳过权重 {k}: 形状不匹配 {source_shape} vs {target_shape}")
                    skip_count += 1
            else:
                # 处理架构变化 (num_experts 4->8)
                if 'expert' in k:
                    # 尝试找到对应的专家
                    parts = k.split('.')
                    for i, part in enumerate(parts):
                        if part.isdigit() and 'expert' in parts[i-1]:
                            expert_idx = int(part)
                            if expert_idx >= 4:  # 旧的4专家，新的8专家
                                # 映射到前4个专家
                                new_expert_idx = expert_idx % 4
                                new_key = k.replace(f'.{expert_idx}.', f'.{new_expert_idx}.')
                                if new_key in model_state_dict:
                                    filtered_state_dict[new_key] = v
                                    adapt_count += 1
                                    print(f"      🔧 映射专家 {k} -> {new_key}")
                                    break
                    else:
                        print(f"      ⚠️ 忽略不存在键: {k}")
                else:
                    print(f"      ⚠️ 忽略不存在键: {k}")
        
        # 加载适配后的权重
        if filtered_state_dict:
            model_state_dict.update(filtered_state_dict)
            model_ours.load_state_dict(model_state_dict, strict=False)
            print(f"   ✅ 适配加载成功: {len(filtered_state_dict)}个权重 ({adapt_count}个适配, {skip_count}个跳过)")
        else:
            print("   ❌ 无法加载任何权重，使用随机初始化")
    
    print("   ✅ Ultra-LSNT 就位")'''
    
    if old_loading in content:
        content = content.replace(old_loading, new_loading)
        print("✅ 已替换权重加载逻辑")
    else:
        print("ℹ️  权重加载逻辑未找到或已更新")
    
    # 2. 添加配置检查
    config_check = '''    # 检查模型配置是否匹配
    print("   🔍 检查模型配置...")
    if hasattr(model_ours, 'config'):
        model_hidden_dim = model_ours.config.hidden_dim
        print(f"     当前模型 hidden_dim = {model_hidden_dim}")
        
    # 尝试从检查点读取配置
    config_from_ckpt = None
    if 'config' in ckpt:
        config_from_ckpt = ckpt['config']
        print(f"     检查点配置: {config_from_ckpt}")
    elif os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_from_ckpt = json.load(f)
            print(f"     配置文件 hidden_dim = {config_from_ckpt.get('hidden_dim', '未知')}")
        except:
            pass'''
    
    # 在权重加载前插入配置检查
    insert_point = "ckpt_path = 'checkpoints_ts/main/best_model.pth'"
    if insert_point in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if insert_point in line:
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                config_lines = config_check.split('\n')
                for config_line in config_lines:
                    lines.insert(i+1, indent_str + config_line)
                content = '\n'.join(lines)
                print("✅ 已添加配置检查")
                break
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建权重适配工具
        weight_tool = '''#!/usr/bin/env python3
# GBDT实验权重适配工具
import torch
import json

def adapt_weights_for_gbdt():
    """
    为GBDT实验适配模型权重
    解决 hidden_dim=128->256 和 num_experts=4->8 的问题
    """
    print("权重适配工具")
    
    # 创建修复版本
    with open("run_gbdt_robustness_fixed.py", "w") as f:
        f.write('''
# 这是修复后的GBDT鲁棒性测试脚本
# 自动处理权重不匹配问题

import torch
import json

def load_ultra_lsnt_for_gbdt(input_dim, ts_config, device):
    """
    为GBDT对比实验加载Ultra-LSNT模型（修复版本）
    """
    # 加载模型配置
    config_path = 'checkpoints_ts/main/model_config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 强制使用检查点的hidden_dim（如果是128）
    from ultra_lsnt_timeseries import LSNTConfig, UltraLSNTForecaster
    
    # 使用检查点的配置
    config = LSNTConfig(**config_dict)
    config.input_dim = input_dim
    
    model = UltraLSNTForecaster(config, ts_config).to(device)
    
    # 加载权重
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # 简单加载（因为配置一致）
    model.load_state_dict(ckpt_state_dict, strict=False)
    print(f"✅ GBDT对比模型加载完成")
    
    return model

# 替换原脚本中的模型加载部分
if __name__ == "__main__":
    print("修复版本已生成: run_gbdt_robustness_fixed.py")
    print("使用方法：替换原脚本中的模型加载部分")
''')
    
    print("已创建修复版本: run_gbdt_robustness_fixed.py")
    return True

if __name__ == "__main__":
    adapt_weights_for_gbdt()
'''
        
        with open("gbdt_weight_adapter.py", "w") as f:
            f.write(weight_tool)
        print("已创建权重适配工具: gbdt_weight_adapter.py")
        
        return True
    else:
        print(f"ℹ️  {filepath} 无需修改")
        return True

if __name__ == "__main__":
    success = fix_gbdt_robustness()
    sys.exit(0 if success else 1)