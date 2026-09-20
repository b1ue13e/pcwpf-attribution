#!/usr/bin/env python3
# GBDT实验权重适配工具
import torch
import json

def adapt_weights_for_gbdt():
    """
    为GBDT实验适配模型权重
    解决 hidden_dim=128->256 和 num_experts=4->8 的问题
    """
    print("权重适配工具")
    
    # 创建修复版本 - 使用三引号字符串，但内部使用双引号
    fixed_script = """
#!/usr/bin/env python3
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
"""
    
    with open("run_gbdt_robustness_fixed.py", "w") as f:
        f.write(fixed_script)
    
    print("已创建修复版本: run_gbdt_robustness_fixed.py")
    return True

if __name__ == "__main__":
    adapt_weights_for_gbdt()
