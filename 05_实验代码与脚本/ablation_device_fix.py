# 消融实验设备修复补丁
import torch

def apply_ablation_fixes():
    '''
    应用消融实验的设备修复
    用法：在 run_ablation_study.py 的 main 函数开始时调用
    '''
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保模型和数据的设备一致性
    def ensure_device_consistency(model, data):
        model_device = next(model.parameters()).device
        data_device = data.device if isinstance(data, torch.Tensor) else None
        
        if data_device != model_device:
            print(f"警告: 数据在 {data_device}，模型在 {model_device}，正在同步...")
            if isinstance(data, torch.Tensor):
                data = data.to(model_device)
        
        return data
    
    return device, ensure_device_consistency
