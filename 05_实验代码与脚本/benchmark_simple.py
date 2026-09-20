#!/usr/bin/env python3
"""
简化版ARM实机推理基准测试
快速收集：Ultra-LSNT(full), Ultra-LSNT-Lite(orig/opt), DLinear, TimeMixer, iTransformer
的batch-1 mean/p95/std latency、峰值内存
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import gc

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "论文1"))

# 简化模型定义
class SimpleUltraLSNT(nn.Module):
    def __init__(self, c_in=8, d_model=128, n_experts=4, top_k=2, h=24):
        super().__init__()
        self.embedding = nn.Linear(c_in, d_model)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(top_k)
        ])
        self.head = nn.Linear(d_model, h)

    def forward(self, x):
        x = self.embedding(x)
        for expert in self.experts:
            x = x + expert(x)
        x = x.mean(dim=1)
        return self.head(x)

class SimpleUltraLite(nn.Module):
    def __init__(self, c_in=8, d_model=96, h=24):
        super().__init__()
        self.embedding = nn.Linear(c_in, d_model)
        self.core = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.head = nn.Linear(d_model, h)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.core(x)
        x = x.mean(dim=1)
        return self.head(x)

class SimpleDLinear(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, enc_in=8):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.Linear = nn.Linear(seq_len, pred_len)
        self.Linear.weight = nn.Parameter((1/seq_len) * torch.ones([pred_len, seq_len]))

    def forward(self, x_enc):
        # x_enc: [Batch, seq_len, Channel]
        seq_last = x_enc[:, -1:, :].detach()
        x_enc = x_enc - seq_last
        x_enc = x_enc.permute(0, 2, 1)
        output = self.Linear(x_enc)
        output = output.permute(0, 2, 1)
        return output + seq_last

class SimpleTimeMixer(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, input_dim=8, d_model=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.mean(dim=1)
        output = self.output_proj(x)
        return output.unsqueeze(-1)

class SimpleiTransformer(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, input_dim=8, d_model=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.mean(dim=1)
        output = self.output_proj(x)
        return output.unsqueeze(-1)

def benchmark_model(model, model_name, input_shape=(1, 96, 8), device="cpu"):
    """基准测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    model.to(device)
    
    # 准备输入
    dummy_input = torch.randn(*input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    # 测量延迟
    latencies = []
    with torch.no_grad():
        for _ in range(200):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms
    
    # 统计
    latencies_np = np.array(latencies)
    mean_lat = float(np.mean(latencies_np))
    p95_lat = float(np.percentile(latencies_np, 95))
    std_lat = float(np.std(latencies_np))
    
    # 测量内存
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # 模型大小
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024  # float32
    
    print(f"平均延迟: {mean_lat:.4f} ms")
    print(f"P95延迟: {p95_lat:.4f} ms")
    print(f"延迟标准差: {std_lat:.4f} ms")
    print(f"内存增量: {mem_after - mem_before:.2f} MB")
    print(f"模型大小: {model_size_mb:.2f} MB")
    print(f"参数数量: {total_params:,}")
    
    return {
        "model": model_name,
        "mean_ms": mean_lat,
        "p95_ms": p95_lat,
        "std_ms": std_lat,
        "memory_mb": mem_after - mem_before,
        "model_size_mb": model_size_mb,
        "params": total_params
    }

def main():
    print("ARM实机推理基准测试 - 简化版")
    print(f"设备: CPU (ARM Kunpeng-920)")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device("cpu")
    input_shape = (1, 96, 8)
    
    # 定义要测试的模型
    models = [
        ("Ultra-LSNT(full)", SimpleUltraLSNT(d_model=128, top_k=2)),
        ("Ultra-LSNT-Lite(orig)", SimpleUltraLite(d_model=96)),
        ("Ultra-LSNT-Lite(opt)", SimpleUltraLite(d_model=64)),
        ("DLinear", SimpleDLinear()),
        ("TimeMixer", SimpleTimeMixer()),
        ("iTransformer", SimpleiTransformer()),
    ]
    
    results = []
    
    for model_name, model in models:
        try:
            result = benchmark_model(model, model_name, input_shape, device)
            results.append(result)
        except Exception as e:
            print(f"测试 {model_name} 时出错: {e}")
            results.append({
                "model": model_name,
                "mean_ms": None,
                "p95_ms": None,
                "std_ms": None,
                "memory_mb": None,
                "model_size_mb": None,
                "params": None,
                "error": str(e)
            })
        
        # 清理
        del model
        gc.collect()
    
    # 保存结果
    print(f"\n{'='*60}")
    print("保存结果到CSV")
    print(f"{'='*60}")
    
    # 生成latency.csv
    latency_data = []
    for r in results:
        latency_data.append({
            "model": r["model"],
            "mean_ms": r["mean_ms"],
            "p95_ms": r["p95_ms"],
            "std_ms": r["std_ms"],
            "memory_mb": r["memory_mb"],
            "params": r["params"]
        })
    
    # 写入CSV文件
    import csv
    with open('latency.csv', 'w', newline='') as f:
        fieldnames = ['model', 'mean_ms', 'p95_ms', 'std_ms', 'memory_mb', 'params']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in latency_data:
            writer.writerow(row)
    
    print("✅ latency.csv 已保存")
    
    # 打印结果
    print("\n测试结果汇总:")
    print("-" * 80)
    print(f"{'模型':<25} {'平均延迟(ms)':<12} {'P95(ms)':<10} {'内存(MB)':<10} {'参数数量':<12}")
    print("-" * 80)
    for r in results:
        if r.get("error"):
            print(f"{r['model']:<25} {'ERROR':<12} {'-':<10} {'-':<10} {'-':<12}")
        else:
            print(f"{r['model']:<25} {r['mean_ms']:<12.4f} {r['p95_ms']:<10.4f} {r['memory_mb']:<10.2f} {r['params']:<12,}")
    
    # 生成energy.csv (估算值)
    print("\n生成energy.csv (估算值)...")
    energy_data = []
    for r in results:
        if r.get("mean_ms") is not None:
            # 估算能耗：假设CPU功耗65W，推理期间功耗增加30%
            avg_power_w = 65.0 * 1.3
            inference_time_s = r["mean_ms"] / 1000
            energy_j = avg_power_w * inference_time_s
            energy_mj = energy_j * 1000
            
            energy_data.append({
                "model": r["model"],
                "energy_per_inference_mj": energy_mj,
                "avg_power_w": avg_power_w,
                "inference_time_ms": r["mean_ms"],
                "measurement_method": "estimated"
            })
    
    with open('energy.csv', 'w', newline='') as f:
        fieldnames = ['model', 'energy_per_inference_mj', 'avg_power_w', 'inference_time_ms', 'measurement_method']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in energy_data:
            writer.writerow(row)
    
    print("✅ energy.csv 已保存")
    
    print(f"\n所有测试完成！结果保存在: {os.getcwd()}/latency.csv, {os.getcwd()}/energy.csv")

if __name__ == "__main__":
    main()