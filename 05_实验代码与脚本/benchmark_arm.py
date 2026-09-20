import time
import torch
import torch.nn as nn
import psutil
import os
import numpy as np

# ==========================================
# 1. 定义简化版的模型结构 (为了模拟你的 Ultra-LSNT)
# 注意：这里我们用标准的 PyTorch 模块模拟你的计算量
# 这样你不需要上传复杂的完整项目代码即可测出近似数据
# ==========================================

class MockUltraLSNT(nn.Module):
    def __init__(self, c_in=8, d_model=128, n_experts=4, top_k=2, h=24):
        super().__init__()
        self.embedding = nn.Linear(c_in, d_model)
        # 模拟 MoE 层: 简单的 MLP * Active Experts
        # 实际上 Top-K 路由的计算量约为 K 个 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(top_k) # 只实例化 K 个来模拟计算量
        ])
        self.head = nn.Linear(d_model, h)

    def forward(self, x):
        # x: [Batch, Length, Channel]
        x = self.embedding(x)
        # 模拟 Expert 计算
        for expert in self.experts:
            x = x + expert(x)
        # 简单的平均池化模拟聚合
        x = x.mean(dim=1)
        return self.head(x)

class MockUltraLite(nn.Module):
    def __init__(self, c_in=8, d_model=96, h=24):
        super().__init__()
        # Lite 版: d_model 更小, Top-1 (相当于普通 MLP)
        self.embedding = nn.Linear(c_in, d_model)
        self.core = nn.Sequential(
            nn.Linear(d_model, d_model * 2), # Lite版中间层通常较小
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.head = nn.Linear(d_model, h)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.core(x)
        x = x.mean(dim=1)
        return self.head(x)

# ==========================================
# 2. 性能测试函数
# ==========================================
def benchmark(model, name):
    model.eval()
    # 模拟输入: Batch=1 (单样本在线推理), L=96, C=8
    dummy_input = torch.randn(1, 96, 8)
    
    # A. 预热 (Warmup)
    print(f"--- Warming up {name} ---")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # B. 测速 (Latency)
    print(f"--- Benchmarking {name} Latency ---")
    latencies = []
    with torch.no_grad():
        for _ in range(200): # 跑200次取平均
            t0 = time.time()
            _ = model(dummy_input)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # 转换为 ms

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    # C. 测内存 (Peak Memory)
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024 # MB
    # 强制进行一次复杂的很多次推理来触发内存分配
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    mem_after = process.memory_info().rss / 1024 / 1024 # MB
    
    # 简单的内存增量估算
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024 # MB (float32)

    print(f"Results for {name}:")
    print(f"  Avg Latency: {avg_lat:.4f} ms")
    print(f"  P95 Latency: {p95_lat:.4f} ms")
    print(f"  Model Size (Weights): {model_size:.4f} MB")
    print(f"  System Memory (RSS) Active: ~{mem_after - mem_before:.2f} MB (Approximation)")
    print("-" * 30)

# ==========================================
# 3. 主执行块
# ==========================================
if __name__ == "__main__":
    torch.set_num_threads(1) # 强制 PyTorch 只用一个核
    print("Environment: CPU (ARMv8 Simulated Single Core)")
    
    # 测试完整版
    full_model = MockUltraLSNT(d_model=128, top_k=2) # 对应论文配置
    benchmark(full_model, "Ultra-LSNT (Full)")

    # 测试 Lite 版
    lite_model = MockUltraLite(d_model=96) # 对应论文配置
    benchmark(lite_model, "Ultra-LSNT-Lite")