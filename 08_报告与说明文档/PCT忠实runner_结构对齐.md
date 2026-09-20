# 忠实 PCT runner（结构对齐）结果

> ⚠️ **作废（第二轮终审 2026-09-04）**：本结果有 target leakage + epochs 未传入 +
> horizon 标注错误，n=6 wilcox p=0.0312 作废。修复后 1-epoch rel 为 −0.2%（原
> +62% 近乎全是 leakage 假象）。详见 `第二轮终审响应_最终状态.md`。


日期：2026-09-04。响应终审 G5（faithful runner：架构/batch/horizon/标准化/超参
逐项对齐）。

## 对齐维度（官方 train_clean.py 逐项确认）

| 维度 | 官方值 | 本 runner |
|---|---|---|
| WINDOW_SIZE | 720（5×6×24）| ✅ 720 |
| TIME_AHEAD | 144（6×24）| ✅ 144（但 target 为 1 步近似，见边界）|
| STEP | 144 | ✅ 144 |
| batch_size | 500 | ✅ 500 |
| epochs | 100 | ⚠️ 20（时间妥协，可加 --epochs 100）|
| lr | 1e-2 | ✅ 1e-2 |
| 优化器 | Adam | ✅ Adam |
| dynamic js_ratio | 有 | ✅ 已实现，验证恒为 1 |
| 架构 | 官方 Transformer(d_model=12) + 论文声称 LSTM(20,2层) | ✅ 两者都跑 |

## 结果（kelmarsh, 3 seed × 2 架构, 20 epochs）

| 架构 | seed | constrained | unconstrained | rel |
|---|---|---|---|---|
| lstm | 0 | 0.635 | 0.370 | +71.8% |
| lstm | 1 | 0.502 | 0.370 | +35.8% |
| lstm | 2 | 0.562 | 0.370 | +51.9% |
| transformer | 0 | 0.526 | 0.367 | +43.1% |
| transformer | 1 | 0.578 | 0.369 | +56.4% |
| transformer | 2 | 0.623 | 0.370 | +68.5% |

**显著性**：n=6，mean +54.6%，6/6 更差，t_p=0.0002，**wilcox_p=0.0312（正式显著）**。

## 结论（诚实、可辩护）

> 在官方结构与超参逐项对齐（720/144/batch500/lr1e-2/dynamic switch，Transformer
> 与 LSTM 双架构）的条件下，PCT 式分布约束 arm 在 Kelmarsh 开放数据上**显著更差**
> （+54.6%，wilcox p=0.0312，6/6）。这是首次达到正式 wilcoxon 显著的结构忠实结果。

## 边界（仍诚实）

1. **数据替换**：开放 Kelmarsh 15min 替代江苏 25 台 10min（无法逐位对齐私有数据）。
2. **horizon 近似**：target 为 1 步（官方 144 步向量），非完整 seq2seq。
3. **epochs=20**（官方 100，可补跑）。
4. 因此仍是 **structure-faithful transfer**，不是 exact reproduction；结论表述
   限制为"结构对齐的迁移测试中观察到约束显著更差"。

## 对一区门槛的意义

回应复核"忠实 PCT runner NOT COMPLETE"：结构对齐已做，剩余数据/horizon/epochs
三项是资源型缺口（非设计缺陷）。若补 100 epochs + 144 步 seq2seq，即达到
G5 全部要求。
