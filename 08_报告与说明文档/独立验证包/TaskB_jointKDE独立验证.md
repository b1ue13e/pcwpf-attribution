# Task B — joint-KDE 实现性质的独立验证

> 材料：本文件 + 公开来源（论文、官方仓库）。
> 你的任务：独立判定官方代码中"联合概率密度"实现的数学性质。
> 请先完成判定再写报告，不要在本项目任何文档里寻找"期望答案"。

## 1. 背景

某论文（<https://www.nature.com/articles/s41598-025-34348-x>）用预测功率与风速的
**联合分布**构造训练正则项，论文方法节写的是标准的二维核密度估计（2D KDE）。

其公开官方代码仓库：<https://github.com/daxin007/PCT-WPF>

仓库中相关实现的结构（逐字转录，仅变量名简化；请以仓库当前版本为准）：

```python
def marginal_pdf(values, bins, a, b=1e-10):
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / a).pow(2))   # 高斯核
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + b
    pdf = pdf / normalization
    return pdf, kernel_values

def joint_pdf(kernel_values1, kernel_values2):
    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    pdf = joint_kernel_values / normalization
    return pdf
```

其中：
- `values` 形状为 `(1, N, 1)`（N 个配对样本，每样本一个标量）；
- `bins` 形状为 `(n_bins,)`；
- `kernel_values` 返回形状为 `(1, N, n_bins)`——注意：**不是向量，是样本×bin 矩阵**；
- `joint_pdf` 中 `normalization` 为二维求和（见仓库原版）。

## 2. 你的任务

逐一回答，每题都要有推导：

1. 写出 `matmul(kernel_values1.transpose(1,2), kernel_values2)` 的矩阵形状，
   并写出结果第 $(i,j)$ 个元素的求和表达式（用 $N$ 个配对样本的核函数值表示）。
2. 该元素是"两个**边际**密度的乘积" $\hat p(b_i)\,\hat p(b_j)$，还是
   "配对样本上的二维 product-kernel 联合密度"
   $\propto \sum_s \phi(b_i; w_s)\,\phi(b_j; p_s)$？两者在数学上是否相等？
3. 该实现是否蕴含"功率与风速边际独立"？为什么？
4. 该实现与标准 2D KDE（product kernel，对 $N$ 个配对样本求和）在数学上是否等价
   （同核、同带宽、同归一化下）？若有差异，差异在归一化还是结构？
5. 若有人写"官方实现把联合分布退化成两个边际密度的乘积、破坏了风速-功率依赖"，
   你的判定是：该说法**成立 / 不成立 / 部分成立**？请给出结论和依据。

## 3. 报告模板

```markdown
# TaskB 报告 — <你的名字> — <日期>

## Q1 形状与元素表达式
...

## Q2 边际乘积 vs 配对 2D product-kernel（是否相等）
...

## Q3 是否蕴含边际独立
...

## Q4 与标准 2D KDE 是否等价
...

## Q5 对"退化为边际乘积"说法的判定（成立/不成立/部分成立 + 依据）
...

## 声明
我基于上述材料独立完成判定；未查阅本项目任何结论性文档。
```

## 4. 提交

文件命名 `TaskB_报告_你的名字.md`，交给协调人。
