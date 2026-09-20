# TaskA 报告 — 杜俊涛 — 2026-09-04

## Q1 F.kl_div 点态定义（附文档出处）

**出处**：
查阅自 PyTorch 官方文档 `torch.nn.KLDivLoss` / `torch.nn.functional.kl_div`（查验环境版本为 PyTorch 2.11.0 / 官方稳定版文档规范）。

**点态数学定义**：
对于具有相同形状的输入张量 $y_{\text{pred}}$（对应函数参数 `input`）与 $y_{\text{true}}$（对应函数参数 `target`），当默认参数 `log_target=False` 时，PyTorch 定义的点态损失（pointwise loss）为：

$$L(y_{\text{pred}}, y_{\text{true}}) = y_{\text{true}} \cdot \left(\log y_{\text{true}} - y_{\text{pred}}\right)$$

PyTorch 明确规定：算子期望第一入参 `input` 处于对数概率空间（in log space，即 $y_{\text{pred}} = \log \hat{y}$），因此代入后点态公式等价于：

$$L(\log \hat{y}, y_{\text{true}}) = y_{\text{true}} \cdot \left(\log y_{\text{true}} - \log \hat{y}\right) = y_{\text{true}} \log \frac{y_{\text{true}}}{\hat{y}}$$

> **PyTorch 官方 Note 明确提示**：  
> *"As all the other losses in PyTorch, this function expects the first argument, `input`, to be the output of the model (e.g. the neural network) and the second, `target`, to be the observations in the dataset. This differs from the standard mathematical notation $KL(P \parallel Q)$ where $P$ denotes the distribution of the observations and $Q$ denotes the model."*

---

## Q2 推导

**调用语句**：
`F.kl_div(p.log(), q, reduction='sum')`

**推导过程**：
1. **参数对应**：
   - 入参 `input` $= \log p$
   - 入参 `target` $= q$
2. **代入 Q1 点态定义**：
   对离散分布各个分量（或连续分布求和网格各点）$i$：
   $$L_i = q_i \cdot \left(\log q_i - (\log p)_i\right) = q_i \left(\log q_i - \log p_i\right) = q_i \log \frac{q_i}{p_i}$$
3. **结合求和约简（`reduction='sum'`）**：
   $$\text{Loss} = \sum_i L_i = \sum_i q_i \log \frac{q_i}{p_i}$$
4. **比对标准离散 KL 散度数学定义**：
   离散概率分布 $U$ 与 $V$ 之间的 Kullback–Leibler 散度定义为：
   $$KL(U \parallel V) \triangleq \sum_i U_i \log \frac{U_i}{V_i}$$
   将上式与求和结果对比，显然有 $U = q$，$V = p$。

**推导结论**：
`F.kl_div(p.log(), q, reduction='sum')` 计算的是 **$KL(q \parallel p)$**，而非 $KL(p \parallel q)$。

---

## Q3 论文公式方向

**出处**：
论文原文（*Scientific Reports* 2026, 16:4237, DOI: 10.1038/s41598-025-34348-x）"Methods" 节中给出的公式 (4) 与公式 (5)：

$$KL(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx \tag{4}$$

$$JS(P \parallel Q) = \frac{1}{2} KL\left(P \Big\parallel \frac{P+Q}{2}\right) + \frac{1}{2} KL\left(Q \Big\parallel \frac{P+Q}{2}\right) \tag{5}$$

设混合平均分布 $M = \frac{P+Q}{2}$，则两个 KL 项分别为：
- **第一项**：$KL(P \parallel M)$。其中 $P$（模型输出分布）在前作为基准概率测度（对其求期望），平均分布 $M$ 在后作为比值分母；
- **第二项**：$KL(Q \parallel M)$。其中 $Q$（目标观测分布）在前作为基准概率测度，平均分布 $M$ 在后作为比值分母。

两个 KL 项各自的被比较方向均为：**“各自主体分布在前，混合平均分布 $M$ 在后”**。

---

## Q4 是否一致（精确记号）

**判定**：**不一致（方向彻底相反）**。

**官方代码实际计算分析**：
官方公开仓库 `PCT-WPF`（`train_clean.py` 第 344-352 行）实现如下：
```python
def kl_div(p, q):
    kl = F.kl_div(p.log(), q, reduction='sum')
    return kl

def js_div(p: Tensor, q: Tensor) -> Tensor:
    m = 0.5 * (p + q)
    js = 0.5 * (kl_div(p, m) + kl_div(q, m))
    return js
```
在训练流程中（`train_clean.py` 第 284/292 行）：
`js_loss = js_div(pc0, pc)`
其中 `pc0` 传入为模型预测分布 $p$，`pc` 传入为真实目标分布 $q$。

根据 Q2 的推导，函数 `kl_div(u, v)` 实际计算的是 $KL(v \parallel u)$：
1. 传入 `kl_div(p, m)` 时：第一参数为 $p$，第二参数为 $m$，实际执行的是 $F.\text{kl\_div}(p.\log(), m)$，其数学值为：
   $$KL(m \parallel p) = KL\left(\frac{p+q}{2} \Big\parallel p\right)$$
2. 传入 `kl_div(q, m)` 时：第一参数为 $q$，第二参数为 $m$，实际执行的是 $F.\text{kl\_div}(q.\log(), m)$，其数学值为：
   $$KL(m \parallel q) = KL\left(\frac{p+q}{2} \Big\parallel q\right)$$

**官方实际实现的精确记号**：
官方代码所实现的散度为：

$$JS_{\text{actual}}(p, q) = \frac{1}{2} KL\left(\frac{p+q}{2} \Big\parallel p\right) + \frac{1}{2} KL\left(\frac{p+q}{2} \Big\parallel q\right) = \frac{1}{2} KL(M \parallel p) + \frac{1}{2} KL(M \parallel q)$$

与论文公式 (5) $JS(P \parallel Q) = \frac{1}{2} KL(P \parallel M) + \frac{1}{2} KL(Q \parallel M)$ 相比，两个 KL 项内部的前后方向全部颠倒。

---

## Q5 官方实现应如何命名

**推荐学术命名**：
**“逆向 Jensen–Shannon 散度”**（Reverse Jensen–Shannon Divergence，简称 **Reverse-JS**）或 **“对偶 JS 散度”**（Dual JS Divergence）。

**命名理由**：
在信息几何与 $f$-散度（$f$-divergence）体系中，标准 JS 散度是由前向 KL 散度 $KL(\cdot \parallel M)$ 构建的；而官方代码由于 PyTorch 参数位置的语义理解偏差，实际上使用了逆向 KL 散度 $KL(M \parallel \cdot)$ 构建对称混合散度。在生成元对偶意义下，该构造精确对应 Reverse-KL 混合，因此应精确命名为 Reverse-JS 或 Dual-JS。

---

## Q6 定性含义（一两句）

标准 JS 中模型预测分布 $p$ 位于前项充当积分测度，优化时倾向于均值覆盖（Mode-covering）；而在官方 Reverse-JS 实现中，模型预测分布 $p$ 处于对数分母且由混合分布 $M = \frac{p+q}{2}$ 加权，导致模型在真实数据分布 $q$ 具有密度的任何区域都被强行要求不能为零（Zero-avoiding / 严厉惩罚低估），彻底改变了分布正则项的梯度几何与惩罚特性。

---

## 声明

我基于上述材料独立完成判定；未查阅本项目任何结论性文档。
