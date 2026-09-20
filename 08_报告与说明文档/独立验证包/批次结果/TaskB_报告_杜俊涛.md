# TaskB 报告 — 杜俊涛 — 2026-09-04

## Q1 形状与元素表达式

### 1. 矩阵形状推导
在官方代码实现中：
- 单变量核计算函数 `marginal_pdf` 返回的 `kernel_values` 形状为 `(1, N, n_bins)`。其中 $N$ 为配对样本数量，$n_{\text{bins}}$ 为网格 bin 划分数；
- 设输入为第一特征（如风速 $w$）与第二特征（如功率 $p$），对应的核矩阵分别为 `kernel_values1`（形状 `(1, N, n_bins_1)`）和 `kernel_values2`（形状 `(1, N, n_bins_2)`）；
- `kernel_values1.transpose(1, 2)` 将第 1 维与第 2 维转置，形状变为 `(1, n_bins_1, N)`；
- 批矩阵乘法 `torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)`：
  - 前两维度维度匹配：`(n_bins_1, N) × (N, n_bins_2)`；
  - 缩并内积维度为样本维度 $N$；
  - 最终张量形状为：**`(1, n_bins_1, n_bins_2)`**（若两个特征 bin 数均为 $B$，则形状为 `(1, B, B)`）。

### 2. 第 $(i, j)$ 个元素的求和表达式
记第 $s$ 个配对样本（$s \in \{1, 2, \dots, N\}$）在两特征上的观测值分别为 $w_s$ 和 $p_s$；
记网格分箱点分别为 $b_{1, i}$（$i \in \{1, \dots, n_{\text{bins}_1}\}$）与 $b_{2, j}$（$j \in \{1, \dots, n_{\text{bins}_2}\}$）；
单点高斯核函数记为 $\phi_a(\Delta) = \exp\left(-\frac{1}{2}\left(\frac{\Delta}{a}\right)^2\right)$。

矩阵乘法第 $(i, j)$ 个元素是由 `kernel_values1.transpose(1, 2)` 的第 $i$ 行与 `kernel_values2` 的第 $j$ 列进行向量内积得到的，其显式数学表达式为：

$$\left[\text{joint\_kernel\_values}\right]_{i, j} = \sum_{s=1}^N \text{kernel\_values1}[0, s, i] \cdot \text{kernel\_values2}[0, s, j] = \sum_{s=1}^N \phi_{a_1}(w_s - b_{1, i}) \cdot \phi_{a_2}(p_s - b_{2, j})$$

---

## Q2 边际乘积 vs 配对 2D product-kernel（是否相等）

### 1. 本质判定
该元素是**“配对样本上的二维 product-kernel 联合密度”**（非归一化形式），而**不是**“两个边际密度的乘积”。

### 2. 数学对比与不等性证明
- **配对样本二维 Product-Kernel 联合密度**：
  在统计学中，基于 $N$ 个二维配对观测样本 $\{(w_s, p_s)\}_{s=1}^N$ 的标准二维乘积核密度估计定义为：
  $$\hat{p}_{\text{joint}}(b_{1, i}, b_{2, j}) \propto \sum_{s=1}^N \phi_{a_1}(w_s - b_{1, i}) \cdot \phi_{a_2}(p_s - b_{2, j})$$
  该式为**单重求和**，样本索引 $s$ 在风速核与功率核中严格配对绑定。
- **两个边际密度的乘积**：
  若计算两个经验边际密度的乘积 $\hat{p}_w(b_{1, i}) \cdot \hat{p}_p(b_{2, j})$，其展开形式为：
  $$\hat{p}_w(b_{1, i}) \cdot \hat{p}_p(b_{2, j}) \propto \left(\sum_{s=1}^N \phi_{a_1}(w_s - b_{1, i})\right) \cdot \left(\sum_{s'=1}^N \phi_{a_2}(p_{s'} - b_{2, j})\right) = \sum_{s=1}^N \sum_{s'=1}^N \phi_{a_1}(w_s - b_{1, i}) \cdot \phi_{a_2}(p_{s'} - b_{2, j})$$
  该式为**双重求和**（共 $N^2$ 项交叉乘积），两维度的样本索引完全解耦。

### 3. 是否相等
**两者在数学上不相等**（除非两变量在样本集上恰好完全独立，且核宽趋于特殊极限；一般情况下柯西-施瓦茨不等式及方差非零性保证 $\sum_s u_s v_s \neq \frac{1}{N} \sum_s u_s \sum_{s'} v_{s'}$）。

---

## Q3 是否蕴含边际独立

**结论：不蕴含边际独立**。

**数学依据**：
1. 若一个联合分布蕴含两变量边际独立，则对于所有网格点必须恒成立 $P(W=b_{1,i}, P=b_{2,j}) = P(W=b_{1,i}) P(P=b_{2,j})$，对应联合概率矩阵必须为秩 1 矩阵（由两边际向量的外积生成）；
2. 官方实现计算的是 Gram 矩阵内积：
   $$J = K_1^T K_2, \quad K_1 \in \mathbb{R}^{N \times n_1},\ K_2 \in \mathbb{R}^{N \times n_2}$$
   当样本中风速与功率存在强物理依赖（如典型的 S 型风电功率曲线）时，$K_1$ 与 $K_2$ 具有高度协同的行对应关系，矩阵 $J$ 的代数秩 $\min(N, n_1, n_2) > 1$，且概率质量高度浓缩在经验功率曲线对应的特定对角/流形区域，绝不满足秩 1 外积特征；
3. 样本维度的求和点对点锁定了每一个时刻 $s$ 的风速 $w_s$ 与功率 $p_s$，完整保留了经验样本对之间的联合协方差与高阶依赖结构。

---

## Q4 与标准 2D KDE 是否等价

### 1. 结构等价性
**在数学结构上完全等价**。  
标准二维乘积核（Product Kernel）高斯 KDE 的理论公式为：
$$\hat{f}(w, p) = \frac{1}{N \cdot 2\pi a_1 a_2} \sum_{s=1}^N \exp\left(-\frac{(w - w_s)^2}{2a_1^2}\right) \exp\left(-\frac{(p - p_s)^2}{2a_2^2}\right)$$
官方实现的分子为：
$$\text{joint\_kernel\_values}_{i, j} = \sum_{s=1}^N \exp\left(-\frac{(w_s - b_{1, i})^2}{2a_1^2}\right) \exp\left(-\frac{(p_s - b_{2, j})^2}{2a_2^2}\right)$$
两者在求和核心结构、核函数类型、带宽作用方式以及配对样本累加机制上是**逐点严格一一对应**的。

### 2. 差异所在
差异**仅在归一化方式（离散网格归一化 vs 连续域积分归一化）**，结构上无任何差异：
- 标准连续 KDE 的积分归一化依赖高斯积分常数 $\frac{1}{2\pi a_1 a_2}$，使得连续全空间积分 $\iint \hat{f}(w, p) dw dp = 1$；
- 官方代码面向离散网格评价（准备用于后续离散 JS 散度计算），采用了网格元素求和归一化：
  $$\text{pdf} = \frac{\text{joint\_kernel\_values}}{\sum_{i, j} \text{joint\_kernel\_values}_{i, j}}$$
  使得所有网格点上的概率质量和严格为 1（满足离散概率质量函数 PMF 的定义）。

---

## Q5 对"退化为边际乘积"说法的判定

### 1. 独立判定结论
**该说法完全不成立（错误）**。

### 2. 判定依据与致误根源分析
1. **数学推导驳斥**：
   如 Q1 与 Q2 所示，官方代码 `torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)` 执行的是样本维度 $N$ 上的单重内积：
   $$\sum_{s=1}^N \phi(w_s - b_i) \phi(p_s - b_j)$$
   这在数学定义上正是标准配对 2D 联合核密度估计的分子。若要退化为边际乘积，必须是将两个已经对样本维度求和的向量做外积（即 $\left[\sum_s \phi(w_s)\right] \cdot \left[\sum_{s'} \phi(p_{s'})\right]$），而代码中根本没有这样的操作。
2. **审查怀疑致误根源**：
   提出该批评的人很可能是仅凭函数名字（`kernel_values` 由名为 `marginal_pdf` 的函数返回）产生了字面误解，误以为从 `marginal_pdf` 出来的张量就是“边际概率”，从而想当然地认为两个输出相乘就是“边际密度的乘积”。但实际上：
   - 函数 `marginal_pdf` 内部计算了边际 PDF，同时在其 `return` 语句中一并返回了原始的样本级未缩并核矩阵 `kernel_values`（形状为 $(1, N, n_{\text{bins}})$）；
   - `joint_pdf` 消费的是这个保持样本配对未缩并的 $N \times n_{\text{bins}}$ 矩阵，并利用矩阵乘法实现了样本维度的配对缩并。
3. **结论**：
   官方实现并没有破坏风速-功率依赖，也没有将联合分布退化为边际乘积。它在离散网格归一化的意义下，忠实实现了样本配对的二维乘积核联合概率密度估计。

---

## 声明

我基于上述材料独立完成判定；未查阅本项目任何结论性文档。
