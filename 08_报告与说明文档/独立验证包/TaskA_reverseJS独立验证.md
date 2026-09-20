# Task A — reverse-JS 方向独立验证

> 材料：本文件 + 公开来源（论文、官方仓库、PyTorch 文档）。
> 你的任务：独立判定一行官方代码与论文公式是否一致。请先完成判定再写报告，
> 不要在本项目任何文档里寻找"期望答案"。

## 1. 背景

某论文提出用 Jensen–Shannon 散度作为训练正则项，约束模型输出分布与目标分布。
论文方法节给出的公式是标准 JS：

$$JS(P\|Q) = \frac{1}{2}\,KL(P\|M) + \frac{1}{2}\,KL(Q\|M), \qquad M=\frac{P+Q}{2}$$

论文原文：<https://www.nature.com/articles/s41598-025-34348-x>
（看 Methods / 公式部分即可。）

其公开官方代码仓库：<https://github.com/daxin007/PCT-WPF>
（重点文件 `train_clean.py`。）

官方仓库中与该正则项相关的实现为（逐字转录，仅变量名简化）：

```python
def js_div(p_output, q_target, get_softmax=True):
    ...
    kl_divergence = F.kl_div(p_output.log(), q_target, reduction='sum')
    ...
    js_divergence = (kl_divergence + kl_divergence_2) / 2.0
    return js_divergence
```

（`p_output` 为模型输出经 softmax 后的分布；`q_target` 为预计算的目标分布。
若仓库版本与此转录有出入，以仓库为准，并在报告中说明。）

## 2. 你的任务

按以下问题逐一回答，每个回答都要有推导或出处：

1. 写出 PyTorch `F.kl_div(input, target)` 的点态定义
   （参考 PyTorch 官方文档，注明你查的版本）。
2. 据此推导：`F.kl_div(p.log(), q, reduction='sum')` 计算的是
   $KL(p\|q)$ 还是 $KL(q\|p)$？
3. 论文公式中的两个 KL 项，各自的被比较方向是什么（谁在前谁在后）？
4. 官方这一项与论文公式的对应项，方向是否一致？若不一致，用精确记号写出
   官方实际实现的是什么（例如写出 $KL(\cdot\|\cdot)$ 形式）。
5. 若要给官方实现一个准确的名字（不能笼统叫"非标准 KL"），你会叫什么？
6. 若论文里 $q$ 是目标分布、$p$ 是模型输出分布，方向不一致在训练中意味着
   什么？用一两句定性地讲，不要推测数值后果。

## 3. 报告模板

```markdown
# TaskA 报告 — <你的名字> — <日期>

## Q1 F.kl_div 点态定义（附文档出处）
...

## Q2 推导
...

## Q3 论文公式方向
...

## Q4 是否一致（精确记号）
...

## Q5 官方实现应如何命名
...

## Q6 定性含义（一两句）
...

## 声明
我基于上述材料独立完成判定；未查阅本项目任何结论性文档。
```

## 4. 提交

文件命名 `TaskA_报告_你的名字.md`，交给协调人。
