# Task E — 训练实验独立复核（协议 + 代码审查 + 独立重跑）

> 材料：本文件 + `重跑预注册协议.md` + 两个脚本。
> 你的任务：审查两个训练脚本是否忠于其预注册协议，并在你自己的环境重跑。
> 本任务排最后：只有 Task A–D 结论允许继续推进实证层时才做。协调人会告知是否开工。

## 0. 先读这份，再看代码

必须严格按此顺序：
1. 读 `重跑预注册协议.md`（这是**冻结**的实验规范）；
2. 再读代码 `pct_faithful_runner.py` 与 `pitfn_audit.py`；
3. **只审查代码是否忠实实现协议**，不审查协议本身好不好。

## 1. 审查清单（在报告里逐条打勾/打叉）

### 1a. PCT runner（pct_faithful_runner.py）
对照协议"实验 1"逐条核验：
- [ ] target 是否为**窗口外下一点**（无输入-目标泄漏：target 不得出现在输入通道中）
- [ ] `--epochs` 是否真正传入训练循环（而非硬编码 100）
- [ ] 打印的 `horizon` 与真实预测步数一致（协议是 horizon=1）
- [ ] window=720、batch=500、lr=1e-2 与协议一致
- [ ] 按架构分层（transformer 与 lstm 分开统计，不合并成 6 伪样本）
- [ ] 约束项使用了动态 js_ratio 开关（若有）

### 1b. PI-TFN（pitfn_audit.py）
对照协议"实验 2"：
- [ ] `P^phys = 0.5·ρ·A·Cp·v³` 是否做了 **W→kW 换算（/1000）** 再 `/p_rated`
- [ ] rated mask 是否基于**预测功率 > 额定**（而非风速 > 额定风速）
- [ ] Cp 是否有 Betz 上界（16/27）
- [ ] rotor area 是物理值还是数据反推（协议要求物理值，代码里应注明假设）

## 2. 独立重跑

> ⚠️ **先看运行前提**：两个脚本使用**相对路径**读数据，必须从脚本所在目录运行：
> ```bash
> cd 05_实验代码与脚本
> ```
> 数据文件在 `../04_数据集/prepared_priority_datasets/kelmarsh_uk_2016.csv`（脚本已内置路径）。

在你自己的环境（Python 3.11 + torch，CPU 或 GPU 均可，1 seed 各跑 1–2 epoch
做 smoke 即可；若时间允许按协议跑完整 5 seed）：

```bash
# PCT runner（1 seed smoke）
python pct_faithful_runner.py --site kelmarsh --seed 0 --epochs 2 --arch transformer --out your_result.txt
python pct_faithful_runner.py --site kelmarsh --seed 0 --epochs 2 --arch lstm --out your_result.txt

# PI-TFN（1 seed smoke）
python pitfn_audit.py --site kelmarsh --seed 0 --epochs 2 --out your_result.txt
```

数据：`../04_数据集/prepared_priority_datasets/kelmarsh_uk_2016.csv`（脚本已内置路径）。

> 若因为依赖/环境跑不起来，**这也是有效发现**：请记录报错与环境信息，
> 不要自行改我们的脚本。报告"无法复现"本身是重要结论。

## 3. 报告模板

```markdown
# TaskE 报告 — <你的名字> — <日期>

## 环境
- OS / Python / torch 版本 / CPU or GPU

## 1a PCT runner 审查
（逐条打勾，注明你依据的代码行号）

## 1b PI-TFN 审查
（逐条打勾 + 行号）

## 重跑结果
- smoke 是否跑通
- 输出关键几行（RMSE / rel）
- 若不通过：报错与定位

## 审查结论
- 代码是否忠于预注册协议（是/否/部分，列不符点）
- 你是否发现除协议声明外的其它问题（自由写，尤其欢迎）

## 声明
我只审查了"代码 vs 协议"的一致性并独立重跑；未使结果偏向任何方向。
```

## 4. 提交

- `TaskE_报告_你的名字.md`
- 你的结果文件（如 `your_result.txt`）
