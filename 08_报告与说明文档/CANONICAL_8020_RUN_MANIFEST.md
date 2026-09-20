# Canonical 80/20 重跑 · Run Manifest

- 启动时间：2026-08-31 18:00（北京时间）
- 主机：interactive730（root@192.168.17.251:25457），2 × RTX 4090
- 远程工作区：`/root/paper1/`，日志 `/root/paper1/canonical_run.log`，产物 `05_实验代码与脚本/results/canonical_8020/`

## 协议（canonical，全流程唯一）

| 项 | 值 |
| :--- | :--- |
| 数据 | `wind_final.csv`（121,528 行，15-min，新疆风场） |
| 划分 | 时序 80/20：train 97,222 行 / test 24,306 行；val 从 train 尾部切出（不触碰 test） |
| 窗口 | L=96, H=24 → test 24,187 窗口 = 580,488 预测点 |
| 种子 | 42, 123, 2025, 3407, 777（5 seeds） |
| 骨干 | Ultra-LSNT、iTransformer（同一脚本 `run_pi_moe_full_training_evidence.py --backbone`） |
| 变体 | raw / clip-only / FG-MoE（导出脚本三分支） |
| 训练 | FG 微调 8 epochs，lr=2e-5，wd=0.005，aux=0.005，λ₂=0.03（与论文套件默认一致） |

## 关键补丁（仅远端，`ultra_lsnt_timeseries.py`）

`create_dataloaders` 增加环境变量覆盖：`ULTRA_TRAIN_RATIO=0.8`、`ULTRA_VAL_RATIO=0.0`；
val_ratio≤0 时从 **train 尾部**切出验证集（test 不动）。原文件备份为 `ultra_lsnt_timeseries.py.bak_canonical`。
补丁脚本：`/root/paper1/05_实验代码与脚本/patch_split_8020.py`（本地副本 `tmp/patch_split_8020.py`）。

## 环境修复记录

1. `LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/usr/local/cuda/lib64`（否则 NGC torch 导入失败）
2. venv 内 `protobuf==3.20.3`（否则 onnx 与系统 protobuf 冲突）
3. venv 继承系统 site-packages + 安装 `lightgbm 4.7.0 / pandas 2.2.3 / pypower 5.1.21`

## 验证记录

- 划分自检：train 97,103 / val 513 / test 24,187 窗口（580,488 点）✅ 与论文 Table 1 一致
- 前一轮（70/15/15 默认）全套件复现：调度 headline 数字精确命中（slack 791.06→567.03，cost 30.88→30.44M）

## 后续阶段

- Phase E：baseline（DLinear/LightGBM/Persistence/PatchTST/TimeMixer）+ 插补对照在 80/20 下重评
- Phase D：用 canonical 预测归档重跑 120 天调度（5 seeds 全量，含 seed 稳定性均值±std）
