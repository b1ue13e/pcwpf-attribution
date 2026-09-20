================================================================================
Ultra-LSNT SOTA Search 完整运行包
================================================================================

包含文件说明：
----------------
1. run_ultra_lsnt_sota_search_4090.py  - 主入口脚本（配置搜索调度器）
2. run_ultra_lsnt_sota_push_4090.py    - 训练执行脚本（被主脚本调用）
3. run_ultra_lsnt_sota_ensemble_4090.py - 模型集成脚本（可选）
4. ultra_lsnt_sota_push.py             - SOTA模型定义（核心依赖）
5. ultra_lsnt_timeseries.py            - 时间序列模型基础类（核心依赖）

使用方法：
----------------

1. 新疆数据快速验证（推荐先跑这个）：

python run_ultra_lsnt_sota_search_4090.py \
    --train-data xinjiang_wind_2019 \
    --target power \
    --train-name Xinjiang \
    --presets "balanced" \
    --seeds "42" \
    --epochs 40 \
    --pretrain-epochs 15 \
    --batch-size 128 \
    --device cuda \
    --num-workers 4 \
    --out-dir results/xinjiang_main/sota_search

2. 新疆数据完整搜索：

python run_ultra_lsnt_sota_search_4090.py \
    --train-data xinjiang_wind_2019 \
    --target power \
    --train-name Xinjiang \
    --presets "balanced,wide,deep" \
    --seeds "42,52,62" \
    --epochs 60 \
    --pretrain-epochs 20 \
    --batch-size 128 \
    --device cuda \
    --num-workers 4 \
    --export-best-edge-ready \
    --out-dir results/xinjiang_main/sota_search

预期输出：
----------------
results/xinjiang_main/sota_search/
├── leaderboard.csv          - 所有配置排行榜
├── best_run.json            - 最优配置详情
└── balanced_seed42/
    └── main_Xinjiang/
        ├── test_metrics.csv - 测试指标（R², RMSE, MAE）
        ├── stage_summary.json
        └── checkpoints/

关键参数：
----------------
--train-data: 数据集别名（xinjiang_wind_2019 / wind_turbine_primary）
--train-name: 数据集名称（Xinjiang / TurbinePrimary）
--presets:    预设配置（balanced, wide, deep, aggressive）
--seeds:      随机种子列表
--epochs:     总训练轮数
--export-best-edge-ready: 导出边缘部署版本

注意事项：
----------------
1. 确保已安装依赖：pip install torch pandas numpy
2. 如显存不足，减小 --batch-size 到 64
3. 快速测试可加 --max-train-batches 10
4. 所有数据集独立训练，不共享检查点

================================================================================
