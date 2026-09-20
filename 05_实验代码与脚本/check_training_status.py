# -*- coding: utf-8 -*-
import json
import datetime

# 读取训练历史
with open('checkpoints_ts/exp_wind_us/history.json', 'r') as f:
    history = json.load(f)

print('=== Ultra-LSNT 风电预测训练状态 ===')
print(f'当前时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'总训练轮次: {len(history)}')
print(f'当前轮次: {history[-1]["epoch"]}')
print()

print('最新训练结果 (第{}轮):'.format(history[-1]['epoch']))
print('训练损失: {:.6f}'.format(history[-1]['train']['loss']))
print('验证R2:   {:.4f}'.format(history[-1]['val']['R2']))
print('验证RMSE: {:.4f}'.format(history[-1]['val']['RMSE']))
print('验证MAE:  {:.4f}'.format(history[-1]['val']['MAE']))
print()

print('训练趋势分析:')
# 取每10轮的数据点
step = max(1, len(history) // 10)
print('轮次  训练损失    验证R2     验证RMSE   跳过率')
print('----  ----------  ----------  ----------  --------')
for i in range(0, len(history), step):
    epoch = history[i]['epoch']
    train_loss = history[i]['train']['loss']
    val_r2 = history[i]['val']['R2']
    val_rmse = history[i]['val']['RMSE']
    skip_rate = history[i]['efficiency']['avg_skip_rate']
    print(f'{epoch:3d}  {train_loss:10.6f}  {val_r2:10.4f}  {val_rmse:10.4f}  {skip_rate:8.3f}')

print()
print('最佳性能轮次:')
best_r2_epoch = max(history, key=lambda x: x['val']['R2'])
print(f'最佳R2: 第{best_r2_epoch["epoch"]}轮 - R2 = {best_r2_epoch["val"]["R2"]:.4f}')
print(f'最低RMSE: 第{min(history, key=lambda x: x["val"]["RMSE"])["epoch"]}轮 - RMSE = {min(history, key=lambda x: x["val"]["RMSE"])["val"]["RMSE"]:.4f}')
print(f'最低MAE: 第{min(history, key=lambda x: x["val"]["MAE"])["epoch"]}轮 - MAE = {min(history, key=lambda x: x["val"]["MAE"])["val"]["MAE"]:.4f}')

print()
print('模型效率:')
print(f'平均计算比例: {history[-1]["efficiency"]["avg_compute_ratio"]}')
print(f'平均跳过率: {history[-1]["efficiency"]["avg_skip_rate"]:.3f}')
print(f'平均慢速比例: {history[-1]["efficiency"]["avg_slow_ratio"]}')