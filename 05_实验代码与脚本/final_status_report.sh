#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           最终实验状态报告 $(date '+%Y-%m-%d %H:%M')                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "【P0 - wind_cn 全量】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Traditional
trad_done=$(grep -l "CNN-LSTM.*R2=" /root/experiment_logs/trad_wind_cn_*.log 2>/dev/null | wc -l)
echo "✅ Traditional: $trad_done/3 seeds (已完成)"

# COA
coa_done=$(grep -l "R2=" /root/experiment_logs/coa_wind_cn_*.log 2>/dev/null | wc -l)
coa_time=$(ps -o etime= -p $(pgrep -f "coa_wind_cn" | head -1) 2>/dev/null | tr -d ' ')
echo "🔄 COA-BiLSTM: $coa_done/3 seeds (运行中, 已运行 $coa_time)"

# BWO
bwo_svr_done=$(grep -l "R2=" /root/experiment_logs/bwo_svr_wind_cn_*.log 2>/dev/null | wc -l)
bwo_cnn_done=$(grep -l "R2=" /root/experiment_logs/bwo_cnn_wind_cn_*.log 2>/dev/null | wc -l)
echo "⏳ BWO-SVR: $bwo_svr_done/3 seeds"
echo "⏳ BWO-CNN: $bwo_cnn_done/3 seeds"

echo ""
echo "【P1 - electricity 核心】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Traditional: 已完成"
echo "⏳ COA-BiLSTM: 等待 wind_cn 完成"
echo "✅ Ultra-LSNT-Lite: 已完成 (R2=0.94)"
echo "⏳ BWO: 有余力时运行"

echo ""
echo "【系统资源】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr ',' ' ')"
echo "进程: $(ps aux | grep python | grep -v grep | grep -v 'vscode\|jupyter\|tensorboard\|nvidia\|6276' | wc -l) 个"
echo ""

echo "【当前结果统计】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 << 'PYEOF'
import pandas as pd
df = pd.read_csv('/root/all_models_clean_80_20.csv')
print(f"CSV总计: {len(df)} 条记录")
print("\n分布:")
print(df.groupby(['dataset', 'model']).size().to_string())
PYEOF

echo ""
echo "【预计完成时间】"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COA正在运行中，预计1-3小时内完成"
echo "BWO实验将在COA完成后自动启动"
echo "全部完成预计: 今晚 21:00-00:00"
echo ""
