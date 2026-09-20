#!/bin/bash
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     优先级实验监控 $(date '+%H:%M:%S')                               ║"
echo "║     P0: wind_cn 全量 | P1: electricity 核心                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 统计wind_cn进度
echo "【wind_cn 进度】"
echo "─────────────────────────────────────────────────────────────"

# COA
coa_done=$(for f in /root/experiment_logs/coa_wind_cn_*.log; do if grep -q "R2=" "$f" 2>/dev/null; then echo 1; fi; done | wc -l)
echo "  COA-BiLSTM: $coa_done / 3"

# BWO-SVR
bwo_svr_done=$(for f in /root/experiment_logs/bwo_svr_wind_cn_*.log; do if grep -q "R2=" "$f" 2>/dev/null; then echo 1; fi; done | wc -l)
echo "  BWO-SVR: $bwo_svr_done / 3"

# BWO-CNN
bwo_cnn_done=$(for f in /root/experiment_logs/bwo_cnn_wind_cn_*.log; do if grep -q "R2=" "$f" 2>/dev/null; then echo 1; fi; done | wc -l)
echo "  BWO-CNN: $bwo_cnn_done / 3"

# Traditional
echo "  Traditional: ✅ 已完成"

echo ""
echo "【electricity 进度】"
echo "─────────────────────────────────────────────────────────────"
coa_done=$(for f in /root/experiment_logs/coa_electricity_*.log; do if grep -q "R2=" "$f" 2>/dev/null; then echo 1; fi; done | wc -l)
echo "  COA-BiLSTM: $coa_done / 3"
echo "  Traditional: ✅ 已完成"

echo ""
echo "【系统状态】"
echo "─────────────────────────────────────────────────────────────"
echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr ',' ' ')"
echo "  进程: $(ps aux | grep python | grep -v grep | grep -v 'vscode\|jupyter\|tensorboard\|nvidia\|6276' | wc -l) 个"
echo ""
