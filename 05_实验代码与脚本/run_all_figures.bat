@echo off
chcp 65001 >nul
echo ========================================
echo 开始生成所有图片...
echo ========================================
echo.

echo [1/4] 生成 fig_comparison.png 和 fig_ablation.png...
python plot_paper_figs.py
if %errorlevel% neq 0 (
    echo 错误：plot_paper_figs.py 执行失败
    pause
    exit /b 1
)
echo.

echo [2/4] 生成 fig_architecture.png...
python plot_arch.py
if %errorlevel% neq 0 (
    echo 错误：plot_arch.py 执行失败
    pause
    exit /b 1
)
echo.

echo [3/4] 生成 fig_battle_dlinear.png...
python battle_dlinear.py
if %errorlevel% neq 0 (
    echo 警告：battle_dlinear.py 执行失败（可能需要训练模型）
)
echo.

echo [4/4] 检查 results.png...
if exist "checkpoints_ts\main\results.png" (
    echo ✓ results.png 已存在
) else (
    echo ⚠ results.png 不存在，需要运行训练代码生成
    echo   运行: python ultra_lsnt_timeseries.py
)
echo.

echo ========================================
echo 完成！
echo ========================================
pause
