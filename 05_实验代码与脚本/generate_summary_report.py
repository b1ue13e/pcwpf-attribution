"""
生成实验汇总报告
"""
import json
import os
from datetime import datetime

def generate_report():
    print("生成实验汇总报告...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Ultra-LSNT 补充实验汇总报告")
    report_lines.append("=" * 70)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 检查各个实验的结果文件
    result_files = {
        '脉冲噪声实验': 'battle_dlinear_results.json',
        '漂移噪声测试': 'gbdt_drift_results.json',
        '量化噪声测试': 'gbdt_quantization_results.json',
        'Mamba对比': 'mamba_results.json',
        '三维分析': 'performance_metrics_3d.json'
    }
    
    for experiment, filename in result_files.items():
        if os.path.exists(filename):
            report_lines.append(f"✅ {experiment}: 已完成")
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                report_lines.append(f"   - {key}: {value}")
            except:
                report_lines.append(f"   - 结果文件存在但无法读取")
        else:
            report_lines.append(f"❌ {experiment}: 未完成或结果文件缺失")
    
    # 检查图表文件
    chart_files = ['3d_performance_scatter.png', 'performance_analysis_grid.png']
    report_lines.append("")
    report_lines.append("图表文件:")
    for chart in chart_files:
        if os.path.exists(chart):
            report_lines.append(f"✅ {chart}")
        else:
            report_lines.append(f"❌ {chart} (未生成)")
    
    # 总结
    report_lines.append("")
    report_lines.append("实验完成状态总结:")
    completed = sum(1 for f in result_files.values() if os.path.exists(f))
    total = len(result_files)
    report_lines.append(f"实验完成度: {completed}/{total} ({completed/total*100:.1f}%)")
    
    # 建议
    report_lines.append("")
    report_lines.append("下一步建议:")
    report_lines.append("1. 将实验结果整合到论文中")
    report_lines.append("2. 更新鲁棒性分析部分，包括非高斯噪声测试结果")
    report_lines.append("3. 在相关章节添加Mamba对比数据")
    report_lines.append("4. 在经济效益分析中加入三维性能分析图")
    
    # 保存报告
    with open('experiment_summary_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("✅ 汇总报告已生成: experiment_summary_report.txt")
    
    # 打印报告摘要
    print("\n" + "=" * 70)
    for line in report_lines[:20]:
        print(line)

if __name__ == '__main__':
    generate_report()
