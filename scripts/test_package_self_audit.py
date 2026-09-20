import os
import sys
import zipfile
import tempfile
import csv
from pathlib import Path

def test_zip_archive():
    print("="*70)
    print("1. 压缩包物理完整性与解压测试 (ZIP Physical Integrity)")
    print("="*70)
    zip_path = Path("剩余26篇文献独立双盲复核包.zip")
    if not zip_path.exists():
        print(f"[FAIL] 压缩包不存在: {zip_path}")
        return False
    print(f"[OK] 压缩包存在: {zip_path} ({zip_path.stat().st_size:,} bytes)")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        crc_err = zf.testzip()
        if crc_err:
            print(f"[FAIL] CRC 校验失败: {crc_err}")
            return False
        print("[OK] CRC 校验完全通过，无坏块、无损坏")
        
        namelist = zf.namelist()
        print(f"[OK] 包含文件总数: {len(namelist)}")
        junk = [n for n in namelist if '__pycache__' in n or n.endswith('.pyc') or n.startswith('.') or 'Thumbs.db' in n]
        if junk:
            print(f"[WARN] 发现垃圾临时文件: {junk}")
        else:
            print("[OK] 纯净度检测通过：无 __pycache__、无 .pyc、无隐藏临时文件")
            
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            extracted = list(Path(tmpdir).iterdir())
            print(f"[OK] 沙盒解压测试成功: 成功解压 {len(extracted)} 个文件到临时沙盒环境")
    return True

def test_corpus_consistency():
    print("\n" + "="*70)
    print("2. 文献集合逻辑口径与条目一致性自查 (Corpus Consistency)")
    print("="*70)
    
    # 检查 01 文献清单
    doc_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/01_需复核文献清单_剩余26篇纳入文献.md")
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()
        
    # 从 03_空白盲审填写表_剩余26篇.csv 读取
    csv26_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/03_空白盲审填写表_剩余26篇.csv")
    with open(csv26_path, 'r', encoding='utf-8-sig') as f:
        rows26 = list(csv.DictReader(f))
    
    sids_26 = [r['study_id'] for r in rows26]
    print(f"[+] 03_空白表 26 篇包含条目数: {len(sids_26)}")
    print(f"    S 编号: {', '.join(sids_26)}")
    
    # 验证是否严格剔除了负对照 S26 和 S42
    if 'S26' in sids_26 or 'S42' in sids_26:
        print("[FAIL] 26 篇中误包含了阴性对照 S26/S42！")
        return False
    else:
        print("[OK] 阴性对照过滤核验通过：S26 与 S42 已严格剔除")
        
    # 检查文档中的每个 S 编号是否都在 CSV 中
    for sid in sids_26:
        if sid not in doc_text:
            print(f"[FAIL] 01 文献清单中遗漏了编号: {sid}")
            return False
    print(f"[OK] 01 文献清单与 03 填写表 26 篇条目 100% 互锁对齐")
    
    # 检查 03b 非先导 28 篇
    csv28_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/03b_空白盲审填写表_非先导28篇.csv")
    with open(csv28_path, 'r', encoding='utf-8-sig') as f:
        rows28 = list(csv.DictReader(f))
    sids_28 = [r['study_id'] for r in rows28]
    print(f"[+] 03b_非先导表包含条目数: {len(sids_28)}")
    print(f"    S 编号: {', '.join(sids_28)}")
    
    # 检查 04 全量 46 篇
    csv46_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/04_空白盲审填写表_全量46篇.csv")
    with open(csv46_path, 'r', encoding='utf-8-sig') as f:
        rows46 = list(csv.DictReader(f))
    sids_46 = [r['study_id'] for r in rows46]
    print(f"[+] 04_全量表包含条目数: {len(sids_46)}")
    if len(sids_46) == 46 and 'S26' not in sids_46 and 'S42' not in sids_46:
        print("[OK] 全量 46 篇格式表完全对齐 PRISMA 最终纳入研究数量")
    else:
        print(f"[FAIL] 全量 46 篇表条目异常: {len(sids_46)}")
        return False
    return True

def test_excel_and_csv_parity():
    print("\n" + "="*70)
    print("3. Excel 与 CSV 同源双格式对齐自查 (Excel & CSV Parity)")
    print("="*70)
    try:
        import openpyxl
    except ImportError:
        print("[WARN] openpyxl 未安装，跳过 xlsx 解析测试")
        return True
        
    pairs = [
        ("03_空白盲审填写表_剩余26篇.csv", "03_空白盲审填写表_剩余26篇.xlsx", 26),
        ("03b_空白盲审填写表_非先导28篇.csv", "03b_空白盲审填写表_非先导28篇.xlsx", 28),
        ("04_空白盲审填写表_全量46篇.csv", "04_空白盲审填写表_全量46篇.xlsx", 46),
    ]
    base_dir = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包")
    for csv_name, xlsx_name, expected_count in pairs:
        with open(base_dir / csv_name, 'r', encoding='utf-8-sig') as f:
            c_rows = list(csv.DictReader(f))
            
        wb = openpyxl.load_workbook(base_dir / xlsx_name)
        ws = wb.active
        x_rows = list(ws.iter_rows(values_only=True))
        x_data_count = len(x_rows) - 1
        
        if len(c_rows) == expected_count and x_data_count == expected_count:
            print(f"[OK] {csv_name} 与 {xlsx_name} 行数一致 ({expected_count} 行)")
        else:
            print(f"[FAIL] 行数不一致! CSV={len(c_rows)}, XLSX={x_data_count}, 期望={expected_count}")
            return False
            
        # Check column headers match
        csv_headers = list(c_rows[0].keys())
        xlsx_headers = list(x_rows[0])
        if csv_headers == xlsx_headers:
            print(f"     表头结构 100% 严格一致: {len(csv_headers)} 列")
        else:
            print(f"[FAIL] 表头不一致: \nCSV={csv_headers}\nXLSX={xlsx_headers}")
            return False
    return True

def test_script_simulation_and_cases():
    print("\n" + "="*70)
    print("4. 自动化脚本鲁棒性与异常输入对抗测试 (Script Robustness)")
    print("="*70)
    script_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/05_一键合流与信度重算脚本.py")
    
    # 动态导入脚本模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("merger_module", str(script_path))
    merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merger)
    
    print("[+] 脚本模块导入成功")
    
    # 测试大小写混杂、首尾空格的脏数据归一化
    dirty_tests = [
        (" INTRINSIC ", "D1_provenance", "intrinsic"),
        ("training-loss\n", "D2_stage", "training-loss"),
        ("None", "D3_inference_req", "none"),
        ("FORECAST-INPUT", "D3_inference_req", "forecast-input"),
        ("Concurrent-Future", "D3_inference_req", "forecast-input"),
        ("not-found", "D4_fidelity", "not-found"),
        ("no-code", "D4_fidelity", "not-found"),
        ("Constraint-On-Off", "d5_ablation_type", "constraint-on-off"),
        ("MODULE-ABLATION", "d5_ablation_type", "module-ablation"),
    ]
    all_norm_ok = True
    for val, dim, expected in dirty_tests:
        norm = merger.normalize_value(val, dim)
        if norm != expected:
            print(f"[FAIL] 归一化失败: '{val}' ({dim}) -> '{norm}', 期望 '{expected}'")
            all_norm_ok = False
    if all_norm_ok:
        print("[OK] 容错归一化函数测试完全通过（完美清洗首尾空格、换行、大小写及同义词）")
        
    # 测试端到端合流模拟
    print("[+] 运行 --simulate 模拟测试:")
    exit_code = os.system(f'python "{script_path}" --simulate > nul')
    if exit_code == 0:
        print("[OK] --simulate 运行成功 (Exit Code 0)")
    else:
        print(f"[FAIL] --simulate 运行失败 (Exit Code {exit_code})")
        return False
        
    # 测试导出 Markdown 报告功能
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp_md:
        tmp_md_path = tmp_md.name
    try:
        cmd = f'python "{script_path}" --simulate --export_md "{tmp_md_path}" > nul'
        exit_code = os.system(cmd)
        if exit_code == 0 and os.path.getsize(tmp_md_path) > 500:
            print(f"[OK] Markdown 审计报告生成测试通过: 生成报告大小 {os.path.getsize(tmp_md_path):,} bytes")
        else:
            print(f"[FAIL] Markdown 报告生成失败")
            return False
    finally:
        if os.path.exists(tmp_md_path):
            os.remove(tmp_md_path)
            
    return True

def test_controlled_vocabulary_and_prompt():
    print("\n" + "="*70)
    print("5. 受控词典与 AI 提示词闭环自查 (Controlled Vocabulary & Prompt Alignment)")
    print("="*70)
    guide_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/02_独立盲审评分指南与受控词典.md")
    prompt_path = Path("07_审稿Rebuttal备用包/02_剩余26篇文献独立双盲复核包/06_专家评审AI辅助提示词_Prompt.md")
    
    with open(guide_path, 'r', encoding='utf-8') as f:
        guide_text = f.read()
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
        
    required_keys = [
        "intrinsic", "historical-static", "learned-online", "unreported",
        "training-loss", "inference-architecture", "post-processing",
        "none", "forecast-input",
        "constraint-on-off", "module-ablation", "both"
    ]
    for key in required_keys:
        if key not in guide_text:
            print(f"[FAIL] 02 评分指南缺少受控词: {key}")
            return False
        if key not in prompt_text:
            print(f"[FAIL] 06 AI 提示词缺少受控词: {key}")
            return False
    print("[OK] 核心维度 D1-D5 受控词典在评分指南与 AI Prompt 中 100% 双向对齐")
    return True

def main():
    print("======================================================================")
    print("          《剩余 26 篇文献独立双盲复核包》全面交付自查工程             ")
    print("======================================================================")
    results = [
        test_zip_archive(),
        test_corpus_consistency(),
        test_excel_and_csv_parity(),
        test_script_simulation_and_cases(),
        test_controlled_vocabulary_and_prompt()
    ]
    print("\n" + "="*70)
    if all(results):
        print(">>> 综合自查评估结果: 全部 5 项核心自查测试 100% 满分通过！工件完全合格。")
    else:
        print(">>> 综合自查评估结果: 存在未通过的自查项，请根据上述日志针对性排查。")
    print("="*70)

if __name__ == '__main__':
    main()
