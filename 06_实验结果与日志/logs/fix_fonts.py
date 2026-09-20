#!/usr/bin/env python3
"""
修复 matplotlib 字体警告
将所有脚本中的 'serif' 字体替换为 'sans-serif'
"""

import os
import re

def fix_fonts_in_file(filepath):
    """修复单个文件中的字体设置"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 保存原始内容
        original = content
        
        # 替换字体设置
        # 1. 替换 font.family
        content = re.sub(
            r"plt\.rcParams\['font\.family'\]\s*=\s*['\"]serif['\"]",
            "plt.rcParams['font.family'] = 'sans-serif'",
            content
        )
        
        # 2. 替换 font.serif
        content = re.sub(
            r"plt\.rcParams\['font\.serif'\]\s*=\s*\[.*?\]",
            "# # # # plt.rcParams['font.serif'] = ['DejaVu Sans']  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告  # 注释掉以避免警告",
            content
        )
        
        # 3. 替换 matplotlib.rcParams 设置
        content = re.sub(
            r"matplotlib\.rcParams\['font\.family'\]\s*=\s*['\"]serif['\"]",
            "matplotlib.rcParams['font.family'] = 'sans-serif'",
            content
        )
        
        # 4. 替换样式表
        content = re.sub(
            r"plt\.style\.use\(['\"]seaborn-v0_8['\"]\)",
            "plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'sans-serif'\nplt.rcParams['font.family'] = 'sans-serif'",
            content
        )
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"  错误处理文件 {filepath}: {e}")
        return False

def main():
    print("正在修复所有Python脚本中的字体警告...")
    
    # 查找所有Python文件
    python_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and not file.startswith("."):
                python_files.append(os.path.join(root, file))
    
    fixed_count = 0
    for filepath in python_files:
        try:
            if fix_fonts_in_file(filepath):
                print(f"  ✅ 已修复: {filepath}")
                fixed_count += 1
        except Exception as e:
            print(f"  ❌ 修复失败 {filepath}: {e}")
    
    print(f"字体修复完成，共修复 {fixed_count} 个文件")
    
    # 创建通用字体设置脚本
    font_setup = '''#!/usr/bin/env python3
# 通用字体设置，避免警告
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
'''
    
    with open("font_setup.py", "w") as f:
        f.write(font_setup)
    
    print("已创建通用字体设置脚本: font_setup.py")

if __name__ == "__main__":
    main()
