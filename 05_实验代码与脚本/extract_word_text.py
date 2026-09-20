#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET

# 解析Word文档XML
tree = ET.parse('_tmp_docx_extract/word/document.xml')
root = tree.getroot()

# Word文档的命名空间
namespaces = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
}

# 提取所有文本
print("=== Word文档内容 ===\n")

# 提取段落
for para in root.findall('.//w:p', namespaces):
    text_parts = []
    for t in para.findall('.//w:t', namespaces):
        if t.text:
            text_parts.append(t.text)
    para_text = ''.join(text_parts)
    if para_text.strip():
        print(para_text)

print("\n=== 表格数量 ===")
tables = root.findall('.//w:tbl', namespaces)
print(f"总共{len(tables)}个表格")
