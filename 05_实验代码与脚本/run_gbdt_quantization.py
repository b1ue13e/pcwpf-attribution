# 量化噪声版本的GBDT对比
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

with open('run_gbdt_robustness.py', 'r') as f:
    content = f.read()

content = content.replace("noise_type = 'gaussian'", "noise_type = 'quantization'")

with open('run_gbdt_quantization_temp.py', 'w') as f:
    f.write(content)

exec(open('run_gbdt_quantization_temp.py').read())
