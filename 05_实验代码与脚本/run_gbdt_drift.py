# 漂移噪声版本的GBDT对比
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 修改原脚本中的噪声类型
with open('run_gbdt_robustness.py', 'r') as f:
    content = f.read()

# 将噪声类型从 gaussian 改为 drift
content = content.replace("noise_type = 'gaussian'", "noise_type = 'drift'")

# 写入临时文件
with open('run_gbdt_drift_temp.py', 'w') as f:
    f.write(content)

# 执行修改后的脚本
exec(open('run_gbdt_drift_temp.py').read())
