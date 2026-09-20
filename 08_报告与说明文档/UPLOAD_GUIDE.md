# 上传和运行指南 (Upload & Run Guide)
======================================

## 📦 生成的文件

打包完成后，你会看到两个文件：

```
cloud_4090_cross_domain_bundle_20260330_024852.zip          (23.0 MB)
cloud_4090_cross_domain_bundle_20260330_024852.zip.sha256   (校验文件)
```

**你应该上传:** `cloud_4090_cross_domain_bundle_20260330_024852.zip`

---

## ☁️ 上传到云端

### 方式1: SCP (推荐)

```bash
# 上传到你的服务器
scp cloud_4090_cross_domain_bundle_20260330_024852.zip username@your-server-ip:/home/username/

# 示例
scp cloud_4090_cross_domain_bundle_20260330_024852.zip root@192.168.1.100:/root/experiments/
```

### 方式2: FileZilla (图形界面)

1. 打开 FileZilla
2. 连接到你的服务器 (主机、用户名、密码、端口22)
3. 本地站点: 找到 zip 文件
4. 远程站点: 选择上传目录
5. 拖拽上传

### 方式3: 云存储 (阿里云OSS/腾讯云COS/AWS S3)

```bash
# 阿里云OSS
ossutil cp cloud_4090_cross_domain_bundle_*.zip oss://your-bucket/experiments/

# AWS S3
aws s3 cp cloud_4090_cross_domain_bundle_*.zip s3://your-bucket/experiments/
```

---

## 🖥️ 云端运行步骤

### 第1步: 解压

```bash
# SSH登录服务器
ssh username@your-server-ip

# 进入上传目录
cd /home/username/experiments

# 解压
unzip cloud_4090_cross_domain_bundle_*.zip
cd cloud_4090_cross_domain_bundle
```

### 第2步: 检查环境

```bash
# 检查Python
python --version  # 需要 3.8+

# 检查CUDA
nvidia-smi  # 确认10张4090都可见

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 第3步: 安装依赖

```bash
# 安装依赖
pip install torch numpy pandas matplotlib seaborn scipy

# 或者使用requirements.txt
pip install -r requirements.txt
```

### 第4步: 快速测试

```bash
# 运行快速测试（验证环境）
bash run_quick_smoke.sh

# 如果显示 "Quick test completed!" 则环境正常
```

### 第5步: 运行完整实验

```bash
# 运行所有实验（10张GPU，约24小时）
bash run_cross_domain_full_suite_4090.sh

# 后台运行（推荐）
nohup bash run_cross_domain_full_suite_4090.sh > experiment.log 2>&1 &

# 查看实时日志
tail -f experiment.log
```

---

## 📊 监控实验进度

### 查看GPU使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 查看实验进度

```bash
# 统计已完成任务数
ls results/cross_domain/*/*.json 2>/dev/null | wc -l

# 查看日志
tail -f logs_cross_domain/*.log
```

### 实验完成后的结果聚合

```bash
# 聚合所有结果
python aggregate_cross_domain_results.py \
    --results-root results/cross_domain \
    --output-dir results/cross_domain

# 查看汇总报告
cat results/cross_domain/cross_domain_summary_report.md
```

---

## 📁 下载结果

实验完成后，下载结果文件：

```bash
# 压缩结果
tar -czvf cross_domain_results.tar.gz results/cross_domain/

# 下载到本地
scp username@your-server-ip:/home/username/experiments/cloud_4090_cross_domain_bundle/cross_domain_results.tar.gz ./
```

---

## ⚠️ 常见问题

### 问题1: 上传速度慢

**解决**: 使用分卷压缩
```bash
# 本地分卷压缩
zip -s 5m cloud_4090_cross_domain_bundle_*.zip --out split_bundle.zip

# 上传后合并
cat split_bundle.z* > merged_bundle.zip
```

### 问题2: 磁盘空间不足

**解决**: 清理不必要的文件
```bash
# 删除旧的实验结果
rm -rf results/
rm -rf logs_*

# 查看磁盘空间
df -h
```

### 问题3: CUDA版本不匹配

**解决**: 安装对应版本的PyTorch
```bash
# CUDA 11.8
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 12.1
pip install torch==2.0.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 问题4: 某个任务失败

**解决**: 查看详细日志
```bash
# 找到失败的日志
grep -l "ERROR\|error\|failed" logs_cross_domain/*.log

# 查看具体错误
cat logs_cross_domain/adapt_ultra_ratio0.01_seed42.log
```

### 问题5: 显存不足 (OOM)

**解决**: 减小batch size
```bash
export BATCH_SIZE=64
bash run_cross_domain_full_suite_4090.sh
```

---

## 📞 技术支持

如果遇到问题：

1. 检查日志: `logs_cross_domain/*.log`
2. 检查GPU: `nvidia-smi`
3. 检查磁盘: `df -h`
4. 检查内存: `free -h`

---

## ✅ 检查清单

上传前确认：
- [ ] 压缩包大小约 23 MB
- [ ] 有校验文件 (.sha256)
- [ ] 服务器有 10 张 RTX 4090
- [ ] 服务器磁盘空间 > 50 GB
- [ ] 服务器已安装 CUDA 11.8+

运行前确认：
- [ ] Python 3.8+
- [ ] PyTorch with CUDA
- [ ] 10 GPUs visible: `nvidia-smi`
- [ ] 快速测试通过

---

## 🎯 预期结果

实验完成后，你会得到：

```
results/cross_domain/
├── domain_adaptation/          (135个JSON文件)
├── multisite_pretraining/      (60个JSON文件)
├── guardrail_ablation/         (64个JSON文件)
├── standardization/            (30个JSON文件)
├── summary_*.csv               (4个汇总表格)
├── cross_domain_summary_report.md  (综合报告)
└── *.png                       (可视化图表)
```

**关键发现示例**:
- Ultra-LSNT (5%数据): R² ≈ 0.75
- iTransformer (5%数据): R² ≈ 0.45
- 物理护栏: Cut Region Violation 从 80% 降至 5%

---

**准备好上传了吗？** 🚀
