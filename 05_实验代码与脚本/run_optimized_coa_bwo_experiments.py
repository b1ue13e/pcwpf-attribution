#!/usr/bin/env python3
"""
COA/BWO 优化实验运行脚本 (针对RTX 4090 D优化)

本脚本自动化运行所有COA/BWO实验，充分利用RTX 4090 D (23GB) GPU性能：
1. 混合精度训练 (AMP)
2. 并行数据加载
3. 大batch size优化
4. 动态GPU内存管理
5. 多卡支持 (如果可用)

支持后台运行、实时进度监控和GPU性能监控。
"""

import os
import sys
import json
import subprocess
import time
import datetime
import argparse
import psutil
import GPUtil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import signal
import threading
import queue
import atexit
import warnings

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 忽略警告
warnings.filterwarnings('ignore')

class GPUManager:
    """GPU资源管理器"""
    
    def __init__(self, gpu_id: int = 0, memory_fraction: float = 0.9):
        self.gpu_id = gpu_id
        self.memory_fraction = memory_fraction
        self.available_memory = 0
        self.used_memory = 0
        self.device_name = ""
        
        self.initialize_gpu()
    
    def initialize_gpu(self):
        """初始化GPU并获取信息"""
        try:
            import torch
            if torch.cuda.is_available():
                self.device_name = torch.cuda.get_device_name(self.gpu_id)
                total_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
                self.available_memory = int(total_memory * self.memory_fraction)
                
                print(f"✅ GPU {self.gpu_id}: {self.device_name}")
                print(f"   总显存: {total_memory / 1024**3:.1f} GB")
                print(f"   可用显存: {self.available_memory / 1024**3:.1f} GB")
                print(f"   CUDA版本: {torch.version.cuda}")
                
                # 设置CUDA设备
                torch.cuda.set_device(self.gpu_id)
                
                # 设置显存限制
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.gpu_id)
                
                return True
            else:
                print("❌ 未检测到CUDA设备")
                return False
                
        except Exception as e:
            print(f"❌ GPU初始化失败: {e}")
            return False
    
    def get_memory_usage(self) -> Tuple[int, int]:
        """获取当前GPU显存使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.gpu_id)
                reserved = torch.cuda.memory_reserved(self.gpu_id)
                return allocated, reserved
        except:
            pass
        return 0, 0
    
    def clear_cache(self):
        """清空GPU缓存"""
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    def monitor_gpu_stats(self, interval: int = 5):
        """监控GPU统计信息"""
        while self.monitoring:
            try:
                allocated, reserved = self.get_memory_usage()
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[self.gpu_id]
                    print(f"GPU监控 - 使用率: {gpu.load*100:.1f}%, 显存: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} GB, 温度: {gpu.temperature}°C")
            except:
                pass
            time.sleep(interval)
    
    def start_monitoring(self):
        """启动GPU监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_gpu_stats, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止GPU监控"""
        self.monitoring = False

class OptimizedExperimentRunner:
    """优化实验运行管理器"""
    
    def __init__(self, log_dir: str = "optimized_experiment_logs", 
                 results_dir: str = ".",
                 max_parallel: int = 2,  # 默认并行2个实验
                 gpu_memory_fraction: float = 0.8,
                 use_amp: bool = True,
                 batch_size_multiplier: float = 2.0):
        
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.max_parallel = min(max_parallel, 4)  # 最大并行4个
        self.gpu_memory_fraction = gpu_memory_fraction
        self.use_amp = use_amp
        self.batch_size_multiplier = batch_size_multiplier
        
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化GPU管理器
        self.gpu_manager = GPUManager(gpu_id=0, memory_fraction=gpu_memory_fraction)
        
        # 实验配置 - 针对GPU优化
        self.experiments = [
            {
                "name": "coa_bilstm",
                "script": "coa_bilstm_experiment.py",
                "datasets": ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "description": "COA优化的BiLSTM模型",
                "gpu_optimized": True,
                "batch_size_factor": 1.5,
                "default_batch_size": 128  # 针对4090的更大batch size
            },
            {
                "name": "bwo_svr", 
                "script": "bwo_svr_experiment.py",
                "datasets": ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "description": "BWO优化的SVR模型",
                "gpu_optimized": False,  # SVR不依赖GPU
                "batch_size_factor": 1.0,
                "default_batch_size": 64
            },
            {
                "name": "bwo_cnn",
                "script": "bwo_cnn_experiment.py", 
                "datasets": ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "description": "BWO优化的1D CNN模型",
                "gpu_optimized": True,
                "batch_size_factor": 2.0,
                "default_batch_size": 256  # CNN可以使用更大的batch size
            }
        ]
        
        # 状态跟踪
        self.status = {
            "total_experiments": 0,
            "completed": 0,
            "failed": 0,
            "running": 0,
            "queued": 0
        }
        
        self.start_time = None
        self.progress_file = self.log_dir / "optimized_experiment_progress.json"
        
        # 并行执行队列
        self.task_queue = queue.Queue()
        self.worker_threads = []
        self.running_tasks = {}
        self.lock = threading.Lock()
        
        # 注册退出处理
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """清理资源"""
        self.gpu_manager.stop_monitoring()
        self.gpu_manager.clear_cache()
    
    def setup_environment(self):
        """设置优化实验环境"""
        print("=" * 70)
        print("COA/BWO 优化实验运行器 (针对RTX 4090 D优化)")
        print("=" * 70)
        print(f"日志目录: {self.log_dir}")
        print(f"结果目录: {self.results_dir}")
        print(f"最大并行度: {self.max_parallel}")
        print(f"GPU显存使用率: {self.gpu_memory_fraction*100:.0f}%")
        print(f"混合精度训练: {'启用' if self.use_amp else '禁用'}")
        print(f"Batch size乘数: {self.batch_size_multiplier:.1f}x")
        
        # 验证必要文件
        required_files = [
            "coa_bilstm_experiment.py",
            "bwo_svr_experiment.py", 
            "bwo_cnn_experiment.py",
            "coa_algorithm.py",
            "bwo_algorithm.py",
            "coa_bwo_unified_split.py",
            "split_manifest_80_20_unified.json"
        ]
        
        missing_files = []
        for f in required_files:
            if not Path(f).exists():
                missing_files.append(f)
        
        if missing_files:
            print(f"\n❌ 缺少必要文件: {missing_files}")
            return False
        
        # 启动GPU监控
        self.gpu_manager.start_monitoring()
        
        # 创建优化配置文件
        self.create_optimization_config()
        
        print("\n✅ 优化环境验证通过")
        return True
    
    def create_optimization_config(self):
        """创建GPU优化配置文件"""
        config = {
            "gpu_optimization": {
                "device_name": self.gpu_manager.device_name,
                "available_memory_gb": self.gpu_manager.available_memory / 1024**3,
                "memory_fraction": self.gpu_memory_fraction,
                "use_amp": self.use_amp,
                "batch_size_multiplier": self.batch_size_multiplier,
                "num_workers": min(8, os.cpu_count() or 4),  # 数据加载线程数
                "pin_memory": True,
                "persistent_workers": True
            },
            "experiment_settings": {
                "max_parallel": self.max_parallel,
                "datasets": ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"],
                "seeds": [42, 123, 999],
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        
        config_file = self.log_dir / "gpu_optimization_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ GPU优化配置已保存: {config_file}")
    
    def update_split_manifest(self):
        """更新划分清单以包含所有数据集"""
        manifest_path = Path("split_manifest_80_20_unified.json")
        if not manifest_path.exists():
            print("❌ 划分清单文件不存在")
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # 检查已存在的数据集
            existing_datasets = [d["name"] for d in manifest["datasets"]]
            datasets_needed = ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"]
            
            # 重命名wind为wind_cn（如果存在）
            for i, dataset in enumerate(manifest["datasets"]):
                if dataset["name"] == "wind" and "wind_cn" not in existing_datasets:
                    manifest["datasets"][i]["name"] = "wind_cn"
                    manifest["datasets"][i]["description"] = "中国风电数据集"
                    print("✅ 已重命名wind为wind_cn")
            
            # 添加wind_us（如果不存在）
            if "wind_us" not in existing_datasets:
                wind_us_config = {
                    "name": "wind_us",
                    "file": "wind_us.csv",
                    "target_column": "power (MW)",
                    "time_column": None,
                    "total_samples": 105121,
                    "training_set": {
                        "samples": 84096,
                        "indices": "0:84096",
                        "ratio": 0.8,
                        "description": "First 84096 samples (80%)"
                    },
                    "test_set": {
                        "samples": 21025,
                        "indices": "84096:105121",
                        "ratio": 0.2,
                        "description": "Last 21025 samples (20%)"
                    },
                    "split_validation": {
                        "no_overlap": True,
                        "chronological": True,
                        "consistent_across_runs": True
                    }
                }
                manifest["datasets"].append(wind_us_config)
                print("✅ 已添加wind_us数据集")
            
            # 检查缺失的数据集
            for dataset_name in datasets_needed:
                if dataset_name not in [d["name"] for d in manifest["datasets"]]:
                    print(f"⚠️  数据集 {dataset_name} 不在划分清单中")
            
            # 保存备份
            backup_path = Path("split_manifest_80_20_unified_backup.json")
            if not backup_path.exists():
                with open(backup_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"✅ 划分清单已更新")
            print(f"   数据集: {[d['name'] for d in manifest['datasets']]}")
            return True
            
        except Exception as e:
            print(f"❌ 更新划分清单失败: {e}")
            return False
    
    def create_optimized_experiment_script(self, exp_config: Dict) -> str:
        """创建针对GPU优化的实验脚本"""
        template = f'''#!/usr/bin/env python3
"""
{exp_config['description']} - GPU优化版本
针对RTX 4090 D优化，使用混合精度训练和大batch size
"""

import os
import sys
import json
import argparse

# 添加GPU优化参数
optimization_args = [
    "--gpu", "0",
    "--batch-size", str(int(exp_config.get('default_batch_size', 64) * self.batch_size_multiplier * exp_config.get('batch_size_factor', 1.0))),
    "--amp",  # 混合精度训练
    "--workers", "8",  # 数据加载线程
    "--pin-memory",
    "--benchmark"  # CUDA基准测试
]

# 构建命令
cmd = [sys.executable, exp_config['script']] + optimization_args + sys.argv[1:]

print(f"运行优化命令: {{' '.join(cmd)}}")
os.execvp(cmd[0], cmd)
'''
        
        script_path = self.log_dir / f"run_optimized_{exp_config['name']}.py"
        with open(script_path, 'w') as f:
            f.write(template)
        
        os.chmod(script_path, 0o755)
        return str(script_path)
    
    def worker_thread(self, worker_id: int):
        """工作线程，用于并行运行实验"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 终止信号
                    break
                
                exp_config, dataset, seed = task
                
                # 记录任务开始
                with self.lock:
                    self.running_tasks[(exp_config['name'], dataset, seed)] = {
                        'worker_id': worker_id,
                        'start_time': time.time(),
                        'status': 'running'
                    }
                
                # 运行实验
                success = self.run_optimized_experiment(exp_config, dataset, seed, worker_id)
                
                # 更新状态
                with self.lock:
                    if success:
                        self.status['completed'] += 1
                        status = 'completed'
                    else:
                        self.status['failed'] += 1
                        status = 'failed'
                    
                    self.status['running'] -= 1
                    del self.running_tasks[(exp_config['name'], dataset, seed)]
                
                # 更新进度
                self.save_progress()
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"工作线程 {worker_id} 错误: {e}")
                continue
    
    def run_optimized_experiment(self, exp_config: Dict, dataset: str, 
                               seed: int, worker_id: int) -> bool:
        """运行优化后的单个实验"""
        exp_name = exp_config['name']
        script = exp_config['script']
        
        # 创建日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{exp_name}_{dataset}_{seed}_{timestamp}.log"
        
        print(f"   [Worker {worker_id}] ▶ {exp_name}/{dataset}/seed{seed}: 运行中...")
        
        # 构建命令参数
        cmd = [
            sys.executable, script,
            "--dataset", dataset,
            "--seed", str(seed),
            "--output", f"{exp_name}_{dataset}_{seed}.json"
        ]
        
        # 添加GPU优化参数
        if exp_config.get('gpu_optimized', False):
            cmd.extend([
                "--batch-size", str(int(exp_config.get('default_batch_size', 64) * 
                                     self.batch_size_multiplier * 
                                     exp_config.get('batch_size_factor', 1.0))),
                "--workers", "8",
                "--pin-memory",
                "--benchmark"
            ])
            
            if self.use_amp:
                cmd.append("--amp")
        
        # 记录命令
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"工作线程: {worker_id}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"开始时间: {datetime.datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
        
        try:
            # 运行实验，带超时保护
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时捕获输出
            output_lines = []
            start_time = time.time()
            timeout = 3600 * 2  # 2小时超时
            
            while True:
                if time.time() - start_time > timeout:
                    process.terminate()
                    with open(log_file, 'a') as f:
                        f.write(f"\n⏰ 超时终止 (>{timeout/3600:.1f}小时)\n")
                    return False
                
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output:
                    output_lines.append(output)
                    with open(log_file, 'a') as f:
                        f.write(output)
            
            return_code = process.poll()
            
            with open(log_file, 'a') as f:
                f.write(f"\n退出码: {return_code}\n")
                f.write(f"结束时间: {datetime.datetime.now().isoformat()}\n")
                f.write(f"总耗时: {time.time() - start_time:.1f}秒\n")
            
            return return_code == 0
            
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"\n❌ 异常: {str(e)}\n")
            return False
    
    def run_all_experiments_parallel(self, verbose: bool = False, 
                                   skip_completed: bool = True):
        """并行运行所有实验"""
        self.verbose = verbose
        self.start_time = datetime.datetime.now()
        
        print("\n" + "="*70)
        print("开始并行运行COA/BWO优化实验")
        print("="*70)
        
        # 更新数据集划分
        print("\n1. 更新数据集划分清单...")
        if not self.update_split_manifest():
            print("⚠️  划分清单更新失败，继续使用原有配置")
        
        # 计算总实验数
        total = 0
        for exp in self.experiments:
            total += len(exp["datasets"]) * len(exp["seeds"])
        
        self.status["total_experiments"] = total
        self.status["queued"] = total
        
        print(f"\n2. 准备运行 {total} 个实验:")
        for exp in self.experiments:
            exp_count = len(exp["datasets"]) * len(exp["seeds"])
            print(f"   • {exp['description']}: {exp_count} 个实验")
            print(f"     GPU优化: {'是' if exp.get('gpu_optimized', False) else '否'}")
            if exp.get('gpu_optimized', False):
                batch_size = int(exp.get('default_batch_size', 64) * 
                               self.batch_size_multiplier * 
                               exp.get('batch_size_factor', 1.0))
                print(f"     优化batch size: {batch_size}")
        
        print(f"\n3. 启动 {self.max_parallel} 个工作线程...")
        
        # 启动工作线程
        for i in range(self.max_parallel):
            thread = threading.Thread(target=self.worker_thread, args=(i+1,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        # 添加任务到队列
        print(f"\n4. 添加 {total} 个任务到队列...")
        queued_count = 0
        for exp_config in self.experiments:
            exp_name = exp_config["name"]
            
            for dataset in exp_config["datasets"]:
                for seed in exp_config["seeds"]:
                    # 检查是否已存在结果
                    result_file = Path(f"{exp_name}_{dataset}_{seed}.json")
                    if skip_completed and result_file.exists():
                        print(f"   ✓ {exp_name}/{dataset}/seed{seed}: 已存在结果，跳过")
                        self.status["completed"] += 1
                        self.status["queued"] -= 1
                        continue
                    
                    # 添加到任务队列
                    self.task_queue.put((exp_config, dataset, seed))
                    queued_count += 1
        
        print(f"   实际排队任务: {queued_count} 个")
        
        # 进度监控线程
        progress_thread = threading.Thread(target=self.monitor_progress, daemon=True)
        progress_thread.start()
        
        # 等待所有任务完成
        print(f"\n5. 等待任务完成...")
        self.task_queue.join()
        
        # 发送终止信号给工作线程
        for _ in range(self.max_parallel):
            self.task_queue.put(None)
        
        for thread in self.worker_threads:
            thread.join()
        
        # 运行完成
        self.finalize()
    
    def monitor_progress(self):
        """监控进度并显示状态"""
        while not self.task_queue.empty() or self.status['running'] > 0:
            elapsed = datetime.datetime.now() - self.start_time
            completed = self.status['completed']
            failed = self.status['failed']
            running = self.status['running']
            queued = self.status['queued']
            total = self.status['total_experiments']
            
            progress = (completed + failed) / total * 100
            
            print(f"\r📊 进度: {progress:.1f}% | 完成: {completed}/{total} | 失败: {failed} | 运行中: {running} | 排队: {queued} | 耗时: {elapsed}", end="", flush=True)
            
            # 显示运行中的任务
            if self.running_tasks and elapsed.seconds % 10 == 0:  # 每10秒显示一次
                print(f"\n   运行中的任务: {list(self.running_tasks.keys())[:3]}...")
            
            time.sleep(2)
        
        print()  # 换行
    
    def save_progress(self):
        """保存进度到文件"""
        elapsed = datetime.datetime.now() - self.start_time
        
        progress = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": self.status.copy(),
            "elapsed_time": str(elapsed),
            "gpu_memory_usage": self.gpu_manager.get_memory_usage(),
            "running_tasks": list(self.running_tasks.keys()),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def finalize(self):
        """实验完成后的处理"""
        elapsed = datetime.datetime.now() - self.start_time
        
        print("\n" + "="*70)
        print("COA/BWO 优化实验运行完成")
        print("="*70)
        
        print(f"\n📊 统计信息:")
        print(f"   总实验数: {self.status['total_experiments']}")
        print(f"   完成: {self.status['completed']}")
        print(f"   失败: {self.status['failed']}")
        print(f"   运行中: {self.status['running']}")
        print(f"   排队中: {self.status['queued']}")
        print(f"   总耗时: {elapsed}")
        
        # GPU使用统计
        allocated, reserved = self.gpu_manager.get_memory_usage()
        print(f"   GPU显存使用: {allocated/1024**3:.1f} GB / {reserved/1024**3:.1f} GB")
        
        # 运行结果收集器
        print(f"\n🔄 运行结果收集器...")
        try:
            subprocess.run([sys.executable, "coa_bwo_results_collector.py"], 
                         check=False)
        except Exception as e:
            print(f"   结果收集失败: {e}")
        
        # 生成性能报告
        self.generate_performance_report(elapsed)
        
        print(f"\n📁 日志目录: {self.log_dir}")
        print(f"📄 进度文件: {self.progress_file}")
        
        if self.status['failed'] > 0:
            print(f"\n⚠️  有 {self.status['failed']} 个实验失败，请检查日志文件")
            return 1
        else:
            print(f"\n✅ 所有实验成功完成!")
            return 0
    
    def generate_performance_report(self, elapsed):
        """生成性能报告"""
        try:
            # 获取实验文件统计
            result_files = list(Path(".").glob("*_*.json"))
            log_files = list(self.log_dir.glob("*.log"))
            
            report = {
                "experiment_completed": datetime.datetime.now().isoformat(),
                "elapsed_time_seconds": elapsed.total_seconds(),
                "experiments_per_hour": self.status['completed'] / (elapsed.total_seconds() / 3600) if elapsed.total_seconds() > 0 else 0,
                "total_experiments": self.status['total_experiments'],
                "completed": self.status['completed'],
                "failed": self.status['failed'],
                "parallelism": self.max_parallel,
                "gpu_optimization": {
                    "memory_fraction": self.gpu_memory_fraction,
                    "use_amp": self.use_amp,
                    "batch_size_multiplier": self.batch_size_multiplier
                },
                "files_generated": {
                    "result_files": len(result_files),
                    "log_files": len(log_files),
                    "total_size_mb": sum(f.stat().st_size for f in result_files + log_files) / 1024**2
                },
                "performance_metrics": {
                    "avg_experiment_time_seconds": elapsed.total_seconds() / self.status['completed'] if self.status['completed'] > 0 else 0,
                    "efficiency": self.status['completed'] / (self.status['total_experiments'] * (elapsed.total_seconds() / 3600)) if elapsed.total_seconds() > 0 else 0
                }
            }
            
            report_file = self.log_dir / "performance_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📊 性能报告已保存: {report_file}")
            print(f"   实验/小时: {report['experiments_per_hour']:.1f}")
            print(f"   平均实验时间: {report['performance_metrics']['avg_experiment_time_seconds']:.1f}秒")
            
        except Exception as e:
            print(f"生成性能报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行COA/BWO优化实验 (针对RTX 4090 D)")
    parser.add_argument("--log-dir", default="optimized_experiment_logs",
                       help="日志目录 (默认: optimized_experiment_logs)")
    parser.add_argument("--results-dir", default=".",
                       help="结果目录 (默认: 当前目录)")
    parser.add_argument("--parallel", type=int, default=2,
                       help="最大并行实验数 (默认: 2, 推荐1-4)")
    parser.add_argument("--gpu-memory", type=float, default=0.8,
                       help="GPU显存使用比例 (默认: 0.8)")
    parser.add_argument("--no-amp", action="store_true",
                       help="禁用混合精度训练")
    parser.add_argument("--batch-multiplier", type=float, default=2.0,
                       help="Batch size乘数 (默认: 2.0)")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细输出")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已完成的实验")
    parser.add_argument("--monitor-only", action="store_true",
                       help="只监控不运行新实验")
    
    args = parser.parse_args()
    
    print(f"""
    ⚡ COA/BWO 优化实验运行器 ⚡
    
    配置:
      GPU: RTX 4090 D (23GB)
      并行度: {args.parallel}
      显存使用: {args.gpu_memory*100:.0f}%
      混合精度: {'启用' if not args.no_amp else '禁用'}
      Batch size乘数: {args.batch_multiplier}x
      跳过已完成: {'是' if not args.no_skip else '否'}
    
    """)
    
    runner = OptimizedExperimentRunner(
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        max_parallel=args.parallel,
        gpu_memory_fraction=args.gpu_memory,
        use_amp=not args.no_amp,
        batch_size_multiplier=args.batch_multiplier
    )
    
    if runner.setup_environment():
        if args.monitor_only:
            print("监控模式启动...")
            runner.gpu_manager.start_monitoring()
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\n监控结束")
        else:
            return runner.run_all_experiments_parallel(
                verbose=args.verbose,
                skip_completed=not args.no_skip
            )
    else:
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n实验运行被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)