import argparse
import subprocess
import time
import json
import threading
import psutil
import os
import sys

def get_gpu_memory(handle):
    try:
        import pynvml
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    args = parser.parse_args()

    cmd = args.command
    if not cmd:
        sys.exit(1)
    
    # Try initializing NVML
    nvml_handle = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        pass

    start_time = time.time()
    
    try:
        proc = subprocess.Popen(cmd)
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)

    peak_rss = 0
    peak_gpu = 0

    while proc.poll() is None:
        try:
            # CPU Memory
            p = psutil.Process(proc.pid)
            mem = p.memory_info().rss
            children = p.children(recursive=True)
            for child in children:
                try:
                    mem += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            mem_mb = mem / (1024 * 1024)
            if mem_mb > peak_rss:
                peak_rss = mem_mb
            
            # GPU Memory (Global used on device 0 as proxy, imperfect but standard for single-task)
            if nvml_handle:
                gpu_mem = get_gpu_memory(nvml_handle)
                if gpu_mem > peak_gpu:
                    peak_gpu = gpu_mem
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        time.sleep(0.1)

    end_time = time.time()
    runtime = end_time - start_time
    
    if nvml_handle:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except:
            pass

    stats = {
        "runtime_seconds": runtime,
        "peak_cpu_memory_mb": peak_rss,
        "peak_gpu_memory_mb": peak_gpu,
        "exit_code": proc.returncode
    }
    
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
