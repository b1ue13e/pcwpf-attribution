try:
    import mamba_ssm
    print("Mamba 已安装: mamba_ssm")
except ImportError:
    print("警告: mamba_ssm 未安装，将使用简化实现")
