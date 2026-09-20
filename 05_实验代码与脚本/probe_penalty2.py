import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import real_train_audit as R

# 1 站 1 seed，重 penalty 长训，看 mse vs envelope vs js 是否分化
dev = torch.device("cpu")
for kind in ["mse", "mse+envelope", "mse+js"]:
    r = R.train_one("wind_cn", kind, 200.0, 2.44, 11.50, 0, dev, 42,
                    seq_len=48, pred_len=12, epochs=20, penalty_weight=1.0)
    print(f"{kind:14s} R2={r['R2']:.4f} asym={r['ramp_asym_pt']:+7.2f}pt "
          f"over={r['over%']:.3f} neg={r['neg%']:.3f}", flush=True)
