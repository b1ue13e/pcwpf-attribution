import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultra_lsnt_timeseries import LSTMBaseline
import real_train_audit as R

df = pd.read_csv("../04_数据集/prepared_priority_datasets/wind_cn_core_15min.csv")
ws = df["wind_speed"].to_numpy(float); p = df["power"].to_numpy(float)
v = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p); ws, p = ws[v], p[v]
ws_n = (ws - ws.mean()) / (ws.std() + 1e-6); p_n = (p - p.mean()) / (p.std() + 1e-6)
seq, pred = 96, 24
X, Y, vp_audit, vr_audit = R.build_windows(ws_n, p_n, np.zeros((len(ws), 1), np.float32), seq, pred)
vr_phys = vr_audit * ws.std() + ws.mean()
n = int(len(X) * 0.7)
Xtr, Ytr = torch.from_numpy(X[:n]), torch.from_numpy(Y[:n])
Xte, Yte = torch.from_numpy(X[n:]), torch.from_numpy(Y[n:])
p_rated, v_in, v_rated = 200.0, 2.44, 11.5


def run(kind, w, ep):
    torch.manual_seed(42); np.random.seed(42)
    m = LSTMBaseline(2, 32, pred).cpu(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for e in range(ep):
        perm = torch.randperm(len(Xtr))
        for bi in range(0, len(Xtr), 64):
            xb, yb = Xtr[perm[bi:bi + 64]], Ytr[perm[bi:bi + 64]]
            pr, _ = m(xb); mse = F.mse_loss(pr, yb)
            if kind == "envelope":
                vv = xb[:, -1, 0].numpy() * ws.std() + ws.mean()
                env = R.power_curve_pu(np.repeat(vv[:, None], pred, 1), v_in, v_rated) * p_rated
                over = np.clip(pr.detach().numpy() - env, 0, None)
                loss = mse + w * torch.tensor(float((over ** 2).mean()), dtype=pr.dtype)
            elif kind == "js":
                pw = pr.detach().numpy().flatten()
                pg = np.linspace(0, 1, 100)
                ppdf = R.kde(np.clip(pw, 0, 1), pg, 0.03) + 1e-9
                tpdf = R.kde(np.clip(p_n, 0, 1), pg, 0.03) + 1e-9
                loss = mse + w * torch.tensor(R.js_div(ppdf, tpdf), dtype=pr.dtype)
            else:
                loss = mse
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        P = []
        for bi in range(0, len(Xte), 128):
            pr, _ = m(Xte[bi:bi + 128]); P.append(pr.numpy())
    pk = np.concatenate(P) * p.std() + p.mean(); tk = Yte.numpy() * p.std() + p.mean()
    err = pk - tk
    r = np.abs(np.diff(vr_phys, axis=1, prepend=vr_phys[:, :1])).mean(1)
    ramp = (vr_phys.mean(1) < v_rated) & (r > np.quantile(r, 0.9))
    above = vr_phys.mean(1) >= v_rated
    rb = float(np.clip(-err[ramp], 0, None).mean()) / max(tk[ramp].mean(), 1e-6)
    ab = float(np.clip(-err[above], 0, None).mean()) / max(tk[above].mean(), 1e-6)
    return (rb - ab) * 100


for kind in ["mse", "envelope", "js"]:
    a = run(kind, 1.0, 25)
    print(f"{kind:10s} w=1.0 ep=25 asym={a:+8.2f}pt", flush=True)
