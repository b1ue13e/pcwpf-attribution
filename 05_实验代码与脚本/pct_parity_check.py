"""Minimum parity check: PCT official 'JS constraint' vs a true joint-KDE constraint.

Faithfully ports PCT train_clean.py's marginal_pdf / joint_pdf / js_div, then
compares against a correct 2D joint KDE. Goal: quantify whether the official
'outer product' implementation materially differs from a joint density on
(power, wind).

No full training needed: the constraint is a deterministic (pred_power, wind) -> loss
mapping. We evaluate both on real windowed data (audit split) with the model
replaced by a simple baseline prediction (persistence + fitted power curve), then
report:
  - normalized loss value difference
  - gradient alignment (does official implementation push predictions the same way?)
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402

# ---------- faithful ports of PCT train_clean.py ----------
def marginal_pdf(values, bins, a, b=1e-10):
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / a).pow(2))
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + b
    pdf = pdf / normalization
    return pdf, kernel_values + torch.tensor(1e-10)


def joint_pdf_official(kernel_values1, kernel_values2):
    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1)
    pdf = joint_kernel_values / normalization
    return pdf


def kl_div(p, q):
    return F.kl_div(p.log(), q, reduction="sum")


def js_div(p, q):
    m = 0.5 * (p + q)
    js = 0.5 * (kl_div(p, m) + kl_div(q, m))
    return js


# ---------- correct true joint KDE (2D gaussian on concatenated [power, wind]) ----------
def true_joint_pdf(power, windspeed, bins_p, bins_w, h_p, h_w):
    # power/windspeed: (1, N, 1)
    rp = power - bins_p.unsqueeze(0).unsqueeze(0)      # (1,N,Bp)
    rw = windspeed - bins_w.unsqueeze(0).unsqueeze(0)  # (1,N,Bw)
    # 2D gaussian kernel in (rand_p, rand_w): need (1,N,Bp,1) + (1,N,1,Bw)
    rp2 = (rp / h_p).pow(2).unsqueeze(-1)              # (1,N,Bp,1)
    rw2 = (rw / h_w).pow(2).unsqueeze(-2)              # (1,N,1,Bw)
    k = torch.exp(-0.5 * (rp2 + rw2)) + 1e-10           # (1,N,Bp,Bw), same floor as official
    pdf = torch.mean(k, dim=1)                         # (1,Bp,Bw)
    pdf = pdf / (torch.sum(pdf, dim=(1, 2), keepdim=True) + 1e-10)
    return pdf


def run_site(site, bin_num=50, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = __import__("pandas").read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)
    i1 = int(n * 0.8)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_te, p_te = ws[i1:], p[i1:]

    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)

    def pu(a):
        return np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)

    # build (pred_power, wind) pairs on audit split: baseline pred = fitted power curve
    # + small noise (simulating a model output that needs the constraint)
    v_t = ws_te
    pred = power_curve_pu(v_t, v_in, v_rated) * p_rated
    # normalize like the official code does to its 'average' tensor
    power = torch.tensor(pu(p_te), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)   # target
    predp = torch.tensor(pu(pred), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    wind = torch.tensor(((v_t - v_t.mean()) / (v_t.std() + 1e-8)), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    bins_p = torch.linspace(float(power.min()), float(power.max()), bin_num)
    bins_w = torch.linspace(float(wind.min()), float(wind.max()), bin_num)

    # bandwidths (official style: sigma * bins^(-1/5) * 0.1)
    bw_sigma = torch.pow(torch.tensor(float(bin_num)), -1 / 5) * 0.1
    h_p = power.std() * bw_sigma
    h_w = wind.std() * bw_sigma

    # ---- official target power curve (pc) ----
    _, kp_t = marginal_pdf(power, bins_p, h_p)
    _, kw_t = marginal_pdf(wind, bins_w, h_w)
    pc_target = joint_pdf_official(kp_t, kw_t)

    # ---- official predicted pc ----
    _, kp_q = marginal_pdf(predp, bins_p, h_p)
    _, kw_q = marginal_pdf(wind, bins_w, h_w)
    pc_pred_official = joint_pdf_official(kp_q, kw_q)

    loss_official = float(js_div(pc_pred_official, pc_target))

    # ---- true joint KDE both ----
    joint_target = true_joint_pdf(power, wind, bins_p, bins_w, h_p, h_w)
    joint_pred = true_joint_pdf(predp, wind, bins_p, bins_w, h_p, h_w)
    loss_true = float(js_div(joint_pred, joint_target))

    # ---- gradient check: direction of constraint gradient wrt pred power at 0 ----
    predp.requires_grad_(True)
    _, kp_g = marginal_pdf(predp, bins_p, h_p)
    _, kw_g = marginal_pdf(wind, bins_w, h_w)
    pc_g_official = joint_pdf_official(kp_g, kw_g)
    g_official = torch.autograd.grad(js_div(pc_g_official, pc_target), predp)[0]
    predp2 = predp.detach().clone().requires_grad_(True)
    pc_g_true = true_joint_pdf(predp2, wind, bins_p, bins_w, h_p, h_w)
    g_true = torch.autograd.grad(js_div(pc_g_true, joint_target), predp2)[0]

    # alignment = cosine similarity of the two gradients
    align = float(torch.nn.functional.cosine_similarity(g_official.flatten(), g_true.flatten(), dim=0))
    return dict(site=site,
                loss_official=loss_official, loss_true=loss_true,
                loss_ratio=loss_official / max(loss_true, 1e-12),
                grad_cos=align)


def main():
    rows = []
    for site in ["wind_cn", "kelmarsh", "smarteole", "penmanshiel", "wind_us"]:
        r = run_site(site)
        rows.append(r)
        print(f"{site:12s} official_js={r['loss_official']:.4f} true_js={r['loss_true']:.4f} "
              f"ratio={r['loss_ratio']:.2f} grad_cos={r['grad_cos']:+.3f}")
    import pandas as pd
    out = pd.DataFrame(rows)
    out.to_csv("../tmp/pct_parity_check.csv", index=False, encoding="utf-8-sig")
    print("\nsaved -> ../tmp/pct_parity_check.csv")
    print("\n=== 判读 ===")
    print("loss_ratio >> 1 或 << 1  -> official 与 true joint 的 loss 量级系统性偏离")
    print("grad_cos 接近 -1 或远小于 +1 -> official 约束把预测推向的方向与真 joint 相反/无关")


if __name__ == "__main__":
    main()
