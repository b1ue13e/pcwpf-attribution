"""kl_div numerical comparison: PCT official `F.kl_div(p.log(), q)` vs standard KL.

Standard KL(p||q) = sum p * (log p - log q) = sum p.log()*p - p.log()*q ... precisely:
    KL(p||q) = sum_i p_i * (log p_i - log q_i)

PyTorch F.kl_div(input, target, reduction='sum') computes:
    sum_i target_i * (log target_i - input_i)
i.e. F.kl_div(log_q, p) actually equals KL(p||q). But PCT calls F.kl_div(p.log(), q),
which computes sum_i q_i * (log q_i - log p_i) = KL(q||p) — the *reversed* direction.

So official 'js_div' = 0.5*(KL(q||p) + KL(q||m))-style but with wrong argument order,
AND on joint histograms that are already the outer-product artifact.

This script quantifies, on real audit data, the difference between:
  (A) official:  js = 0.5*(F.kl_div(p.log(), m) + F.kl_div(q.log(), m))   [as in PCT]
  (B) standard:  js = 0.5*(kl_std(p, m) + kl_std(q, m)),  kl_std = sum p*(log p - log m)
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_train_audit import SITES, fit_curve, power_curve_pu  # noqa: E402


def marginal_pdf(values, bins, a, b=1e-10):
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / a).pow(2))
    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + b
    pdf = pdf / normalization
    return pdf, kernel_values + torch.tensor(1e-10)


def joint_pdf_official(k1, k2):
    j = torch.matmul(k1.transpose(1, 2), k2)
    norm = torch.sum(j, dim=(1, 2)).view(-1, 1, 1)
    return j / norm


def kl_std(p, q):
    # KL(p||q) = sum p * (log p - log q)
    return torch.sum(p * (p.log() - q.log()))


def js_official(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction="sum") + F.kl_div(q.log(), m, reduction="sum"))


def js_standard(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_std(p, m) + kl_std(q, m))


def run_site(site, bin_num=50):
    import pandas as pd
    df = pd.read_csv(SITES[site])
    ws = df["wind_speed"].to_numpy(np.float64)
    p = df["power"].to_numpy(np.float64)
    ok = (ws >= 0) & (ws <= 40) & np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[ok], p[ok]
    n = len(ws)
    i1 = int(n * 0.8)
    ws_tr, p_tr = ws[:i1], p[:i1]
    ws_te, p_te = ws[i1:], p[i1:]
    p_rated, v_in, v_rated = fit_curve(ws_tr, p_tr)

    pu = lambda a: np.clip(a / max(p_rated, 1e-6), 0.0, 1.0)
    pred = power_curve_pu(ws_te, v_in, v_rated) * p_rated

    power = torch.tensor(pu(p_te), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    predp = torch.tensor(pu(pred), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    wind = torch.tensor(((ws_te - ws_te.mean()) / (ws_te.std() + 1e-8)),
                        dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    bins_p = torch.linspace(float(power.min()), float(power.max()), bin_num)
    bins_w = torch.linspace(float(wind.min()), float(wind.max()), bin_num)
    bw = torch.pow(torch.tensor(float(bin_num)), -1 / 5) * 0.1
    h_p = power.std() * bw
    h_w = wind.std() * bw

    _, kp_t = marginal_pdf(power, bins_p, h_p)
    _, kw_t = marginal_pdf(wind, bins_w, h_w)
    pc_t = joint_pdf_official(kp_t, kw_t)
    _, kp_q = marginal_pdf(predp, bins_p, h_p)
    _, kw_q = marginal_pdf(wind, bins_w, h_w)
    pc_q = joint_pdf_official(kp_q, kw_q)

    off = float(js_official(pc_q, pc_t))
    std = float(js_standard(pc_q, pc_t))
    return dict(site=site, js_official=off, js_standard=std,
                ratio=off / max(std, 1e-12),
                signed_diff=off - std)


def main():
    rows = []
    for site in ["wind_cn", "kelmarsh", "smarteole", "penmanshiel", "wind_us"]:
        r = run_site(site)
        rows.append(r)
        print(f"{site:12s} js_official={r['js_official']:.4f} js_standard={r['js_standard']:.4f} "
              f"ratio={r['ratio']:.3f} diff={r['signed_diff']:+.4f}")
    import pandas as pd
    out = pd.DataFrame(rows)
    out.to_csv("../tmp/pct_kldiv_check.csv", index=False, encoding="utf-8-sig")
    print("\nsaved -> ../tmp/pct_kldiv_check.csv")
    print("\n=== 判读 ===")
    print("js_official 为负 -> 说明 F.kl_div(p.log(), q) 已导致 KL 表达式为负（KL 应 >= 0），")
    print("即非规范 KL 不仅在数值上偏离，还破坏 KL 的非负性；ratio 偏离 1 表示系统性偏差。")


if __name__ == "__main__":
    main()
