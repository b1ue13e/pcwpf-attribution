"""
Controlled Prospective Ablation Matrix (3D Disentanglement Suite).

Designed to causally disentangle:
1. Genuine Domain Physical Knowledge
2. Information Asymmetry (unconstrained model reading power only vs prior reading wind speed)
3. Non-specific Regularization (scale-matched permuted/scrambled pseudo-prior)
4. Time Proxy Mismatch (lagged wind persistence proxy decay over horizon)

Baselines:
- Arm A (Info Symmetry): DLinear with dual input [Power, Wind] unconstrained.
- Arm B (Pseudo-Prior): Curve & Distribution penalties with scrambled/permuted wind-power mapping (matched lambda & gradient scale).
- Arm C (Lag Scan): Prior queried using wind at t-k (k in [0, 6, 12, 18, 36]).
"""

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from core import Forecaster, starts, windows, fit_prior, infer, CFG

class DualChannelDLinear(nn.Module):
    """
    DLinear that takes both Power (channel 0) and Wind Speed (channel 1),
    providing strict information symmetry against physics priors that ingest wind speed.
    """
    def __init__(self, L, H):
        super().__init__()
        self.avg = nn.AvgPool1d(25, stride=1)
        self.seasonal_p = nn.Linear(L, H)
        self.trend_p = nn.Linear(L, H)
        self.proj_w = nn.Linear(L, H)
        nn.init.constant_(self.seasonal_p.weight, 1.0 / L)
        nn.init.constant_(self.trend_p.weight, 1.0 / L)
        nn.init.constant_(self.proj_w.weight, 1.0 / L)

    def forward(self, x):
        p = x[:, :, 0]
        w = x[:, :, 1]
        pad = torch.cat([p[:, :1].repeat(1, 12), p, p[:, -1:].repeat(1, 12)], 1)
        trend = self.avg(pad[:, None]).squeeze(1)
        seasonal = p - trend
        out_p = self.seasonal_p(seasonal) + self.trend_p(trend)
        out_w = self.proj_w(w)
        return out_p + out_w

def fit_scrambled_prior(base_prior, seed=42):
    """
    Construct scale-matched pseudo-prior by permuting the wind-speed to power mapping.
    This preserves the distribution of target values and gradient scale while destroying
    the physical relationship.
    """
    rng = np.random.default_rng(seed)
    scrambled = copy.deepcopy(base_prior)
    perm = rng.permutation(len(base_prior['means']))
    scrambled['means'] = base_prior['means'][perm]
    scrambled['hist'] = base_prior['hist'][perm]
    return scrambled

def controlled_penalty(pred, x, family, factor, prior, lag_step=0):
    """
    Lag-aware penalty calculation.
    lag_step: how many steps in the past to sample wind speed from (0 means x[:, -1, 1]).
    """
    if family == 'none':
        return pred.sum() * 0.0
    if family == 'bounds':
        return (F.relu(-pred)**2 + F.relu(pred - factor)**2).mean()
    
    # Time proxy index selection
    query_pos = max(0, x.shape[1] - 1 - lag_step)
    idx = (x[:, query_pos, 1] * 15.0).long().clamp(0, len(prior['means']) - 1)
    
    if family == 'curve':
        target = prior['means'][idx, None] * factor
        return ((pred - target) ** 2).mean()
    
    if family == 'distribution':
        p = torch.exp(-0.5 * ((pred[:, :, None] / factor - prior['grid']) / 0.06) ** 2).mean(1) + 1e-8
        p = p / p.sum(1, keepdim=True)
        q = prior['hist'][idx] + 1e-8
        q = q / q.sum(1, keepdim=True)
        m = (p + q) / 2.0
        return 0.5 * (p * (p.log() - m.log()) + q * (q.log() - m.log())).sum(1).mean()
    
    raise ValueError(f"Unknown family: {family}")

def run_ablation_selftest():
    print("Running controlled ablation sanity checks...")
    # 1. DualChannelDLinear forward & backward
    x = torch.randn(8, 36, 2, requires_grad=True)
    m = DualChannelDLinear(36, 36)
    out = m(x)
    assert out.shape == (8, 36)
    out.sum().backward()
    assert (x.grad.abs().sum((0, 1)) > 0).all(), "Both channels must receive gradients!"
    
    # 2. Scrambled prior scale matching
    dummy_prior = {
        'means': np.linspace(0.1, 0.9, 31).astype(np.float32),
        'hist': np.ones((31, 40), dtype=np.float32) / 40.0,
        'grid': np.linspace(-0.1, 1.2, 40).astype(np.float32)
    }
    scrambled = fit_scrambled_prior(dummy_prior, seed=42)
    assert len(scrambled['means']) == len(dummy_prior['means'])
    assert not np.array_equal(scrambled['means'], dummy_prior['means']), "Scrambled means should be permuted!"
    assert np.isclose(scrambled['means'].sum(), dummy_prior['means'].sum()), "Total sum and scale must be invariant!"
    
    # 3. Lagged penalty query
    pred = torch.rand(8, 36)
    pri_t = {k: torch.as_tensor(v) for k, v in dummy_prior.items()}
    loss_lag0 = controlled_penalty(pred, x, 'curve', 1.0, pri_t, lag_step=0)
    loss_lag10 = controlled_penalty(pred, x, 'curve', 1.0, pri_t, lag_step=10)
    assert loss_lag0.item() >= 0 and loss_lag10.item() >= 0
    print("PASS: Controlled ablation suite verified successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true', help='Run sanity verification')
    args = parser.parse_args()
    if args.selftest or True:
        run_ablation_selftest()
