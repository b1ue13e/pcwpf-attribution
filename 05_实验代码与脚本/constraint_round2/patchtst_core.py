"""
patchtst_core.py
================
Modern Channel-Independent PatchTST (Nie et al., ICLR 2023) backbone 
under the prospective constraint auditing protocol.
Evaluates:
- 3 Sites: Kelmarsh, Hill of Towie, Penmanshiel
- 4 Methods: Unconstrained, Bounds, Power Curve, Distribution Prior
- 5 Seeds: [42, 123, 456, 789, 1024]
- Real-world BSC 30-minute dynamic balancing price settlement
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# Configure stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))

from core import starts, windows, perturb, fit_prior, penalty, infer

# -----------------------------------------------------------------------------
# PatchTST Architecture
# -----------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            x = x + self.mean
            return x

class PatchTST(nn.Module):
    def __init__(self, L=36, H=36, patch_len=6, stride=4, d_model=64, n_heads=4, d_ff=128, e_layers=2, dropout=0.1):
        super().__init__()
        self.L = L
        self.H = H
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (L - patch_len) // stride + 1
        
        self.revin = RevIN(num_features=1, affine=False)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Linear(self.n_patches * d_model, H)

    def forward(self, x):
        # Channel-independent on target generation: x[:, :, 0:1]
        p = x[:, :, 0:1]
        p_norm = self.revin(p, 'norm').squeeze(-1) # (B, L)
        
        # Unfold patches
        patches = p_norm.unfold(dimension=-1, size=self.patch_len, step=self.stride) # (B, n_patches, patch_len)
        
        # Embedding + Positional Encoding
        tokens = self.patch_embed(patches) + self.pos_embed # (B, n_patches, d_model)
        
        # Transformer Encoding
        enc_out = self.encoder(tokens)
        
        # Flatten and Project to Horizon
        out_norm = self.head(enc_out.reshape(enc_out.size(0), -1)) # (B, H)
        
        # Denormalize
        out = self.revin(out_norm.unsqueeze(-1), 'denorm').squeeze(-1) # (B, H)
        return out

# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------
def train_patchtst(seed, family, lam, factor, control, trainxy, valxy, prior, device, epochs=30):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PatchTST(L=CFG['lookback'], H=CFG['horizon']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    xt, yt = [torch.as_tensor(a, device=device) for a in trainxy]
    vp = valxy[1]
    best = float('inf')
    state = None
    stale = 0
    history = []
    pri = {k: torch.as_tensor(v, device=device) for k, v in prior.items()}
    rng = np.random.default_rng(seed)

    batch_size = CFG['batch_size']
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(xt))
        n_batches = int(np.ceil(len(xt) / batch_size))
        for ix in np.array_split(perm, n_batches):
            x_b = xt[ix]
            y_b = yt[ix]
            opt.zero_grad()
            pred = model(x_b)
            loss = F.mse_loss(pred, y_b) + lam * penalty(pred, x_b, family, factor, pri)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss in PatchTST')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_pred = infer(model, valxy[0], device)
        val_mse = float(np.mean((val_pred - vp) ** 2))
        history.append(val_mse)

        if val_mse < best:
            best = val_mse
            state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= CFG['patience']:
            break

    model.load_state_dict(state)
    return model, best, history

# -----------------------------------------------------------------------------
# Experiment Matrix Runner
# -----------------------------------------------------------------------------
def run_patchtst_matrix(site, data_dir, out_dir, epochs=30):
    data_path = Path(data_dir) / f'{site}.csv'
    frame = pd.read_csv(data_path, index_col='time', parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    
    split_cfg = CFG['splits'][site]
    x, y, ids = starts(frame, split_cfg)
    trainxy = windows(x, y, ids['train'])
    valxy = windows(x, y, ids['val'])
    prior = fit_prior(frame, split_cfg['train_end'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = [
        ('none', 0.0, 1.0, 'plain'),
        ('bounds', 0.01, 1.0, 'plain'),
        ('curve', 0.01, 1.0, 'plain'),
        ('distribution', 0.01, 1.0, 'plain'),
    ]
    seeds = CFG['seed_list']

    print(f"\n==========================================")
    print(f"Running PatchTST Matrix for site: {site}")
    print(f"Device: {device} | Seeds: {seeds}")
    print(f"==========================================")

    results = []
    for seed in seeds:
        seed_dir = Path(out_dir) / site / 'patchtst' / f'seed{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        for family, lam, factor, control in configs:
            tag = f'{family}_l{lam}_f{factor}_{control}'
            pt_path = seed_dir / f'{tag}.pt'
            meta_path = seed_dir / f'{tag}.json'

            if meta_path.exists() and pt_path.exists():
                print(f"[{site}] seed{seed} {tag} already exists. Skipping training.")
                res = json.loads(meta_path.read_text())
                results.append(res)
                continue

            t0 = time.time()
            model, best_val, hist = train_patchtst(
                seed, family, lam, factor, control,
                trainxy, valxy, prior, device, epochs=epochs
            )
            elapsed = time.time() - t0
            torch.save(model.state_dict(), pt_path)

            row = {
                'site': site,
                'backbone': 'patchtst',
                'seed': seed,
                'tag': tag,
                'family': family,
                'lambda': lam,
                'factor': factor,
                'control': control,
                'val_mse': best_val,
                'val_rmse': float(np.sqrt(best_val)),
                'elapsed_s': elapsed
            }
            meta_path.write_text(json.dumps(row, indent=2))
            results.append(row)
            print(f"[{site}] seed{seed} {tag} -> val_rmse={row['val_rmse']:.5f} ({elapsed:.1f}s)")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(Path(out_dir) / site / 'patchtst' / 'validation_summary.csv', index=False)
    print(f"Validation summary saved for {site}.")
    return summary_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', default='kelmarsh', choices=['kelmarsh', 'hill_of_towie', 'penmanshiel', 'all'])
    parser.add_argument('--data', default=str(ROOT / '04_数据集' / 'constraint_rebuild_v2'))
    parser.add_argument('--out', default=str(ROOT / '06_实验结果与日志' / 'constraint_round2'))
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    sites = ['kelmarsh', 'hill_of_towie', 'penmanshiel'] if args.site == 'all' else [args.site]
    for s in sites:
        run_patchtst_matrix(s, args.data, args.out, epochs=args.epochs)
