from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultra_lsnt_timeseries import LSNTConfig, UltraLSNTBlock


@dataclass
class SotaPushConfig:
    input_dim: int
    seq_len: int = 96
    pred_len: int = 24
    hidden_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 4
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.15
    router_z_loss_coef: float = 0.01
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.05
    dual_mode_expansion: int = 4
    skip_threshold: float = 0.5
    target_skip_rate: float = 0.12
    skip_budget_weight: float = 1.0
    slow_path_penalty_weight: float = 0.15
    easy_alignment_weight: float = 0.2
    kernel_sizes: Tuple[int, ...] = (3, 7, 15)
    linear_kernel: int = 25
    target_idx: int = -1

    def to_dict(self) -> Dict:
        return asdict(self)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self._cached_mean: torch.Tensor | None = None
        self._cached_std: torch.Tensor | None = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, unbiased=False, keepdim=True) + self.eps)
        self._cached_mean = mean
        self._cached_std = std
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denorm_target(self, y: torch.Tensor, target_idx: int = -1) -> torch.Tensor:
        if self._cached_mean is None or self._cached_std is None:
            raise RuntimeError("RevIN statistics are not available. Call norm() first.")
        mean = self._cached_mean[..., target_idx]
        std = self._cached_std[..., target_idx]
        if self.affine:
            weight = self.weight[..., target_idx]
            bias = self.bias[..., target_idx]
            y = (y - bias) / (weight + self.eps)
        return y * std + mean


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.transpose(1, 2)
        x = self.avg(x)
        return x.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class MultiScaleTemporalStem(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        *,
        num_heads: int,
        dropout: float,
        kernel_sizes: Sequence[int],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.conv_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                )
                for k in kernel_sizes
            ]
        )
        self.branch_fuse = nn.Sequential(
            nn.Linear(hidden_dim * (len(kernel_sizes) + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]

        conv_in = x.transpose(1, 2)
        branches = [x]
        for conv in self.conv_branches:
            branches.append(conv(conv_in).transpose(1, 2))
        x = self.branch_fuse(torch.cat(branches, dim=-1))

        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attn(query, x, x, need_weights=False)
        pooled = pooled.squeeze(1)
        return self.norm(pooled + x.mean(dim=1))


class HorizonDecoder(nn.Module):
    def __init__(self, hidden_dim: int, pred_len: int, dropout: float):
        super().__init__()
        self.pred_len = pred_len
        self.horizon_embed = nn.Parameter(torch.randn(1, pred_len, hidden_dim) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.size(0)
        ctx = context.unsqueeze(1).expand(batch, self.pred_len, -1)
        dec_in = torch.cat([ctx, self.horizon_embed.expand(batch, -1, -1)], dim=-1)
        return self.net(dec_in).squeeze(-1)


class PredictOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module, temperature: float = 0.35):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _ = self.model(x, temperature=self.temperature)
        return pred


class SotaUltraLSNTForecaster(nn.Module):
    def __init__(self, config: SotaPushConfig):
        super().__init__()
        self.config = config
        self.target_idx = config.target_idx

        self.revin = RevIN(config.input_dim, affine=True)
        self.stem = MultiScaleTemporalStem(
            config.input_dim,
            config.hidden_dim,
            config.seq_len,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kernel_sizes=config.kernel_sizes,
        )
        self.moe_cfg = LSNTConfig(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_blocks=config.num_blocks,
            num_experts=config.num_experts,
            top_k=config.top_k,
            dropout=config.dropout,
            router_z_loss_coef=config.router_z_loss_coef,
            router_aux_loss_coef=config.router_aux_loss_coef,
            router_jitter_noise=config.router_jitter_noise,
            skip_threshold=config.skip_threshold,
            target_skip_rate=config.target_skip_rate,
            skip_budget_weight=config.skip_budget_weight,
            slow_path_penalty_weight=config.slow_path_penalty_weight,
            easy_alignment_weight=config.easy_alignment_weight,
            dual_mode_expansion=config.dual_mode_expansion,
            probabilistic_mode="none",
            heteroscedastic_moe=False,
        )
        self.blocks = nn.ModuleList([UltraLSNTBlock(self.moe_cfg) for _ in range(config.num_blocks)])
        self.token_norm = nn.LayerNorm(config.hidden_dim)
        self.pool = AttentionPooling(config.hidden_dim, num_heads=config.num_heads, dropout=config.dropout)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.decoder = HorizonDecoder(config.hidden_dim, config.pred_len, config.dropout)
        self.last_token_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.pred_len),
        )

        self.decomp = SeriesDecomposition(config.linear_kernel)
        self.linear_seasonal = nn.Linear(config.seq_len, config.pred_len)
        self.linear_trend = nn.Linear(config.seq_len, config.pred_len)
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.pred_len),
            nn.Sigmoid(),
        )
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, temperature: float = 0.35, return_stats: bool = False):
        x = self.revin.norm(x)

        target_hist = x[:, :, [self.target_idx]]
        seasonal, trend = self.decomp(target_hist)
        linear_out = self.linear_seasonal(seasonal.squeeze(-1)) + self.linear_trend(trend.squeeze(-1))

        tokens = self.stem(x)
        total_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        block_stats: List[Dict] = []

        batch_size, seq_len, hidden_dim = tokens.shape
        for block in self.blocks:
            flat = tokens.reshape(batch_size * seq_len, hidden_dim)
            flat, stats = block(flat, temperature=temperature)
            tokens = flat.reshape(batch_size, seq_len, hidden_dim)
            total_aux_loss = total_aux_loss + stats["aux_loss"]
            block_stats.append(stats)

        tokens = self.token_norm(tokens)
        pooled = self.pool(tokens)
        last_token = tokens[:, -1, :]
        context = self.context_proj(torch.cat([pooled, last_token], dim=-1))

        nonlinear_out = self.decoder(context) + 0.25 * self.last_token_head(last_token)
        gate = self.fusion_gate(context)
        pred_norm = nonlinear_out + gate * linear_out
        pred_norm = self.output_dropout(pred_norm)
        pred = self.revin.denorm_target(pred_norm, target_idx=self.target_idx)

        if return_stats:
            summary = {
                "avg_compute_ratio": float(np.mean([s["router"]["compute_ratio"] for s in block_stats])) if block_stats else 1.0,
                "avg_skip_rate": float(np.mean([s["dual_mode"]["skip_rate"] for s in block_stats])) if block_stats else 0.0,
                "avg_slow_ratio": float(np.mean([s["dual_mode"]["slow_mode_ratio"] for s in block_stats])) if block_stats else 1.0,
            }
            return pred, total_aux_loss, {"efficiency": summary, "block_stats": block_stats, "gate_mean": gate.mean().item()}
        return pred, total_aux_loss


def partial_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    skip_prefixes: Iterable[str] = (),
) -> Dict[str, int]:
    own_state = model.state_dict()
    matched = 0
    skipped = 0
    loaded_state: Dict[str, torch.Tensor] = {}
    prefixes = tuple(skip_prefixes)

    for key, value in state_dict.items():
        if prefixes and any(key.startswith(prefix) for prefix in prefixes):
            skipped += 1
            continue
        if key not in own_state:
            skipped += 1
            continue
        if own_state[key].shape != value.shape:
            skipped += 1
            continue
        loaded_state[key] = value
        matched += 1

    own_state.update(loaded_state)
    model.load_state_dict(own_state, strict=False)
    return {"matched": matched, "skipped": skipped, "total_incoming": len(state_dict)}


def average_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("Cannot average an empty list of state dicts.")

    averaged: Dict[str, torch.Tensor] = {}
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key].detach().cpu() for sd in state_dicts]
        first = tensors[0]
        if torch.is_floating_point(first):
            stacked = torch.stack([t.to(dtype=torch.float32) for t in tensors], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
