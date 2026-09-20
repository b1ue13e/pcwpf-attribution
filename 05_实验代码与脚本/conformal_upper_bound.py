"""Conformal Upper Prediction Bound for Realized Power (partial-identification version).

Honest scope (2026-09-03 decision dec-a57748cdfd637a47):

  * We do NOT learn the latent operational feasible upper bound U*.
    With only wind + realized power, U* is NOT identified: infinitely many
    U* are compatible with the same observed Y (knowing Y <= U* only gives a
    downward witness).

  * What we DO learn: the conformal upper prediction bound
        Uhat(alpha) = Q_{1-alpha}(Y | H)  +  qhat(alpha)
    i.e. a calibrated upper bound for REALIZED power Y, not for U*.

  * Partial identification: because  Y <= U* <= P_rated,  any constraint risk is
    only partially identified.  The observed side (false exclusion) is a
    DETERMINISTIC lower bound:  Uhat < Y  =>  Uhat < U*  (certain FE).
    The unobserved side (false admission) cannot be bounded without further
    assumptions on the conditional distribution of U* - Y.

  * Therefore the honest claim is: for a target false-exclusion (witness) risk
    alpha, the conformal bound achieves FER ~ alpha on realized power, while a
    deterministic proxy ceiling has uncontrolled FER.  This is a coverage claim
    about a realized-power forecast bound, NOT an identification of the physical
    feasible ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class _QuantLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_len, quantiles):
        super().__init__()
        self.quantiles = list(quantiles)
        self.pred_len = pred_len
        self.nq = len(quantiles)
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, pred_len * self.nq)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(x.size(0), self.pred_len, self.nq)


def _pinball(pred_q, y, qs):
    y = y.unsqueeze(-1)
    err = y - pred_q
    q = torch.tensor(qs, device=pred_q.device, dtype=pred_q.dtype)
    return torch.where(err >= 0, q, q - 1.0).mul(err).mean()


def partial_id_metrics(Uhat: np.ndarray, Y: np.ndarray, ramp_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Partial-identification metrics on the OBSERVED side only.

    Returns:
      FER_lb   : P(Uhat < Y)  — deterministic LOWER bound on false exclusion
                 (Uhat < Y  =>  Uhat < U*, so the true FE >= this).
      LEC_lb   : E[(Y-Uhat)^+ | FE] — lower bound on clipped legitimate energy.
      coverage : observed coverage of the realized-power bound (1 - FER_lb).
      sharpness: mean ceiling height (a conservatism proxy, not a risk bound).
    """
    fe = (Uhat < Y).astype(np.float64)
    fer = float(fe.mean()) if fe.size else 0.0
    lec = float(np.where(fe > 0, (Y - Uhat) * fe, 0.0).mean()) if fe.any() else 0.0
    ramp_fer = float(fe[ramp_mask].mean()) if ramp_mask is not None and ramp_mask.any() else float("nan")
    return {
        "FER_lb": float(round(fer, 6)),
        "LEC_lb": float(round(lec, 6)),
        "coverage": float(round(1.0 - fer, 6)),
        "sharpness": float(round(float(Uhat.mean()), 6)) if Uhat.size else 0.0,
        "ramp_FER_lb": float(round(ramp_fer, 6)) if ramp_fer == ramp_fer else float("nan"),
    }


@dataclass
class ConformalUpperBound:
    """Conformal upper prediction bound for realized power (NOT for U*)."""

    quantiles: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95)
    hidden_dim: int = 32
    lr: float = 1e-3
    epochs: int = 15
    batch_size: int = 64
    device: str = "auto"
    _model: Optional[_QuantLSTM] = None
    _qhat: Dict[float, float] = field(default_factory=dict)
    _qhat_by_regime: Dict[float, Dict[int, float]] = field(default_factory=dict)
    _has_regime: bool = False
    _fitted: bool = False

    def fit(self, X, Y, seed=42):
        self.device = self.device if self.device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed); np.random.seed(seed)
        C = X.shape[2]
        H = Y.shape[1]
        self._model = _QuantLSTM(C, self.hidden_dim, H, self.quantiles).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        Xt = torch.from_numpy(X.astype(np.float32))
        Yt = torch.from_numpy(Y.astype(np.float32))
        for _ in range(self.epochs):
            self._model.train()
            perm = torch.randperm(len(Xt))
            for bi in range(0, len(Xt), self.batch_size):
                idx = perm[bi:bi + self.batch_size]
                xb, yb = Xt[idx].to(self.device), Yt[idx].to(self.device)
                loss = _pinball(self._model(xb), yb, self.quantiles)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
        self._fitted = True
        return self

    def _predict(self, X):
        self._model.eval()
        out = []
        with torch.no_grad():
            for bi in range(0, len(X), 256):
                xb = torch.from_numpy(X[bi:bi + 256].astype(np.float32)).to(self.device)
                out.append(self._model(xb).cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, X.shape[1], len(self.quantiles)))

    def _level_idx(self, alpha):
        return int(np.argmin([abs(q - (1.0 - alpha)) for q in self.quantiles]))

    def calibrate(self, X_cal, Y_cal, alphas):
        q_cal = self._predict(X_cal)
        for alpha in alphas:
            qi = self._level_idx(alpha)
            base = q_cal[:, :, qi]
            # one-sided non-negative conformity score on realized power
            score = np.maximum(Y_cal - base, 0.0).flatten()
            n = len(score)
            if n == 0:
                self._qhat[float(alpha)] = 0.0
                continue
            k = min(int(np.ceil((1.0 - alpha) * (n + 1))), n - 1)
            k = max(k, 0)
            self._qhat[float(alpha)] = float(np.partition(score, k)[k])
        return self

    # ---- regime-conditional + block conformal (fixes under-coverage & seq correlation) ----
    def calibrate_regime(self, X_cal, Y_cal, regime_ids, alphas):
        """Regime-conditional split conformal.

        regime_ids : (N,) int array, one regime label per CALIBRATION sample.
                     Each regime gets its OWN conformal shift so coverage holds
                     within each regime (fixes kelmarsh-style under-coverage in
                     one regime dragging the pooled quantile).

        Block structure: within a regime we use, per sample, the MAX of the H
        one-sided scores (block score), which restores exchangeability at the
        SAMPLE level under overlapping multi-step targets (H windows share
        features), then take the empirical (1-alpha) quantile over samples.
        """
        q_cal = self._predict(X_cal)
        self._qhat_by_regime = {}
        for alpha in alphas:
            qi = self._level_idx(alpha)
            base = q_cal[:, :, qi]
            per_slot_score = np.maximum(Y_cal - base, 0.0)     # (N, H)
            block_score = per_slot_score.max(axis=1)           # (N,) sample-level score
            self._qhat_by_regime[float(alpha)] = {}
            for g in np.unique(regime_ids):
                sel = block_score[regime_ids == g]
                ng = len(sel)
                if ng == 0:
                    self._qhat_by_regime[float(alpha)][int(g)] = 0.0
                    continue
                k = min(int(np.ceil((1.0 - alpha) * (ng + 1))), ng - 1)
                k = max(k, 0)
                self._qhat_by_regime[float(alpha)][int(g)] = float(np.partition(sel, k)[k])
        self._has_regime = True
        return self

    def bound_regime(self, X, alpha, regime_ids):
        """Ceiling with per-regime conformal shift. regime_ids must align with X."""
        qi = self._level_idx(alpha)
        base = self._predict(X)[:, :, qi]
        out = base.copy()
        for g in np.unique(regime_ids):
            shift = self._qhat_by_regime.get(float(alpha), {}).get(int(g), 0.0)
            out[regime_ids == g] += shift
        return out

    def bound(self, X, alpha):
        """Conformal upper prediction bound for realized power at risk alpha."""
        qi = self._level_idx(alpha)
        base = self._predict(X)[:, :, qi]
        return base + self._qhat.get(float(alpha), 0.0)

    def audit(self, X, Y, alpha, ramp_mask=None):
        return partial_id_metrics(self.bound(X, alpha), Y, ramp_mask)


def partial_id_fe_bounds(Uhat, Y):
    """Explicit partial-identification statement:
    true false-exclusion FE_true satisfies  FE_true >= P(Uhat < Y)  (observed lower
    bound).  The upper bound on FE_true is NOT identified without assumptions on
    U* - Y.  This function returns the deterministic quantities only."""
    fe_lb = float((Uhat < Y).astype(np.float64).mean()) if Uhat.size else 0.0
    return {"FE_true_lower_bound": float(round(fe_lb, 6)),
            "note": "FE_true >= this; upper bound unidentified without assumptions on U* - Y"}
