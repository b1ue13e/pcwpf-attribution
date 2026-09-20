"""CCF — Causal Chance-Constrained Feasibility (Contribution 3).

A standalone method for the "constraint observability gap" problem:

    Replace a deterministic proxy ceiling c(H_t) with the conditional quantile
    Q_alpha(U* | H_t), trained with pinball loss and calibrated with split
    conformal prediction, so that the deployed ceiling achieves a *declared*
    false-exclusion risk alpha instead of an uncontrolled, implicit one.

Three-step pipeline (all estimation on disjoint slices):
    fit(X_tr, Y_tr)                 -> quantile forecaster (pinball)
    calibrate(X_cal, Y_cal, alphas) -> split-conformal correction per alpha
    ceiling(X, alpha)               -> Uhat = Q_{1-alpha}(X) + qhat(alpha)

Theory (see Contribution 1 note):
    * alpha is the risk budget / false-exclusion trade-off knob;
    * Bayes-optimal for asymmetric (c_FE, c_FA) cost with alpha* = c_FE/(c_FE+c_FA);
    * offline metrics only certify the false-exclusion side (feasible-witness
      semantics): Uhat < Y => certain FE; Uhat >= Y does NOT certify safety.

Design invariants (no train/test leakage):
    scaler, feature transformation and quantile model fit on TRAIN only;
    conformal quantile qhat computed on CALIBRATION only;
    evaluator sees TEST only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quantile forecaster
# ---------------------------------------------------------------------------
class _QuantLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, pred_len: int, quantiles: Sequence[float]):
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


# ---------------------------------------------------------------------------
# Metrics (offline, feasible-witness semantics)
# ---------------------------------------------------------------------------
def fe_metrics(Uhat: np.ndarray, Y: np.ndarray, ramp_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Uhat, Y, optional ramp_mask are (B, H) arrays; returns the audit metrics."""
    fe = (Uhat < Y).astype(np.float64)
    fer = float(fe.mean()) if fe.size else 0.0
    lec = float(np.where(fe > 0, (Y - Uhat) * fe, 0.0).mean()) if fe.any() else 0.0
    cov = 1.0 - fer
    sharp = float(Uhat.mean()) if Uhat.size else 0.0
    ramp_fer = float(fe[ramp_mask].mean()) if ramp_mask is not None and ramp_mask.any() else float("nan")
    return {
        "FER": float(round(fer, 6)),
        "LEC": float(round(lec, 6)),
        "coverage": float(round(cov, 6)),
        "sharpness": float(round(sharp, 6)),
        "ramp_FER": float(round(ramp_fer, 6)) if ramp_fer == ramp_fer else float("nan"),
    }


# ---------------------------------------------------------------------------
# CCF method
# ---------------------------------------------------------------------------
@dataclass
class CCF:
    """Causal Chance-Constrained Feasibility ceiling estimator."""

    quantiles: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95)
    hidden_dim: int = 32
    lr: float = 1e-3
    epochs: int = 15
    batch_size: int = 64
    device: str = "auto"
    _model: Optional[_QuantLSTM] = None
    _qhat: Dict[float, float] = field(default_factory=dict)   # alpha -> conformal shift
    _fitted: bool = False

    # ---- training ----
    def fit(self, X: np.ndarray, Y: np.ndarray, seed: int = 42) -> "CCF":
        """X: (N, L, C), Y: (N, H) per-unit power targets."""
        self.device = self.device if self.device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed); np.random.seed(seed)
        L, C = X.shape[1], X.shape[2]
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

    # ---- inference ----
    def _predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "call fit() first"
        self._model.eval()
        out = []
        with torch.no_grad():
            for bi in range(0, len(X), 256):
                xb = torch.from_numpy(X[bi:bi + 256].astype(np.float32)).to(self.device)
                out.append(self._model(xb).cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, X.shape[1], len(self.quantiles)))

    def _level_idx(self, alpha: float) -> int:
        target = 1.0 - alpha
        return int(np.argmin([abs(q - target) for q in self.quantiles]))

    # ---- calibration ----
    def calibrate(self, X_cal: np.ndarray, Y_cal: np.ndarray, alphas: Sequence[float]) -> "CCF":
        """Split-conformal: for each alpha, shift the base quantile so empirical
        coverage on the calibration slice is at least 1-alpha (one-sided)."""
        q_cal = self._predict(X_cal)
        for alpha in alphas:
            qi = self._level_idx(alpha)
            base = q_cal[:, :, qi]
            score = np.maximum(Y_cal - base, 0.0).flatten()   # one-sided non-negative
            n = len(score)
            if n == 0:
                self._qhat[float(alpha)] = 0.0
                continue
            k = min(int(np.ceil((1.0 - alpha) * (n + 1))), n - 1)
            k = max(k, 0)
            self._qhat[float(alpha)] = float(np.partition(score, k)[k])
        return self

    # ---- ceiling ----
    def ceiling(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """Deployed ceiling at risk level alpha: Q_{1-alpha}(X) + qhat(alpha)."""
        assert self._fitted, "call fit() first"
        qi = self._level_idx(alpha)
        base = self._predict(X)[:, :, qi]
        qhat = self._qhat.get(float(alpha), 0.0)
        return base + qhat

    # ---- audit ----
    def audit(self, X: np.ndarray, Y: np.ndarray, alpha: float,
              ramp_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        return fe_metrics(self.ceiling(X, alpha), Y, ramp_mask)
