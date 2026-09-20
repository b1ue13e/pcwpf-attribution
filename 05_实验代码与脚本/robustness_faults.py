from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch


FAULT_GAUSSIAN = "gaussian"
FAULT_DRIFT = "drift"
FAULT_QUANTIZATION = "quantization"
FAULT_MISSINGNESS = "missingness"
FAULT_CLEAN = "clean"

ALL_FAULTS = (
    FAULT_GAUSSIAN,
    FAULT_DRIFT,
    FAULT_QUANTIZATION,
    FAULT_MISSINGNESS,
)


def available_faults() -> Sequence[str]:
    return ALL_FAULTS


def parse_faults(raw: str) -> List[str]:
    faults = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not faults:
        raise ValueError("No valid fault types were provided.")
    unknown = [x for x in faults if x not in ALL_FAULTS and x != FAULT_CLEAN]
    if unknown:
        raise ValueError(f"Unsupported fault type(s): {unknown}. Allowed: {list(ALL_FAULTS)}")
    return faults


def severity_to_missing_rate(severity: float) -> float:
    # Calibrated so severity=0.6 gives ~0.30 missing rate (matches prior supplementary protocol).
    return float(np.clip(0.05 + 0.4166667 * float(severity), 0.0, 0.8))


def _rand(shape: torch.Size, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.rand(shape, device=device)
    cpu_rand = torch.rand(shape, generator=generator, device="cpu")
    return cpu_rand.to(device=device)


def _randn(shape: torch.Size, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, device=device)
    cpu_randn = torch.randn(shape, generator=generator, device="cpu")
    return cpu_randn.to(device=device)


def inject_gaussian(
    x: torch.Tensor,
    severity: float,
    alpha: float = 1.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if severity <= 0:
        return x
    std = x.std(dim=1, keepdim=True) + 1e-8
    noise = _randn(x.shape, device=x.device, generator=generator).to(dtype=x.dtype) * (float(alpha) * float(severity)) * std
    return x + noise


def inject_drift(x: torch.Tensor, severity: float, alpha: float = 1.5) -> torch.Tensor:
    if severity <= 0:
        return x
    bsz, seq_len, _ = x.shape
    if seq_len <= 1:
        return x
    std = x.std(dim=1, keepdim=True) + 1e-8
    ramp = torch.linspace(0.0, 1.0, steps=seq_len, device=x.device, dtype=x.dtype).view(1, seq_len, 1)
    drift = ramp * (float(alpha) * float(severity)) * std
    return x + drift


def inject_quantization(x: torch.Tensor, severity: float) -> torch.Tensor:
    if severity <= 0:
        return x
    std = x.std(dim=1, keepdim=True) + 1e-8
    step = (0.05 + 0.45 * float(severity)) * std
    step = torch.clamp(step, min=1e-6)
    return torch.round(x / step) * step


def inject_missingness(
    x: torch.Tensor,
    severity: float,
    generator: torch.Generator | None = None,
    imputation: str = "ffill_zero",
) -> torch.Tensor:
    if severity <= 0:
        return x
    if imputation != "ffill_zero":
        raise ValueError(f"Unsupported imputation mode: {imputation}")

    out = x.clone()
    p = severity_to_missing_rate(float(severity))
    miss_mask = _rand(out.shape, device=out.device, generator=generator) < p
    out[miss_mask] = float("nan")

    # Forward fill over temporal axis, then zero-fill any remaining NaN.
    seq_len = out.shape[1]
    for t in range(1, seq_len):
        missing = torch.isnan(out[:, t, :])
        out[:, t, :] = torch.where(missing, out[:, t - 1, :], out[:, t, :])
    out[torch.isnan(out)] = 0.0
    return out


def apply_fault_tensor(
    x: torch.Tensor,
    fault: str,
    severity: float,
    *,
    alpha: float = 1.5,
    generator: torch.Generator | None = None,
    missing_imputation: str = "ffill_zero",
) -> torch.Tensor:
    f = fault.lower().strip()
    if f in ("none", FAULT_CLEAN):
        return x
    if f == FAULT_GAUSSIAN:
        return inject_gaussian(x, severity=severity, alpha=alpha, generator=generator)
    if f == FAULT_DRIFT:
        return inject_drift(x, severity=severity, alpha=alpha)
    if f == FAULT_QUANTIZATION:
        return inject_quantization(x, severity=severity)
    if f == FAULT_MISSINGNESS:
        return inject_missingness(
            x,
            severity=severity,
            generator=generator,
            imputation=missing_imputation,
        )
    raise ValueError(f"Unsupported fault type: {fault}")


def sample_fault_uniform(faults: Iterable[str], rng: np.random.Generator) -> str:
    faults = [f for f in faults if f and f != FAULT_CLEAN]
    if not faults:
        return FAULT_CLEAN
    idx = int(rng.integers(0, len(faults)))
    return faults[idx]
