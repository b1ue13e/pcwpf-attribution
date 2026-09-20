"""Unit tests for physgate.py (Route A: calibration-free feasibility guardrail).

Run from 05_实验代码与脚本:
    ../.venv/Scripts/python.exe -m unittest test_physgate -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

import physgate as pg

THETA = pg.TurbineEnvelope(p_rated=200.0, v_in=3.0, v_rated=11.5, v_out=25.0)


def _ar1_increments(n: int, rho: float, sigma_delta: float, seed: int = 0) -> np.ndarray:
    """Increments following d_t = rho * d_{t-1} + eps_t with std(d) = sigma_delta."""
    rng = np.random.default_rng(seed)
    eps_std = sigma_delta * math.sqrt(1.0 - rho * rho)
    eps = rng.normal(0.0, eps_std, size=n)
    d = np.empty(n)
    d[0] = eps[0] / math.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        d[t] = rho * d[t - 1] + eps[t]
    return d


class TestEnvelope(unittest.TestCase):
    def test_piecewise_values(self):
        v = torch.tensor([2.0, 3.0, 5.0, 11.5, 20.0, 24.999, 25.0, 30.0])
        env = pg.power_envelope_pu(v, THETA)
        expected_mid = ((5.0 - 3.0) / (11.5 - 3.0)) ** 3
        self.assertTrue(torch.allclose(env, torch.tensor([0.0, 0.0, expected_mid, 1.0, 1.0, 1.0, 0.0, 0.0]), atol=1e-6))

    def test_validation_rejects_bad_constants(self):
        with self.assertRaises(ValueError):
            pg.TurbineEnvelope(200.0, 11.5, 3.0, 25.0).validate()
        with self.assertRaises(ValueError):
            pg.TurbineEnvelope(-200.0, 3.0, 11.5, 25.0).validate()


class TestBand(unittest.TestCase):
    def test_eta_two_physical_limits(self):
        v = torch.tensor([10.0])
        s = torch.tensor([1.0])
        up0, _ = pg.proxy_band(v, s, torch.tensor([0.0]), horizon=4, q=2.0)
        up1, _ = pg.proxy_band(v, s, torch.tensor([1.0]), horizon=4, q=2.0)
        band0 = up0 - v
        band1 = up1 - v
        # rho=0 -> h^0.5 ; rho=1 -> h^1.0 ; q=2, sigma=1
        self.assertTrue(torch.allclose(band0, 2.0 * torch.tensor([1.0, math.sqrt(2), math.sqrt(3), 2.0]), atol=1e-5))
        self.assertTrue(torch.allclose(band1, 2.0 * torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-5))

    def test_lower_band_clipped_at_zero(self):
        v = torch.tensor([1.0])
        s = torch.tensor([2.0])
        _, lo = pg.proxy_band(v, s, torch.tensor([0.5]), horizon=4, q=2.0)
        self.assertTrue((lo >= 0.0).all())


class TestRollingStats(unittest.TestCase):
    def test_ar1_parameter_recovery(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(1)
        B, L = 200, 2000
        d = _ar1_increments(B * L, rho=0.6, sigma_delta=0.8, seed=2).reshape(B, L)
        v = 10.0 + np.cumsum(d, axis=1)
        x = torch.tensor(v, dtype=torch.float32)
        sigma, rho = pg.rolling_turbulence_stats(x, window=96)
        self.assertAlmostEqual(float(sigma.mean()), 0.8, delta=0.08)
        self.assertAlmostEqual(float(rho.mean()), 0.6, delta=0.05)

    def test_numpy_torch_consistency(self):
        d = _ar1_increments(300, rho=0.4, sigma_delta=0.7, seed=3)
        v = 8.0 + np.cumsum(d)
        st, rt = pg.rolling_turbulence_stats(torch.tensor(v, dtype=torch.float32), window=96)
        sn, rn = pg.rolling_turbulence_stats_np(v, window=96)
        self.assertAlmostEqual(float(st[0]), float(sn[0]), delta=1e-2)
        self.assertAlmostEqual(float(rt[0]), float(rn[0]), delta=1e-2)

    def test_flatline_run_masked(self):
        rng = np.random.default_rng(4)
        v = 10.0 + np.cumsum(rng.normal(0, 0.5, size=200))
        v[-40:] = v[-41]  # stuck sensor: 40-step flatline at the tail
        s_masked, _ = pg.rolling_turbulence_stats(torch.tensor(v, dtype=torch.float32), window=96)
        naive = np.std(np.diff(v[-96:]))
        self.assertGreater(float(s_masked[0]), 0.3)
        self.assertGreater(float(s_masked[0]), float(naive))  # masked one must not be dragged down

    def test_all_flat_series_returns_zero(self):
        v = torch.full((2, 200), 12.0)
        sigma, rho = pg.rolling_turbulence_stats(v, window=96)
        self.assertTrue(torch.allclose(sigma, torch.zeros(2)))
        self.assertTrue(torch.allclose(rho, torch.zeros(2)))

    def test_corrupted_stream_stays_finite(self):
        rng = np.random.default_rng(5)
        v = 10.0 + np.cumsum(rng.normal(0, 0.6, size=300))
        v[10:14] = np.nan              # dropout
        v[60:80] = v[59]               # long flatline
        v[30], v[31] = 60.0, -20.0     # spike outliers
        x = torch.tensor(v, dtype=torch.float32)
        sigma, rho = pg.rolling_turbulence_stats(x, window=96)
        self.assertTrue(torch.isfinite(sigma).all() and (sigma >= 0).all())
        self.assertTrue(torch.isfinite(rho).all() and ((rho >= 0) & (rho <= 1)).all())
        up, lo = pg.proxy_band(x[-1:] if x.dim() == 1 else x[:, -1], sigma, rho, horizon=8)
        self.assertTrue(torch.isfinite(up).all() and (up <= pg.DEFAULT_MAX_WIND_MPS).all())
        self.assertTrue(torch.isfinite(lo).all())


class TestBandCoverage(unittest.TestCase):
    def test_q2_one_sided_coverage(self):
        """Empirical exceedance of v_{t+h} over v_up should be ~ 1-Phi(2) = 2.3%."""
        torch.manual_seed(7)
        B, W, H = 5000, 96, 4
        d = _ar1_increments(B * (W + H), rho=0.5, sigma_delta=1.0, seed=8).reshape(B, W + H)
        v = 10.0 + np.cumsum(d, axis=1)
        x = torch.tensor(v, dtype=torch.float32)
        sigma, rho = pg.rolling_turbulence_stats(x[:, :W], window=W)
        up, _ = pg.proxy_band(x[:, W - 1], sigma, rho, horizon=H, q=2.0)
        true_future = x[:, W : W + H]
        coverage = (true_future > up).float().mean(dim=0)
        nominal = 1.0 - 0.5 * (1.0 + math.erf(2.0 / math.sqrt(2.0)))  # 1-Phi(2)
        for h in range(H):
            c = float(coverage[h])
            self.assertGreater(c, 0.005, f"h={h+1}: coverage {c:.4f} too low")
            self.assertLess(c, 0.05, f"h={h+1}: coverage {c:.4f} too high")
        self.assertAlmostEqual(nominal, 0.02275, places=4)


class TestPhysgateLoss(unittest.TestCase):
    def test_feasible_prediction_zero_loss(self):
        pred = torch.full((1, 4), 0.4)
        v_last = torch.tensor([10.0])
        sigma = torch.tensor([0.5])
        rho = torch.tensor([0.5])
        loss = pg.physgate_loss(pred, v_last, sigma, rho, THETA)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_above_band_penalized_and_grad_flows(self):
        pred = torch.tensor([[0.9, 0.9, 0.9, 0.9]], requires_grad=True)
        v_last = torch.tensor([2.0])   # v_up = 2 + 2*0.5*h^eta < v_rated -> tight envelope
        sigma = torch.tensor([0.5])
        rho = torch.tensor([0.5])
        loss = pg.physgate_loss(pred, v_last, sigma, rho, THETA)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue((pred.grad != 0).all())

    def test_cut_region_penalty(self):
        pred = torch.tensor([[0.3, 0.3, 0.3, 0.3]])
        v_last = torch.tensor([30.0])  # v_lo > v_out for h>=1 -> cut region
        sigma = torch.tensor([0.1])
        rho = torch.tensor([0.0])
        loss, terms = pg.physgate_loss(pred, v_last, sigma, rho, THETA, return_terms=True)
        self.assertAlmostEqual(float(terms["cut"]), float((pred ** 2).mean()), places=6)
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
