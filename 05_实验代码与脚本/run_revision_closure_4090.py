#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "supplementary_evidence"
LOG_ROOT = ROOT / "logs"


@dataclass
class StepResult:
    name: str
    ok: bool
    return_code: int
    elapsed_sec: float
    log_file: str
    command: List[str]
    skipped: bool = False
    note: str = ""


def run_step(name: str, cmd: List[str], log_dir: Path, dry_run: bool = False) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if dry_run:
        log_file.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return StepResult(
            name=name,
            ok=True,
            return_code=0,
            elapsed_sec=0.0,
            log_file=str(log_file),
            command=cmd,
        )

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        f.write("[cmd] " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        proc.wait()
        rc = int(proc.returncode)

    return StepResult(
        name=name,
        ok=(rc == 0),
        return_code=rc,
        elapsed_sec=time.time() - t0,
        log_file=str(log_file),
        command=cmd,
    )


def maybe_skip(name: str, note: str) -> StepResult:
    return StepResult(
        name=name,
        ok=True,
        return_code=0,
        elapsed_sec=0.0,
        log_file="",
        command=[],
        skipped=True,
        note=note,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="One-click 4090 revision closure runner: fairness -> dispatch -> decision summary -> guardrail -> matrix."
    )
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seeds", type=str, default="42,123,2025")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--faults", type=str, default="gaussian,drift,quantization,missingness")
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--mapping-methods", type=str, default="affine,isotonic,quantile")
    ap.add_argument("--risk-q", type=float, default=0.9)
    ap.add_argument("--ramp-q", type=float, default=0.9)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--audit-tag", type=str, default="revision_closure_round1")
    ap.add_argument("--skip-fairness", action="store_true")
    ap.add_argument("--skip-dispatch", action="store_true")
    ap.add_argument("--skip-decision-summary", action="store_true")
    ap.add_argument("--skip-pi-moe-train", action="store_true")
    ap.add_argument("--skip-guardrail", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--force-pi-moe-train", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = LOG_ROOT / f"revision_closure_4090_{run_stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    steps: List[StepResult] = []

    def execute_or_stop(step: StepResult) -> None:
        steps.append(step)
        if (not step.ok) and (not args.continue_on_error):
            raise RuntimeError(f"Step failed: {step.name} (rc={step.return_code})")

    # 1) Fairness closure
    if args.skip_fairness:
        execute_or_stop(maybe_skip("fairness_closure", "Skipped by --skip-fairness"))
    else:
        cmd = [
            args.python,
            "run_fairness_closure_fullfault_4090.py",
            "--device",
            args.device,
            "--seeds",
            args.seeds,
            "--epochs",
            str(args.epochs),
            "--faults",
            args.faults,
            "--severity",
            str(args.severity),
        ]
        execute_or_stop(run_step("fairness_closure", cmd, log_dir, dry_run=args.dry_run))

    # 2) Dispatch closure
    if args.skip_dispatch:
        execute_or_stop(maybe_skip("dispatch_closure", "Skipped by --skip-dispatch"))
    else:
        cmd = [
            args.python,
            "run_dispatch_closure_mapping_decision_4090.py",
            "--device",
            args.device,
            "--mapping-methods",
            args.mapping_methods,
            "--risk-q",
            str(args.risk_q),
            "--ramp-q",
            str(args.ramp_q),
        ]
        execute_or_stop(run_step("dispatch_closure", cmd, log_dir, dry_run=args.dry_run))

    # 3) Decision-aware summary (table + mapping ranking bump figure)
    if args.skip_decision_summary:
        execute_or_stop(maybe_skip("decision_aware_summary", "Skipped by --skip-decision-summary"))
    else:
        cmd = [
            args.python,
            "build_decision_aware_dispatch_evidence.py",
            "--fairness-csv",
            str(out_dir / "fairness_closure_fullfault_hard_table.csv"),
            "--dispatch-l1-csv",
            str(out_dir / "dispatch_raw_vs_mapped_summary.csv"),
            "--matrix4-csv",
            str(out_dir / "main4_matrix4_dispatch_value.csv"),
            "--dispatch-agg-csv",
            str(out_dir / "dispatch_closure_mapping_decision_aggregate.csv"),
            "--rank-csv",
            str(out_dir / "dispatch_closure_mapping_decision_ranking_stability.csv"),
            "--out-summary-csv",
            str(out_dir / "decision_aware_robustness_summary.csv"),
            "--out-figure-prefix",
            str(out_dir / "fig_mapping_ranking_sensitivity"),
            "--out-meta",
            str(out_dir / "decision_aware_robustness_summary.meta.json"),
        ]
        execute_or_stop(run_step("decision_aware_summary", cmd, log_dir, dry_run=args.dry_run))

    # 4) PI-MoE full training (optional if checkpoint missing)
    pi_ckpt = out_dir / "pi_moe_full_checkpoint.pth"
    if args.skip_pi_moe_train:
        execute_or_stop(maybe_skip("pi_moe_full_training", "Skipped by --skip-pi-moe-train"))
    elif (not args.force_pi_moe_train) and pi_ckpt.exists():
        execute_or_stop(maybe_skip("pi_moe_full_training", f"Checkpoint exists: {pi_ckpt}"))
    else:
        cmd = [args.python, "run_pi_moe_full_training_evidence.py"]
        execute_or_stop(run_step("pi_moe_full_training", cmd, log_dir, dry_run=args.dry_run))

    # 5) Guardrail physics closure
    if args.skip_guardrail:
        execute_or_stop(maybe_skip("guardrail_physics_closure", "Skipped by --skip-guardrail"))
    else:
        cmd = [
            args.python,
            "run_guardrail_physics_closure_4090.py",
            "--device",
            args.device,
            "--faults",
            args.faults,
            "--severity",
            str(args.severity),
            "--ramp-q",
            str(args.ramp_q),
        ]
        execute_or_stop(run_step("guardrail_physics_closure", cmd, log_dir, dry_run=args.dry_run))

    # 6) Rebuild matrices
    if args.skip_build:
        execute_or_stop(maybe_skip("build_main4_matrices", "Skipped by --skip-build"))
    else:
        cmd = [
            args.python,
            "build_main4_matrices_from_evidence.py",
            "--out_dir",
            str(out_dir),
            "--audit_tag",
            args.audit_tag,
        ]
        execute_or_stop(run_step("build_main4_matrices", cmd, log_dir, dry_run=args.dry_run))

    summary = {
        "script": "run_revision_closure_4090.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_stamp": run_stamp,
        "log_dir": str(log_dir),
        "config": vars(args),
        "steps": [
            {
                "name": s.name,
                "ok": s.ok,
                "return_code": s.return_code,
                "elapsed_sec": s.elapsed_sec,
                "log_file": s.log_file,
                "command": s.command,
                "skipped": s.skipped,
                "note": s.note,
            }
            for s in steps
        ],
        "outputs": {
            "fairness_raw": str(out_dir / "fairness_closure_fullfault_raw.csv"),
            "dispatch_agg": str(out_dir / "dispatch_closure_mapping_decision_aggregate.csv"),
            "decision_aware_summary_csv": str(out_dir / "decision_aware_robustness_summary.csv"),
            "mapping_ranking_figure_pdf": str(out_dir / "fig_mapping_ranking_sensitivity.pdf"),
            "physics_closure": str(out_dir / "physics_closure.csv"),
            "main4_matrix2": str(out_dir / "main4_matrix2_perf_robust_transfer.csv"),
            "main4_matrix3": str(out_dir / "main4_matrix3_hardware_safety_joint.csv"),
            "main4_matrix4": str(out_dir / "main4_matrix4_dispatch_value.csv"),
        },
    }
    summary_path = log_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_count = sum(1 for s in steps if s.ok)
    fail_count = sum(1 for s in steps if not s.ok)
    print(f"[done] steps={len(steps)}, ok={ok_count}, failed={fail_count}")
    print(f"[done] summary: {summary_path}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
