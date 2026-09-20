#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
STATE = ROOT / ".agent_state"
MANIFEST = ROOT / ".agents" / "verification_manifest.json"

TASK_RE = re.compile(r"^- \[(?P<mark>[ x!\-])\]\s+(?P<id>T[A-Za-z0-9_-]+)\s+(?P<state>TODO|DOING|VERIFIED|BLOCKED)\b", re.M)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_command(cmd: str, cwd: Path):
    cp = subprocess.run(cmd, cwd=cwd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return cp.returncode, cp.stdout


def check_canonical_numerical_consistency(root: Path, failures: list[str]):
    """Cross-artifact canonical numerical consistency audit asserting exact 5-seed provenance."""
    # 1. Check main.tex abstract
    main_tex = root / "01_论文投稿材料" / "active_manuscript" / "main.tex"
    if main_tex.exists():
        t = main_tex.read_text(encoding="utf-8")
        if "1.28\\%" not in t:
            failures.append("main.tex Abstract lacks canonical Penmanshiel reduction 1.28\\%")
        if "61.2\\%" not in t:
            failures.append("main.tex Abstract lacks canonical scrambled prior surge 61.2\\%")
        if "and remain consistent" not in t:
            failures.append("main.tex Abstract grammar check failed: expected 'gains ... and remain consistent'")
        if "remains consistent with non-specific" in t:
            failures.append("main.tex Abstract grammar error persists: 'remains consistent with non-specific'")
        if "1.22\\%" in t:
            failures.append("main.tex Abstract contains stale Penmanshiel reduction 1.22\\%")

    # 2. Check 08_discussion.tex
    disc_tex = root / "01_论文投稿材料" / "active_manuscript" / "sections" / "08_discussion.tex"
    if disc_tex.exists():
        t = disc_tex.read_text(encoding="utf-8")
        if "1.28\\%$ on Penmanshiel" not in t:
            failures.append("08_discussion.tex lacks canonical '1.28\\%$ on Penmanshiel'")
        if "+61.2\\%" not in t:
            failures.append("08_discussion.tex lacks canonical scrambled surge '+61.2\\%'")
        if "1.22\\%$ on Penmanshiel" in t:
            failures.append("08_discussion.tex contains stale '1.22\\%$ on Penmanshiel'")
        if "+68.1\\%" in t:
            failures.append("08_discussion.tex contains stale scrambled surge '+68.1\\%'")

    # 3. Check Table 4 zero-bid diagnostic
    t4_tex = root / "01_论文投稿材料" / "active_manuscript" / "tables" / "table4_zero_bid_diagnostic.tex"
    if t4_tex.exists():
        t = t4_tex.read_text(encoding="utf-8")
        for val in ["0.08804", "0.08913", "0.09852", "0.09726"]:
            if val not in t:
                failures.append(f"table4_zero_bid_diagnostic.tex lacks canonical value {val}")
        for stale in ["0.08812", "0.08924", "0.09881", "0.09760"]:
            if stale in t:
                failures.append(f"table4_zero_bid_diagnostic.tex contains stale value {stale}")

    # 4. Check 06_boundaries.tex
    bnd_tex = root / "01_论文投稿材料" / "active_manuscript" / "sections" / "06_boundaries.tex"
    if bnd_tex.exists():
        t = bnd_tex.read_text(encoding="utf-8")
        if "0.09852" not in t:
            failures.append("06_boundaries.tex lacks canonical Penmanshiel reference baseline 0.09852")
        for stale in ["0.08924", "0.09022", "0.09257", "0.08876", "0.09043", "0.10036", "0.10473", "0.09881"]:
            if stale in t:
                failures.append(f"06_boundaries.tex contains stale value {stale}")

    # 5. Check 07_alignment_forensics.tex
    align_tex = root / "01_论文投稿材料" / "active_manuscript" / "sections" / "07_alignment_forensics.tex"
    if align_tex.exists():
        t = align_tex.read_text(encoding="utf-8")
        for val in ["0.08804", "0.08923", "5,764.03", "0.08913", "0.09028", "5,941.22"]:
            if val not in t:
                failures.append(f"07_alignment_forensics.tex lacks canonical value {val}")
        for stale in ["0.08812", "0.08910", "5,831.79", "0.08924", "0.09042", "6,006.09"]:
            if stale in t:
                failures.append(f"07_alignment_forensics.tex contains stale value {stale}")

    # 6. Check Figure 4 plot script
    p4_py = root / "05_实验代码与脚本" / "plot_learning_and_corruption.py"
    if p4_py.exists():
        t = p4_py.read_text(encoding="utf-8")
        if "ax.axhline(0.09852" not in t or "ax.axhline(0.08804" not in t:
            failures.append("plot_learning_and_corruption.py lacks canonical axhline baseline 0.09852 / 0.08804")
        if "ax.axhline(0.09881" in t or "ax.axhline(0.08812" in t:
            failures.append("plot_learning_and_corruption.py contains stale axhline baseline 0.09881 / 0.08812")

    # 7. Check Figure 2 generation script
    fig2_py = root / "05_实验代码与脚本" / "generate_figure2_causal_ablation.py"
    if not fig2_py.exists():
        failures.append("Missing Figure 2 reproduction generator: generate_figure2_causal_ablation.py")


def main() -> int:
    p = argparse.ArgumentParser(description="Fail closed when a long task lacks completion evidence.")
    p.add_argument("--no-run-commands", action="store_true", help="validate state only")
    args = p.parse_args()

    failures = []
    notes = []
    required_state = ["TASK_LEDGER.md", "EVIDENCE_LEDGER.md", "DECISION_LOG.md", "RUN_STATE.json"]
    for name in required_state:
        if not (STATE / name).exists():
            failures.append(f"missing state file: .agent_state/{name}")

    if failures:
        print("VERIFICATION_GATE: FAIL")
        for x in failures:
            print(f"- {x}")
        return 2

    manifest = load_json(MANIFEST, {})
    task_text = (STATE / "TASK_LEDGER.md").read_text(encoding="utf-8-sig")
    evidence = (STATE / "EVIDENCE_LEDGER.md").read_text(encoding="utf-8-sig")
    tasks = list(TASK_RE.finditer(task_text))
    if not tasks:
        failures.append("no parseable tasks found in TASK_LEDGER.md")

    if manifest.get("require_all_nonblocked_tasks_verified", True):
        for m in tasks:
            state = m.group("state")
            tid = m.group("id")
            if state in {"TODO", "DOING"}:
                failures.append(f"open task: {tid} is {state}")

    if manifest.get("require_evidence_for_verified_tasks", True):
        for m in tasks:
            if m.group("state") == "VERIFIED":
                tid = m.group("id")
                if f"Task-ID: {tid}" not in evidence:
                    failures.append(f"verified task lacks evidence entry: {tid}")

    if manifest.get("require_independent_review", True):
        review = STATE / "REVIEW_REPORT.md"
        if not review.exists():
            failures.append("independent review missing: .agent_state/REVIEW_REPORT.md")
        else:
            rt = review.read_text(encoding="utf-8-sig")
            if not re.search(r"\b(PASS|PASS_WITH_BLOCKERS)\b", rt):
                failures.append("independent review has no PASS/PASS_WITH_BLOCKERS verdict")

    if manifest.get("require_verification_report", False):
        vr = STATE / "VERIFICATION_REPORT.md"
        if not vr.exists() or not re.search(r"\bPASS\b", vr.read_text(encoding="utf-8-sig")):
            failures.append("required verification report missing or not PASS")

    for rel in manifest.get("required_files", []):
        if not (ROOT / rel).exists():
            failures.append(f"required file missing: {rel}")

    for item in manifest.get("forbidden_patterns", []):
        pattern = item.get("pattern", "")
        paths = item.get("paths", ["."])
        if not pattern:
            continue
        rx = re.compile(pattern)
        for rel in paths:
            pth = ROOT / rel
            candidates = [pth] if pth.is_file() else [p for p in pth.rglob("*") if p.is_file() and ".git" not in p.parts and ".agent_state" not in p.parts]
            for fp in candidates:
                try:
                    txt = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if rx.search(txt):
                    failures.append(f"forbidden pattern {pattern!r} found in {fp.relative_to(ROOT)}")
                    break

    # Automated Cross-Artifact Canonical Numerical Consistency Check
    check_canonical_numerical_consistency(ROOT, failures)

    command_results = []
    if not args.no_run_commands:
        for spec in manifest.get("commands", []):
            if not spec.get("enabled", True):
                continue
            name = spec.get("name", spec.get("command", "command"))
            cmd = spec.get("command", "")
            required = spec.get("required", True)
            cwd = ROOT / spec.get("cwd", ".")
            if not cmd:
                continue
            code, out = run_command(cmd, cwd)
            command_results.append((name, cmd, code, out[-5000:]))
            if required and code != 0:
                failures.append(f"verification command failed: {name} (exit {code})")

    report = STATE / "GATE_REPORT.md"
    lines = [f"# Verification Gate Report\n", f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"]
    for name, cmd, code, out in command_results:
        lines += [f"## Command: {name}", f"`{cmd}`", f"Exit: {code}", "```", out, "```", ""]
    if failures:
        lines += ["## Verdict", "FAIL", "", "## Failures"] + [f"- {x}" for x in failures]
    else:
        lines += ["## Verdict", "PASS"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        print("VERIFICATION_GATE: FAIL")
        for x in failures:
            print(f"- {x}")
        return 1

    print("VERIFICATION_GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
