from __future__ import annotations

import subprocess
import sys


def _run(script: str) -> None:
    print(f"[run] {script}")
    subprocess.check_call([sys.executable, script])


def main() -> None:
    scripts = [
        "plot_paper_figs.py",
        "plot_arch.py",
        "plot_multi_domain.py",
        "plot_dlinear_victory.py",
        "plot_universal_robustness_from_log.py",
        "plot_gbdt_battle_from_log.py",
        "plot_expert_heatmap.py",
        "run_expert_physics.py",
        "battle_economics.py",
        "ensure_paper_pdfs.py",
    ]
    for s in scripts:
        _run(s)
    print("Done. Outputs are in ./figure (PNG+PDF).")


if __name__ == "__main__":
    main()
