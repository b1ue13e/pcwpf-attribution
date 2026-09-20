# Constraint rebuild v1

This is a new controlled experiment branch. It does not overwrite the historical results.

Local preparation from the workspace root:

```powershell
.\.venv\Scripts\python.exe 05_实验代码与脚本/constraint_rebuild/download_sources.py
.\.venv\Scripts\python.exe 05_实验代码与脚本/constraint_rebuild/prepare.py
.\.venv\Scripts\python.exe 05_实验代码与脚本/constraint_rebuild/benchmark.py --selftest
```

Training can be invoked with explicit `--data`, `--out`, `--site`, `--backbone`, `--seed`. The remote layout is `code/`, `data/`, `results/`, `logs/` under `/root/paper1/constraint_rebuild_v1`.

For the provided container runtime, use `LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. This fixes existing container library conflicts without modifying global packages. LightGBM 4.6.0 was installed with `--no-deps --target .../vendor` on 25329.

Run development training first. Copy the two hosts' validation tables to one result root. `evaluate.py --select --root RESULT_ROOT` requires all development candidates and writes the selection lock. Transfer this lock to both nodes before running the external site or any test evaluation. The external λ choices are transferred; they are not selected using external test errors.

`classical.py` evaluates persistence, lagged empirical curve, ridge and LightGBM. `dlinear_baseline.py` is an explicitly univariate DLinear comparator (added after the initial matrix; unchanged predeclared data splits, no test tuning). `summarize.py RESULT_ROOT` writes the aggregate CSV and exploratory paired time-block intervals.

`rebuild_corpus.py` retrieves records and creates blank human coding worksheets. `validate_coding.py` rejects unfilled evidence or absent independent coder IDs; passing IDs alone is not proof of human independence, which authors must substantiate. Do not use AI-generated worksheets as human-coded results.

Important limitations are recorded in the execution status report. In particular: measured net negatives are retained but the task is nonnegative generation; nameplate exceedances are descriptive; synthetic perturbations are not field calibration; SMARTEOLE is not a historically pristine holdout; normalized operation scores are not money. The archive field `origin_ns` denotes first target timestamp; decision issue time is 10 minutes earlier.

The full prediction archives remain on the two hosts. Local copies contain all neural/DLinear checkpoints, metrics, validation choices, raw-data provenance, and verified prediction samples.
