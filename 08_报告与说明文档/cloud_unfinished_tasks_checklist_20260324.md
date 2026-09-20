# Cloud Unfinished Tasks Checklist

This checklist summarizes the items from the expanded dataset plan that are still not fully closed. It is organized by priority and by whether the current codebase is already cloud-ready.

## 1. What is already done

- `Wind (CN)` main paper chain is complete:
  - full search
  - Jump Gate evidence
  - dispatch rerun
  - FG-MoE / clip-only / safety chain
- `Wind (US)` secondary external evidence is complete:
  - cross-domain transfer table
  - bidirectional cross-site warm-start summaries
- `GEFCom` secondary boundary benchmark is complete.

These do **not** need to be rerun unless you want refreshed numbers.

## 2. Must-do if you want the dataset roadmap fully closed

### Task A. Wind-turbine dataset as an observed fault bank

- Priority: `P1`
- Status: `not finished`
- Cloud-ready: `partially`

### What is already done

- Data audit is complete.
- Dataset alias is ready:
  - `wind_turbine_primary`
- Prepared file exists:
  - `prepared_priority_datasets/wind_turbine_primary.csv`
- Only smoke-level training exists:
  - `results/turbine_alias_smoke/`
  - `results/smoke_sota_gate_fix*/main_TurbinePrimary/`

### What is still missing

- full training on turbine data
- observed fault segment extraction
- real fault replay results
- turbine-side safety diagnostics
- turbine-side `raw vs clip-only vs FG-MoE` comparison

### Recommended data split inside the `wind-turbine dataset`

- Better supervised candidates:
  - `11.csv`
  - `12.csv`
  - `13.csv`
  - `14.csv`
  - `17.csv`
- Better anomaly-mining candidates:
  - `15.csv`
  - `16.csv`
  - `18.csv`
  - `19.csv`
  - `20.csv`

### Important blocker

The current safety scripts are still hard-coded to `wind_final.csv` and are **not yet parameterized** for turbine aliases:

- `run_pi_moe_full_training_evidence.py`
- `run_clip_only_guardrail_control.py`
- `run_safety_deployment.py`
- `run_fairness_closure_fullfault_4090.py`

So this task is only **partially** cloud-ready right now.  
You can already run the turbine forecasting baseline on cloud, but the full turbine fault-bank and safety chain needs one more local code patch before upload.

### Cloud-ready part you can already run

```bash
python run_ultra_lsnt_sota_search_4090.py \
  --train-data wind_turbine_primary \
  --target power \
  --train-name TurbinePrimary \
  --device cuda \
  --presets balanced,wide,deep,aggressive \
  --seeds 42,52,62 \
  --epochs 80 \
  --pretrain-epochs 30 \
  --transfer-epochs 12 \
  --out-dir results/sota_search_turbine_primary_fullrun \
  --export-best-edge-ready
```

### Expected outputs

- `results/sota_search_turbine_primary_fullrun/leaderboard.csv`
- `results/sota_search_turbine_primary_fullrun/best_run.json`
- `results/sota_search_turbine_primary_fullrun/*/main_TurbinePrimary/test_metrics.csv`
- `results/sota_search_turbine_primary_fullrun/*/main_TurbinePrimary/stage_summary.json`

### Still needed after the local patch

- `results/observed_fault_bank_turbine/observed_fault_bank.csv`
- `results/observed_fault_bank_turbine/observed_fault_bank_summary.json`
- `results/.../fgmoe_clip_only_control_turbine.csv`
- `results/.../safety_deployment_turbine.csv`
- `results/.../fairness_closure_fullfault_turbine_*.csv`

## 3. Useful but lower-priority

### Task B. Inner Mongolia lightweight sanity-check

- Priority: `P2`
- Status: `not done`
- Cloud-ready: `no`

### Why it is not ready

- Audit says it is **not** compatible with the current `(L,H)=(96,24)` protocol.
- Source files are mainly Excel, not directly consumable by the current trainer.
- It is better suited to a different single-variable protocol than to the paper's main controller-side setup.

### Recommendation

For the current paper, skip this unless you want one extra appendix sanity point.

### If you insist on running it

You would first need:

1. preprocess one clean Inner Mongolia file into a regular CSV
2. choose a different window protocol such as daily-history / short-horizon forecasting
3. add a dedicated alias or preprocessing script before cloud upload

## 4. Optional appendix expansion

### Task C. One extra load benchmark from the three local load datasets

- Priority: `P2`
- Status: `not done`
- Cloud-ready: `no`

### Current state

- `GEFCom` is already done and in the paper.
- The three additional load datasets are still only audited, not integrated into training.

### Best candidate if you want exactly one more load dataset

- `short-term power-system load dataset`

Reason:

- among the three local load datasets, it has the cleanest audit footprint
- it is the least awkward candidate for a single auxiliary benchmark

### Why it is not cloud-ready yet

- source files are Excel-based
- the current training pipeline expects CSV after preprocessing
- target-column and time-grid normalization still need one explicit preprocessing step

### Recommendation

Only do this if you want a stronger appendix. It is not necessary for the current submission because `GEFCom` already fills the "not only one private wind site" role at the cross-domain level.

## 5. Already done enough; do not spend more time here

### Task D. Air Quality

- Priority: `skip`
- Status: `already done as secondary evidence`

It is already present in:

- `results/supplementary_evidence/multi_domain_transfer_matrix_all_models.csv`
- Appendix transfer table in the manuscript

Do **not** spend more cloud time here unless you intentionally want to reposition the paper as a broader corrupted spatiotemporal forecasting paper.

### Task E. More duplicate load datasets

- Priority: `skip`

Do not run all three local load datasets. One extra load benchmark is the upper limit that still keeps the paper focused.

## 6. Recommended execution order

If you want the most value for the least extra time, run in this order:

1. `TurbinePrimary` full search baseline on cloud
2. local patch to parameterize turbine fault-bank and safety scripts
3. turbine observed-fault replay + `raw / clip-only / FG-MoE`
4. only if time remains: one extra load benchmark
5. skip Inner Mongolia unless you specifically want a lightweight appendix point

## 7. Bottom line

The only **truly missing high-value item** from your original roadmap is the `wind-turbine dataset -> observed fault bank + turbine-side safety chain`.

Everything else is either:

- already done
- optional appendix expansion
- or not worth the cloud time for this paper right now
