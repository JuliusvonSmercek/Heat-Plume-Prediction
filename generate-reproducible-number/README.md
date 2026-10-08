# Generate reproducible paper numbers

Runs the six pLGCNN paper cases at two RWPT resolutions and writes metrics under `results/`.

Steady-state datasets: `steady-state-large` (formerly old/varyK) and `steady-state-small` (formerly correct).

## Layout

```
config/rwpt-high-res/   # 200k samples, 10k steps
config/rwpt-low-res/    # 40k samples, 4k steps
results/<variant>/<run>/{step1,step2,step3,data_prep,timings.json}
```

Configs under `config/` are committed; edit them directly if needed. Each run uses its own `data_prep` folder.

## Run

From the repo root (prefer **tmux**):

```bash
tmux new-session -d -s paper-baseline \
  'cd /path/to/Heat-Plume-Prediction && .venv/bin/python generate-reproducible-number/run_paper_baseline.py; echo DONE; exec bash'
```

Edit `GPUS = [1, 2, 3]` at the top of `run_paper_baseline.py` to choose devices.

| Flag | Effect |
|------|--------|
| `--dry-run` | Print planned stages only |
| `--only name1,name2` | Subset of runs |
| `--variants rwpt-low-res` | Subset of variants |

## Skip / resume

- **Missing** `results/<variant>/<run>/` → full train from scratch (step1 if pred) → step2 → step3 → tests
- **Folder exists** but missing `step3/measurements.yaml` or `timings.json` → test only (no-viz timed, then viz)
- Both artifacts present → skip
