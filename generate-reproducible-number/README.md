# Generate reproducible paper numbers

Runs the six pLGCNN paper cases at three RWPT budgets (`rwpt-high-res`, `rwpt-default`, `rwpt-low-res`) and writes metrics, plot specs and timing under `results/`.

Steady-state datasets: `steady-state-large` (formerly old/varyK) and `steady-state-small` (formerly correct).

## Layout

```
config/rwpt-high-res/       # 200k samples, max_steps_per_day 1.0 (≈10k steps), Courant 1
config/rwpt-default/        # 40k samples, max_steps_per_day 0.4 (≈4k steps), Courant 1
config/rwpt-low-res/        # 10k samples, max_steps_per_day 0.4 (≈4k steps), target_courant: 2
results/<variant>/<run>/{step1,step2,step3,data_prep,timing/,timings.yaml}
```

One YAML per run (`config/<variant>/<run>.yaml`). The variants differ only in the RWPT settings (`step2.general`). The runner sets the pipeline and device per call. Each run uses its own `data_prep` folder.

## Run

From the repo root (prefer **tmux**):

```bash
tmux new-session -d -s paper-baseline \
  'cd /path/to/Heat-Plume-Prediction && .venv/bin/python generate-reproducible-number/run_paper_baseline.py; echo DONE; exec bash'
```

| Flag | Effect |
|------|--------|
| `--dry-run` | Print planned stages only |
| `--only name1,name2` | Subset of runs |
| `--variants rwpt-default` | Subset of variants (also `rwpt-high-res`, `rwpt-low-res`) |
| `--gpus 1,2` | GPUs to use, one job per GPU at a time (default: `GPUS` in the script) |
| `--from-step2` | Reuse existing step1 models; rerun step2 → step3 |
| `--reuse-step1 rwpt-default` | New pred_vel runs of the other variants copy that variant's step1 model (and `timing/step1_train.yaml`) instead of training it; step1 does not depend on the RWPT settings. The source variant runs first; on several GPUs a target waits until the source model exists. `timings.yaml` records `step1_source`. |
| `--progress-bars` | Show tqdm bars (default: off, one log line per epoch) |
| `--no-notify` | Do not send Pushover notifications |

### Notifications

At the start, after every stage and after every run, the runner prints a one-line `NOTIFY …` message with the
estimated remaining time and finish time, e.g.
`default/seasonal-pred_vel step1_train ✓ 8h 20m · rest ~4d 6h (Fri 14:00)`.
If `.pushover` in the repo root (gitignored) holds a Pushover URL, the message is also sent there:

```
https://api.pushover.net/1/messages.json?token=<app token>&user=<user key>
```

The query parameters are posted as form fields together with `title` and `message`. The estimate uses the stage
durations measured so far in this invocation (same dataset; step2 also same variant) and rough defaults
(`baseline_progress.py`) until then, divided by the number of GPUs.

## Skip / resume

- **Missing** `results/<variant>/<run>/` → train from scratch: step1 train (if pred; copied with `--reuse-step1`) → step2 → step3 train
- **Folder exists** but missing `step3/measurements.yaml` or `timings.yaml` → test only (re-evaluate existing models)
- Both present → skip

`step*/measurements.yaml` holds the metrics of the best-epoch model on the full train/val/test fields (batch size 1).

## Visualization

The runner uses `visualize` / `visualize_epochs` as set in the config YAMLs. Plot data is always written as
`<name>.npy` + `<name>.yaml`.

- `visualize: true`: images are rendered right away, inside the timed stage (`timing/<stage>.yaml` still reports
  `total_without_visualization_s`).
- `visualize: false` (the committed configs): no images during the run; the step's specs are rendered in a separate,
  untimed call afterwards:

| After | Rendered |
|-------|----------|
| step1 train | `python -m code.plotting results/<variant>/<run>/step1` |
| step2 | `python -m code.plotting results/<variant>/<run>/step2` |
| step3 train | `python -m code.plotting results/<variant>/<run>/step3` |

Its duration goes to `render_s` in `timings.yaml`, not into the stage times; a rendering failure only prints a warning.
Restyle by editing a spec yaml and re-running the same command (e.g. with `--format pdf`).

The prior-vs-truth plots (`step2/RUN_<i>/thermal_prior_vs_true_{prior,true,error}`: three separate images in °C;
MAE/RMSE in the error yaml's `meta`) are rendered with the rest of step2.

## Timing

Every `python -m code` call writes a timing tree, moved to `results/<variant>/<run>/timing/<stage>.yaml`
(`step1_train`, `step2`, `step3_train`; `step1_test` / `step3_test` in test-only mode).
Training stages report `training_s`, `per_epoch_s` and `inference_{train,val,test}_per_sample_s` (pure forward pass)
separately from evaluation and plot writing.
`timings.yaml` keeps the per-stage wall clock including process start-up (`wall_s`), the rendering time (`render_s`) and the code version (`git: {commit, dirty}`; `dirty` = uncommitted changes to tracked files).

## Collect numbers

```bash
.venv/bin/python generate-reproducible-number/collect_paper_numbers.py   # → paper_numbers.yaml
```

`timing_overview` at the top lists per variant/run: training time and per-epoch time (step1/step3),
inference per test sample (step1/step3), step2 total / per datapoint / RWPT per datapoint / step1
inference inside step2, the same without plot writing (`*_without_visualization_s`), and the wall clock.
