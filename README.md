# Time-aware ML-Based Modeling of Heat Flow around Groundwater Heat Pumps

## Abstract

This project introduces a hybrid machine learning and physics-based approach to model heat plumes in groundwater systems. Building upon the **LGCNN** architecture, we extend the methodology to account for time-dependent (seasonal) factors and probabilistic streamline estimation. The method effectively combines Convolutional Neural Networks (CNNs) for local heterogeneity with explicit numerical solvers for global flow dependencies.


## Methodology

The pipeline has three ordered steps that together predict the temperature field $T(\mathbf{x})$. Configured via YAML, a single run can train or evaluate any subset of these steps.

### 1. Local Velocity Estimation
A U-Net predicts the heterogeneous groundwater velocity field $\mathbf{v}(\mathbf{x})$
- inputs: Hydraulic pressure gradient $\nabla p$, Permeability field $k$, Heat pump locations $i$
- outputs: X-velocity, Y-velocity

### 2. Global Streamline Computation
RWPT thermal prior (not a full PFLOTRAN run). The injection rate in the RWPT solver is scaled by `0.25` to align with PFLOTRAN ground truth.

| Channel | Name | Description |
|---------|------|-------------|
| `7` | RWPT | GPU-computed thermal plume |

- inputs: X-velocity, Y-velocity, heat-pump positions
- outputs: 1 tensor (channel `7`)

### 3. Temperature Regression
A second CNN maps physical priors (and selected raw channels) to $T(\mathbf{x})$.


## Installation

### Prerequisites
* Python >=3.12
* NVIDIA GPU + CUDA recommended for step 2. CPU works for install and light runs; set `run_configuration.device` to `cpu` in the YAML if needed.

### Setup
```sh
# HPC (optional): module load python/3.12.9
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dataset Setup
~76 GB from DaRUS; expect hours depending on network. If the dataset is still private, put the API key in `.darus_apikey` at the **repo root**. Submodule deps are only in `darus_data_download/requirements.txt`. Re-running `get_data.py` will continue the download.

```sh
git submodule update --init --recursive --force
cd darus_data_download
pip install -r requirements.txt
python scripts/get_data.py
unzip scripts/data/datasets-100-heat-pumps-synthetic-permeability-fields-simulation-raw-3-1-data-points/random_perm_3dp.zip
mkdir -p ../datasets ../results
mv dataset_giant_100hp_varyK ../datasets/
mv scripts/data/student-thesis-on-transient-adaptation-of-lgcnn/datasets/* ../datasets/
mv scripts/data/student-thesis-on-transient-adaptation-of-lgcnn/probabilistic-lgcnn/results/* ../results/
cd ..
```

Where the configs look for data (`paths:` block of every YAML, relative to the repo root):

| Key | Default | Content |
|-----|---------|---------|
| `paths.datasets_raw` | `../../datasets` | raw datasets, one folder per `run_configuration.dataset` (e.g. `seasonal`, `steady-state-large`, `steady-state-small`) |
| `paths.datasets_prep` | `./datasets_prep` | preprocessed tensors (created automatically) |
| `paths.results` | `./results` | models, metrics, plots, timing |

Either move the downloaded datasets to `../../datasets/<name>` or point `paths.datasets_raw` at them; the folder name must match `run_configuration.dataset`.


## Usage

### Running a pipeline

Everything is driven by one YAML file and one entry point (run from the repo root):

```bash
python -m code settings/seasonal.yaml        # add -v for debug logging
```

`run_configuration.pipeline` is an ordered list of actions; each runs in the listed order:

| Action | Modes | What it does |
|--------|-------|--------------|
| `clean` | `data_prep`, `results-step1`, `results-step2`, `results-step3` | delete preprocessed data or a step's results |
| `step1` | `train`, `test`, `finetune`, `hopt` | velocity U-Net |
| `step2` | `run` | RWPT thermal prior for every datapoint (uses the step1 model if `use_velocity_model: true`, else the true velocities) |
| `step3` | `train`, `test`, `finetune`, `hopt` | temperature U-Net |

`test` and `finetune` load `results/<run_name>/<step>/model.pt`; `hopt` runs an Optuna search over `hopt_parameters`.

```yaml
run_configuration:
  run_name: seasonal            # → results/seasonal/
  dataset: seasonal             # → <datasets_raw>/seasonal/
  device: cuda:0                # or cpu
  use_velocity_model: true      # step2 uses step1 predictions
  pipeline:
    - clean: data_prep
    - step1: train
    - step2: run
    - step3: train
```

Every option is documented in [`settings/template.yaml`](settings/template.yaml).

Progress bars (tqdm) can be tuned with environment variables, which is useful in tmux or when piping into a log file. Training also logs one line per epoch (losses, learning rate, duration), so nothing is lost without bars:

```bash
TQDM_DISABLE=1 python -m code settings/seasonal.yaml     # no bars
TQDM_NCOLS=80 python -m code settings/seasonal.yaml      # fixed width instead of following the terminal
TQDM_MININTERVAL=60 python -m code settings/seasonal.yaml  # redraw at most once a minute
```

Note: tqdm treats *any* non-empty `TQDM_DISABLE` (also `0` or `false`) as "disable"; unset it to get the bars back.

### Provided configs

| Config | Dataset |
|--------|---------|
| `settings/seasonal.yaml`, `settings/seasonal_hopt.yaml` | legacy PFLOTRAN `seasonal` (ambient 10.6 °C) |
| `settings/steady-state-large.yaml`, `settings/steady-state-small.yaml` | steady-state `steady-state-large` / `steady-state-small` |
| `settings/step-1-single-heat-plume.yaml`, `settings/full_dataset.yaml` | DaRUS-converted datasets (ambient 10.0 °C) |
| `settings/test-*.yaml` | tiny smoke versions (1 epoch) used by the tests |
| `settings/physical_parameters-*.yaml` | shared physical parameters, pulled in via `include:` |

### Outputs

```
results/<run_name>/
  config.yaml              # copy of the config of the last call
  config_resolved.yaml     # the same with includes and defaults resolved
  log_<timestamp>.txt
  timing.yaml              # timing of the last call (see Timing)
  step1/ step3/
    model.pt, info.yaml, model_structure.txt
    measurements.yaml      # MSE/MAE/Huber/L∞/PAT/SSIM on train/val/test (+ training time)
    events.out.tfevents.*  # TensorBoard: tensorboard --logdir results/<run_name>
    val_0_input_*, val_0_output_*_out / *_error, test_*   # plot specs (+ .png if visualize)
  step2/
    RUN_<i>/rwpt/L<level>_*                               # RWPT stage plots (off; enable via viz_dir in streamline_main.py)
    RUN_<i>-Streamline-TemperatureApproximation.*         # thermal prior plot spec (+ .png if visualize)
    RUN_<i>/thermal_prior_vs_true_{prior,true}.*          # prior / truth in °C: plot specs (+ .png if visualize)
    RUN_<i>/thermal_prior_vs_true_error.yaml (+ .png)     # |prior − true| computed from the two npy; MAE / RMSE / max under meta
```

### Visualization

Every plot is written as plot data: `<name>.npy` (raw values) + `<name>.yaml` (how to draw it).
Each step's `visualize` setting decides whether the images are rendered as well:

| Value | Output |
|-------|--------|
| `false` | plot specs only (`.npy` + `.yaml`) |
| `true` | plot specs + rendered images (`.png`) |

```yaml
general_configuration:
  step1:
    general:
      visualize: true
      visualize_epochs: false   # extra plots after improving epochs (rendered if visualize)
  step2:
    general:                    # step2 settings: visualize + RWPT parameters (samples, max_steps_per_day, ...)
      visualize: false
```

Plotting lives in the self-contained package [`code/plotting/`](code/plotting) (numpy, yaml, matplotlib, scipy only — the folder can be copied elsewhere). Render or re-render specs at any time, e.g. after `visualize: false`:

```bash
python -m code.plotting results/<run_name>/step2                 # all specs in a folder (recursive)
python -m code.plotting results/<run_name>/step3/val_0_output_Temperature\ [C]_out.yaml --format pdf --dpi 300
```

To restyle a figure, edit its yaml and render again. A spec looks like this (shortened):

```yaml
data: RUN_0-Streamline-TemperatureApproximation.npy
kind: panel
colormap: {name: jp_temperature_upperlinear, vmin: 10.6, vmax: 15.6, colorbar: true, ...}
axes: {show: true, show_labels: false, xlabel: y [m], ylabel: x [m], extent: [0, 320, 1280, 0], ...}
cutout: {enabled: false, rows: [100, 400], cols: [100, 400]}
display: {interpolation: nearest, marker_dilation_px: 0}
output: {format: png, dpi: 160, figsize: [8.4, 6.5], bbox_inches: tight}
```

Multi-panel figures store one npy + yaml per panel plus `<stem>.figure.yaml` (layout, title, panel list). A panel can also be derived from other panels' data instead of storing its own npy, e.g. the prior-vs-truth error panel:

```yaml
data: {op: abs_diff, of: [thermal_prior_vs_true_prior.npy, thermal_prior_vs_true_true.npy]}   # ops: abs_diff, diff
```

Unknown keys are rejected, so typos fail loudly.

### Timing

Every `python -m code` call writes `results/<run_name>/timing.yaml` (also when the run fails or is interrupted). It starts with the run context (host, GPU, pipeline) and a `summary` per step, followed by the full section tree:

```yaml
summary:
  step1_train:
    total_s: 10.60
    training_s: 5.04
    epochs_run: 1
    per_epoch_s: 5.04
    inference_test_per_sample_s: 0.0127     # pure model forward, batch size 1
    visualization_s: 0.0
    total_without_visualization_s: 10.60
  step2_run:
    total_s: 10.09
    datapoints: 26
    per_datapoint_s: 0.209
    step1_inference_per_datapoint_s: 0.0203
    rwpt_per_datapoint_s: 0.0589
    ...
step1_train:
  total_s: 10.60
  preprocessing: {total_s: 3.95}
  training:
    epoch_train_pass: {total_s: 2.80}
    ...
```

Sections repeated in a loop (epochs, datapoints, samples, RWPT iterations) are aggregated into `calls`, `mean_s`, `min_s` and `max_s`; `other_s` is time not covered by a sub-section. On GPU, section boundaries synchronize CUDA so asynchronous kernels are charged to the right section. The file is overwritten by the next call, so copy it if you run several configs into the same `run_name`.

### Reproducing the paper numbers

[`generate-reproducible-number/`](generate-reproducible-number/README.md) trains and evaluates all paper cases at three RWPT budgets on several GPUs, keeps each stage's `timing.yaml`, and collects metrics plus a timing overview into `paper_numbers.yaml`.

### Tests

```sh
python -m unittest discover -s tests -t tests
```

Covers config parsing, plot specs and rendering, the timer, step2 APIs / tiny RWPT, U-Net train+infer, hyperparameter search, and minimal seasonal / steady-state pipelines via `settings/test-seasonal.yaml` and `settings/test-steady-state-large.yaml`. The seasonal smoke test runs step1→2→3 on CPU; the steady-state smoke test runs step1 only on CPU (full step2/3 needs CUDA). Pipeline tests are skipped if the raw datasets are missing under `datasets_raw`; they take several minutes.

### Data conversion

Upstream data come from [VampireMan](https://github.com/JuliaPelzer/VampireMan) (PFLOTRAN). To convert a VampireMan dataset into this pipeline's layout:

```bash
python code/converter.py [path_to_vampireman_dataset] [path_to_save_dataset]
```

DaRUS NPZ datasets (`step-1-single-heat-plume`, `full_dataset`) are converted with `code/convert_npz_dataset.py` (see its docstring).


## Acknowledgements

This project is based on the research from [Heat-Plume-Prediction](https://github.com/JuliaPelzer/Heat-Plume-Prediction) and utilizes the [VampireMan](https://github.com/JuliaPelzer/VampireMan) data generation tool.
