# Time-aware ML-Based Modeling of Heat Flow around Groundwater Heat Pumps

## Abstract

This project introduces a hybrid machine learning and physics-based approach to model heat plumes in groundwater systems. Building upon the **LGCNN** architecture, we extend the methodology to account for time-dependent (seasonal) factors and probabilistic streamline estimation. The method effectively combines explicit numerical solvers for global flow dependencies with a  recurrent neural network for sequence prediction.

## Methodology

Our approach utilizes a three-step pipeline to predict the final temperature distribution $T(\mathbf{x})$.

### 1. Local Velocity Estimation
A U-Net based CNN predicts the heterogeneous groundwater velocity field $\mathbf{v}(\mathbf{x})$
- inputs: Hydraulic pressure gradient $\nabla p$, Permeability field $k$, Heat pump locations $i$
- outputs: X-velocity, Y-velocity

### 2. Global Streamline Computation
* **Physical Solver Approximation:** A direct physics-based solver estimates a baseline temperature map based on material properties. Note: To align with PFLOTRAN ground truth, the injection rate in this solver is scaled by a factor of `0.25`.

| Channel | Name | Description |
|---------|------|-------------|
| `1` | Sum Position | Accumulated streamline density |
| `2` | Relative Uncertainty | Std/mean of ensemble |
| `3` | Time-Faded Position | Temporal decay weighting |
| `4` | Max Time-Faded | Peak influence over time |
| `5` | Seasonal Position | Multi-season aggregation |
| `6` | Max Seasonal | Peak seasonal influence |
| `7` | Direct Solver | GPU-computed thermal plume |

- inputs: X-velocity, Y-velocity
- outputs: 7 tensors

### 3. Temperature Regression
A second CNN aggregates the physical priors, derived from the 7 computed tensors, to predict the final temperature distribution $T(\mathbf{x})$.


## Usage

This branch serves soley the execution of Step3 with a Convolutional LSTM Network.

### Testing

Download the contents of sequence_prediction from Darus https://doi.org/10.18419/DARUS-5877.

Adjust the paths in the `config.yaml` to your dataset and model destination.


Run:
```bash
python main.py [config.yaml]
```

## Training

For training a new model set run_configuration.pipelin.step3 to "train".

## Important Note
Full integration of Steps 1 to 3 is ongoing and will be continued at a new branch (https://github.com/JuliusvonSmercek/Heat-Plume-Prediction/tree/probabilistic-lgcnn).


## Acknowledgements

This project is based on the research from [Heat-Plume-Prediction](https://github.com/JuliaPelzer/Heat-Plume-Prediction) and utilizes the [VampireMan](https://github.com/JuliaPelzer/VampireMan) data generation tool.
