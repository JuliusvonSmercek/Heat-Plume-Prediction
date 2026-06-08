# Training, Hyperparameter Search and Evaluation of LGCNN (and UNet $_{3dp}$)
This repository contains code for hyperparameter optimization and training of LGCNN and a vanilla UNet model using PyTorch. It also includes functionality for evaluating the trained model and reproducing results from the associated research paper.

Pre-trained models and raw datasets are part of the supplementary material of the paper. The preparation of the datasets for the first step or the full pipeline is done automatically, when `main.py` is run and the datasets is not yet prepared or not in the correct folder (see below). For Step 3, the data has to be prepared manually, see Step 2.

## Table of Contents
1. Some important files
2. Getting started
3. Training of an LGCNN (or another model)
4. Inference of an LGCNN
5. Postprocessing, incl. evaluation of metrics
6. Results

## Some important files in code/
- **`main.py`**: Contains code for training a single model (Step 1 and 3 of LGCNN, and vanilla UNet_3dp) or conducting a hyperparameter search.
- **`LGCNN_step2.py`**: Contains code for calculating streamlines (Step 2) and preparing inputs for Step 3.
- **`eval_metrics.py`**: Computes metrics for the train, test, validation, and scaling datasets using a given model.
- **`default_HPS_options.yaml`**: Defines the default search ranges for hyperparameter optimization. Needs to be copied to the hyperparameter search folder and renamed to `HPS_options.yaml` to be used, also used for setting the model parameters if no hyperparameter search is performed.
- **`runs/example`**: Contains example files for the command line arguments and hyperparameters. These files need to be copied to the folder where a new model is trained.

## Getting started
- install the requirements: via`pip install -r requirements.txt`
- download the raw datasets, optional: trained models
- make separate folders for the raw and prepared datasets and the models, e.g., `datasets`, `datasets_prep` and `runs` as in the example
- `cd` into the code folder

## Training of LGCNN (or another model)
Check the path variables

    PATH_DATA_RAW = Path("../datasets_raw") # TODO: change to your raw data directory
    PATH_DATA_PREP = Path("../datasets_prep") # TODO: change to your prepared data directory
    PATH_MODELS_DIR = Path("../runs") # TODO: change to your models/ results directory

in `main.py` and `LGCNN_step2.py`.

### Step 1 (predict v)
- make a new folder for the model in `runs/`, e.g., `NAME_OF_DIR_PREDICT_V`
- define a `command_line_argument.yaml` in `NAME_OF_DIR_PREDICT_V` acc. to the example in `example/command_line_arguments_v.yaml`, and `HPS_options.yaml` in `NAME_OF_DIR_PREDICT_V` acc. to the example in `example/HPS_options.yaml` (hyperparameters of your model or of the hyperparameter search)
- run `python main.py --destination NAME_OF_DIR_PREDICT_V`
- if you want to run a hyperparameter search, make sure to add the parameter `--hsearch True` and to give more than one value for the hyperparameters in `HPS_options.yaml`

###  Step 2 (calculate streamlines and prepare inputs for Step 3)
- open `LGCNN_step2.py` and set 
    - the path variables, 
    - the `dataset_name`, 
    - whether it is the synthetic-permeability(randomK)-dataset or the realK one (`randomK=True/False`), 
    - if the streamline calculation should be based on the predicted velocity field or the original velocity field (`based_on_pred=True/False`). If thsi is set to `True`, you also need a model-path in `model`
    - the method is by default set to `RK45` (RungaKutta 4(5)), but you can also set it to `RK23` (RungaKutta 2(3)) or `Radau` (implicit RK method)
- run `python LGCNN_step2.py`

### Step 3 (predict T)
- check that the required prepared dataset exists (download or run Step 2)
- define a `command_line_argument.yaml` in `NAME_OF_DIR_PREDICT_T` acc. to example in `example/command_line_arguments_T.yaml` and `HPS_options.yaml` (same as in Step 1)
- make sure that `data_pred`in `command_line_arguments.yaml` points to the prepared dataset from Step 2, i.e., that the path is correct, e.g., `datasets_prep/dataset_100hp_giant_real_fixP0_0025 inputs_ixydk+s_outer outputs_t`
- run `python main.py --destination NAME_OF_DIR_PREDICT_T`
- if you want to run a hyperparameter search, make sure to add the parameter `--hsearch True` 

## Inference of LGCNN
- make sure, that you have a trained model for Step 1 and Step 3, e.g., `NAME_OF_DIR_PREDICT_V` and `NAME_OF_DIR_PREDICT_T`
- prepare a new dataset for Step 2 based on the outputs of Step 1 and the predicted velocities (see parameters in `LGCNN_step2.py`: `based_on_pred=True` and `model=NAME_OF_DIR_PREDICT_V`)
- in `NAME_OF_DIR_PREDICT_T`, set `case: test` in `command_line_arguments.yaml` to only run inference and set `visu: True` to visualize the results
- run `python main.py --destination NAME_OF_DIR_PREDICT_T`

## Postprocessing
- set `visu: True` during training / inference in `command_line_arguments.yaml`
- set `case: test` in `command_line_arguments.yaml` to only run inference

### Evaluation
- run `eval_metrics.py` with a proper dataset and model paths to evaluate metrics as in the paper

## Results
This section contains the most relevant results from the paper and information to reproduce them. To generate them, download the raw datasets and trained models from the supplementary material of the paper. The hyperparameters used for the training of the models are in the `HPS_options.yaml` files in the models' folders.

The metrics were calculated with `code/eval_metrics.py` with respective dataset-name, model-name called in the file, which are listed below.

### Vanilla UNet $_{3dp}$ (pki $ \rightarrow $ T)
- on synthetic permeability fields
- based on 
    - dataset: dataset_giant_100hp_varyK inputs_pki outputs_t
    - model: UNet $_{3dp}$

| Model        | Case | Huber  | $L_\infty$ | MAE    | MSE    | PAT   | SSIM   |
|--------------|-------|--------|------------|--------|--------|--------|--------|
| UNet $_{3DP}$| train | 0.0020 | 2.4954     | 0.0404 | 0.0040 | 6.43   | 0.8281 |
|              | val   | 0.0269 | 5.2901     | 0.1365 | 0.0574 | 38.82  | 0.5717 |
|              | test  | 0.0235 | 4.8642     | 0.1314 | 0.0492 | 39.05  | 0.5794 |

### LGCNN Step 1 (pki $ \rightarrow $ v $_x$ v $_y$)
- on synthetic permeability fields
- based on 
    - dataset: dataset_giant_100hp_varyK inputs_pki outputs_xy, giant_double_100hp_v2 inputs_pki outputs_xy (scaling test)
    - model: LGCNN_Step1_randomK

| Model  | Output | Case     | Huber   | $L_\infty$ | MAE     | MSE      | SSIM    |
|--------|--------|----------|---------|------------|---------|----------|---------|
| Step 1 | $v_x$  | train    | 9.4732  | 132.6672   | 9.9620  | 171.0032 | 0.9972  |
|        | $v_y$  | train    | 11.4005 | 223.4601   | 11.8902 | 275.6190 | 0.9937  |
|        | $v_x$  | val      | 22.2241 | 343.9907   | 22.7179 | 1102.3721| 0.9905  |
|        | $v_y$  | val      | 26.6078 | 274.8036   | 27.1026 | 1524.5099| 0.9841  |
|        | $v_x$  | test     | 21.8237 | 190.8046   | 22.3178 | 972.5668 | 0.9911  |
|        | $v_y$  | test     | 32.2488 | 256.2519   | 32.7444 | 2031.3357| 0.9812  |
|        | $v_x$  | scaling  | 24.4314 | 294.0457   | 24.9261 | 1204.1154| 0.9911  |
|        | $v_y$  | scaling  | 25.7847 | 367.6891   | 26.2795 | 1463.8218| 0.9820  |

### LGCNN Step 3 (kiv $_x$ v $_y$ s s $_o$ $ \rightarrow $ T)
- based on 
    - dataset: dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t, giant_double_100hp_v2 inputs_ixydk+s_outer outputs_t (scaling test)
    - model: LGCNN_Step3_randomK

| Model  | Case     | Huber    | $L_\infty$ | MAE     | MSE      | PAT   | SSIM   |
|--------|----------|----------|------------|---------|----------|-------|--------|
| Step 3 | train    | 1.87e-5  | 2.2536     | 0.0014  | 3.95e-5  | 0.03  | 0.9997 |
|        | val      | 0.0027   | 2.8857     | 0.0369  | 0.0054   | 8.61  | 0.9283 |
|        | test     | 0.0021   | 2.8990     | 0.0347  | 0.0041   | 7.54  | 0.9304 |
|        | scaling  | 0.0007   | 3.0250     | 0.0168  | 0.0014   | 2.05  | 0.9510 |

### LGCNN full pipeline (pki $ \rightarrow $ T)
- based on 
    - dataset: dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t prep_with_LGCNN_step1_randK, giant_double_100hp_v2 inputs_ixydk+s_outer outputs_t prep_with_LGCNN_step1_randK (scaling test)
    - model: LGCNN_Step3_randomK

| Model  | Case     | Huber    | $L_\infty$ | MAE     | MSE      | PAT   | SSIM   |
|--------|----------|----------|------------|---------|----------|-------|--------|
| Pipeline  | train    | 0.0121  | 5.1006     | 0.0642  | 0.0272  | 18.17 | 0.8700 |
|           | val      | 0.0188  | 4.2264     | 0.0967  | 0.0411  | 28.96 | 0.7625 |
|           | test     | 0.0147  | 4.2120     | 0.0905  | 0.0307  | 28.92 | 0.7637 |
|           | scaling  | 0.0065  | 4.9366     | 0.0413  | 0.0141  | 10.87 | 0.8654 |


### LGCNN Domain Transfer to real permeability fields
- based on 
    - raw datasets: dataset_100hp_giant_real_fixP0_0025, dataset_100hp_scaling_real_fixP0_0025 (scaling test)
    - prepared datasets: see above
    - model: LGCNN_Step1_realK, LGCNN_Step3_realK

| Model  | Output | Case     | Huber   | $L_\infty$ | MAE     | MSE       | SSIM    |
|--------|--------|----------|---------|------------|---------|-----------|---------|
| Step 1 | $v_x$  | train    | 13.6890 | 122.7981   | 14.1819 | 340.6713  | 0.9973  |
|        | $v_y$  | train    | 9.0680  | 71.2576    | 9.5619  | 126.7222  | 0.9991  |
|        | $v_x$  | val      | 14.9187 | 106.8607   | 15.4095 | 380.3762  | 0.9939  |
|        | $v_y$  | val      | 10.1675 | 74.5570    | 10.6605 | 148.2475  | 0.9993  |
|        | $v_x$  | scaling  | 109.5079| 459.1620   | 110.0078| 13570.0000| 0.9462  |
|        | $v_y$  | scaling  | 17.1118 | 240.9406   | 17.6051 | 616.5186  | 0.9965  |

| Model  | Case     | Huber   | $L_\infty$ | MAE     | MSE     | PAT   | SSIM   |
|--------|----------|---------|------------|---------|---------|-------|--------|
| Step 3  | train    | 0.0002  | 0.7704     | 0.0139  | 0.0005  | 0.43  | 2.9107 |
|         | val      | 0.0005  | 0.8222     | 0.0175  | 0.0010  | 2.18  | 0.9672 |
|         | scaling  | 0.0004  | 0.8052     | 0.0189  | 0.0008  | 0.92  | 0.9497 |
| Pipeline  | train    | 0.0049  | 2.5437     | 0.0534  | 0.0100  | 17.58 | 2.4923 |
|           | val      | 0.0137  | 2.3194     | 0.0841  | 0.0275  | 27.79 | 0.7510 |
|           | scaling  | 0.0022  | 2.0511     | 0.0394  | 0.0044  | 10.02 | 0.8708 |
