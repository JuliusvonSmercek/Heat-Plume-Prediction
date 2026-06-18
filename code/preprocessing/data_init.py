from torch.utils.data import DataLoader, Subset
from pathlib import Path
import torch
import logging

from preprocessing.datasets.dataset import DataPoint, DataPointSequence, DatasetBasis
from preprocessing.datasets.dataset_cuts_jit import ChainedSubsetBatchSampler, SimulationDatasetCuts, SimulationDatasetCutsSequential
from utils.utils_args import load_time_steps, get_run_ids_from_prep
import utils.logging as log

def _label_has_midcell_above_threshold(label: torch.Tensor, threshold: float) -> bool:
    if label.dim() < 2:
        return False
    height = label.shape[-2]
    width = label.shape[-1]
    h_third = height // 3
    w_third = width // 3
    if h_third == 0 or w_third == 0:
        return False
    mid = label[..., h_third:2*h_third, w_third:2*w_third]
    return torch.any(mid > threshold).item()

def _select_overfit_datapoints(data_prep: Path, candidate_indices: list, max_points: int, threshold: float = 0.5) -> list:
    labels_dir = data_prep / "Labels"
    run_ids = get_run_ids_from_prep(labels_dir)
    if candidate_indices is None or len(candidate_indices) == 0:
        candidate_indices = list(range(len(run_ids)))

    selected = []
    for idx in candidate_indices:
        if idx < 0 or idx >= len(run_ids):
            log.warning(f"Overfit candidate index {idx} is out of range for {len(run_ids)} datapoints.")
            continue
        run_id = run_ids[idx]
        label_path = labels_dir / f"RUN_{run_id}.pt"
        label = torch.load(label_path, map_location="cpu")
        if _label_has_midcell_above_threshold(label, threshold):
            selected.append(idx)
            if len(selected) >= max_points:
                break
    return selected

def _select_overfit_cutout_indices(dataset, max_points: int, threshold: float = 0.5) -> list:
    selected = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if _label_has_midcell_above_threshold(label, threshold):
            selected.append(i)
            if len(selected) >= max_points:
                break
    return selected

def init_data(args:dict, datapoint_test: int, datapoint_validate: int, datapoint_train: list, log_path:str, tmp_bool_cutouts:bool=False, ):
    
    log.configure_logging(log_path=log_path, level=logging.INFO, clear_handlers=False)
    datasets = {}
    network = args.get("network", "unet").lower()
    
    if "time_steps_to_predict" in args and args["time_steps_to_predict"] is not None:
      time_steps_to_predict = args["time_steps_to_predict"]
    else:
      time_steps_to_predict = load_time_steps(Path(args["data_raw"],"RUN_"+str(args["order_data"][0]), "pflotran.h5"))
      
    overfit = args.get("overfit", False)
    

    if not overfit:
      # Check batchsize  
      if 1 == args["batchsize"]:
          print("Warning: Batch size is 1, do you want to overfit?")
          #exit(1)
          
      # Train dataset
      if not tmp_bool_cutouts: 
          if network in ["convlstm", "rnn", "lstm"]:
            dataset_train = DataPointSequence(
                args["data_prep"],
                i=datapoint_train,
                time_steps_to_predict=time_steps_to_predict,
                max_simulation_timestep=args.get("max_simulation_timestep", None),
            )
            log.info("dataset_train is DataPointSequence")
            dataset_train_full_dp = dataset_train
          else:
            dataset_train = DataPoint(args["data_prep"], i=datapoint_train)
      else:
          if network in ["convlstm", "rnn", "lstm"]:
              dataset_train = SimulationDatasetCutsSequential(args["data_prep"], time_steps_to_predict, args.get("max_simulation_timestep", None), skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=datapoint_train, log_path=args["destination"] / "training.log")
              dataset_train_full_dp = DataPointSequence(args["data_prep"], i = datapoint_train, time_steps_to_predict=time_steps_to_predict)
          else:
              dataset_train = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], ids=datapoint_train, box_size=args["len_box"])
     
      # Validation dataset
      if network in ["convlstm", "rnn", "lstm"]:
            dataset_val = DataPointSequence(
                args["data_prep"],
                i=datapoint_validate,
                time_steps_to_predict=time_steps_to_predict,
                max_simulation_timestep=args.get("max_simulation_timestep", None),
            )
      else:
          dataset_val = DataPoint(args["data_prep"], i=datapoint_validate)
    
      datasets["train_full_dp"] = dataset_train_full_dp
      datasets["train"] = dataset_train
      datasets["val"] = dataset_val
      
      # Test dataset
      try:
          if network in ["convlstm", "rnn", "lstm"]:
              dataset_test = DataPointSequence(
                  args["data_prep"],
                  i=datapoint_test,
                  time_steps_to_predict=time_steps_to_predict,
                  max_simulation_timestep=args.get("max_simulation_timestep", None),
              )
          else:
              dataset_test = DataPointSequence(args["data_prep"], i=datapoint_test)
          datasets["test"] = dataset_test
      except:
          print("Warning: No test dataset found.")
          pass
    else:
    
      overfit_count = int(args.get("overfit_on", 1))
      overfit_count = max(1, overfit_count)
      # Check batchsize
      if 1 != args["batchsize"]:
          print("Warning: Batch size is not 1, do you really want to overfit?")
          exit(1)
      print("Overfitting mode: Using only training data for all datasets.")
      # assert ...
      
      # Cutouts or no cutouts
      if not tmp_bool_cutouts: 
          if network in ["convlstm", "rnn", "lstm"]:
            dataset_train = DataPointSequence(
                args["data_prep"],
                i=datapoint_train,
                time_steps_to_predict=time_steps_to_predict,
                max_simulation_timestep=args.get("max_simulation_timestep", None),
            )
            log.info("dataset_train is DataPointSequence")
            
          else:
            dataset_train = DataPoint(args["data_prep"], i=datapoint_train)
      else:
          if network in ["convlstm", "rnn", "lstm"]:
              dataset_train_all = SimulationDatasetCutsSequential(args["data_prep"], time_steps_to_predict,args.get("max_simulation_timestep", None), skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=datapoint_train, log_path=args["destination"] / "training.log")
          else:
              dataset_train_all = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], ids=datapoint_train, box_size=args["len_box"])

          selected_cutouts = _select_overfit_cutout_indices(dataset_train_all, overfit_count, threshold=0.5)
          if len(selected_cutouts) == 0:
              log.warning("No cutouts matched overfit criteria. Falling back to first cutout.")
              selected_cutouts = [0]
          if len(selected_cutouts) < overfit_count:
              log.warning(f"Only found {len(selected_cutouts)} cutouts matching overfit criteria (requested {overfit_count}).")
          dataset_train = Subset(dataset_train_all, selected_cutouts)
          args["overfit_on"] = list(range(len(selected_cutouts)))
          log.info(f"Overfit cutouts selected (cutout indices): {selected_cutouts}")
              
      datasets["train"] = dataset_train
      datasets["val"] = dataset_train
      datasets["test"] = dataset_train

    dataset_train_meta = datasets["train"].dataset if hasattr(datasets["train"], "dataset") else datasets["train"]
    return dataset_train_meta.input_channels, dataset_train_meta.output_channels, datasets

def custom_collate_with_metadata(batch):
    """Custom collate function that handles (x, y, metadata) tuples"""
    if len(batch[0]) == 3:
        # Batch contains (x, y, metadata)
        x_list = [item[0] for item in batch]
        y_list = [item[1] for item in batch]
        metadata_list = [item[2] for item in batch]
        
        # Stack x and y normally
        x_stacked = torch.stack(x_list)
        y_stacked = torch.stack(y_list)
        
        # Return as tuple with metadata list
        return x_stacked, y_stacked, metadata_list
    else:
        # Batch contains (x, y) only - use default collate
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

def construct_dataloader(batch_size: int, dataset: DatasetBasis, shuffle=True) -> DataLoader:
    if dataset.__class__.__name__ in ["SimulationDatasetCutsSequential"]:
        sampler = ChainedSubsetBatchSampler(
            n_chains=len(dataset.chains),
            n_subsets=len(dataset.subsets),
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,  # new, before: 0
        pin_memory=True,  # new
        persistent_workers=True,  # new: makes progress approx. 10 times faster
        )
    else:
        return DataLoader(
            dataset,
            batch_size=min(len(dataset), batch_size),
            shuffle=shuffle,
            drop_last=True,
            num_workers=4,  # new, before: 0
            pin_memory=True,  # new
            persistent_workers=True,  # new: makes progress approx. 10 times faster
            prefetch_factor=8  # new,
        )
