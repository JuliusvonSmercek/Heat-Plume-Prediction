import gc
import logging
from pathlib import Path
from pyexpat import model
import time
from dataclasses import dataclass
from torch import device, manual_seed
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
import torch
from torch.nn import Module, modules, MSELoss, HuberLoss, L1Loss
from torch.optim import Adam, Optimizer, LBFGS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset
import utils.logging as log

from postprocessing.visualization import reverse_norm_one_dp_outputs, visualize_inputs, visualize_outputs
from processing.networks.model import weights_init as model_weights_init
from processing.networks.convLSTM import weights_init as convlstm_weights_init
from processing.networks.convLSTM import Seq2Seq
from utils.utils_args import save_yaml
from processing.loss_fcts import SSIMLoss, FocalMSE, PATLoss, MaxNormLoss


@dataclass
class Solver(object):
    model: Module
    train_dataset: Dataset
    val_dataset: Dataset
    loss_func: modules.loss._Loss = MSELoss()
    learning_rate: float = 1e-4
    batchsize: int = 32
    opt: Optimizer = Adam
    optimizer_switch: bool = False
    finetune: bool = False
    best_model_params: dict = None
    metrics: dict = None
    global_step: int = 0

    def __post_init__(self):
        self.opt = self.opt(self.model.parameters(), self.learning_rate, weight_decay=1e-4)
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}

        if not self.finetune:
            if isinstance(self.model, Seq2Seq):
                self.model.apply(convlstm_weights_init)
                nn.init.xavier_uniform_(self.model.final_conv.weight)
                nn.init.constant_(self.model.final_conv.bias, 0.0)
            else:
                self.model.apply(model_weights_init)
        self.metrics: dict = {"Huber": HuberLoss(), "MSE": MSELoss(), "MAE": L1Loss(), "MaxLoss": MaxNormLoss(), "SSIM": SSIMLoss(), "PAT": PATLoss(0.1)}

    def save_epoch_metrics_yaml(
        self,
        destination: Path,
        filename: str,
        epoch: int,
        train_epoch_loss: float,
        val_epoch_loss: float,
        other_losses_train: dict,
        other_losses_val: dict,
        no_params: int = None,
        max_epochs: int = None,
        training_time: float = None,
        checkpoint_type: str = None,
    ):
        """Save a lightweight metrics snapshot for a specific checkpoint epoch."""
        metrics = {
            "current_epoch": epoch,
            "train": dict(other_losses_train),
            "val": dict(other_losses_val),
        }
        metrics["train"]["train loss"] = train_epoch_loss
        metrics["val"]["val loss"] = val_epoch_loss

        if checkpoint_type is not None:
            metrics["checkpoint_type"] = checkpoint_type
        if no_params is not None:
            metrics["no_params"] = no_params
        if max_epochs is not None:
            metrics["max_epochs"] = max_epochs
        if training_time is not None:
            metrics["training_time [s]"] = training_time

        save_yaml(metrics, destination / filename)


    def train(self, train_dataloader, val_dataloader, args: dict):
        self.args = args
        
        # initialize logging
        log_path = args["destination"] / "training.log"
        log.configure_logging(log_path=log_path, level=logging.INFO, clear_handlers=True)
        
        # start training
        manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        
        
        device = args["device"]
        
        #gate_logging = args.get("gate_logging", False)
        gate_logging = False
        writer = SummaryWriter(args["destination"])
        if gate_logging:
            if hasattr(self.model, "enable_gate_logging"):
                self.model.enable_gate_logging(writer, log_every=args.get("gate_log_every", 1000))
                log.info("Gate logging enabled.")
        self.log_gradients = gate_logging

        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        log.info(f"Total Parameters: {total_params:,}, Trainable: {trainable_params:,}")

        # Log receptive field information if available
        if hasattr(self.model, 'calculate_receptive_field'):
            try:
                receptive_field_dict = self.model.calculate_receptive_field()
                rf_info = receptive_field_dict
                # Log the detailed receptive field table
                log.info(f"[RECEPTIVE FIELD] Stage{'':<21} k    d    s   k_eff    RF   jump")
                log.info(f"[RECEPTIVE FIELD] {'-'*58}")
                for stage in rf_info['stages']:
                    log.info(
                        f"[RECEPTIVE FIELD] {stage['stage']:<25} {stage['kernel']:>4} {stage['dilation']:>4} "
                        f"{stage['stride']:>4} {stage['k_eff']:>6} {stage['rf']:>6} {stage['jump']:>6}"
                    )
                log.info(f"[RECEPTIVE FIELD] {'-'*58}")
                log.info(
                    f"[RECEPTIVE FIELD] Total RF{'':<17} {rf_info['total_rf']:>6}  "
                    f"(input size: {rf_info['input_size'][0]}x{rf_info['input_size'][1]})"
                )
                log.info(f"[RECEPTIVE FIELD] RF covers {rf_info['rf_coverage_pct']:.1f}% of the input width")
            except Exception as e:
                log.warning(f"Could not calculate receptive field: {e}")

        # if optimizer_switch is True, switch to LBFGS optimizer after 90% of epochs
        self.epoch_switch_optimizer = args["epochs"] + 1
        if self.optimizer_switch:
            self.epoch_switch_optimizer = int(0.9 * self.epoch_switch_optimizer)

        self.best_model_params = None
        self.best_val_model_params = None
        start_epoch = 0

        # Assume noisy data

        scheduler_config = args["scheduler"]
        if self.finetune:
            log.info(
                f"Finetune mode: keeping optimizer LR from model_parameters.lr ({self.opt.param_groups[0]['lr']:.2e}) "
                f"instead of scheduler.init_lr ({scheduler_config.init_lr:.2e})."
            )
        else:
            self.opt.param_groups[0]["lr"] = scheduler_config.init_lr

        if scheduler_config.type == "ReduceLROnPlateau":
          scheduler = ReduceLROnPlateau(
            self.opt,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            threshold=scheduler_config.threshold,
            min_lr=scheduler_config.min_lr
          )
          early_stop_patience = scheduler_config.early_stop_patience if scheduler_config.early_stop_patience is not None else scheduler_config.patience * 2
        elif scheduler_config.type == "StepLR":
          scheduler = StepLR(
            self.opt,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma
          )
          early_stop_patience = scheduler_config.early_stop_patience if scheduler_config.early_stop_patience is not None else 10 * 2
        elif scheduler_config.type == "constant":
          scheduler = LambdaLR(self.opt, lr_lambda=lambda epoch: 1.0)
          early_stop_patience = scheduler_config.early_stop_patience
        else:
          raise ValueError(f"Unknown scheduler type: {scheduler_config.type}")
        print(f"Early stopping patience set to {early_stop_patience} epochs.")
        early_stop_counter = 0
        min_delta = 1e-7
        
        epochs = tqdm(range(start_epoch, start_epoch + args["epochs"]), "CNN Training Epochs")
        vis_interval = args.get("visualize_interval", 200)
        visualize = args.get("visualization", True)
        if visualize:
            vis_dataloader = DataLoader(
                        train_dataloader.dataset,
                        batch_size=train_dataloader.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                        drop_last=False)
            vis_dataloader_val = DataLoader(
                val_dataloader.dataset,
                batch_size=val_dataloader.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False)
        try:
            for epoch in epochs:
                if epoch == self.epoch_switch_optimizer:
                    self.opt = LBFGS(self.model.parameters(), history_size=20, line_search_fn="strong_wolfe")
                    log.info(f"Switched to LBFGS optimizer at epoch {epoch}.")

                # Training
                
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(train_dataloader, device, writer)
                
                # Validation
                self.model.eval()
                
                print(f"initial lr: {self.opt.param_groups[0]['lr']}")

                torch.cuda.empty_cache()
                with torch.no_grad():
                    val_epoch_loss, other_losses_val = self.run_epoch(val_dataloader, device, writer)
                    #val_epoch_loss, other_losses_val = train_epoch_loss, other_losses_train # TODO 
                if True: # realK
                    val_epoch_loss = other_losses_val["Huber"] # TODO for realK
                    
                old_lr = self.opt.param_groups[0]["lr"]
                if scheduler_config.type == "ReduceLROnPlateau":
                    scheduler.step(train_epoch_loss)
                    new_lr = self.opt.param_groups[0]["lr"]
                    logging.info(
                        f"[LR CHECK] epoch={epoch} "
                        f"train_loss={train_epoch_loss:.4e} "
                        f"val_loss={val_epoch_loss:.4e} "
                        f"best={scheduler.best:.4e} "
                        f"bad_epochs={scheduler.num_bad_epochs}/{scheduler.patience} "
                        f"lr={old_lr:.2e}→{new_lr:.2e}"
                    )
                else:
                    scheduler.step()
                    new_lr = self.opt.param_groups[0]["lr"]
                    logging.info(
                        f"[LR CHECK] epoch={epoch} "
                        f"train_loss={train_epoch_loss:.4e} "
                        f"val_loss={val_epoch_loss:.4e} "
                        f"lr={new_lr:.2e}"
                    )

                # Logging
                for metric_name, metric_value in other_losses_val.items():
                    writer.add_scalar(f"val {metric_name}", metric_value, epoch)
                for metric_name, metric_value in other_losses_train.items():
                    writer.add_scalar(f"train {metric_name}", metric_value, epoch)

                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                current_lr = self.opt.param_groups[0]['lr']
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.4e}, val loss: {val_epoch_loss:.4e}, lr: {current_lr:.2e}")

                # Keep best model
                if self.best_model_params is None or train_epoch_loss < (self.best_model_params["loss"] - min_delta):
                    self.best_model_params = {
                        "epoch": epoch,
                        "loss": train_epoch_loss,
                        "val loss": val_epoch_loss,
                        "state_dict": deepcopy(self.model.state_dict()),
                        "optimizer": self.opt.state_dict(),
                        # "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }
                    self.model.save(args["destination"])
                    self.save_epoch_metrics_yaml(
                        destination=args["destination"],
                        filename="measurements_best_train.yaml",
                        epoch=epoch,
                        train_epoch_loss=train_epoch_loss,
                        val_epoch_loss=val_epoch_loss,
                        other_losses_train=other_losses_train,
                        other_losses_val=other_losses_val,
                        no_params=total_params,
                        max_epochs=args["epochs"],
                        training_time=time.perf_counter() - start_time,
                        checkpoint_type="best_train",
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    log.info(f"No improvement for {early_stop_counter}/{early_stop_patience} epochs.")

                    if early_stop_patience <= early_stop_counter:
                        log.info(f"\nEarly stopping triggered! No improvement in validation loss for {early_stop_patience} consecutive epochs.")
                        break
                    
                if self.best_val_model_params is None or val_epoch_loss < (self.best_val_model_params["loss"] - min_delta):
                    self.best_val_model_params = {
                        "epoch": epoch,
                        "loss": val_epoch_loss,
                        "state_dict": deepcopy(self.model.state_dict()),
                        "training time in sec": (time.perf_counter() - start_time)
                    }
                    path = args["destination"] / "val"
                    path.mkdir(parents=True, exist_ok=True)
                    self.model.save(path)
                    self.save_epoch_metrics_yaml(
                        destination=path,
                        filename="measurements_best_val.yaml",
                        epoch=epoch,
                        train_epoch_loss=train_epoch_loss,
                        val_epoch_loss=val_epoch_loss,
                        other_losses_train=other_losses_train,
                        other_losses_val=other_losses_val,
                        no_params=total_params,
                        max_epochs=args["epochs"],
                        training_time=time.perf_counter() - start_time,
                        checkpoint_type="best_val",
                    )
                
                overfit = args.get("overfit", False)
                if visualize and epoch % vis_interval == 0: # and early_stop_counter < vis_interval and self.best_model_params is not None:
                  with torch.no_grad():
                    best_model_tmp = deepcopy(self.model)
                    best_model_tmp.load_state_dict(self.best_model_params["state_dict"])
                    best_model_tmp.to(args["device"])
                    
                    if epoch == 0:
                        visualize_inputs(train_dataloader, args, amount_datapoints_to_visu=1, plot_path=args["destination"] / f"train_temp{epoch}", pic_format="png")
                        visualize_outputs(best_model_tmp, train_dataloader, args, plot_path=args["destination"] / f"train_best_e{epoch}", amount_datapoints_to_visu=1, pic_format="png", plot_true=True)
                        #visualize_inputs(vis_dataloader_val, args, amount_datapoints_to_visu=1, plot_path=args["destination"] / f"val_temp{epoch}", pic_format="png")
                        #visualize_outputs(best_model_tmp, vis_dataloader_val, args, plot_path=args["destination"] / f"val_e{epoch}", amount_datapoints_to_visu=1, pic_format="png")
                    else:
                        visualize_outputs(best_model_tmp, train_dataloader, args, plot_path=args["destination"] / f"train_best_e{epoch}", amount_datapoints_to_visu=1, pic_format="png", plot_true=False)
                        #if not overfit:
                           # visualize_outputs(best_model_tmp, vis_dataloader_val, args, plot_path=args["destination"] / f"val_e{epoch}", amount_datapoints_to_visu=1, pic_format="png", plot_true=False)
                        
        except KeyboardInterrupt:
            log.info("\nTraining interrupted by user.")

            try:
                with torch.no_grad():
                  model_tmp = deepcopy(self.model)
                  model_tmp.load_state_dict(self.best_model_params["state_dict"])
                  model_tmp.to(args["device"])
                  model_tmp.save(args["destination"], model_name=f"interim_model_e{epoch}.pt")
                  visualize_outputs(model_tmp, val_dataloader, args, plot_path=args["destination"] / f"plot_val_interim_e{epoch}", amount_datapoints_to_visu=2, pic_format="png")
            except Exception as e:
                logging.error(e)

            try:
                choice = input("Enter new LR to continue, or press Enter to stop: ")
                if choice:
                    new_lr = float(choice)
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    log.info(f"Resuming with new LR: {new_lr}")
            except Exception:
                pass
        finally:
            if writer is not None:
                writer.close()

        if self.best_model_params is not None:
            self.model.load_state_dict(self.best_model_params["state_dict"])
            self.opt.load_state_dict(self.best_model_params["optimizer"])
            log.info(f"Best model was found in epoch {self.best_model_params['epoch']}.")
            logging.info(f"Best model was found in epoch {self.best_model_params['epoch']}/{args["epochs"]}.")
            return self.best_model_params["loss"]
        else:
            logging.warning("Training stopped before any model could be saved.")
            return float('inf')

    def run_epoch(self, dataloader: DataLoader, device: str, writer: SummaryWriter = None):
        overfit = self.args.get("overfit", False) if hasattr(self, "args") else False
        overfit_on = self.args.get("overfit_on", None) if hasattr(self, "args") else None
        epoch_loss = 0.0
        if overfit and overfit_on:
            y_pred = None
            y_reduced = None
            for i, (x, y) in enumerate(dataloader):
                if i in overfit_on:  # only process selected batches
                    x = x.to(device)
                    y = y.to(device)
                    
                    self.opt.zero_grad()
                    y_pred = self.model(x)
                    required_size = y_pred.shape[2:]
                    start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
                    y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

                    #time_steps = y_pred.shape[2]
                    # for t in range(time_steps):
                    #     logging.info(f"Timestep {t}: y_true min/max: {y_reduced[:,:,t].min().item():.6e}/{y_reduced[:,:,t].max().item():.6e}, y_pred min/max: {y_pred[:,:,t].min().item():.6e}/{y_pred[:,:,t].max().item():.6e}")

                    loss = self.loss_func(y_pred, y_reduced)

                    if self.model.training:
                        loss.backward()
                        
                        self.opt.step()
                        self.global_step += 1

                    epoch_loss += loss.detach().item()
                    
                    if self.log_gradients:
                        total_norm = 0.0
                        log_hists = (self.global_step % 100) == 0
                        for name, p in self.model.named_parameters():
                            if p.grad is None:
                                continue
                            #if log_hists:
                                #writer.add_histogram(f"grad/{name}", p.grad.detach().abs(), self.global_step)
                            total_norm += p.grad.detach().data.norm(2).item() ** 2

                        writer.add_scalar("grad/global_norm", total_norm ** 0.5, self.global_step)
                        writer.add_scalar("loss/focal_mse", FocalMSE()(y_pred, y_reduced), self.global_step)
                        writer.add_scalar("loss/ssim", SSIMLoss()(y_pred, y_reduced), self.global_step)
                        
            epoch_loss /= len(overfit_on)

            

            # Calculate metrics
            metric_values = {}
            if y_pred is not None and y_reduced is not None:
                for metric_name, metric in self.metrics.items():
                    metric_values[metric_name] = metric(y_pred, y_reduced).detach().item()
                    
            

            return epoch_loss, metric_values


        # free up memory
        gc.collect()
        torch.cuda.empty_cache()
        processed_batches = 0
        sublist_predictions_cache = {}  # Cache predictions from previous sublists for chaining
        
        epoch_metrics = {name: 0.0 for name in self.metrics}
        
        for batch_idx, (x,y,metadata_list) in tqdm(enumerate(dataloader), "Processing batches", total=len(dataloader)):
            
            subset_idx = metadata_list["subset_idx"][0].item()
            chain_ids = metadata_list["chain_idx"].tolist()
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Build init_frame from cache
            init_frame = None
            if subset_idx > 0:
                H, W = x.shape[3], x.shape[4]
                init_frame = torch.zeros(x.shape[0], 1, H, W, device=x.device, dtype=x.dtype)
                for batch_i, chain_id in enumerate(chain_ids):
                        cache_key = (chain_id, subset_idx -1)
                        if cache_key in sublist_predictions_cache:
                            prev_pred = torch.sigmoid(sublist_predictions_cache[cache_key].to(device))
                            init_frame[batch_i, 0] = prev_pred[0, -1]
                        else:
                            logging.warning(f"Cache miss for chain {chain_id}, subset {subset_idx -1 }. Using zeros for init_frame.")
    
            processed_batches += 1
            x = x.to(device, non_blocking=True) # Use non_blocking transfers
            y = y.to(device, non_blocking=True)
            
            if self.model.training:
                self.opt.zero_grad(set_to_none=True)
                if self.opt.__class__.__name__ == "LBFGS":
                    def closure():
                        """ closure function for optimizer LBFGS """
                        self.opt.zero_grad()
                        y_pred_logits = self.model(x, init_frame)
                        required_size = y_pred_logits.shape[2:]
                        start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
                        y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

                        loss = self.loss_func(y_pred_logits, y_reduced)
                        loss.backward()
                        return loss
                    self.opt.step(closure)

            y_pred_logits = self.model(x, init_frame)
            required_size = y_pred_logits.shape[2:]
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]
            
            if self.loss_func.__class__.__name__ == "CustomLoss":
                loss = self.loss_func(y_pred_logits, y_reduced, writer, self.global_step)
            else:
                loss = self.loss_func(y_pred_logits, y_reduced)
            
                
            y_pred = torch.sigmoid(y_pred_logits)
            
            if self.model.training:
                loss.backward()
                if self.log_gradients:
                    total_norm = 0.0
                    log_hists = (self.global_step % 100) == 0

                    # Track per-layer gradient norms grouped by component
                    enc_norms = {}
                    enc_max_abs = {}
                    bottleneck_norm = 0.0
                    bottleneck_max_abs = 0.0
                    proj_norm = 0.0
                    proj_max_abs = 0.0
                    enc_dec_proj_norm = 0.0
                    enc_dec_proj_max_abs = 0.0
                    dec_norms = {}
                    dec_max_abs = {}
                    other_norm = 0.0

                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue

                        grad_norm = p.grad.detach().data.norm(2).item() ** 2
                        total_norm += grad_norm
                        
                        grad_abs_argmax = p.grad.detach().data.abs().argmax()
                        grad_max_abs = p.grad.detach().data.flatten()[grad_abs_argmax].item()

                        # --- Encoder blocks inside ConvLSTMCell ---
                        if "convLSTMcell.encoders" in name:
                            # extract enc index e.g. "convLSTMcell.encoders.0.0.weight" -> 0
                            try:
                                enc_idx = int(name.split("encoders.")[1].split(".")[0])
                                enc_norms[enc_idx] = enc_norms.get(enc_idx, 0.0) + grad_norm
                                enc_max_abs[enc_idx] = max(enc_max_abs.get(enc_idx, 0.0), grad_max_abs, key=abs)
                            except (IndexError, ValueError):
                                other_norm += grad_norm

                        # --- Bottleneck ---
                        elif "convLSTMcell.bottleneck" in name:
                            bottleneck_norm += grad_norm
                            if abs(grad_max_abs) > abs(bottleneck_max_abs):
                                bottleneck_max_abs = grad_max_abs
                                
                        # --- Gate projection ---
                        elif "convLSTMcell.proj" in name:
                            proj_norm += grad_norm
                            if abs(grad_max_abs) > abs(proj_max_abs):
                                proj_max_abs = grad_max_abs

                        # --- Encoder-decoder projection --- 
                        elif "enc_dec_projection" in name:
                            enc_dec_proj_norm += grad_norm
                            if abs(grad_max_abs) > abs(enc_dec_proj_max_abs):
                                enc_dec_proj_max_abs = grad_max_abs
                            
                        # --- Decoder blocks ---
                        elif "decoder_blocks" in name:
                            try:
                                dec_idx = int(name.split("decoder_blocks.")[1].split(".")[0])
                                dec_norms[dec_idx] = dec_norms.get(dec_idx, 0.0) + grad_norm
                                dec_max_abs[dec_idx] = max(dec_max_abs.get(enc_idx, 0.0), grad_max_abs, key=abs)
                            except (IndexError, ValueError):
                                other_norm += grad_norm

                        else:
                            other_norm += grad_norm

                    # --- Log global norm ---
                    writer.add_scalar("grad/global_norm", total_norm ** 0.5, self.global_step)

                    # --- Log per encoder block ---
                    for enc_idx, norm_sq in enc_norms.items():
                        writer.add_scalar(f"grad/enc/{enc_idx}/norm", norm_sq ** 0.5, self.global_step)
                        
                    for enc_idx, maximum in enc_max_abs.items():
                        writer.add_scalar(f"grad/enc/{enc_idx}/max", maximum, self.global_step)

                    # --- Log bottleneck and proj ---
                    writer.add_scalar("grad/bottleneck/norm", bottleneck_norm ** 0.5, self.global_step)
                    writer.add_scalar("grad/bottleneck/max", bottleneck_max_abs, self.global_step)
                    writer.add_scalar("grad/proj/norm", proj_norm ** 0.5, self.global_step)
                    writer.add_scalar("grad/proj/max", proj_max_abs, self.global_step)
                    
                    # --- Log encoder-decoder projection ---
                    writer.add_scalar("grad/enc_dec_proj/norm", enc_dec_proj_norm ** 0.5, self.global_step)
                    writer.add_scalar("grad/enc_dec_proj/max", enc_dec_proj_max_abs, self.global_step)

                    # --- Log per decoder block ---
                    for dec_idx, norm_sq in dec_norms.items():
                        writer.add_scalar(f"grad/dec/{dec_idx}/norm", norm_sq ** 0.5, self.global_step)
                        
                    for dec_idx, maximum in dec_max_abs.items():
                        writer.add_scalar(f"grad/enc/{dec_idx}/max", maximum, self.global_step)

                    # --- Log ratio enc[0] / enc[-1] to detect vanishing ---
                    if 0 in enc_norms and len(enc_norms) > 1:
                        last_enc_idx = max(enc_norms.keys())
                        ratio = (enc_norms[0] ** 0.5) / (enc_norms[last_enc_idx] ** 0.5 + 1e-8)
                        writer.add_scalar("grad/enc/ratio_first_last", ratio, self.global_step)
                        
                    # --- Log ratio dec[0] / dec[-1] to detect vanishing ---
                    if 0 in dec_norms and len(dec_norms) > 1:
                        last_dec_idx = max(dec_norms.keys())
                        ratio = (dec_norms[0] ** 0.5) / (dec_norms[last_dec_idx] ** 0.5 + 1e-8)
                        writer.add_scalar("grad/dec/ratio_first_last", ratio, self.global_step)
                        
                    if self.global_step % 100 == 1:
                        writer.add_scalar("loss/focal_mse", FocalMSE()(y_pred_logits, y_reduced), self.global_step)
                        writer.add_scalar("loss/ssim", SSIMLoss()(y_pred, y_reduced), self.global_step)
                        writer.add_scalar("loss/mse", MSELoss()(y_pred, y_reduced), self.global_step)
                    
                    
                    writer.add_scalar("data/target_std", y_reduced.std(), self.global_step)

                    # --- Optional histograms ---
                    if log_hists:
                        for name, p in self.model.named_parameters():
                            if p.grad is not None:
                                writer.add_histogram(f"grad_hist/{name}", p.grad.detach().abs(), self.global_step)

                #nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                if self.opt.__class__.__name__ == "LBFGS":
                    self.opt.step(closure)
                else:
                    self.opt.step()
                self.global_step += 1

            # Accumulate loss
            epoch_loss += loss.item()

            with torch.no_grad():
                 # Calculate metrics
                if y_pred is not None and y_reduced is not None:
                    #logging.info(f"Norm of dataset: {dataloader.dataset.norm}")
                    y_pred_rev = reverse_norm_one_dp_outputs(y_pred, dataloader.dataset.norm)
                    y_reduced_rev = reverse_norm_one_dp_outputs(y_reduced, dataloader.dataset.norm)
                    
                    for metric_name, metric in self.metrics.items():
                        if metric_name == "SSIM":
                            epoch_metrics[metric_name] += metric(y_pred, y_reduced) #.detach().item()
                        else:
                            epoch_metrics[metric_name] += metric(y_pred_rev, y_reduced_rev).detach().item()
                        #logging.info(f"Metric {metric_name}: {epoch_metrics[metric_name]:.4e}")
                        
                if True:
                #if self.sublist_mode: TODO
                    for batch_i, chain_id in enumerate(chain_ids):
                        cache_key = (chain_id, subset_idx)
                        sublist_predictions_cache[cache_key] = y_pred_logits[batch_i].detach().cpu()
                        
                    if subset_idx > 0:
                        for chain_id in chain_ids:
                            sublist_predictions_cache.pop((chain_id, subset_idx -1), None)
                            


        # Average out the results
        num_batches = len(dataloader)
        
        epoch_loss /= processed_batches
        for name in epoch_metrics:
            epoch_metrics[name] /= processed_batches

        return epoch_loss, epoch_metrics

    def save_metrics_separate_yaml(self, train_dataloader, val_dataloader, test_dataloader, destination: Path, no_params:int, max_epochs:int, training_time:float, device: str, current_epoch: int = None):
        # prepare data as dict
        self.model.eval()
        torch.cuda.empty_cache()
        
        if self.model.eval:
            print("Model in eval mode")
            
            with torch.no_grad():
                train_epoch_loss, other_losses_train = self.run_epoch(train_dataloader, device, writer=None)
        else:
            train_epoch_loss, other_losses_train = self.run_epoch(train_dataloader, device, writer=None)
        
        torch.cuda.empty_cache()
        
        if self.model.eval:
            print("Model in eval mode")
            with torch.no_grad():
                val_epoch_loss, other_losses_val = self.run_epoch(val_dataloader, device, writer=None)
        else:
            val_epoch_loss, other_losses_val = self.run_epoch(val_dataloader, device, writer=None)
        
        if test_dataloader is not None:
            if self.model.eval:
                print("Model in eval mode")
                with torch.no_grad():
                    test_epoch_loss, other_losses_test = self.run_epoch(test_dataloader, device, writer=None)
            else:
                test_epoch_loss, other_losses_test = self.run_epoch(test_dataloader, device, writer=None)
        
        metrics = {}
        metrics["no_params"] = no_params
        metrics["max_epochs"] = max_epochs
        metrics["training_time [s]"] = training_time
        if current_epoch is not None:
            metrics["current_epoch"] = current_epoch
        if self.best_model_params is not None:
            metrics["best_epoch"] = self.best_model_params["epoch"]
        metrics["train"] = other_losses_train
        metrics["train"]["train loss"] = train_epoch_loss
        metrics["val"] = other_losses_val
        metrics["val"]["val loss"] = val_epoch_loss
        if test_dataloader is not None:
            metrics["test"] = other_losses_test
            metrics["test"]["test loss"] = test_epoch_loss

        # save data as yaml
        save_yaml(metrics, destination / "measurements.yaml")
        
def log_grad_stats(model, logger=logging):
    logger.info("Gradient and Parameter statistics:")
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        grad = p.grad.detach()
        logger.info(
            f"[GRAD] {name:40s} "
            f"min={grad.min():+.2e} "
            f"max={grad.max():+.2e} "
            f"mean={grad.mean():+.2e} "
            f"std={grad.std():+.2e} "
        )
        logger.info(
            f"[PARAM] {name:40s} "
            f"min={p.min():+.2e} "
            f"max={p.max():+.2e} "
            f"mean={p.mean():+.2e} "
            f"std={p.std():+.2e} "
        )
