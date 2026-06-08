import torch.nn as nn
import torch
from torch import max, abs, zeros, sum
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

    
class CombiLoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self, alpha: float = 1., second_loss:nn.Module = nn.L1Loss()):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha}) with {self.secondary_loss_function}"

    def forward(self, predictions, labels):
        eval_second = self.secondary_loss_function(predictions, labels)

        return self.alpha * self.mse(predictions, labels) + (1. - self.alpha) * eval_second
    

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.min = 0
        self.max = 1

    def forward(self, predictions, labels):
        # predictions/labels: [B, C, T, H, W]
        batch_size  = predictions.shape[0]
        num_channels = predictions.shape[1]
        num_timesteps = predictions.shape[2]
        ssim_total = 0.0

        for b in range(batch_size):
            for c in range(num_channels):
                for t in range(num_timesteps):
                    ssim_val = ssim(
                        predictions[b, c, t].detach().cpu().numpy(),  # [H, W]
                        labels[b, c, t].detach().cpu().numpy(),        # [H, W]
                        data_range=self.max - self.min
                    )
                    ssim_total += ssim_val

        return ssim_total / (batch_size * num_channels * num_timesteps)
    

class LinfLoss(nn.Module):
    def __init__(self):
        super(LinfLoss, self).__init__()

    def forward(self, output, target):
        return max(abs(output - target))


class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    pat = torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])
    """

    def __init__(self, pat_threshold: float):
        super(PATLoss, self).__init__()
        self.pat_threshold = pat_threshold

    def forward(self, output, label):
        assert output.shape == label.shape, f"Output and label must have the same shape, got {output.shape} vs {label.shape}"
        if output.ndim == 4:
            output = output.unsqueeze(0)
            label = label.unsqueeze(0)
        assert output.ndim == 5, f"Expected output shape [B,C,T,H,W], got {output.shape}"
        assert output.size(1) == 1, f"PATLoss expects C=1, got C={output.size(1)}"

        threshold = torch.as_tensor(self.pat_threshold, device=output.device, dtype=output.dtype)
        if threshold.ndim != 0:
            raise ValueError(f"pat_threshold must be a scalar for C=1, got shape {tuple(threshold.shape)}")

        pat = (abs(output - label) > threshold).float().mean()
        return pat * 100
    
class WeightedMSE(nn.Module):
    def __init__(self, threshold=0.4, hot_weight=5.0, normalize_by_weights=True):
        super().__init__()
        self.threshold = threshold
        self.hot_weight = hot_weight
        self.normalize_by_weights = normalize_by_weights

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

        w = torch.ones_like(target)
        w[target > self.threshold] = self.hot_weight

        weighted_sq_error = w * (pred - target) ** 2
        if self.normalize_by_weights:
            return weighted_sq_error.sum() / (w.sum() + 1e-12)
        
        return weighted_sq_error.mean()

class FocalMSE(nn.Module): 
    def __init__(self, gamma=3.0,):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred_logits, target):
        if pred_logits.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")
        pred = torch.sigmoid(pred_logits)
        error = (pred - target) ** 2
        weight = (target - 0.5).abs()  # zero weight at background
        weight = weight ** self.gamma
        weight = weight / (weight.mean() + 1e-8)  # normalize
        return (weight * error).mean()
    
class CombinedFocalMSE(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

        weight = (target - 0.5).abs() ** self.gamma  # zero weight at background
        weight = weight / (weight.mean() + 1e-8)  # normalize
        
        error = (pred - target) ** 2
        focal_mse = (weight * error).mean()
        
        mse = error.mean()
        return self.alpha * focal_mse + (1 - self.alpha) * mse
    
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, delta=0.1, delta_warmup_steps=1000):
        super().__init__()
        assert alpha + beta + delta - 1.0 < 1e-6, (
            f"Coefficients must sum to 1.0, got {alpha+beta+delta:.3f}"
        )
        self.alpha = alpha   # BCE (logits)
        self.beta  = beta    # gradient
        self.delta = delta   # second-order gradient
        self.delta_warmup_steps = delta_warmup_steps
        self.register_buffer("step", torch.tensor(0))

    def _delta_effective(self):
        if self.delta_warmup_steps <= 0:
            return self.delta
        progress = min(self.step.item() / self.delta_warmup_steps, 1.0)
        return self.delta * progress

    def gradient_loss(self, pred, target):
        dx = lambda x: x[..., 1:]    - x[..., :-1]
        dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
        return F.l1_loss(dx(pred), dx(target)) + F.l1_loss(dy(pred), dy(target))

    def second_order_gradient_loss(self, pred, target):
        dx  = lambda x: x[..., 1:]    - x[..., :-1]
        dy  = lambda x: x[..., 1:, :] - x[..., :-1, :]
        d2x = lambda x: dx(x)[..., 1:]    - dx(x)[..., :-1]
        d2y = lambda x: dy(x)[..., 1:, :] - dy(x)[..., :-1, :]
        return F.l1_loss(d2x(pred), d2x(target)) + F.l1_loss(d2y(pred), d2y(target))

    def forward(self, pred_logits, target, writer=None, global_step=None):
        delta_eff  = self._delta_effective()
        alpha_eff  = self.alpha + (self.delta - delta_eff)

        # BCE operates on logits — numerically stable, handles saturation
        bce = F.binary_cross_entropy_with_logits(pred_logits, target)

        # Spatial terms operate on sigmoid output
        pred = torch.sigmoid(pred_logits)
        grad  = self.gradient_loss(pred, target)
        grad2 = self.second_order_gradient_loss(pred, target) if delta_eff > 0 else 0.0

        loss = (
              alpha_eff  * bce
            + self.beta  * grad
            + delta_eff  * grad2
        )
        
        if writer is not None and global_step is not None:
            writer.add_scalar("loss/alpha_eff", alpha_eff,  global_step)
            writer.add_scalar("loss/beta",      self.beta,  global_step)
            writer.add_scalar("loss/delta_eff", delta_eff,  global_step)

        if self.training:
            self.step += 1

        return loss
    
class BinaryCrossEntropy(nn.Module):
    def __init__(self,gamma=2.0):
        super().__init__()
        
        self.gamma = gamma
    
    def forward(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()
    
class FocalMSE_FP(nn.Module):
    def __init__(self, gamma=2.0, fp_penalty=3.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=True):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target) ** 2

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation ** self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return (weight * error).mean()
    
class FocalMSE_FP_2D(nn.Module):
    def __init__(self, gamma=2.0, fp_penalty=2.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=True):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target) ** 2

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation ** self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return weight * error
    
class FocalMAE_FP(nn.Module):
    def __init__(self, gamma=2.0, fp_penalty=2.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=True):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target).abs()

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation ** self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return (weight * error).mean()
    
class FocalMAE(nn.Module):
    def __init__(self, gamma=2.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.bg_value = bg_value

    def forward(self, pred, target, logits=True):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target).abs()

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        weight = target_deviation ** self.gamma


        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return (weight * error).mean()
 
class FocalMAE_FP_2D(nn.Module):
    def __init__(self, gamma=1.0, fp_penalty=3.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=True):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target).abs()

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation ** self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return weight * error
        
class MaxNormLoss(nn.Module):
    def forward(self, y_pred, y_true):
        if torch.min(y_pred) < 0:
            y_pred = torch.sigmoid(y_pred)
        return (y_pred - y_true).abs().max()
    
class BCEMAELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, eps=1e-8):
        """
        gamma: focal BCE focusing parameter
        alpha: weight for BCE vs F1
               total_loss = alpha * BCE + (1 - alpha) * F1_loss
        eps: numerical stability
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_logits, target):
        # --- FOCAL BCE (on logits, stable) ---
        
        print(f"pred_logits min: {pred_logits.min()}, max: {pred_logits.max()}")
        print(f"target min: {target.min()}, max: {target.max()}")
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none'
        )
        pt = torch.exp(-bce)
        focal_bce = ((1 - pt) ** self.gamma * bce).mean()

        # --- SOFT F1 (on probabilities) ---
        pred = torch.sigmoid(pred_logits)

        f1_loss = (pred-target).abs().mean()

        # --- COMBINE ---
        loss = self.alpha * focal_bce + (1 - self.alpha) * f1_loss
        return loss
    
class BCEMAELoss_2D(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, eps=1e-8):
        """
        gamma: focal BCE focusing parameter
        alpha: weight for BCE vs MAE
               total_loss = alpha * BCE + (1 - alpha) * MAE
        eps: numerical stability
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_logits, target):
        # --- FOCAL BCE (on logits, stable) ---
        print(f"pred_logits min: {pred_logits.min()}, max: {pred_logits.max()}")
        print(f"target min: {target.min()}, max: {target.max()}")
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none'
        )
        pt = torch.exp(-bce)
        focal_bce = ((1 - pt) ** self.gamma * bce)

        # --- SOFT F1 (on probabilities) ---
        pred = torch.sigmoid(pred_logits)

        f1_loss = (pred-target).abs()

        # --- COMBINE ---
        loss = self.alpha * focal_bce + (1 - self.alpha) * f1_loss
        print(f"BCAELoss2D size: {loss.size()}")
        
        print(f"BCAELoss2D mean: {loss.mean()}")
        return loss
    
class MAE_Logits(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        error = (pred - target).abs()
        return error.mean()
    