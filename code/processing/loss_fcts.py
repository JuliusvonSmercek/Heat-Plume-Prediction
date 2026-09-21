from code.utils import logging as log  # noqa: F401
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn import HuberLoss, L1Loss, MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure


class CombiLoss(nn.Module):
    """
    Combines MSE and a secondary loss (e.g. MAE) with ratio alpha.
    Status: Autograd safe. Suitable for training.
    """

    def __init__(self, alpha: float = 0.75, second_loss: nn.Module = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss if second_loss is not None else nn.L1Loss()
        self.alpha = alpha
        self.name = f"CombiLoss (a={alpha}) with {self.secondary_loss_function.__class__.__name__}"

    def forward(self, predictions, labels):
        eval_second = self.secondary_loss_function(predictions, labels)
        return self.alpha * self.mse(predictions, labels) + (1.0 - self.alpha) * eval_second


class SSIMLoss(nn.Module):
    """
    WARNING: returns inverted value compared to old implementation
    Structural Similarity Index Measure.
    Status: Autograd safe. Suitable for training and evaluation.
    """

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, predictions, labels):
        return 1.0 - self.ssim(predictions, labels)


class LinfLoss(nn.Module):
    """
    L-infinity loss (Chebyshev distance).
    Status: Autograd safe (subgradient exists). Suitable for training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.amax(torch.abs(output - target))


class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    pat = torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])
    Status: Evaluation Metric ONLY (Zero gradients). Do NOT use for training.
    """

    def __init__(self, pat_thresholds: list):
        super().__init__()
        self.register_buffer("thresholds", torch.tensor(pat_thresholds).view(1, -1, 1, 1))

    def forward(self, output, label):
        if output.dim() == 3:
            output = output.unsqueeze(1)
            label = label.unsqueeze(1)

        abs_diff = torch.abs(output - label)
        above_thresh = abs_diff > self.thresholds
        pat = above_thresh.to(torch.float32).mean(dim=(2, 3))

        return (pat * 100).mean()


# YAML ``train_loss`` names → zero-arg factories. Add new training losses here only.
_TRAIN_LOSSES: dict[str, Callable[[], nn.Module]] = {
    "mae": L1Loss,
    "mse": MSELoss,
    "huber": HuberLoss,
    "combi": CombiLoss,
}


def get_train_loss(name: str) -> nn.Module:
    """Instantiate a training loss by YAML name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _TRAIN_LOSSES:
        raise ValueError(f"Unknown train_loss '{name}'. Choose from: {sorted(_TRAIN_LOSSES)}")
    return _TRAIN_LOSSES[key]()


# Post-training report metrics (measurements.yaml). PAT is channel-dependent — see ``make_pat_loss``.
DEFAULT_PAT_THRESHOLD: float = 0.1
_EVAL_METRICS: dict[str, Callable[[], nn.Module]] = {
    "Huber": HuberLoss,
    "Linf": LinfLoss,
    "MAE": L1Loss,
    "MSE": MSELoss,
    "SSIM": SSIMLoss,
}
# Per-channel on denormalized labels (SSIM stays full-field on normalized tensors).
CHANNELWISE_EVAL_NAMES: tuple[str, ...] = tuple(name for name in _EVAL_METRICS if name != "SSIM")
EVAL_METRIC_NAMES: tuple[str, ...] = (*_EVAL_METRICS.keys(), "PAT")


def get_eval_metrics(device: str | torch.device) -> dict[str, nn.Module]:
    """Instantiate fixed evaluation metrics and move them to ``device``."""
    return {name: factory().to(device) for name, factory in _EVAL_METRICS.items()}


def make_pat_loss(num_channels: int, device: str | torch.device) -> PATLoss:
    """PAT metric sized to the prediction channel count."""
    return PATLoss(pat_thresholds=[DEFAULT_PAT_THRESHOLD] * num_channels).to(device)
