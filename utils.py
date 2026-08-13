"""Shared losses and metrics used by the MorVess entry points.

This module restores the public imports used by the training and evaluation
scripts.  It intentionally has no dependency on MedPy so that the basic Dice
metric remains available in a minimal inference environment.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss with support for the ignore label ``-100``."""

    def __init__(self, n_classes: int, smooth: float = 1e-5) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight: list[float] | None = None,
        softmax: bool = False,
    ) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        if inputs.ndim != 4 or target.ndim != 3:
            raise ValueError(
                "DiceLoss expects logits shaped [B, C, H, W] and labels shaped [B, H, W]."
            )
        if inputs.shape[0] != target.shape[0] or inputs.shape[2:] != target.shape[1:]:
            raise ValueError(f"Prediction shape {tuple(inputs.shape)} does not match target {tuple(target.shape)}.")

        valid = target.ne(-100)
        safe_target = target.clamp_min(0).long()
        one_hot = F.one_hot(safe_target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).to(inputs.dtype)
        weights = weight or [1.0] * self.n_classes

        loss = inputs.new_zeros(())
        for class_index, class_weight in enumerate(weights):
            score = inputs[:, class_index]
            truth = one_hot[:, class_index]
            mask = valid[:, 0]
            intersection = torch.sum(score * truth * mask)
            denominator = torch.sum(score.square() * mask) + torch.sum(truth.square() * mask)
            loss = loss + class_weight * (1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth))
        return loss / self.n_classes


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return binary Dice for one case, with explicit empty-mask behaviour."""

    pred_mask = np.asarray(pred, dtype=bool)
    gt_mask = np.asarray(gt, dtype=bool)
    pred_size = int(pred_mask.sum())
    gt_size = int(gt_mask.sum())
    if pred_size == 0 and gt_size == 0:
        return 1.0
    if pred_size == 0 or gt_size == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred_mask, gt_mask).sum() / (pred_size + gt_size))


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-image, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, (3, 3), (1, 1), (1, 1))


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Differentiable 2-D skeleton approximation used for a clDice surrogate."""

    image = image.clamp(0.0, 1.0)
    skeleton = F.relu(image - _soft_open(image))
    for _ in range(iterations):
        image = _soft_erode(image)
        delta = F.relu(image - _soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Binary soft-clDice loss for foreground channel 1 of two-class logits."""

    if logits.shape[1] < 2:
        raise ValueError("soft_cldice_loss requires logits with a foreground channel.")
    prediction = torch.softmax(logits, dim=1)[:, 1:2]
    target_fg = target.eq(1).unsqueeze(1).to(prediction.dtype)
    skeleton_pred = soft_skeletonize(prediction, iterations)
    skeleton_target = soft_skeletonize(target_fg, iterations)
    smooth = 1e-5
    precision = (skeleton_pred * target_fg).sum((1, 2, 3)) / (skeleton_pred.sum((1, 2, 3)) + smooth)
    sensitivity = (skeleton_target * prediction).sum((1, 2, 3)) / (skeleton_target.sum((1, 2, 3)) + smooth)
    cldice = 2.0 * precision * sensitivity / (precision + sensitivity + smooth)
    return 1.0 - cldice.mean()


def normalized_thickness_l1(prediction_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant thickness loss described in Eq. (16) of the paper."""

    prediction = F.softplus(prediction_logits)
    target = target.clamp_min(0.0)
    prediction = prediction / prediction.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    target = target / target.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return F.l1_loss(prediction, target)

