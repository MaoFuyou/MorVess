"""Stage-I / Stage-II trainer for MorVess.

The training loop implements the five loss terms and progressive schedules
used by the MorVess framework.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DiceLoss, normalized_thickness_l1, soft_cldice_loss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - allows running without tensorboard installed
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _resize_target(target: torch.Tensor, shape: tuple[int, int], mode: str) -> torch.Tensor:
    return F.interpolate(target.unsqueeze(1).float(), size=shape, mode=mode, align_corners=False if mode != "nearest" else None)


def _configure_stage(model: nn.Module, stage: str) -> None:
    core = _unwrap(model)
    if stage == "stage1":
        return
    for parameter in core.parameters():
        parameter.requires_grad = False
    for parameter in core.sam.mask_decoder.parameters():
        parameter.requires_grad = True


def _losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], dice_loss: DiceLoss) -> dict[str, torch.Tensor]:
    mask_logits = outputs["low_res_logits"]
    labels = batch["low_res_label"].reshape(-1, *batch["low_res_label"].shape[-2:])
    if labels.shape[-2:] != mask_logits.shape[-2:]:
        labels = _resize_target(labels, mask_logits.shape[-2:], "nearest").squeeze(1).long()
    else:
        labels = labels.long()

    cross_entropy = F.cross_entropy(mask_logits, labels)
    dice = dice_loss(mask_logits, labels, softmax=True)
    cldice = soft_cldice_loss(mask_logits, labels)
    losses = {"ce": cross_entropy, "dice": dice, "cldice": cldice}

    distance_logits = outputs.get("low_res_distance_logits")
    if distance_logits is not None:
        boundary = batch["boundary"].reshape(-1, *batch["boundary"].shape[-2:])
        boundary = _resize_target(boundary, distance_logits.shape[-2:], "bilinear")
        losses["distance"] = F.l1_loss(torch.sigmoid(distance_logits), boundary)
    else:
        losses["distance"] = mask_logits.new_zeros(())

    thickness_logits = outputs.get("low_res_thickness_logits")
    if thickness_logits is not None:
        thickness = batch["thickness"].reshape(-1, *batch["thickness"].shape[-2:])
        thickness = _resize_target(thickness, thickness_logits.shape[-2:], "bilinear")
        losses["thickness"] = normalized_thickness_l1(thickness_logits, thickness)
    else:
        losses["thickness"] = mask_logits.new_zeros(())
    return losses


def _make_model_input(images: torch.Tensor) -> torch.Tensor:
    # Dataset layout is [B, five slices, H, W]; SAM expects RGB per slice.
    return images.unsqueeze(2).repeat(1, 1, 3, 1, 1)


def trainer_run(args, model, snapshot_path: str, multimask_output: bool, low_res: int, stage: str = "stage1") -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("MorVess training requires a CUDA-enabled PyTorch installation.")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported training stage: {stage}")

    from datasets.dataset_distance import RandomGenerator, dataset_reader_parse

    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs("training_log", exist_ok=True)
    logging.basicConfig(
        filename=os.path.join("training_log", f"{os.path.basename(snapshot_path)}_{stage}.log"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("MorVess %s trainer", stage)
    logging.info(str(args))

    dataset = dataset_reader_parse(
        base_dir=args.root_path,
        split="train",
        num_classes=args.num_classes,
        transform=RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res]),
    )
    if not len(dataset):
        raise RuntimeError(f"No training samples found under {args.root_path}.")

    def worker_init_fn(worker_id: int) -> None:
        random.seed(args.seed + worker_id)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * args.n_gpu,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 8),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    _configure_stage(model, stage)
    model.train()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after stage configuration.")
    optimizer = optim.AdamW(trainable_parameters, lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.1)
    scheduler = None
    if stage == "stage2":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, len(loader) * 10))

    dice_loss = DiceLoss(args.num_classes + 1)
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(args, "use_amp", True))
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    max_epochs = args.max_epochs
    max_iterations = max_epochs * len(loader)
    warmup_period = getattr(args, "warmup_period", 0)
    weights = {"ce": 0.2, "dice": getattr(args, "dice_param", 0.8), "cldice": 0.3, "distance": 0.2, "thickness": 0.2}
    iteration = 0

    for epoch in tqdm(range(max_epochs), ncols=90, desc=f"MorVess {stage}"):
        for sampled_batch in loader:
            images = _make_model_input(sampled_batch["image"]).cuda(non_blocking=True)
            device_batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in sampled_batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=getattr(args, "use_amp", True)):
                outputs = model(images, multimask_output, args.img_size)
                losses = _losses(outputs, device_batch, dice_loss)
                total = sum(weights[key] * value for key, value in losses.items())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            iteration += 1

            if stage == "stage1":
                if warmup_period and iteration <= warmup_period:
                    learning_rate = args.base_lr * iteration / warmup_period
                else:
                    shifted = max(0, iteration - warmup_period)
                    remaining = max(1, max_iterations - warmup_period)
                    learning_rate = args.base_lr * (1.0 - min(shifted, remaining) / remaining) ** getattr(args, "lr_exp", 7)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            else:
                scheduler.step(iteration)
                learning_rate = optimizer.param_groups[0]["lr"]

            writer.add_scalar("train/lr", learning_rate, iteration)
            writer.add_scalar("train/total_loss", total.item(), iteration)
            for key, value in losses.items():
                writer.add_scalar(f"train/{key}", value.item(), iteration)
            logging.info(
                "epoch=%d iteration=%d total=%.6f ce=%.6f dice=%.6f cldice=%.6f distance=%.6f thickness=%.6f",
                epoch + 1,
                iteration,
                total.item(),
                losses["ce"].item(),
                losses["dice"].item(),
                losses["cldice"].item(),
                losses["distance"].item(),
                losses["thickness"].item(),
            )

        if (epoch + 1) % 20 == 0 or epoch + 1 == max_epochs:
            checkpoint = os.path.join(snapshot_path, f"epoch_{epoch + 1:03d}.pth")
            _unwrap(model).save_parameters(checkpoint)
            logging.info("Saved %s", checkpoint)

    writer.close()
    return "Training Finished!"
