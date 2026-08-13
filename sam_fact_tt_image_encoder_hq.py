"""2.5D FacT wrapper for the MorVess encoder.

This module connects five-slice CT inputs with the adapter modules in
``segment_anything/modeling/image_encoder_hq.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    batch, height, width, channels = x.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height or pad_width:
        x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    windows = x.view(
        batch,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
        channels,
    )
    return windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels), (
        padded_height,
        padded_width,
    )


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
) -> torch.Tensor:
    padded_height, padded_width = padded_hw
    height, width = original_hw
    batch = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    x = windows.view(batch, padded_height // window_size, padded_width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, padded_height, padded_width, -1)
    return x[:, :height, :width, :].contiguous()


def _get_relative_position(query_size: int, key_size: int, relative_position: torch.Tensor) -> torch.Tensor:
    max_relative_distance = 2 * max(query_size, key_size) - 1
    if relative_position.shape[0] != max_relative_distance:
        relative_position = F.interpolate(
            relative_position.reshape(1, relative_position.shape[0], -1).permute(0, 2, 1),
            size=max_relative_distance,
            mode="linear",
        ).reshape(-1, max_relative_distance).permute(1, 0)
    query_coordinates = torch.arange(query_size, device=relative_position.device)[:, None] * max(key_size / query_size, 1.0)
    key_coordinates = torch.arange(key_size, device=relative_position.device)[None, :] * max(query_size / key_size, 1.0)
    relative_coordinates = (query_coordinates - key_coordinates) + (key_size - 1) * max(query_size / key_size, 1.0)
    return relative_position[relative_coordinates.long()]


def _add_decomposed_relative_position(
    attention: torch.Tensor,
    query: torch.Tensor,
    relative_position_height: torch.Tensor,
    relative_position_width: torch.Tensor,
    query_size: Tuple[int, int],
    key_size: Tuple[int, int],
) -> torch.Tensor:
    query_height, query_width = query_size
    key_height, key_width = key_size
    rel_height = _get_relative_position(query_height, key_height, relative_position_height)
    rel_width = _get_relative_position(query_width, key_width, relative_position_width)
    batch, _, channels = query.shape
    query = query.reshape(batch, query_height, query_width, channels)
    height_term = torch.einsum("bhwc,hkc->bhwk", query, rel_height)
    width_term = torch.einsum("bhwc,wkc->bhwk", query, rel_width)
    return (
        attention.view(batch, query_height, query_width, key_height, key_width)
        + height_term[:, :, :, :, None]
        + width_term[:, :, :, None, :]
    ).view(batch, query_height * query_width, key_height * key_width)


class _FactAttention(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, x: torch.Tensor, factor_u: nn.Module, factor_v: nn.Module) -> torch.Tensor:
        batch, height, width, _ = x.shape
        qkv = self.attention.qkv(x, factor_u, factor_v).reshape(
            batch, height * width, 3, self.attention.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.reshape(3, batch * self.attention.num_heads, height * width, -1).unbind(0)
        attention = (query * self.attention.scale) @ key.transpose(-2, -1)
        if self.attention.use_rel_pos:
            attention = _add_decomposed_relative_position(
                attention,
                query,
                self.attention.rel_pos_h,
                self.attention.rel_pos_w,
                (height, width),
                (height, width),
            )
        attention = attention.softmax(dim=-1)
        x = (attention @ value).view(batch, self.attention.num_heads, height, width, -1)
        x = x.permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
        return self.attention.proj(x)


class _FactQKV(nn.Module):
    def __init__(self, qkv: nn.Module, query_factor: nn.Module, value_factor: nn.Module, scale: float) -> None:
        super().__init__()
        self.qkv = qkv
        self.query_factor = query_factor
        self.value_factor = value_factor
        self.dim = qkv.in_features
        self.dropout_query = nn.Dropout(0.1)
        self.dropout_value = nn.Dropout(0.1)
        self.scale = scale

    def forward(self, x: torch.Tensor, factor_u: nn.Module, factor_v: nn.Module) -> torch.Tensor:
        qkv = self.qkv(x)
        query_update = factor_v(self.dropout_query(self.query_factor(factor_u(x))))
        value_update = factor_v(self.dropout_value(self.value_factor(factor_u(x))))
        qkv = qkv.clone()
        qkv[..., : self.dim] += query_update * self.scale
        qkv[..., -self.dim :] += value_update * self.scale
        return qkv


class _FactBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    @staticmethod
    def _adapter(x: torch.Tensor, block: nn.Module, suffix: str, depth: int) -> torch.Tensor:
        batch, height, width, _ = x.shape
        if batch % depth:
            raise ValueError(f"The flattened slice batch ({batch}) must be divisible by d_size ({depth}).")
        normalizer = getattr(block, f"adapter_norm{suffix}")
        down = getattr(block, f"adapter_linear_down{suffix}")
        convolution = getattr(block, f"adapter_conv{suffix}")
        activation = getattr(block, f"adapter_act{suffix}")
        up = getattr(block, f"adapter_linear_up{suffix}")
        shortcut = x
        x = down(normalizer(x))
        x = x.contiguous().view(batch // depth, depth, height, width, block.adapter_channels)
        x = convolution(x.permute(0, 4, 1, 2, 3))
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch, height, width, block.adapter_channels)
        return shortcut + up(activation(x))

    def forward(self, x: torch.Tensor, factor_u: nn.Module, factor_v: nn.Module, depth: int) -> torch.Tensor:
        x = self._adapter(x, self.block, "", depth)
        shortcut = x
        x = self.block.norm1(x)
        if self.block.window_size > 0:
            height, width = x.shape[1:3]
            x, padded_hw = window_partition(x, self.block.window_size)
        x = self.block.attn(x, factor_u, factor_v)
        if self.block.window_size > 0:
            x = window_unpartition(x, self.block.window_size, padded_hw, (height, width))
        x = shortcut + x
        x = self._adapter(x, self.block, "_2", depth)
        return x + self.block.mlp(self.block.norm2(x))


class _FactImageEncoderHQ(nn.Module):
    def __init__(self, image_encoder: nn.Module, factor_u: nn.Module, factor_v: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.factor_u = factor_u
        self.factor_v = factor_v
        self.img_size = image_encoder.img_size

    def forward_features(self, x: torch.Tensor, d_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed
        early_feature = None
        for index, block in enumerate(self.image_encoder.blocks):
            x = block(x, self.factor_u, self.factor_v, d_size)
            if index == 0:
                early_feature = x
        if early_feature is None:
            raise RuntimeError("The image encoder has no transformer blocks.")
        early_feature = self.image_encoder.ln_early(self.image_encoder.proj_early(early_feature.permute(0, 3, 1, 2)))
        last_feature = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        return last_feature, early_feature, last_feature

    def forward(self, x: torch.Tensor, d_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_features(x, d_size)


class Fact_tt_Sam_hq(nn.Module):
    """Apply rank-``r`` FacT and the released 2.5D adapters to a MorVess SAM model."""

    def __init__(self, sam_model: nn.Module, r: int, fact_layer: list[int] | None = None, s: float = 1.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("FacT rank must be positive.")

        base_dimension = sam_model.image_encoder.patch_embed.proj.out_channels
        self.fact_layer = fact_layer or list(range(len(sam_model.image_encoder.blocks)))
        self.query_factors: list[nn.Module] = []
        self.value_factors: list[nn.Module] = []
        self.factor_u = nn.Linear(base_dimension, r, bias=False)
        self.factor_v = nn.Linear(r, base_dimension, bias=False)
        nn.init.zeros_(self.factor_v.weight)

        for name, parameter in sam_model.named_parameters():
            parameter.requires_grad = name.startswith("mask_decoder")
        for name, parameter in sam_model.image_encoder.named_parameters():
            parameter.requires_grad = ".adapter_" in name

        for layer_index, block in enumerate(sam_model.image_encoder.blocks):
            if layer_index not in self.fact_layer:
                continue
            query_factor = nn.Linear(r, r, bias=False)
            value_factor = nn.Linear(r, r, bias=False)
            self.query_factors.append(query_factor)
            self.value_factors.append(value_factor)
            original_attention = block.attn
            original_attention.qkv = _FactQKV(original_attention.qkv, query_factor, value_factor, s)
            block.attn = _FactAttention(original_attention)
            sam_model.image_encoder.blocks[layer_index] = _FactBlock(block)

        sam_model.image_encoder = _FactImageEncoderHQ(sam_model.image_encoder, self.factor_u, self.factor_v)
        self.sam = sam_model

    def forward(self, batched_input: torch.Tensor, multimask_output: bool, image_size: int):
        return self.sam(batched_input, multimask_output, image_size)

    def save_parameters(self, filename: str) -> None:
        if not filename.endswith((".pt", ".pth")):
            raise ValueError("Checkpoints must end in .pt or .pth.")
        state = self.sam.state_dict()
        checkpoint = {
            f"q_FacTs_{index:03d}": module.weight.detach().cpu()
            for index, module in enumerate(self.query_factors)
        }
        checkpoint.update(
            {
                f"v_FacTs_{index:03d}": module.weight.detach().cpu()
                for index, module in enumerate(self.value_factors)
            }
        )
        for key, value in state.items():
            if any(token in key for token in ("mask_decoder", ".adapter_", "factor_u", "factor_v")):
                checkpoint[key] = value.detach().cpu()
        torch.save(checkpoint, filename)

    def load_parameters(self, filename: str) -> None:
        if not filename.endswith((".pt", ".pth")):
            raise ValueError("Checkpoints must end in .pt or .pth.")
        checkpoint = torch.load(filename, map_location="cpu")
        for index, module in enumerate(self.query_factors):
            key = f"q_FacTs_{index:03d}"
            if key in checkpoint:
                module.weight = Parameter(checkpoint[key].to(module.weight.dtype))
        for index, module in enumerate(self.value_factors):
            key = f"v_FacTs_{index:03d}"
            if key in checkpoint:
                module.weight = Parameter(checkpoint[key].to(module.weight.dtype))

        current = self.sam.state_dict()
        compatible = {
            key: value
            for key, value in checkpoint.items()
            if key in current and getattr(value, "shape", None) == current[key].shape
        }
        missing, unexpected = self.sam.load_state_dict(compatible, strict=False)
        if not compatible:
            raise RuntimeError("The checkpoint contains no parameters compatible with the MorVess model.")
