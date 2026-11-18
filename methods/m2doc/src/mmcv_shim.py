"""MMCV Shim for M2Doc and DINO + M2Doc on Python 3.12 / Windows environments.

Provides pure PyTorch fallback shims for mmcv.ops:
- MultiScaleDeformableAttention & multi_scale_deformable_attn_pytorch (via F.grid_sample)
- batched_nms, nms, roi_align, RoIAlign (via torchvision.ops)
Allowing M2Doc, DINO + M2Doc (DINO_w_M2Doc) and MMDetection to run and fine-tune
seamlessly on Windows without C++ compiler toolchains or binary mmcv._ext wheels.
"""

import math
import sys
import types
import warnings
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops


class AutoModule(types.ModuleType):
    """Dynamic module placeholder that safely returns dummy functions for unneeded C++ ops."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__file__ = f"{name}.py"
        self.__path__ = []

    def __getattr__(self, name: str):
        if name in ("__file__", "__name__", "__path__", "__spec__", "__package__"):
            raise AttributeError(name)

        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


class OpsLoader:
    def __init__(self, mod: types.ModuleType):
        self.mod = mod

    def create_module(self, spec):
        return self.mod

    def exec_module(self, module):
        pass


class OpsFinder:
    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None):
        if fullname.startswith("mmcv.ops"):
            if fullname in sys.modules:
                mod = sys.modules[fullname]
            else:
                mod = AutoModule(fullname)
                sys.modules[fullname] = mod
            spec = ModuleSpec(fullname, OpsLoader(mod), is_package=True)
            return spec
        return None


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Pure PyTorch/torchvision implementation of batched NMS."""
    iou_threshold = nms_cfg.get("iou_threshold", 0.5)
    keep = tv_ops.batched_nms(boxes, scores, idxs, iou_threshold)
    max_num = nms_cfg.get("max_num", -1)
    if max_num > 0:
        keep = keep[:max_num]
    dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
    return dets, keep


def nms(boxes, scores, iou_threshold):
    """Pure PyTorch/torchvision implementation of standard NMS."""
    return tv_ops.nms(boxes, scores, iou_threshold)


def roi_align(input, rois, output_size, spatial_scale=1.0, sampling_ratio=0, pool_mode="avg", aligned=True):
    """Torchvision backed RoIAlign implementation."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return tv_ops.roi_align(
        input,
        rois,
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )


class RoIAlign(nn.Module):
    """PyTorch RoIAlign layer wrapping torchvision.ops.roi_align."""

    def __init__(
        self,
        output_size,
        spatial_scale=1.0,
        sampling_ratio=0,
        pool_mode="avg",
        aligned=True,
        use_torchvision=False,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = aligned

    def forward(self, input, rois):
        return roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            aligned=self.aligned,
        )


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch (CPU & CUDA compatible) version of multi-scale deformable attention."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_int, W_int = int(H_), int(W_)
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_int, W_int)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_.to(value_l_.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights.to(value.dtype))
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Pure PyTorch MultiScaleDeformableAttention compatible with MMCV/MMDetection DINO checkpoints."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        value_proj_ratio: float = 1.0,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        self._is_init = True

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


_SHIM_INITIALIZED = False


def setup_m2doc_environment(mmdet_dir: Optional[str] = None):
    """Initializes the MMCV shims (including MultiScaleDeformableAttention) and registers M2Doc mmdetection path."""
    global _SHIM_INITIALIZED
    if _SHIM_INITIALIZED:
        return

    if OpsFinder not in sys.meta_path:
        sys.meta_path.insert(0, OpsFinder)

    ops_mod = AutoModule("mmcv.ops")
    nms_mod = AutoModule("mmcv.ops.nms")
    roi_align_mod = AutoModule("mmcv.ops.roi_align")
    msda_mod = AutoModule("mmcv.ops.multi_scale_deform_attn")

    nms_mod.batched_nms = batched_nms
    nms_mod.nms = nms
    ops_mod.nms = nms_mod
    ops_mod.batched_nms = batched_nms

    roi_align_mod.roi_align = roi_align
    roi_align_mod.RoIAlign = RoIAlign
    ops_mod.roi_align = roi_align
    ops_mod.RoIAlign = RoIAlign

    msda_mod.MultiScaleDeformableAttention = MultiScaleDeformableAttention
    msda_mod.multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_pytorch
    ops_mod.MultiScaleDeformableAttention = MultiScaleDeformableAttention
    ops_mod.multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_pytorch

    sys.modules["mmcv.ops"] = ops_mod
    sys.modules["mmcv.ops.nms"] = nms_mod
    sys.modules["mmcv.ops.roi_align"] = roi_align_mod
    sys.modules["mmcv.ops.multi_scale_deform_attn"] = msda_mod

    try:
        import mmcv

        mmcv.__version__ = "2.1.0"
        mmcv.ops = ops_mod
        import mmcv.cnn.bricks.transformer as trans

        trans.MultiScaleDeformableAttention = MultiScaleDeformableAttention
        from mmengine.registry import MODELS

        if "MultiScaleDeformableAttention" not in MODELS.module_dict:
            MODELS.register_module(module=MultiScaleDeformableAttention, force=True)
    except Exception as e:
        warnings.warn(f"[mmcv_shim] Warning initializing mmcv: {e}")

    if mmdet_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        mmdet_dir = str(base_dir / "mmdetection")

    if mmdet_dir not in sys.path:
        sys.path.insert(0, mmdet_dir)

    try:
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
    except Exception:
        pass

    _SHIM_INITIALIZED = True
