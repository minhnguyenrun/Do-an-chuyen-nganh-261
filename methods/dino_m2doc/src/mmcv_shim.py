"""MMCV Shim for DINO + M2Doc on Python 3.12 / Windows environments.

Provides pure PyTorch fallback shims for mmcv.ops:
- MultiScaleDeformableAttention & multi_scale_deformable_attn_pytorch (via F.grid_sample)
- batched_nms, nms, roi_align, RoIAlign (via torchvision.ops)
Allowing DINO + M2Doc (DINO_w_M2Doc) and MMDetection to run and evaluate
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
        use_torchvision=True,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input, rois):
        return roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.pool_mode,
            self.aligned,
        )


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch implementation of Multi-Scale Deformable Attention via F.grid_sample.

    Args:
        value (Tensor): [bs, num_keys, num_heads, head_dims]
        value_spatial_shapes (Tensor): [num_levels, 2] (H, W for each level)
        sampling_locations (Tensor): [bs, num_queries, num_heads, num_levels, num_points, 2]
        attention_weights (Tensor): [bs, num_queries, num_heads, num_levels, num_points]

    Returns:
        Tensor: [bs, num_queries, num_heads * head_dims]
    """
    bs, _, num_heads, head_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_ = int(H_)
        W_ = int(W_)
        # [bs, H_*W_, num_heads, head_dims] -> [bs*num_heads, head_dims, H_, W_]
        value_l_ = (
            value_list[level]
            .permute(0, 2, 3, 1)
            .reshape(bs * num_heads, head_dims, H_, W_)
        )
        # [bs, num_queries, num_heads, num_points, 2] -> [bs*num_heads, num_queries, num_points, 2]
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level]
            .permute(0, 2, 1, 3, 4)
            .reshape(bs * num_heads, num_queries, num_points, 2)
        )
        # grid_sample: [bs*num_heads, head_dims, num_queries, num_points]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # [bs, num_queries, num_heads, num_levels, num_points] -> [bs*num_heads, 1, num_queries, num_levels*num_points]
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    # [bs*num_heads, head_dims, num_queries, num_levels, num_points]
    stacked_val = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (stacked_val * attention_weights).sum(-1)
    # [bs*num_heads, head_dims, num_queries] -> [bs, num_queries, num_heads * head_dims]
    output = output.view(bs, num_heads, head_dims, num_queries).permute(0, 3, 1, 2)
    return output.reshape(bs, num_queries, num_heads * head_dims)


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module using pure PyTorch fallback."""

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
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.head_dims = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, int(embed_dims * value_proj_ratio))
        self.output_proj = nn.Linear(int(embed_dims * value_proj_ratio), embed_dims)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
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

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

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
            # [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

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
        attention_weights = F.softmax(attention_weights, -1).view(
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
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


_SHIM_INITIALIZED = False


def setup_m2doc_environment(mmdet_dir: Optional[str] = None):
    """Initializes the MMCV shim and mounts M2Doc/MMDetection on sys.path."""
    global _SHIM_INITIALIZED
    if _SHIM_INITIALIZED:
        return

    if not any(isinstance(finder, OpsFinder) for finder in sys.meta_path):
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

    root_dir = Path(__file__).resolve().parents[3]
    if mmdet_dir is None:
        mmdet_dir = str(root_dir / "methods" / "m2doc" / "mmdetection")

    if mmdet_dir not in sys.path:
        sys.path.insert(0, mmdet_dir)

    m2doc_src = str(root_dir / "methods" / "m2doc" / "src")
    if m2doc_src not in sys.path:
        sys.path.insert(0, m2doc_src)

    local_src = str(Path(__file__).resolve().parent)
    if local_src not in sys.path:
        sys.path.insert(0, local_src)

    try:
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
    except Exception:
        pass

    _SHIM_INITIALIZED = True
