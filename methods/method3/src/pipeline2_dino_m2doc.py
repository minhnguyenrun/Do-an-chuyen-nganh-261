"""Pipeline 2: DINO + M2Doc integrated with Book-Level Context Fusion (Method 3).

Tích hợp mô hình đa phương thức DINO + M2Doc (ResNet-50 + BERT Early/Late Fusion + Deformable DINO)
với ngữ cảnh toàn sách C_{book} từ Book-Level Encoder:
- Hỗ trợ Cách A (Feature/Query-level Cross-Attention Fusion): Hòa trộn C_{book} trực tiếp vào
  900 object queries của DINO Decoder trước khi đưa qua DINOHead.
- Hỗ trợ Cách B (Box-level Context Refinement): Tinh chỉnh danh sách bounding boxes, nhãn và
  confidence scores đầu ra của DINO + M2Doc bằng ContextAwareBoxRefiner.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from book_context import (
        ContextAwareBoxRefiner,
        ContextFusionModule,
        HierarchicalBookEncoder,
    )
    from mmcv_shim import setup_m2doc_environment
except ImportError:
    from methods.method3.src.book_context import (
        ContextAwareBoxRefiner,
        ContextFusionModule,
        HierarchicalBookEncoder,
    )
    from methods.method3.src.mmcv_shim import setup_m2doc_environment

# Google Drive ID chính thức của DINO + M2Doc (ResNet-50, 12 epochs, 89.3 mAP trên DocLayNet)
GDRIVE_DINO_M2DOC_R50_ID = "11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF"

M2DOC_CLASSES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

M2DOC_TO_DOCLAYNET = {name: idx + 1 for idx, name in enumerate(M2DOC_CLASSES)}


def download_dino_m2doc_weights(weights_path: Path, gdrive_id: str = GDRIVE_DINO_M2DOC_R50_ID):
    """Tự động tải trọng số DINO + M2Doc (ResNet-50) từ Google Drive nếu chưa tồn tại."""
    if weights_path.exists() and weights_path.stat().st_size > 100_000_000:
        return
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Method 3] Đang tải trọng số DINO + M2Doc từ Google Drive (ID: {gdrive_id})...")
    try:
        import gdown
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    gdown.download(id=gdrive_id, output=str(weights_path), quiet=False)
    if not weights_path.exists() or weights_path.stat().st_size < 100_000_000:
        raise FileNotFoundError(
            f"Tải trọng số DINO + M2Doc thất bại. Vui lòng kiểm tra kết nối hoặc tải thủ công về: {weights_path}"
        )
    print(f"[Method 3] Đã tải xong {weights_path.name} ({weights_path.stat().st_size / (1024**2):.2f} MB)!")


def load_base_dino_m2doc(
    config_path: Optional[Union[str, Path]] = None,
    weights_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> nn.Module:
    """Khởi tạo và nạp trọng số cho mô hình cơ sở DINO_w_M2Doc từ MMDetection."""
    setup_m2doc_environment()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    init_default_scope("mmdet")

    root_dir = Path(__file__).resolve().parents[3]
    if config_path is None:
        config_path = (
            root_dir
            / "methods"
            / "m2doc"
            / "mmdetection"
            / "m2doc_config"
            / "dino-4scale_w_m2doc_r50_doclaynet_12epoch.py"
        )
    config_path = Path(config_path)

    if weights_path is None:
        weights_path = Path(__file__).resolve().parent.parent / "weights" / "dino_m2doc_r50.pth"
    weights_path = Path(weights_path)

    if not weights_path.exists():
        download_dino_m2doc_weights(weights_path)

    cfg = Config.fromfile(str(config_path))
    # Không cần tải lại resnet50 imagenet khi đã có full checkpoint dino_m2doc_r50.pth
    if hasattr(cfg.model, "backbone") and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    model = MODELS.build(cfg.model)
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[Method 3] Đã nạp DINO_w_M2Doc từ {weights_path.name} "
        f"(missing keys: {len(missing)}, unexpected keys: {len(unexpected)})"
    )
    model.to(device)
    model.eval()
    return model


class DINO_M2Doc_BookContextPipeline(nn.Module):
    r"""Mô hình Pipeline 2 hoàn chỉnh: DINO + M2Doc kết hợp Book-Level Encoder & Context Fusion.

    Hỗ trợ 2 cơ chế tích hợp ngữ cảnh liên trang:
    - fusion_mode = "feature" (Cách A): Đưa ngữ cảnh toàn sách C_{book} vào đặc trưng DINO Decoder
      thông qua Cross-Attention + Residual (\widetilde{F}_i = F_i + \alpha F_i') trước khi DINOHead dự đoán.
    - fusion_mode = "box" (Cách B): Tinh chỉnh trực tiếp các bounding box dự đoán bằng ContextAwareBoxRefiner.
    """

    def __init__(
        self,
        base_detector: nn.Module,
        num_classes: int = 11,
        embed_dim: int = 256,
        num_summary_tokens: int = 4,
        fusion_mode: str = "feature",
    ):
        super().__init__()
        self.detector = base_detector
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode

        # Cầu nối: Book-Level Encoder phân cấp (Box -> Page -> Book)
        self.book_encoder = HierarchicalBookEncoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_summary_tokens=num_summary_tokens,
        )

        # Cách A: Feature/Query-level Context Fusion gắn vào đầu ra Decoder của DINO_w_M2Doc
        self.feature_context_fusion = ContextFusionModule(
            embed_dim=embed_dim,
            nhead=8,
            init_alpha=0.0,
        )

        # Cách B: Box-level Context Refinement
        self.box_context_refiner = ContextAwareBoxRefiner(
            num_classes=num_classes,
            embed_dim=embed_dim,
            nhead=8,
        )

        self._active_book_context: Optional[torch.Tensor] = None
        self._active_page_context: Optional[torch.Tensor] = None
        self._hook_Step = None
        self._install_decoder_fusion_hook()

    def _install_decoder_fusion_hook(self):
        r"""Gắn forward hook vào tầng cuối của DINO Decoder để thực thi Cách A:
            \widetilde{F}_i = ContextFusion(F_i, C_{book})
            \hat{B}_i = DINOHead(\widetilde{F}_i)
        """
        if not hasattr(self.detector, "decoder"):
            return

        orig_forward_decoder = self.detector.forward_decoder

        def wrapped_forward_decoder(*args, **kwargs):
            decoder_outputs_dict = orig_forward_decoder(*args, **kwargs)
            if self.fusion_mode in ("feature", "both") and self._active_book_context is not None:
                hidden_states = decoder_outputs_dict["hidden_states"]
                # hidden_states có kích thước (num_decoder_layers=6, bs, num_queries=900, 256)
                last_state = hidden_states[-1]
                fused_last = self.feature_context_fusion(
                    last_state,
                    self._active_book_context,
                    self._active_page_context,
                )
                # Cập nhật tầng cuối trước khi truyền sang DINOHead
                updated_states = torch.cat([hidden_states[:-1], fused_last.unsqueeze(0)], dim=0)
                decoder_outputs_dict["hidden_states"] = updated_states
            return decoder_outputs_dict

        self.detector.forward_decoder = wrapped_forward_decoder

    def configure_trainable_params(self, strategy: str = "context_only") -> Dict[str, int]:
        """Thiết lập chiến lược đóng băng / mở khóa trọng số theo Mục 5 (Phù hợp 8GB VRAM):
        - "context_only" (Bước 2): Đóng băng toàn bộ DINO+M2Doc cơ sở, chỉ train Book Encoder & Context Fusion.
        - "partial_joint" (Bước 3): Mở khóa thêm DINOHead và tầng Late-Fusion cuối của DINO Decoder.
        - "full" (Fine-tune toàn bộ): Mở khóa toàn bộ mô hình.
        """
        # Mặc định đóng băng toàn bộ base detector trước
        for param in self.detector.parameters():
            param.requires_grad = False

        # Luôn mở khóa các module ngữ cảnh mới
        for param in self.book_encoder.parameters():
            param.requires_grad = True
        for param in self.feature_context_fusion.parameters():
            param.requires_grad = True
        for param in self.box_context_refiner.parameters():
            param.requires_grad = True

        if strategy == "partial_joint":
            if hasattr(self.detector, "bbox_head"):
                for param in self.detector.bbox_head.parameters():
                    param.requires_grad = True
            if hasattr(self.detector, "decoder"):
                if hasattr(self.detector.decoder, "line_proj"):
                    self.detector.decoder.line_proj.requires_grad = True
                if hasattr(self.detector.decoder, "layers") and len(self.detector.decoder.layers) > 0:
                    for param in self.detector.decoder.layers[-1].parameters():
                        param.requires_grad = True
        elif strategy == "full":
            for param in self.detector.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {"total_params": total, "trainable_params": trainable, "frozen_params": frozen}

    def encode_book_context(
        self,
        book_page_reps: List[Dict[str, Any]],
        device: Union[str, torch.device] = "cpu",
        max_boxes_per_page: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nén danh sách dự đoán R_1, ..., R_N từ Pipeline 1 thành ngữ cảnh toàn sách C_{book}."""
        N_pages = len(book_page_reps)
        max_m = max(min(max(( len(r["labels"]) for r in book_page_reps ), default=1), max_boxes_per_page), 1)

        spatial_tensor = torch.zeros((N_pages, max_m, 8), dtype=torch.float32, device=device)
        labels_tensor = torch.full((N_pages, max_m), self.num_classes, dtype=torch.long, device=device)
        scores_tensor = torch.zeros((N_pages, max_m), dtype=torch.float32, device=device)
        pad_mask = torch.ones((N_pages, max_m), dtype=torch.bool, device=device)
        page_indices = torch.arange(N_pages, dtype=torch.long, device=device)
        total_pages = torch.full((N_pages,), N_pages, dtype=torch.long, device=device)

        for i, rep in enumerate(book_page_reps):
            m = min(len(rep["labels"]), max_m)
            if m > 0:
                spatial_tensor[i, :m] = torch.as_tensor(rep["spatial_features"][:m], dtype=torch.float32, device=device)
                labels_tensor[i, :m] = torch.as_tensor(rep["labels"][:m], dtype=torch.long, device=device)
                scores_tensor[i, :m] = torch.as_tensor(rep["scores"][:m], dtype=torch.float32, device=device)
                pad_mask[i, :m] = False

        c_book, c_pages = self.book_encoder(
            spatial_features=spatial_tensor,
            labels=labels_tensor,
            scores=scores_tensor,
            page_indices=page_indices,
            padding_mask=pad_mask,
            total_pages=total_pages,
        )
        return c_book, c_pages

    def predict_page_with_context(
        self,
        image_path: Union[str, Path],
        book_context: torch.Tensor,
        page_context_vec: torch.Tensor,
        ocr_provider: Optional[Any] = None,
        threshold: float = 0.3,
        use_ocr: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Chạy suy luận Pipeline 2 (DINO + M2Doc + Context Fusion) cho trang thứ i."""
        from mmengine.structures import InstanceData
        from mmdet.structures import DetDataSample

        image_path = Path(image_path)
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")

        orig_h, orig_w = img.shape[:2]
        target_size = (1025, 1025)
        resized_img = cv2.resize(img, target_size) if (orig_w, orig_h) != target_size else img
        scale_w = target_size[0] / orig_w
        scale_h = target_size[1] / orig_h

        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float().to(device)
        image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem

        data_sample = DetDataSample()
        data_sample.set_metainfo(
            {
                "img_id": image_id,
                "img_path": str(image_path),
                "ori_shape": (orig_h, orig_w),
                "img_shape": target_size,
                "scale_factor": (scale_w, scale_h),
            }
        )

        text_instances = InstanceData()
        if use_ocr and ocr_provider is not None:
            ocr_boxes, ocr_texts = ocr_provider.get_ocr_for_image(
                str(image_path),
                image_id=image_id if isinstance(image_id, int) else None,
                target_size=target_size,
                orig_size=(orig_w, orig_h),
            )
            if ocr_boxes and ocr_texts:
                text_instances["text_bboxes"] = torch.tensor(ocr_boxes, dtype=torch.float32, device=device)
                text_instances["texts"] = ocr_texts
            else:
                text_instances["text_bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
                text_instances["texts"] = []
        else:
            text_instances["text_bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
            text_instances["texts"] = []

        data_sample.text_instances = text_instances
        batch_data = {"inputs": [img_tensor], "data_samples": [data_sample]}

        # Kích hoạt ngữ cảnh toàn sách cho forward hook của DINO Decoder
        self._active_book_context = book_context
        self._active_page_context = page_context_vec.unsqueeze(0) if page_context_vec.dim() == 1 else page_context_vec

        try:
            with torch.no_grad():
                prepped = self.detector.data_preprocessor(batch_data, False)
                results = self.detector.predict(prepped["inputs"], prepped["data_samples"], rescale=False)
        finally:
            self._active_book_context = None
            self._active_page_context = None

        pred_instances = results[0].pred_instances
        pred_bboxes = pred_instances.bboxes.detach().cpu().numpy()
        pred_scores = pred_instances.scores.detach().cpu().numpy()
        pred_labels = pred_instances.labels.detach().cpu().numpy()

        # Nếu dùng Cách B (Box-level refinement), tinh chỉnh tiếp danh sách box có confidence > threshold * 0.5
        keep_mask = pred_scores >= (threshold * 0.5 if self.fusion_mode in ("box", "both") else threshold)
        pred_bboxes = pred_bboxes[keep_mask]
        pred_scores = pred_scores[keep_mask]
        pred_labels = pred_labels[keep_mask]

        if self.fusion_mode in ("box", "both") and len(pred_bboxes) > 0:
            try:
                from pipeline1_docling import DoclingPageScout
            except ImportError:
                from methods.method3.src.pipeline1_docling import DoclingPageScout

            boxes_xywh_scaled = [
                [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in pred_bboxes
            ]
            sp_feats = DoclingPageScout.compute_spatial_features(
                boxes_xywh_scaled, target_size[0], target_size[1]
            )
            sp_t = torch.as_tensor(sp_feats, dtype=torch.float32, device=device).unsqueeze(0)
            lb_t = torch.as_tensor(pred_labels, dtype=torch.long, device=device).unsqueeze(0)
            sc_t = torch.as_tensor(pred_scores, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                ref_out = self.box_context_refiner(
                    sp_t,
                    lb_t,
                    sc_t,
                    book_context,
                    page_context_vec.unsqueeze(0) if page_context_vec.dim() == 1 else page_context_vec,
                )
                ref_probs = torch.softmax(ref_out["cls_logits"][0], dim=-1).cpu().numpy()
                ref_labels = np.argmax(ref_probs, axis=-1)
                ref_deltas = ref_out["box_deltas"][0].cpu().numpy()
                ref_gate = ref_out["conf_multiplier"][0].cpu().numpy()

            # Kết hợp dự đoán gốc của DINO + M2Doc với tín hiệu refinement từ Book Context
            pred_labels = np.where(ref_probs.max(axis=-1) > 0.65, ref_labels, pred_labels)
            pred_scores = np.clip(pred_scores * (0.5 + 0.5 * ref_gate), 0.0, 1.0)
            pred_bboxes[:, 0] += ref_deltas[:, 0] * target_size[0]
            pred_bboxes[:, 1] += ref_deltas[:, 1] * target_size[1]
            pred_bboxes[:, 2] += ref_deltas[:, 2] * target_size[0]
            pred_bboxes[:, 3] += ref_deltas[:, 3] * target_size[1]

        detections = []
        for bbox, score, label_idx in zip(pred_bboxes, pred_scores, pred_labels):
            if score < threshold:
                continue
            raw_label = M2DOC_CLASSES[int(label_idx)] if int(label_idx) < len(M2DOC_CLASSES) else "Text"
            cat_id = M2DOC_TO_DOCLAYNET.get(raw_label, 10)

            x1 = max(0.0, min(float(bbox[0]) / scale_w, float(orig_w)))
            y1 = max(0.0, min(float(bbox[1]) / scale_h, float(orig_h)))
            x2 = max(0.0, min(float(bbox[2]) / scale_w, float(orig_w)))
            y2 = max(0.0, min(float(bbox[3]) / scale_h, float(orig_h)))

            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w < 2 or h < 2:
                continue

            detections.append(
                {
                    "category_id": cat_id,
                    "category_name": raw_label,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "score": round(float(score), 4),
                }
            )

        return {
            "image_id": image_id,
            "file_name": image_path.name,
            "width": orig_w,
            "height": orig_h,
            "detections": detections,
        }
