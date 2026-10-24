"""End-to-End Inference Pipeline: Connecting Pipeline A (BBox Detector) with Pipeline B (Classifier)."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from .model import BBoxMultimodalTransformer


CATEGORY_NAMES = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


class DocumentLayoutFullPipeline:
    """Pipeline hoàn chỉnh 2 giai đoạn (2-Stage Document Layout Pipeline):
    
    - Giai đoạn 1 (Pipeline A): YOLOv8n siêu nhẹ chỉ chuyên định vị và bao quát Bounding Box.
    - Giai đoạn 2 (Pipeline B): Multimodal Transformer dùng đồng thời Ảnh Crop + Chữ + Vị trí 2D
      và cơ chế Self-Attention liên hộp để phân loại nhãn layout chuẩn xác.
    """

    def __init__(
        self,
        detector_weights: Union[str, Path],
        classifier_weights: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_classes: int = 11,
    ):
        self.device = device
        print(f"[Pipeline] Khởi tạo Pipeline 2 giai đoạn trên thiết bị: {device.upper()}")

        # 1. Nạp mô hình Pipeline A: BBox Detector
        self.detector = YOLO(str(detector_weights))

        # 2. Nạp mô hình Pipeline B: Multimodal Transformer Classifier
        self.classifier = BBoxMultimodalTransformer(num_classes=num_classes).to(device)
        if classifier_weights and Path(classifier_weights).is_file():
            state_dict = torch.load(str(classifier_weights), map_location=device)
            self.classifier.load_state_dict(state_dict)
            print(f"[Pipeline B] Đã nạp weights thành công từ: {classifier_weights}")
        else:
            print("[Pipeline B] Khởi tạo mô hình Transformer (chưa nạp weights fine-tuned)")

        self.classifier.eval()

        self.crop_transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_page(
        self,
        image_input: Union[str, Path, Image.Image],
        conf_thresh: float = 0.25,
        iou_nms: float = 0.60,
    ) -> Dict[str, Any]:
        """Dự đoán toàn diện một trang tài liệu qua cả 2 Pipeline."""
        start_t = time.time()

        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")
        img_w, img_h = img.size

        # =========================================================================
        # BƯỚC 1: PIPELINE A - PHÁT HIỆN BOUNDING BOX
        # =========================================================================
        t_det0 = time.time()
        det_results = self.detector(img, conf=conf_thresh, iou=iou_nms, verbose=False, device=self.device)[0]
        det_time = time.time() - t_det0

        candidate_boxes = []
        if len(det_results.boxes) > 0:
            xyxy = det_results.boxes.xyxy.cpu().numpy()
            scores = det_results.boxes.conf.cpu().numpy()
            for b, s in zip(xyxy, scores):
                x1, y1, x2, y2 = b
                w = max(1.0, float(x2 - x1))
                h = max(1.0, float(y2 - y1))
                candidate_boxes.append({
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(s),
                })

        if not candidate_boxes:
            return {
                "width": img_w,
                "height": img_h,
                "inference_time_s": time.time() - start_t,
                "detections": [],
            }

        # =========================================================================
        # BƯỚC 2: TIỀN XỬ LÝ ĐA PHƯƠNG THỨC CHO TỪNG BBOX
        # =========================================================================
        crops_list = []
        spatial_list = []
        text_embeds_list = []

        for item in candidate_boxes:
            x, y, w, h = item["bbox"]
            x0 = max(0.0, float(x))
            y0 = max(0.0, float(y))
            x1 = min(float(img_w), x0 + w)
            y1 = min(float(img_h), y0 + h)

            # 1. Crop ảnh
            crop_patch = img.crop((int(x0), int(y0), max(int(x0) + 1, int(x1)), max(int(y0) + 1, int(y1))))
            crops_list.append(self.crop_transform(crop_patch))

            # 2. Tọa độ hình học 2D
            norm_x0 = x0 / img_w
            norm_y0 = y0 / img_h
            norm_x1 = x1 / img_w
            norm_y1 = y1 / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            aspect_ratio = min(5.0, w / max(h, 1e-4))
            area_ratio = (w * h) / (img_w * img_h)
            spatial_list.append([norm_x0, norm_y0, norm_x1, norm_y1, norm_w, norm_h, aspect_ratio, area_ratio])

            # 3. Vector chữ (giả định 384-dim, có thể tích hợp RapidOCR / MiniLM)
            text_embeds_list.append(torch.zeros(384, dtype=torch.float32))

        # Gom thành batch tensors cho 1 trang (B=1, N=num_boxes)
        tensor_crops = torch.stack(crops_list).unsqueeze(0).to(self.device)  # (1, N, 3, 64, 128)
        tensor_spatial = torch.tensor(spatial_list, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, N, 8)
        tensor_text = torch.stack(text_embeds_list).unsqueeze(0).to(self.device)  # (1, N, 384)

        # =========================================================================
        # BƯỚC 3: PIPELINE B - TRANSFORMER PHÂN LOẠI LIÊN HỘP (SELF-ATTENTION)
        # =========================================================================
        t_cls0 = time.time()
        with torch.no_grad():
            logits = self.classifier(
                crops=tensor_crops,
                text_embeddings=tensor_text,
                spatial_features=tensor_spatial,
                mask=None,
            )  # (1, N, num_classes)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_classes = torch.argmax(probs, dim=-1).cpu().numpy()
            pred_scores = probs.cpu().numpy()
        cls_time = time.time() - t_cls0

        # =========================================================================
        # BƯỚC 4: KẾT HỢP VÀ ĐÓNG GÓI KẾT QUẢ ĐẦU RA
        # =========================================================================
        final_detections = []
        for i, item in enumerate(candidate_boxes):
            class_idx = int(pred_classes[i])
            cat_id = class_idx + 1
            cat_name = CATEGORY_NAMES.get(cat_id, "Text")
            class_conf = float(pred_scores[i, class_idx])
            final_conf = float(item["score"] * class_conf)

            final_detections.append({
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": item["bbox"],
                "score": round(final_conf, 4),
            })

        total_time = time.time() - start_t
        return {
            "width": img_w,
            "height": img_h,
            "inference_time_s": total_time,
            "det_time_s": det_time,
            "cls_time_s": cls_time,
            "detections": final_detections,
        }
