"""Run dots.ocr (1.7B Multilingual Vision-Language Model) layout detection on a processed dataset."""

import argparse
import json
import os
import sys
import time

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

# Ensure local src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parents[1]

# Fix Windows console UTF-8 encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parser import parse_dots_layout_response

DEFAULT_MODEL_ID = "dots-studio/dots.ocr"

PROMPT_LAYOUT_DOCLAYNET = (
    "Please output the layout information from this PDF image, including each layout's bbox and its category. "
    "The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', "
    "'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."
)

PROMPT_LAYOUT_PUBLAYNET = (
    "Please output the layout information from this document image, including each layout's bbox and its category. "
    "The bbox should be in the format [x1, y1, x2, y2]. The layout categories include "
    "['Text', 'Title', 'List', 'Table', 'Figure']. Do not output the corresponding text. The layout result should be in JSON format."
)


def ensure_model_weights(weights_path: Path, model_id: str = DEFAULT_MODEL_ID) -> str:
    """Ensures model weights exist locally or prepares to download them from Hugging Face."""
    has_safetensors = any(weights_path.glob("*.safetensors")) if weights_path.is_dir() else False
    if weights_path.is_dir() and (weights_path / "config.json").exists() and has_safetensors:
        print(f"[dots_ocr] Sử dụng trọng số có sẵn tại: {weights_path}")
        return str(weights_path)

    print(f"[dots_ocr] Chưa tìm thấy trọng số đầy đủ tại: {weights_path}")
    print(f"[dots_ocr] Đang chuẩn bị tải từ Hugging Face Hub: {model_id}...")
    try:
        from huggingface_hub import snapshot_download

        weights_path.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(weights_path),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"[dots_ocr] Tải model thành công về: {local_dir}")
        return str(local_dir)
    except Exception as e:
        print(f"[dots_ocr] Cảnh báo khi tải local snapshot: {e}. Thử nạp trực tiếp qua model ID.")
        return model_id


def load_dots_ocr_model(model_path_or_id: str, device: str = "cpu"):
    """Loads dots.ocr AutoModel and AutoProcessor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[dots_ocr] Khởi tạo mô hình từ: {model_path_or_id} (device={device})")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch, "bfloat16") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        try:
            model.to(device)
        except Exception:
            pass
    else:
        model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path_or_id, trust_remote_code=True)
    return model, processor


def run_on_image(
    model,
    processor,
    image_path: Path,
    device: str = "cpu",
    max_new_tokens: int = 768,
    dataset: str = "doclaynet",
) -> Dict:
    """Runs layout extraction on a single image and returns standard detection format."""
    pil_img = Image.open(image_path).convert("RGB")
    width, height = pil_img.size

    prompt = PROMPT_LAYOUT_PUBLAYNET if dataset.lower() == "publaynet" else PROMPT_LAYOUT_DOCLAYNET

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_w, input_h = None, None
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs and hasattr(image_inputs[0], "size"):
            input_w, input_h = image_inputs[0].size
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        # Direct processor call fallback with chat template
        inputs = processor(
            text=[text],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        )

    if (input_w is None or input_h is None) and "image_grid_thw" in inputs:
        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
        grid = inputs["image_grid_thw"][0]
        input_h = int(grid[1].item()) * patch_size
        input_w = int(grid[2].item()) * patch_size

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs.pop("mm_token_type_ids", None)

    eos_token_id = [151643, 151645, 151673]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=151643,
            do_sample=False,
        )

    # Decode response tokens
    in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    out_ids = generated_ids[0][in_len:]
    output_text = processor.decode(out_ids, skip_special_tokens=True).strip()

    # Parse JSON detections into standard format, rescaling from (input_w, input_h) to (width, height)
    detections = parse_dots_layout_response(
        output_text,
        img_width=width,
        img_height=height,
        dataset=dataset,
        input_width=input_w,
        input_height=input_h,
    )

    image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
    return {
        "image_id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default).",
    )
    parser.add_argument("--image-ids", default="", help="Optional comma-separated image IDs")
    parser.add_argument(
        "--weights",
        default=str(METHOD_DIR / "weights"),
        help="Path to local weights directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output predictions directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: 'cuda' or 'cpu'",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip images that already have a prediction (default).",
    )
    run_mode.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Run inference again and overwrite existing predictions.",
    )
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    # Standard images directory
    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    # Standard output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "outputs" / "predictions" / "dots_ocr" / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect target images with natural sorting
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions],
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )

    if args.image_ids:
        selected_ids = {s.strip() for s in args.image_ids.split(",") if s.strip()}
        image_paths = [p for p in image_paths if p.stem in selected_ids or p.name in selected_ids]

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if args.skip_existing:
        image_paths = [
            path for path in image_paths if not (output_dir / f"{path.stem}.json").exists()
        ]

    print(
        f"[dots_ocr] {len(image_paths)} image(s) to predict from {images_dir} (device: {args.device})"
        f"{' (existing predictions skipped)' if args.skip_existing else ' (force overwrite)'}"
    )

    if not image_paths:
        print(f"[dots_ocr] No work to do. Predictions are in {output_dir}")
        return

    # Resolve weights and load model
    resolved_weights = ensure_model_weights(Path(args.weights))
    model, processor = load_dots_ocr_model(resolved_weights, device=args.device)
    print("[dots_ocr] Model dots.ocr ready for inference!")

    success_count = 0
    total_seconds = 0.0

    for i, img_path in enumerate(image_paths, start=1):
        started = time.time()
        try:
            res = run_on_image(
                model=model,
                processor=processor,
                image_path=img_path,
                device=args.device,
                dataset=args.dataset,
            )
            elapsed = time.time() - started
            total_seconds += elapsed

            out_file = output_dir / f"{img_path.stem}.json"
            out_file.write_text(
                json.dumps(res, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            det_count = len(res["detections"])
            print(
                f"  [{i}/{len(image_paths)}] {img_path.name}: "
                f"{det_count} boxes ({elapsed:.2f}s)"
            )
            success_count += 1
        except Exception as e:
            print(f"  [{i}/{len(image_paths)}] ERROR processing {img_path.name}: {e}")

    avg_speed = (total_seconds / success_count) if success_count > 0 else 0.0
    print(
        f"[dots_ocr] Done. Average {avg_speed:.2f}s/image. "
        f"Predictions: {output_dir}"
    )


if __name__ == "__main__":
    main()
