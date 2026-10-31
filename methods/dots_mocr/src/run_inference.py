"""Run dots.mocr (3B Multimodal Vision-Language Model) layout detection on a processed dataset."""

import argparse
import json
import os
import sys
import time

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from collections import Counter
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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parser import parse_dots_layout_response
from src.common.profiling import ResourceMonitor, save_runtime_summary
from src.common.splits import apply_split, describe, SPLITS

DEFAULT_MODEL_ID = "dots-studio/dots.mocr"

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
    """Return a complete local model directory, or fail with an actionable error."""
    def is_complete_model(path: Path) -> bool:
        config_path = path / "config.json"
        index_path = path / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if config.get("model_type") != "dots_ocr":
            return False
        shards = set(index.get("weight_map", {}).values())
        return bool(shards) and all((path / shard).is_file() for shard in shards)

    # dots.mocr checkpoints keep model_type="dots_ocr" (same DotsOCRForCausalLM architecture).
    if is_complete_model(weights_path):
        print(f"[dots_mocr] Using complete model weights at: {weights_path}")
        return str(weights_path)

    print(f"[dots_mocr] Complete model weights not found at: {weights_path}")
    print(f"[dots_mocr] Attempting to download from Hugging Face Hub: {model_id}...")
    try:
        from huggingface_hub import snapshot_download

        weights_path.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(weights_path),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        downloaded_path = Path(local_dir)
        if is_complete_model(downloaded_path):
            print(f"[dots_mocr] Downloaded model to: {local_dir}")
            return str(local_dir)
        raise RuntimeError(f"Hugging Face returned an incomplete model directory: {local_dir}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain complete dots.mocr weights from {model_id}: {exc}. "
            "Enable Kaggle Internet or attach a Kaggle input containing the full model config and all "
            "model.safetensors shards."
        ) from exc

def load_dots_mocr_model(model_path_or_id: str, device: str = "cpu"):
    """Load the model in GPU-friendly precision with PyTorch SDPA attention."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        target_device = device if ":" in device else "cuda:0"
        print(
            f"[dots_mocr] GPU={torch.cuda.get_device_name(target_device)}, "
            f"VRAM={torch.cuda.get_device_properties(target_device).total_memory / 2**30:.1f} GiB, "
            f"dtype={dtype}"
        )
    else:
        dtype = torch.float32
        target_device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=dtype,
        device_map=target_device if use_cuda else None,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if not use_cuda:
        model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path_or_id, trust_remote_code=True)
    return model, processor

def run_on_images(
    model,
    processor,
    image_paths: List[Path],
    device: str = "cpu",
    max_new_tokens: int = 2048,
    dataset: str = "doclaynet",
) -> List[Dict]:
    """Run one padded multimodal generation batch and return one result per image."""
    if not image_paths:
        return []
    pil_images = [Image.open(path).convert("RGB") for path in image_paths]
    original_sizes = [image.size for image in pil_images]
    prompt = PROMPT_LAYOUT_PUBLAYNET if dataset.lower() == "publaynet" else PROMPT_LAYOUT_DOCLAYNET
    conversations = [
        [{"role": "user", "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": prompt}]}]
        for path in image_paths
    ]
    texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    input_sizes = [None] * len(image_paths)
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(conversations)
        input_sizes = [image.size if hasattr(image, "size") else None for image in image_inputs]
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    except Exception:
        inputs = processor(text=texts, images=pil_images, padding=True, return_tensors="pt")

    if "image_grid_thw" in inputs:
        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
        for i, grid in enumerate(inputs["image_grid_thw"][:len(input_sizes)]):
            if input_sizes[i] is None:
                input_sizes[i] = (int(grid[2].item()) * patch_size, int(grid[1].item()) * patch_size)

    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    inputs.pop("mm_token_type_ids", None)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=[151643, 151645, 151672, 151673],
            pad_token_id=151643,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]
    output_texts = processor.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True)
    results = []
    for path, (width, height), input_size, output_text in zip(image_paths, original_sizes, input_sizes, output_texts):
        input_w, input_h = input_size if input_size is not None else (width, height)
        detections = parse_dots_layout_response(
            output_text.strip(), img_width=width, img_height=height, dataset=dataset,
            input_width=input_w, input_height=input_h,
        )
        image_id = int(path.stem) if path.stem.isdigit() else path.stem
        results.append({
            "image_id": image_id, "file_name": path.name, "width": width, "height": height,
            "detections": detections,
        })
    return results


def run_on_image(model, processor, image_path: Path, device: str = "cpu", max_new_tokens: int = 2048, dataset: str = "doclaynet") -> Dict:
    """Backward-compatible single-image inference wrapper."""
    return run_on_images(model, processor, [image_path], device, max_new_tokens, dataset)[0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument(
        "--split",
        default=None,
        choices=sorted(SPLITS) + ["val", "validation"],
        help="Tap du lieu DocLayNet: train / valid / test (thay cho --dataset doclaynet / doclaynet_val / doclaynet_test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default).",
    )
    parser.add_argument("--image-ids", default="", help="Optional comma-separated image IDs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images per GPU generation batch (default: 1: on an 8 GB GPU batching is slower because it spills VRAM; raise it on larger GPUs).",
    )
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
    args.dataset = apply_split(args.dataset, args.split)
    print(f"[dots_mocr] Tap du lieu: {describe(args.dataset)}", flush=True)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    # Standard images directory
    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    # Standard output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "outputs" / "predictions" / "dots_mocr" / args.dataset
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
        f"[dots_mocr] {len(image_paths)} image(s) to predict from {images_dir} (device: {args.device})"
        f"{' (existing predictions skipped)' if args.skip_existing else ' (force overwrite)'}",
        flush=True,
    )

    if not image_paths:
        print(f"[dots_mocr] No work to do. Predictions are in {output_dir}", flush=True)
        return

    # Resolve weights and load model with resource monitoring
    monitor = ResourceMonitor()
    with monitor.time_model_load():
        resolved_weights = ensure_model_weights(Path(args.weights))
        model, processor = load_dots_mocr_model(resolved_weights, device=args.device)
    print(
        f"[dots_mocr] Model dots.mocr ready for inference in {monitor.model_load_seconds:.2f}s!",
        flush=True,
    )

    success_count = 0
    total_seconds = 0.0
    pending = [image_paths[i:i + args.batch_size] for i in range(0, len(image_paths), args.batch_size)]
    processed = 0

    while pending:
        batch = pending.pop(0)
        try:
            with monitor.time_page() as timer:
                results = run_on_images(
                    model=model,
                    processor=processor,
                    image_paths=batch,
                    device=args.device,
                    dataset=args.dataset,
                )
            elapsed = timer.seconds
            page_seconds = elapsed / len(batch)
            monitor.page_seconds[-1:] = [page_seconds] * len(batch)
            total_seconds += elapsed

            for img_path, res in zip(batch, results):
                res["inference_time_s"] = round(page_seconds, 4)
                out_file = output_dir / f"{img_path.stem}.json"
                out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                processed += 1
                success_count += 1
                cat_counts = Counter(d.get("category_name", d.get("category", "unknown")) for d in res["detections"])
                cats_str = ", ".join(f"{k}: {v}" for k, v in cat_counts.most_common(4))
                summary_tail = f" [{cats_str}]" if cats_str else " [no boxes]"
                avg_so_far = total_seconds / success_count
                remaining_sec = avg_so_far * (len(image_paths) - processed)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))
                print(
                    f"  [{processed}/{len(image_paths)}] {img_path.name}: "
                    f"{len(res['detections'])} boxes ({page_seconds:.2f}s/page | ETA: {eta_str}){summary_tail}",
                    flush=True,
                )
        except torch.cuda.OutOfMemoryError:
            if monitor.page_seconds:
                monitor.page_seconds.pop()
            if len(batch) == 1:
                print(f"  ERROR processing {batch[0].name}: CUDA out of memory even at batch size 1", flush=True)
                continue
            torch.cuda.empty_cache()
            midpoint = max(1, len(batch) // 2)
            pending[0:0] = [batch[:midpoint], batch[midpoint:]]
            print(f"[dots_mocr] GPU memory pressure; retrying batch of {len(batch)} as smaller batches.", flush=True)
        except Exception as e:
            if monitor.page_seconds:
                monitor.page_seconds.pop()
            if len(batch) > 1:
                midpoint = max(1, len(batch) // 2)
                pending[0:0] = [batch[:midpoint], batch[midpoint:]]
                print(f"[dots_mocr] Batch failed ({e}); retrying in smaller batches.", flush=True)
            else:
                print(f"  ERROR processing {batch[0].name}: {e}", flush=True)
                processed += 1

    if monitor.page_seconds:
        summary = monitor.summary(dataset=args.dataset)
        save_runtime_summary("dots_mocr", args.dataset, summary)

    avg_speed = (total_seconds / success_count) if success_count > 0 else 0.0
    print(
        f"[dots_mocr] Done {success_count}/{len(image_paths)} images. Average {avg_speed:.2f}s/image. "
        f"Predictions: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
