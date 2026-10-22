"""Sinh kernel-metadata.json + kernel.py cho tung phuong phap de push len Kaggle.

Moi phuong phap chay trong MOT kernel GPU rieng (tranh xung dot dependency giua
cac method, vd. transformers<4.50 cho dots_ocr/dots_mocr vs transformers>=5.2.0
cho chandra). Kernel se:
  1. git clone repo cong khai tu GitHub.
  2. Tai split `test` cua DocLayNet (4999 anh, dung chung cho moi method).
  3. Cai dependency rieng cua method (+ vai tweak dac thu, xem METHOD_CONFIGS).
  4. Chay run_inference.py --dataset doclaynet_test --limit <N> (N truyen qua bien KAGGLE run).
  5. Don dep repo/dataset/weights nang, chi giu outputs/predictions + outputs/runtime
     trong /kaggle/working/result de Kaggle luu lam kernel output (nhe, keo ve nhanh).

Usage:
    python kaggle/build_kernels.py --limit 20     # sinh kernel ban "validation" (20 anh)
    python kaggle/build_kernels.py --limit 0      # sinh kernel ban "full" (het test split)
"""

import argparse
import json
from pathlib import Path

KAGGLE_USERNAME = "hongnguynngthhjk"
REPO_URL = "https://github.com/minhnguyenrun/DO-AN-261.git"
THIS_DIR = Path(__file__).resolve().parent

#  Moi step la danh sach arg SAU "pip" (khong ghi "pip"/"-m" - generator tu
#  them `[sys.executable, "-m", "pip"]` phia truoc de chac chan dung dung
#  python/pip cua kernel, tranh lech moi truong).
METHOD_CONFIGS = {
    "docling": {
        "title": "DOAN261 Docling",
        "pip_steps": [
            ["install", "-q", "-r", "methods/docling/requirements.txt"],
        ],
    },
    "paddleocr_vl": {
        "title": "DOAN261 PaddleOCR Vl",
        # Khong dung paddlepaddle CPU trong requirements.txt - cai ban GPU theo
        # huong dan trong methods/paddleocr_vl/README.md de tan dung GPU Kaggle.
        "pip_steps": [
            ["install", "-q", "-U", "paddleocr[doc-parser]", "pillow", "tqdm",
             "matplotlib", "opencv-python"],
            ["install", "-q", "paddlepaddle-gpu==3.2.1", "-i",
             "https://www.paddlepaddle.org.cn/packages/stable/cu126/"],
        ],
    },
    "m2doc": {
        "title": "DOAN261 M2doc",
        "pip_steps": [
            ["install", "-q", "-r", "methods/m2doc/requirements.txt"],
        ],
    },
    "dots_ocr": {
        "title": "DOAN261 Dots Ocr",
        "pip_steps": [
            ["install", "-q", "-r", "methods/dots_ocr/requirements.txt"],
        ],
    },
    "dots_mocr": {
        "title": "DOAN261 Dots Mocr",
        "pip_steps": [
            ["install", "-q", "-r", "methods/dots_mocr/requirements.txt"],
        ],
    },
    "chandra": {
        "title": "DOAN261 Chandra",
        "pip_steps": [
            ["install", "-q", "-r", "methods/chandra/requirements.txt"],
            # requirements.txt chi ghi transformers>=4.45.0 (qua long), nhung
            # custom modeling code tren HF can ban moi hon - ep cai ban moi.
            ["install", "-q", "-U", "transformers>=5.2.0"],
        ],
    },
}

KERNEL_TEMPLATE = '''"""Kaggle kernel: chay {method} tren split test cua DocLayNet.

Sinh tu dong boi kaggle/build_kernels.py - sua o do, khong sua file nay truc tiep.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = {repo_url!r}
REPO_DIR = Path("/kaggle/working/DO-AN-261")
METHOD = {method!r}
INFER_LIMIT = {limit!r}
EXISTING_PREDICTIONS_DIR = Path("/kaggle/input/doan261-{method_slug}-preds")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         "scripts/download_datasets/requirements.txt"])
    run([sys.executable, "scripts/download_datasets/download_doclaynet.py",
         "--all", "--splits", "test", "--delete-raw"])

{pip_steps}

    # Phuc hoi prediction da chay tu lan truoc (neu kernel nay duoc gan dataset
    # chua ket qua cu, dung khi can resume mot run bi ngat giua chung).
    predictions_dir = REPO_DIR / "outputs" / "predictions" / METHOD / "doclaynet_test"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if EXISTING_PREDICTIONS_DIR.exists():
        for f in EXISTING_PREDICTIONS_DIR.glob("*.json"):
            shutil.copy(f, predictions_dir / f.name)
        print(f"[resume] Da phuc hoi {{len(list(EXISTING_PREDICTIONS_DIR.glob('*.json')))}} "
              f"prediction tu lan chay truoc.")

    run([
        sys.executable, f"methods/{{METHOD}}/src/run_inference.py",
        "--dataset", "doclaynet_test",
        "--limit", str(INFER_LIMIT),
    ])

    # Chi giu outputs/predictions + outputs/runtime (file JSON nho) de Kaggle luu
    # lam ket qua kernel - khong luu lai dataset anh / weight model (hang GB).
    final_out = Path("/kaggle/working/result")
    final_out.mkdir(exist_ok=True)
    pred_src = REPO_DIR / "outputs" / "predictions" / METHOD
    if pred_src.exists():
        shutil.copytree(pred_src, final_out / "predictions", dirs_exist_ok=True)
    runtime_src = REPO_DIR / "outputs" / "runtime" / METHOD
    if runtime_src.exists():
        shutil.copytree(runtime_src, final_out / "runtime", dirs_exist_ok=True)

    shutil.rmtree(REPO_DIR / "datasets", ignore_errors=True)
    shutil.rmtree(REPO_DIR / "methods" / METHOD / "weights", ignore_errors=True)
    shutil.rmtree(REPO_DIR / ".git", ignore_errors=True)

    # Nen thanh 1 file zip duy nhat - tai hang nghin file JSON le qua Kaggle API
    # rat hay bi rot ket noi, 1 file zip tai on dinh va nhanh hon nhieu.
    shutil.make_archive("/kaggle/working/result", "zip", final_out)
    shutil.rmtree(final_out, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
'''


def build(method: str, config: dict, limit: int) -> None:
    out_dir = THIS_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)

    pip_steps_code = "\n".join(
        "    run([sys.executable, \"-m\", \"pip\"] + {!r})".format(step)
        for step in config["pip_steps"]
    )
    kernel_code = KERNEL_TEMPLATE.format(
        method=method,
        method_slug=method.replace("_", "-"),
        repo_url=REPO_URL,
        limit=limit,
        pip_steps=pip_steps_code,
    )
    (out_dir / "kernel.py").write_text(kernel_code, encoding="utf-8", newline="\n")

    slug = f"doan261-{method.replace('_', '-')}"
    metadata = {
        "id": f"{KAGGLE_USERNAME}/{slug}",
        "title": config["title"],
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build] {out_dir} (limit={limit})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, required=True,
                         help="So anh moi method se chay (0 = toan bo test split)")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_CONFIGS),
                         default=list(METHOD_CONFIGS))
    args = parser.parse_args()

    for method in args.methods:
        build(method, METHOD_CONFIGS[method], args.limit)


if __name__ == "__main__":
    main()
