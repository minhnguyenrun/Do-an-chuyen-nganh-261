"""Kaggle kernel: chay dots_ocr tren split test cua DocLayNet.

Sinh tu dong boi kaggle/build_kernels.py - sua o do, khong sua file nay truc tiep.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = 'https://github.com/minhnguyenrun/DO-AN-261.git'
REPO_DIR = Path("/kaggle/working/DO-AN-261")
METHOD = 'dots_ocr'
INFER_LIMIT = 0
EXISTING_PREDICTIONS_DIR = Path("/kaggle/input/doan261-dots-ocr-preds")


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

    run([sys.executable, "-m", "pip"] + ['install', '-q', '-r', 'methods/dots_ocr/requirements.txt'])

    # Phuc hoi prediction da chay tu lan truoc (neu kernel nay duoc gan dataset
    # chua ket qua cu, dung khi can resume mot run bi ngat giua chung).
    predictions_dir = REPO_DIR / "outputs" / "predictions" / METHOD / "doclaynet_test"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if EXISTING_PREDICTIONS_DIR.exists():
        for f in EXISTING_PREDICTIONS_DIR.glob("*.json"):
            shutil.copy(f, predictions_dir / f.name)
        print(f"[resume] Da phuc hoi {len(list(EXISTING_PREDICTIONS_DIR.glob('*.json')))} "
              f"prediction tu lan chay truoc.")

    run([
        sys.executable, f"methods/{METHOD}/src/run_inference.py",
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
