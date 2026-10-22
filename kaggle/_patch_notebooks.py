"""Mot lan: sua config (DATASET/SPLIT/LIMIT/RUNS) trong cell dau cua moi report
notebook de chay tren doclaynet_test, toan bo anh (~5000)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = {
    "methods/docling/notebooks/docling_report.ipynb": None,  # xu ly rieng (RUNS list)
    "methods/paddleocr_vl/notebooks/paddleocr_vl_report.ipynb": "std",
    "methods/m2doc/notebooks/m2doc_report.ipynb": "std",
    "methods/dots_ocr/notebooks/dots_ocr_report.ipynb": "std",
    "methods/dots_mocr/notebooks/dots_mocr_report.ipynb": "std",
    "methods/chandra/notebooks/chandra_report.ipynb": "std",
}


def set_source(cell, new_text):
    lines = new_text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass
    cell["source"] = lines


def patch_std(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if 'DATASET = "doclaynet"' in src and 'SPLIT = "train"' in src:
            src = src.replace('DATASET = "doclaynet"', 'DATASET = "doclaynet_test"')
            src = src.replace('SPLIT = "train"', 'SPLIT = "test"')
            src = re.sub(r"LIMIT = \d+.*", "LIMIT = 0  # 0: toan bo anh trong tap du lieu", src)
            set_source(cell, src)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[patched] {path}")
    else:
        print(f"[SKIP - pattern not found] {path}")


def patch_docling(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "RUNS = [" in src:
            src = re.sub(
                r"RUNS = \[.*?\]\n",
                'RUNS = [\n    ("doclaynet_test", "test"),         # ~4999 anh (test split)\n]\n',
                src,
                count=1,
                flags=re.DOTALL,
            )
            set_source(cell, src)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[patched] {path}")
    else:
        print(f"[SKIP - pattern not found] {path}")


for rel, kind in TARGETS.items():
    p = ROOT / rel
    if kind == "std":
        patch_std(p)
    else:
        patch_docling(p)
