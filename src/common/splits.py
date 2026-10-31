"""Quy uoc ten cac tap train / valid / test cho DocLayNet.

image_id cua DocLayNet danh lai tu 0 trong MOI split nen moi split nam trong mot
dataset rieng (xem datasets/README.md). Module nay la noi DUY NHAT map ten tap
(`train`, `valid`, `test`) sang ten dataset va ten file annotation, de cac script
chi can `--split valid` thay vi nho `doclaynet_val` + `validation.json`.

Quy tac dung 3 tap:
  - train: chi de huan luyen / fine-tune.
  - valid: chon sieu tham so, nguong, cach map nhan, so sanh phuong an.
  - test : chi de bao cao ket qua cuoi cung, KHONG dung de chinh bat cu thu gi.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitInfo:
    name: str          # ten chuan: train | valid | test
    dataset: str       # thu muc trong datasets/ (va trong outputs/predictions/<method>/)
    annotation: str    # ten file annotation (khong co .json) trong processed/annotations/
    purpose: str


SPLITS = {
    "train": SplitInfo("train", "doclaynet", "train", "huan luyen / fine-tune"),
    "valid": SplitInfo("valid", "doclaynet_val", "validation", "chon sieu tham so, nguong, cach map nhan"),
    "test": SplitInfo("test", "doclaynet_test", "test", "bao cao ket qua cuoi cung"),
}

ALIASES = {"val": "valid", "validation": "valid", "dev": "valid"}

# Ten dataset cu (--dataset) -> ten tap, de tuong thich nguoc voi lenh cu.
DATASET_TO_SPLIT = {info.dataset: info.name for info in SPLITS.values()}


def normalize_split(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in SPLITS:
        raise ValueError(f"Tap '{name}' khong hop le; chon mot trong: train, valid (val), test")
    return key


def resolve(split: str) -> SplitInfo:
    return SPLITS[normalize_split(split)]


def split_of_dataset(dataset: str) -> str | None:
    """Ten tap ung voi mot dataset DocLayNet, None neu khong phai DocLayNet (vd. docbank)."""
    return DATASET_TO_SPLIT.get(dataset)


def apply_split(dataset: str, split: str | None) -> str:
    """Dataset can dung: neu truyen --split (train/valid/test) thi suy ra tu do, nguoc lai giu --dataset."""
    return resolve(split).dataset if split else dataset


def describe(dataset: str) -> str:
    """Nhan hien thi, vd. 'TEST (doclaynet_test)'; dataset la lam 'doclaynet_val' -> 'VALID (...)'."""
    name = split_of_dataset(dataset)
    return f"{name.upper()} ({dataset})" if name else dataset
