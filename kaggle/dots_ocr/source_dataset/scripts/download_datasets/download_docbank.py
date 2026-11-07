"""Tải một phần bộ dữ liệu DocBank từ HuggingFace Hub.

Dataset gốc (liminghao1630/DocBank, ~54GB) gồm:
  - DocBank_500K_ori_img.zip.001 ... .010: MỘT file zip ảnh bị cắt làm 10 phần
    (~5.4GB/phần), nên không thể giải nén từng phần theo cách thông thường.
  - MSCOCO_Format_Annotation.zip (~209MB): annotation dạng COCO (box theo
    vùng, 13 nhãn) cho train/valid/test.

Script này:
  1. Tải MSCOCO_Format_Annotation.zip vào datasets/docbank/raw/.
  2. Stream phần đầu của DocBank_500K_ori_img.zip.001 qua HTTP, đọc tuần tự
     từng local file header của zip và giải nén ảnh .jpg ra
     datasets/docbank/processed/images/ cho tới khi đạt dung lượng chỉ định
     (mặc định 1GB). Không cần tải trọn 54GB.
  3. Lọc annotation COCO, chỉ giữ các ảnh đã tải, lưu vào
     datasets/docbank/processed/annotations/<split>.json.

Usage:
    python download_docbank.py --max-size-gb 1
    python download_docbank.py --max-size-gb 1 --no-extract
"""

import argparse
import json
import struct
import zipfile
import zlib
from pathlib import Path

from huggingface_hub import get_session, hf_hub_download, hf_hub_url
from tqdm import tqdm

REPO_ID = "liminghao1630/DocBank"
IMAGE_PART = "DocBank_500K_ori_img.zip.001"
ANNOTATION_ZIP = "MSCOCO_Format_Annotation.zip"
# file trong ANNOTATION_ZIP -> tên split thống nhất với các dataset khác
SPLIT_FILES = {"train": "500K_train.json", "validation": "500K_valid.json", "test": "500K_test.json"}

THIS_DIR = Path(__file__).resolve().parent
RAW_DIR = THIS_DIR / "../../datasets/docbank/raw"
PROCESSED_DIR = THIS_DIR / "../../datasets/docbank/processed"

LOCAL_HEADER_SIG = 0x04034B50
LOCAL_HEADER_LEN = 30


def download_annotations() -> Path:
    print(f"[download] {ANNOTATION_ZIP}")
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=ANNOTATION_ZIP,
        local_dir=str(RAW_DIR),
    ))


class _Stream:
    """Bọc iterator bytes của HTTP response thành stream đọc được n byte một lúc."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.buf = bytearray()
        self.consumed = 0

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = next(self.chunks, None)
            if chunk is None:
                break
            self.buf += chunk
        data = bytes(self.buf[:n])
        del self.buf[:n]
        self.consumed += len(data)
        return data


def stream_images(max_size_bytes: int) -> list:
    """Stream ảnh từ phần đầu của zip ảnh. Trả về list tên file đã lưu."""
    images_dir = PROCESSED_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(REPO_ID, IMAGE_PART, repo_type="dataset")
    saved = []
    with get_session().stream("GET", url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        stream = _Stream(resp.iter_bytes(chunk_size=1 << 20))
        pbar = tqdm(total=max_size_bytes, unit="B", unit_scale=True, desc="[download] anh")
        while stream.consumed < max_size_bytes:
            header = stream.read(LOCAL_HEADER_LEN)
            if len(header) < LOCAL_HEADER_LEN:
                break
            (sig, _, flag, method, _, _, crc, comp_size, _,
             name_len, extra_len) = struct.unpack("<IHHHHHIIIHH", header)
            if sig != LOCAL_HEADER_SIG:
                break  # hết vùng local file (hoặc gặp central directory)
            if flag & 0x08:
                raise RuntimeError("Zip dung data descriptor, khong stream duoc kich thuoc")

            name = stream.read(name_len).decode("utf-8")
            stream.read(extra_len)
            data = stream.read(comp_size)
            if len(data) < comp_size:
                break  # stream bị cắt giữa chừng, bỏ file dở dang
            pbar.update(LOCAL_HEADER_LEN + name_len + extra_len + comp_size)

            file_name = Path(name).name
            if not file_name.lower().endswith((".jpg", ".png")):
                continue  # entry thư mục
            if method == 8:
                data = zlib.decompress(data, -15)
            elif method != 0:
                raise RuntimeError(f"Phuong thuc nen khong ho tro: {method}")
            if zlib.crc32(data) != crc:
                raise RuntimeError(f"Sai CRC: {name}")

            (images_dir / file_name).write_bytes(data)
            saved.append(file_name)
        pbar.close()

    print(f"[download] Tong cong: {stream.consumed / 1e9:.2f} GB ({len(saved)} anh)")
    return saved


def extract_processed(annotation_zip: Path, image_files: list):
    """Lọc annotation COCO theo các ảnh đã tải, ghi ra processed/annotations/<split>.json.

    image_id trong file gốc được đánh lại từ 1 ở MỖI split (trùng nhau giữa
    các split), nên ở đây đánh lại image_id liên tục trên cả 3 split và đổi
    tên ảnh thành <image_id>.jpg cho thống nhất với DocLayNet. Tên gốc được
    giữ trong trường "orig_file_name".
    """
    images_dir = PROCESSED_DIR / "images"
    annotations_dir = PROCESSED_DIR / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(image_files)
    next_image_id, next_ann_id = 1, 1

    with zipfile.ZipFile(annotation_zip) as zf:
        for split, member in SPLIT_FILES.items():
            print(f"[extract] doc {member} ...")
            with zf.open(member) as f:
                data = json.load(f)

            id_map = {}
            images = []
            for img in data["images"]:
                if img["file_name"] not in wanted:
                    continue
                new_id = next_image_id
                next_image_id += 1
                id_map[img["id"]] = new_id
                file_name = f"{new_id}.jpg"
                (images_dir / img["file_name"]).replace(images_dir / file_name)
                images.append({
                    "id": new_id,
                    "file_name": file_name,
                    "width": img["width"],
                    "height": img["height"],
                    "orig_file_name": img["file_name"],
                })

            annotations = []
            for ann in data["annotations"]:
                if ann["image_id"] not in id_map:
                    continue
                annotations.append({
                    "id": next_ann_id,
                    "image_id": id_map[ann["image_id"]],
                    "category_id": ann["category_id"],
                    "bbox": [float(x) for x in ann["bbox"]],
                    "area": float(ann["area"]),
                })
                next_ann_id += 1
            categories = [{"id": c["id"], "name": c["name"]} for c in data["categories"]]
            del data

            if not images:
                print(f"[extract] {split}: khong co anh nao trong phan da tai, bo qua")
                continue
            coco = {"images": images, "annotations": annotations, "categories": categories}
            out_path = annotations_dir / f"{split}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(coco, f)
            print(f"[extract] {out_path} ({len(images)} anh, {len(annotations)} box)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size-gb", type=float, default=1.0,
                         help="Dung luong anh toi da can tai, tinh bang GB (mac dinh 1.0)")
    parser.add_argument("--no-extract", action="store_true",
                         help="Chi tai annotation zip vao raw/, khong tai anh va khong tao processed/")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    annotation_zip = download_annotations()

    if not args.no_extract:
        image_files = stream_images(int(args.max_size_gb * 1e9))
        if image_files:
            extract_processed(annotation_zip, image_files)


if __name__ == "__main__":
    main()
