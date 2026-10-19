# Datasets

Thư mục này chứa dữ liệu chạy **local only** — không push lên git (xem `.gitignore`
ở thư mục gốc), ngoại trừ file `README.md`. Các thư mục con do script tải dữ liệu tự tạo.

## Cấu trúc

```
datasets/
├── doclaynet/
│   ├── raw/          # file .parquet tải thẳng từ HuggingFace (chưa xử lý)
│   └── processed/    # ảnh (.png) + annotation (COCO-style .json) đã chuyển định dạng
├── publaynet/
│   ├── raw/
│   └── processed/
└── docbank/
    ├── raw/          # MSCOCO_Format_Annotation.zip (annotation gốc, 500K trang)
    └── processed/    # ảnh (.jpg) + annotation (COCO-style .json) của phần đã tải
```

## Nguồn dữ liệu

- **DocLayNet** — `docling-project/DocLayNet-v1.2` trên HuggingFace Hub
  (https://huggingface.co/datasets/docling-project/DocLayNet-v1.2). 11 nhãn layout
  (Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture,
  Section-header, Table, Text, Title), chia theo shard parquet `train/validation/test`.
- **PubLayNet** — `creative-graphic-design/PubLayNet` trên HuggingFace Hub
  (https://huggingface.co/datasets/creative-graphic-design/PubLayNet). Annotation
  theo chuẩn COCO, 5 nhãn layout (Text, Title, List, Table, Figure).
- **DocBank** — `liminghao1630/DocBank` trên HuggingFace Hub
  (https://huggingface.co/datasets/liminghao1630/DocBank), bản phát hành chính thức
  của tác giả (https://github.com/doc-analysis/DocBank). 500K trang paper arXiv,
  annotation COCO với 13 nhãn layout (paragraph, equation, footer, caption, figure,
  table, section, list, reference, abstract, title, author, date).

Do dữ liệu gốc rất lớn (DocLayNet ~40GB, PubLayNet ~108GB, DocBank ~54GB), script tải mặc định chỉ
lấy một phần giới hạn theo kích thước (mặc định 1GB) để đủ dùng cho việc phát triển
và test local.

## Cách tải

```bash
pip install -r scripts/download_datasets/requirements.txt

# tải ~1GB đầu tiên (mặc định), chia theo train/validation/test
python scripts/download_datasets/download_doclaynet.py --max-size-gb 1

# chỉ tải raw, không convert qua processed/
python scripts/download_datasets/download_doclaynet.py --max-size-gb 1 --no-extract

python scripts/download_datasets/download_publaynet.py --max-size-gb 1
python scripts/download_datasets/download_docbank.py --max-size-gb 1

# hoặc tải thử nghiệm ít dung lượng (giới hạn số ảnh và tự động xóa raw sau giải nén, ~1GB tổng cộng cả 2 bộ):
# python scripts/download_datasets/download_doclaynet.py --max-size-gb 0.5 --max-images 250 --delete-raw
# python scripts/download_datasets/download_publaynet.py --max-size-gb 0.5 --max-images 250 --delete-raw
```

Script sẽ:
1. Tải các shard `.parquet` từ HuggingFace Hub vào `raw/`, dừng lại khi đã đạt
   khoảng dung lượng chỉ định (`--max-size-gb`).
2. Trích ảnh + annotation từ các shard đã tải, lưu vào `processed/images/` và
   `processed/annotations/<split>.json` (COCO-style: `images`, `annotations`,
   `categories`).

## Thống kê dữ liệu đã tải

- **DocLayNet**: đã tải 2 shard `train` (~1.04GB raw parquet) từ
  `docling-project/DocLayNet-v1.2`, trích xuất được **1928 ảnh** (`.png`) và
  **24229 bounding box** trên 11 nhãn layout, lưu tại:
  - `datasets/doclaynet/raw/data/train-0000{0,1}-of-00072.parquet`
  - `datasets/doclaynet/processed/images/*.png`
  - `datasets/doclaynet/processed/annotations/train.json` (COCO-style)
- **PubLayNet**: chưa tải (dùng `download_publaynet.py` khi cần).
- **DocBank**: đã stream ~1GB đầu của `DocBank_500K_ori_img.zip.001` từ
  `liminghao1630/DocBank`, được **9908 ảnh** (`.jpg`) và **131838 bounding box**
  trên 13 nhãn layout, lưu tại:
  - `datasets/docbank/raw/MSCOCO_Format_Annotation.zip` (annotation gốc, ~200MB)
  - `datasets/docbank/processed/images/<image_id>.jpg`
  - `datasets/docbank/processed/annotations/{train,validation,test}.json`
    (lần lượt 7926 / 1021 / 961 ảnh; 105135 / 13687 / 13016 box)

  Phân bố nhãn: paragraph 81595, equation 24402, section 9760, caption 3672,
  figure 3357, reference 2743, list 1950, footer 1016, table 1009, author 885,
  abstract 757, title 522, date 170.

### Lưu ý về DocBank

- Ảnh gốc là một file zip bị cắt thành 10 phần, nên script **không** tải nguyên
  file mà stream phần đầu của `.zip.001` qua HTTP và giải nén từng ảnh (đọc tuần tự
  local file header). Ảnh trong zip xếp theo tên file, nên phần đã tải chủ yếu là
  các paper arXiv đầu năm 2014 (tên dạng `1.tar_1401.xxxx...`), không phải mẫu ngẫu
  nhiên của toàn bộ 500K trang.
- `image_id` trong annotation gốc được đánh lại từ 1 ở mỗi split. Script đánh lại
  `image_id` liên tục trên cả 3 split và đổi tên ảnh thành `<image_id>.jpg`; tên file
  gốc giữ trong trường `orig_file_name` của mỗi ảnh.
- Nhãn sinh tự động từ mã nguồn LaTeX (weak supervision), nên có box chồng nhau
  (ví dụ một vùng vừa là `caption` vừa là `paragraph`). Tọa độ box là pixel trên ảnh
  (ảnh đã được resize, cạnh dài 1000px).
- Bộ nhãn khác DocLayNet: muốn đánh giá Docling trên DocBank cần map nhãn
  Docling sang 13 nhãn DocBank (hiện `methods/docling/src/run_inference.py` chỉ map
  sang DocLayNet).
