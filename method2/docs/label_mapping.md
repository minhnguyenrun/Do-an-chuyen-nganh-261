# Nhãn GLM-OCR SDK → Dual-Skip → DocLayNet

Đối chiếu nguồn chính thức ngày 24/10/2026.

SDK GLM-OCR dùng PP-DocLayoutV3 để tìm và gắn nhãn layout. Bộ 25 nhãn bên dưới
thuộc detector đó, không phải bộ nhãn đầu ra của một classifier crop GLM-OCR.
GLM-OCR nhận dạng nội dung với ba task text/table/formula; SDK chuyển nhãn layout
sang task tương ứng. Việc chọn prompt OCR không sinh ra nhãn layout.

Nguồn:
- [GLM-OCR SDK: id2label, label_task_mapping](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/config.yaml)
- [GLM-OCR SDK: PPDocLayoutDetector](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/layout/layout_detector.py)
- [GLM-OCR: giới hạn prompt](https://huggingface.co/zai-org/GLM-OCR#prompt-limited)
- [DocLayNet: 11 lớp và COCO](https://github.com/DS4SD/DocLayNet)

## Mapping hai bước

Cột đầu và task SDK lấy từ nguồn chính thức. Các cột Dual-Skip/DocLayNet là
chính sách quy đổi của dự án, không phải mapping được tác giả hai mô hình công bố.
Giữ nguyên 11 lớp đích của DocLayNet trong chế độ literal. Với đánh giá gộp heading
đã thống nhất, đổi cả Title và Section-header của GT/prediction thành lớp Heading
(ID 8) trước khi tính điểm; đây là taxonomy 10 lớp, không gọi là kết quả 11 lớp gốc.

| GLM-OCR SDK / PP-DocLayoutV3 | Nhãn Dual-Skip | DocLayNet literal | SDK task |
|---|---|---|---|
| `abstract` | `abstract` | Text (10) | text |
| `algorithm` | `algorithm` | Text (10) | text |
| `aside_text` | `aside_text` | Text (10) | abandon |
| `chart` | `chart` | Picture (7) | skip |
| `content` | `content` | Text (10) | text |
| `display_formula` | `formula` | Formula (3) | formula |
| `doc_title` | `title` | Title (11) | text |
| `figure_title` | `figure_title` | Caption (1) | text |
| `footer` | `footer` | Page-footer (5) | abandon |
| `footer_image` | `footer_image` | Page-footer (5) | abandon |
| `footnote` | `footnote` | Footnote (2) | abandon |
| `formula_number` | `formula_number` | Unmapped (0) | text |
| `header` | `header` | Page-header (6) | abandon |
| `header_image` | `header_image` | Page-header (6) | abandon |
| `image` | `image` | Picture (7) | skip |
| `inline_formula` | `formula` | Formula (3) | formula |
| `number` | `number` | Unmapped (0) | abandon |
| `paragraph_title` | `title` | Title (11) | text |
| `reference` | `reference` | Text (10) | abandon |
| `reference_content` | `reference_content` | Text (10) | text |
| `seal` | `seal` | Unmapped (0) | text |
| `table` | `table` | Table (9) | table |
| `text` | `text` | Text (10) | text |
| `vertical_text` | `vertical_text` | Text (10) | text |
| `vision_footnote` | `vision_footnote` | Unmapped (0) | text |

## Giới hạn quy đổi

- `doc_title` và `paragraph_title` cùng thành `title`, nên thông tin phân biệt đã mất.
  Không thể suy ngược ra hai lớp gốc chỉ bằng mapping. Báo cáo gộp heading khắc phục
  khác biệt taxonomy, không chứng minh detector phân biệt được hai loại title.
- `display_formula` và `inline_formula` cùng thành `formula` → Formula.
- `number`: có thể là số trang ở đầu/cuối; không mặc định chọn Page-footer.
- `formula_number`: số hiệu phương trình, không phải một vùng công thức hoàn chỉnh.
- `seal`: DocLayNet không có lớp con dấu; không tự coi mọi seal là Picture.
- `vision_footnote`: cần xác minh vai trò Caption/Footnote/Text theo ngữ cảnh.
- Bốn nhãn trên vẫn là Unmapped (ID 0), được giữ khi đánh giá và có thể tính FP;
  không xóa chúng hoặc bỏ phạt để làm tăng điểm.
- `list_item` là nhãn bổ sung của pipeline → List-item; không có trong 25 nhãn SDK.
- `residual`/`unknown`/`unlabeled` là trạng thái nội bộ. `residual` → Text chỉ là
  fallback đánh giá đang có; không có nghĩa GLM đã xác nhận đó là Text.
- SDK mặc định abandon một số vùng và skip OCR hình/chart. Dự án chỉ ghi nhận
  chính sách SDK làm thông tin tham khảo; vẫn giữ và OCR vùng như pipeline hiện tại.

## Áp dụng trong code

`src/label_mapping.py` là nguồn dùng chung: adapter layout dùng
`normalize_layout_label`, script đánh giá dùng `doclaynet_category`.
`configs/label_mapping.json` là bản xuất để kiểm tra; không đọc file này làm một
cấu hình độc lập có thể lệch code. Mỗi báo cáo đánh giá mới chứa `official_sdk_mapping`.
Các kết quả và báo cáo cũ được giữ nguyên.

Bộ nhãn custom trong bước GLM phân loại residual vẫn là thử nghiệm. Tìm được bộ
nhãn SDK không biến GLM-OCR thành classifier 25 lớp, cũng không tự sửa lỗi JSON.
Muốn đánh giá classifier đó, cần thử crop thật và kiểm tra nhãn trước khi chạy batch.
