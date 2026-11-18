# Thư mục EDA (Exploratory Data Analysis) cho Datasets

Thư mục này chứa các Jupyter Notebooks phục vụ việc khám phá, thống kê và phân tích chuyên sâu các bộ dữ liệu Document Layout Analysis (DLA) trong đồ án **DO-AN-261**.

## 📁 Danh sách Notebooks

| Notebook | Bộ dữ liệu | Mô tả | Trạng thái |
|---|---|---|---|
| [`doclaynet_eda.ipynb`](file:///d:/Semester_261/DoAnChuyenNganh/DO-AN-261/eda/doclaynet_eda.ipynb) | **DocLayNet** | Phân tích 11 nhãn layout, 6 miền tài liệu (domains), mất cân bằng nhãn (class imbalance), mật độ bbox/trang, phân bố kích thước theo chuẩn COCO, bản đồ nhiệt tọa độ không gian (spatial heatmap) và trực quan hóa mẫu tài liệu. | ✅ Sẵn sàng |
| `publaynet_eda.ipynb` | **PubLayNet** | *(Dự kiến)* Phân tích 5 nhãn layout (Text, Title, List, Table, Figure) trên tài liệu y sinh / khoa học. | ⏳ Kế hoạch |
| `docbank_eda.ipynb` | **DocBank** | *(Dự kiến)* Phân tích 13 nhãn layout trên 500K trang arXiv paper. | ⏳ Kế hoạch |

---

## 🎯 Nội dung chi tiết trong `doclaynet_eda.ipynb`

Notebook [`doclaynet_eda.ipynb`](file:///d:/Semester_261/DoAnChuyenNganh/DO-AN-261/eda/doclaynet_eda.ipynb) được xây dựng với 10 phần phân tích chuyên sâu:

1. **Tổng quan dữ liệu**: Nạp annotations định dạng COCO từ `datasets/doclaynet/processed/annotations/train.json` và `datasets/doclaynet_test/processed/annotations/test.json`.
2. **Phân tích miền tài liệu (Document Domains)**: Thống kê số lượng trang tài liệu theo 6 domain (`financial_reports`, `scientific_articles`, `laws_and_regulations`, `government_tenders`, `manuals`, `patents`).
3. **Phân tích mất cân bằng nhãn (Class Imbalance)**: Thống kê số lượng annotation, tỉ lệ %, tỉ lệ mất cân bằng (imbalance ratio) và tỉ lệ hiện diện trên trang (% page presence) cho 11 nhãn.
4. **Tương quan Miền tài liệu & Nhãn cấu trúc (Cross-domain Heatmap)**: Ma trận tỉ lệ % xuất hiện của từng loại layout trong mỗi miền tài liệu.
5. **Mật độ Bounding Box trên mỗi trang (Layout Complexity)**: Mean, Median, Min, Max, P95 và biểu đồ Histogram + KDE + Boxplot theo domain.
6. **Phân tích hình học Bounding Box**:
   - Tỉ lệ khung hình (Aspect Ratio = $w/h$).
   - Phân loại kích thước theo chuẩn COCO (Small $< 32^2$, Medium $32^2 - 96^2$, Large $\ge 96^2$).
7. **Bản đồ nhiệt vị trí không gian ĐẦY ĐỦ 11 Nhãn (Spatial Layout Heatmap)**:
   - Tọa độ tâm chuẩn hóa $(c_x, c_y)$ của **toàn bộ 11 nhãn** (`Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`) + 1 ô tổng hợp Combined Overlay trên lưới 3x4.
8. **Trực quan hóa tài liệu mẫu (Visual Inspection)**: Hàm `visualize_document_sample()` vẽ bounding box ground-truth kèm nhãn màu đặc trưng, an toàn với đường dẫn tiếng Việt trên Windows.
9. **Phân tích Cấp độ Cuốn sách / File Tài liệu Đa trang (Multi-page Book & Document EDA)**:
   - Thuật toán gom cụm các trang thuộc cùng một cuốn sách/tài liệu dựa trên chữ ký cấu trúc (Header/Footer invariance signature).
   - Khảo sát các đặc tính bất biến: Tọa độ $y$ của Header/Footer cố định tuyệt đối qua hàng chục trang ($\sigma_y \approx 0$).
   - Quy luật trang đôi chẵn/lẻ (Facing pages / Even-odd symmetry): Phân bố hai cực (bimodal) của số trang và chân trang.
   - Trực quan hóa so sánh 4 trang side-by-side với các đường dóng ngang guidelines dóng vị trí Header và Footer.
   - Thống kê đối chứng độ lệch chuẩn intra-document vs toàn bộ dataset.
10. **Tổng kết nhận định & Đề xuất kỹ thuật**: 4 bài học chiến lược cho việc huấn luyện mô hình và xây dựng thuật toán hậu xử lý cấp độ tài liệu (Document-level post-processing filter).

---

## 🚀 Hướng dẫn Chạy Notebook

1. **Kích hoạt môi trường Python** của đồ án (khuyến nghị Python 3.12 với các thư viện `pandas`, `numpy`, `matplotlib`, `seaborn`, `pillow`, `opencv-python`).
2. Mở VS Code / Jupyter Lab và điều hướng đến thư mục `eda/`.
3. Mở file [`doclaynet_eda.ipynb`](file:///d:/Semester_261/DoAnChuyenNganh/DO-AN-261/eda/doclaynet_eda.ipynb) và chọn Kernel Python tương ứng.
4. Chạy tuần tự các cells từ trên xuống dưới. Notebook đã được lập trình sẵn cơ chế tự động tìm thư mục gốc dự án (`ROOT`).
