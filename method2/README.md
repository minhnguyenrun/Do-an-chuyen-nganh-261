# Dual-Skip v4 — Notebook chạy trực tiếp

Mở **`Dual_Skip_v4.ipynb`** bằng Jupyter hoặc VS Code, chọn Python kernel rồi chạy lần lượt các cell. Giữ nguyên thư mục `method2/src` và `method2/configs` bên cạnh notebook. GLM-OCR chạy trực tiếp trong tiến trình Python bằng Transformers, không dùng API, HTTP client hay máy chủ riêng.

## Cài đặt

Python 3.10 trở lên. Cài PyTorch phù hợp CPU/CUDA trước nếu máy chưa có; không cần thay bản PyTorch CUDA đang hoạt động. Từ thư mục `method2`:

```powershell
python -m pip install -r requirements-notebook.txt
```

Notebook có cell cài dependency riêng, mặc định tắt (`INSTALL_DEPENDENCIES=False`). Chọn kernel đúng môi trường cài thư viện. Nếu thay phiên bản thư viện sau khi đã import, Restart Kernel rồi chạy lại. Các kiểm thử thuật toán không cần tải model; kiểm thử notebook cần các dependency ở trên.

## Cách chạy notebook

1. Chạy cell xác định thư mục và kiểm tra phiên bản/CPU/GPU.
2. Mặc định `USE_DOCLAYNET_SAMPLE=True`: chạy cell tải mẫu để lấy 50 ảnh split test, rồi notebook xử lý 20 ảnh đầu. Script dùng chế độ `--lightweight`, chỉ lấy từng ảnh và annotation cần dùng; không tải shard parquet. Dữ liệu đã có được tái sử dụng, chạy bị ngắt có thể tiếp tục.
3. Muốn dùng tài liệu riêng, đặt `USE_DOCLAYNET_SAMPLE=False` rồi điền `INPUT_PATH` bằng đường dẫn ảnh PNG/JPEG hoặc PDF. `PAGES=None` chạy tất cả trang PDF, `[1, 3]` chạy trang 1 và 3; ảnh có trang 1. PDF mặc định ở 150 DPI, hỗ trợ đường dẫn tiếng Việt.
4. Chạy cell nạp tài liệu và khởi tạo mô hình, sau đó chạy cell xử lý để xem kết quả từng trang ngay trong notebook.
5. Xem bảng tổng hợp và nhật ký giai đoạn ở cell cuối. Muốn chạy lại tài liệu thì chạy lại cell nạp ảnh/PDF; model vẫn được tái sử dụng nếu cấu hình không đổi.

Cấu hình hiện tại: cả ba adapter PaddleOCR chạy GPU 0 (`gpu:0`), cần thay gói CPU `paddlepaddle` bằng `paddlepaddle-gpu` phù hợp môi trường; GLM dùng CUDA FP16 khi khả dụng và CPU FP32 khi không có CUDA; xử lý từng crop một, giới hạn cạnh dài 1280 pixel và tối đa 1024 token mới. Đây là giới hạn khởi đầu; chạy đồng thời Paddle và GLM trên GPU 4 GB có thể thiếu VRAM. Có thể đổi ba tham số `device` trong `src/runtime.py` về `cpu` để giảm bộ nhớ GPU. Khi OOM, lỗi được hiển thị; giảm `GLM_MAX_CROP_EDGE`/`GLM_MAX_NEW_TOKENS` hoặc đổi `GLM_DEVICE="cpu"`, giải phóng model bằng cell cuối rồi khởi tạo lại.

`GLM_MODEL` nhận model ID `zai-org/GLM-OCR` hoặc đường dẫn thư mục trọng số cục bộ. Lần đầu dùng model ID cần mạng để tải trọng số; các adapter PaddleOCR cũng có thể cần tải model. Sau khi đã có model/cache, suy luận không cần server. Không tự tải hoặc cài model khi chỉ mở notebook.

Chạy thật là mặc định: `MOCK_MODE=False`. Chế độ `True` chỉ kiểm tra luồng với dữ liệu giả lập, được ghi rõ **MOCK** trong kết quả; không dùng để đánh giá chất lượng OCR. Khi model thật lỗi, chương trình không chuyển sang mock.

Cell DocLayNet gọi script có sẵn `scripts/download_datasets/download_doclaynet.py`. Các biến `DOCLAYNET_DOWNLOAD_COUNT=50` và `DOCLAYNET_RUN_COUNT=20` độc lập; ảnh nằm trong `method2/outputs/doclaynet_sample_test/images`, COCO annotation tại `annotations/test.json`. Đây là ảnh do Dataset Viewer cung cấp; tọa độ annotation được quy đổi theo kích thước ảnh thực tải. Mẫu lấy theo thứ tự để thử luồng, không phải mẫu ngẫu nhiên đại diện. Có thể chạy riêng từ thư mục gốc repository:

```powershell
python scripts/download_datasets/download_doclaynet.py --lightweight --splits test --max-images 50 --sample-output-dir method2/outputs/doclaynet_sample_test
```

## Kết quả hiển thị và xuất file

Mỗi trang có:

- Ba ảnh đặt cạnh nhau: layout ban đầu, polygon dòng chữ và layout cuối kèm số thứ tự đọc; màu nhãn nhất quán và có chú giải.
- Bảng vùng và bảng quyết định MERGE/SPLIT/RELABEL/REFINE_BBOX, gồm nguồn, trạng thái, điểm trước/sau và lý do abstain/rollback.
- Nội dung Markdown, crop từng vùng, văn bản OCR, bảng HTML và công thức LaTeX được render.
- Cảnh báo, lỗi suy luận, JSON và provenance đầy đủ có thể mở rộng để xem.

`MAX_REGION_PREVIEWS=20` giới hạn số crop xem trước, không giới hạn xử lý hoặc xuất kết quả. Đặt `None` để xem toàn bộ. `SAVE_OUTPUTS=True` mặc định lưu tại thư mục riêng cho bản có phân loại residual, để giữ kết quả trước đó phục vụ so sánh:

```text
method2/outputs/notebook_residual_glm/<tên tài liệu>/page_0001/
    result.json
    result.md
    layout_original.png
    lines.png
    layout_final.png
    comparison.png
```

`results` chứa các `PageParseResult` trong kernel để phân tích tiếp. `glm_requests` đếm số lượt suy luận GLM cục bộ, không còn là số request HTTP. Nội dung HTML được làm sạch trước khi render. Các trang được nạp lần lượt để không giữ toàn bộ ảnh PDF trong RAM.

### Phân loại vùng chưa có nhãn bằng GLM

`CLASSIFY_UNLABELED_REGIONS=True` mặc định bật bước này. Sau OCR nhẹ, các vùng `residual`/`unknown`/`unlabeled` được đưa qua GLM dưới dạng crop riêng đúng bbox mục tiêu (phóng to tối đa 3 lần, cạnh dài không quá 1280), không ghép trang hoặc vùng lân cận vào ảnh. GLM nhận hướng dẫn ngắn (không gửi nguyên schema) và trả JSON với `label` và `text` được kiểm tra nghiêm ngặt; các nhãn có thể là `text`, `title`, `figure_title` (Caption), `header`, `footer`, `footnote`, `list_item`, `table`, `formula` hoặc `unknown`.

Đây là task phân loại bổ sung dùng cơ chế structured extraction của GLM-OCR, chưa phải chức năng phân loại layout đã được benchmark chính thức. Ba prompt nhận dạng chuẩn vẫn được giữ cho đọc chữ/bảng/công thức. Mỗi lần phân loại dùng chung hạn mức tối đa 30 lượt GLM/trang và giới hạn `min(GLM_MAX_NEW_TOKENS, 768)` token. Decoder chỉ cho phép tiền tố JSON chứa nhãn hợp lệ và dừng khi JSON hoàn chỉnh; phần văn bản vẫn do mô hình nhận dạng. Nếu JSON sai hoặc bị cắt, thử lại tối đa một lần trong cùng ngân sách; giữ cả hai bằng chứng để kiểm tra. Không tự suy nhãn từ schema lỗi hoặc thay nội dung OCR có sẵn. Nếu vẫn lỗi, giữ vùng cần kiểm tra và abstain như trước.

Nhãn chỉ được áp dụng sau kiểm tra vị trí/font/khoảng cách tới hình hoặc bảng/dấu đầu dòng và độ khớp transcript với OCR độc lập. Vẫn giữ quy tắc hơn KEEP ít nhất 0.10 và đạt 0.60. Nhãn `table`/`formula` phải qua thêm lượt parse bằng prompt chuẩn và các validator cấu trúc hiện có. JSON lỗi, `unknown`, nội dung không khớp, hết budget hoặc thiếu bằng chứng sẽ giữ vùng cũ và đánh dấu `needs_review`; không ép mọi vùng thành Text. Đổi nhãn thông thường giữ nguyên nội dung OCR, bbox và lineage. Notebook hiển thị riêng bảng nhãn GLM đề xuất và lịch sử quyết định.

Bật/tắt bằng `routing.classify_unlabeled_regions` trong YAML hoặc biến notebook. Khi tắt semantic refinement hoặc RELABEL cho ablation, bước phân loại cũng được bỏ qua. Để nạp mã mới trong kernel đang mở, Restart Kernel rồi chạy lại notebook; các model/ảnh đã cache được tái sử dụng. Cần chạy lại trên dữ liệu thật để biết F1 có tăng hay không; kết quả cũ không tự thay đổi.

## Mã nguồn và các ràng buộc

- `src/adapters/glm_ocr_local.py`: model/processor nạp một lần, đóng băng trọng số, `inference_mode`, đổi BGR sang RGB và chỉ giải mã token mới; ghi nhận output bị cắt và lỗi thiếu bộ nhớ.
- `src/document_io.py`, `src/notebook_visuals.py`: đọc ảnh/PDF theo trang, hiển thị và xuất overlay.
- `src/pipeline.py`: pipeline bốn giai đoạn, có callback tiến độ tùy chọn `progress_callback(stage, message)`. Notebook gọi bằng `await` để dùng event loop Jupyter có sẵn.

Giữ nguyên schema, nguồn gốc vùng/dòng, hard cross-column guard, bảo toàn dòng, ngân sách tối đa 30 lượt GLM/trang và một vòng sửa đổi. Chỉ chấp nhận nếu phương án vượt KEEP ít nhất 0.10 và đạt 0.60; giả thuyết cạnh tranh gần điểm nhau sẽ abstain. Bảng cần khớp hình học và nội dung cell; công thức cần bằng chứng OCR độc lập; tách vùng phải dựa trên khoảng cách dòng và không mất dòng. Lắp ráp có rollback khi vi phạm ràng buộc.

Vùng khôi phục được lưu riêng trong `recovered_regions`. Vùng cuối có lineage về `original_regions` hoặc `recovered_regions`; `original_regions` và `original_lines` không bị sửa.

## Kiểm thử và chạy ngoài notebook

```powershell
python -m pytest tests -q
# CLI tùy chọn cũng chạy model cục bộ:
python -m src.cli path/to/page.png --glm-device auto --output outputs/real
python -m src.cli path/to/page.png --mock --output outputs/demo
```

Kiểm thử gồm các ràng buộc thuật toán, adapter cục bộ với model/processor tiêm vào, ảnh/PDF nhiều trang, đường dẫn tiếng Việt, ảnh overlay và thực thi toàn bộ notebook bằng kernel mới với mock. Bản notebook đã thực thi để kiểm chứng được lưu trong `outputs/notebook_validation/` và ghi rõ MOCK. Một số sandbox Windows cần quyền loopback để tạo event loop và kernel Jupyter.

CLI đánh giá A0–A9 tiếp tục hỗ trợ predictions đo sẵn hoặc chạy cục bộ A1–A9:

```powershell
python -m src.evaluation.cli path/to/dataset --configurations A0 A9 --output outputs/ablation.json
python -m src.evaluation.cli path/to/dataset --live --configurations A1 A3 A9 --glm-device auto
```

Mỗi file JSON dataset có `image`, `ground_truth` và `predictions` theo cấu hình khi đánh giá output đo sẵn. Ground truth có `revisions`, và tùy chọn `reading_order`, `regions` (`bbox`, `label`), `text`. Callback `run_ablation_suite(dataset_dir, configurations, runner=...)` nhận `runner(record, name, config_overrides)`. A0 cần prediction SDK gốc, không được giả lập bằng pipeline này. Chỉ số thứ tự đọc dùng normalized edit similarity: 1 là hoàn hảo. Over-correction rate hiện là số sửa sai chia tổng số sửa được chấp nhận.

## Giới hạn hiện tại

Đây là PoC zero-training với scoring heuristic chưa hiệu chuẩn trên dataset GT. Bảng nhiều dòng trong một cell hoặc rowspan/colspan được abstain bảo thủ. Thiếu OCR độc lập có thể khiến sửa semantic abstain. CPU có thể chậm; GPU 4 GB vẫn có thể OOM với crop hoặc lượng token lớn. Test mock không chứng minh mô hình thật đã chạy thành công hay tốt hơn baseline. Chỉ kết luận chất lượng sau khi chạy với trọng số thật và dữ liệu GT.

Nguồn sử dụng GLM trực tiếp: [Transformers GLM-OCR](https://huggingface.co/docs/transformers/model_doc/glm_ocr), [model card GLM-OCR](https://huggingface.co/zai-org/GLM-OCR).

Trên Windows, ba adapter Paddle GPU dùng chung một tiến trình Python cục bộ riêng để tránh xung đột DLL cuDNN với PyTorch/GLM. Đây là xử lý nội bộ, không mở API/server. Cell khởi tạo kiểm tra import Paddle trước khi nạp GLM; cell giải phóng đóng cả tiến trình Paddle. Sau khi cập nhật code hoặc cài thư viện, Restart Kernel rồi chạy lại. Tách tiến trình không làm tăng VRAM khả dụng.

Trong tiến trình Paddle, tắt khả năng phát hiện/import Torch của các dependency tùy chọn (ModelScope/Transformers), vì chúng có thể tự nạp Torch ngay cả khi chỉ tạo mô hình Paddle. Việc này chỉ áp dụng trong tiến trình con; GLM trong kernel vẫn dùng Torch. Đã kiểm tra GPU thật trên ảnh DocLayNet 1.png: 6 vùng layout, 33 dòng chữ, 3/3 crop OCR có nội dung. Môi trường thử có cảnh báo Paddle biên dịch với cuDNN 9.9 nhưng dependency đang là 9.5; lần thử này vẫn thành công, chưa xác minh toàn bộ 20 ảnh. Lỗi DLL/import sẽ dừng batch ngay ở trang đầu.

Bộ nhãn chính thức của SDK GLM-OCR (PP-DocLayoutV3) và mapping qua nhãn dự án sang DocLayNet: [bảng đầy đủ](docs/label_mapping.md), [JSON mapping](configs/label_mapping.json). Adapter layout và script đánh giá dùng chung `src/label_mapping.py`; mapping không tạo thêm chức năng phân loại của GLM và không tự bỏ các dự đoán Unmapped.

Bộ đọc JSON phân loại có một chuẩn hóa cú pháp hẹp cho lỗi thực tế: chỉ xóa một dấu ngoặc kép thừa ngay sau khi chuỗi `text` cuối cùng đã đóng (đuôi `""}`). Giữ nguyên nội dung thô trong evidence và ghi `normalized_json:removed_redundant_final_quote`; notebook hiển thị ở cột `normalization`. Không suy nhãn, không sửa chữ, không bỏ kiểm tra độc lập; schema sai, thiếu dấu phẩy, quote bên trong chưa escape và câu trả lời bị cắt vẫn bị từ chối.

Đã chuyển phân loại residual sang crop mục tiêu riêng (`v3-target-only`). Vị trí trên trang chỉ được truyền qua bbox chuẩn hóa trong prompt; không nới ràng buộc OCR/hình học. Cell thử một crop sau khởi tạo GLM cho phép so sánh prompt OCR chuẩn và phân loại trên đúng cùng ảnh, dùng model hiện có; kết quả chẩn đoán ở `outputs/residual_target_probe/`. Đặt `RUN_TARGET_PROBE=True` để thử trước batch.

Notebook c? th?m cell b?o c?o layout (Precision/Recall/Micro-F1/Macro-F1/accuracy ph?n lo?i, IoU 0.5 v? 0.75, baseline/final c?ng taxonomy) v? cell so s?nh final v?i GT DocLayNet (m?c ??nh 3 ?nh). Ch?ng d?ng k?t qu? th?t ?? l?u, kh?ng n?p hay ch?y model; m?c ??nh g?p heading nh? ?? th?ng nh?t. Cell l?u snapshot/CSV/JSON/Markdown v? ?nh so s?nh trong `outputs/evaluation/notebook_<th?i gian>/`. C? th? ch? ch?y cell thi?t l?p ???ng d?n, c?u h?nh, r?i hai cell m?i ?? xem b?o c?o m? kh?ng ch?y l?i OCR. Ch?a c? s? li?u th?c nghi?m Docling/M2Doc c?ng protocol n?n kh?ng g?n s? ?? so s?nh tr?c ti?p.

Cell b?o c?o ? m?c 7 ch?y ??c l?p ngay tr?n kernel m?i: t? t?m method2, import th? vi?n hi?n th? v? l?y ???ng d?n m?c ??nh c?a k?t qu? ?? l?u. Kh?ng c?n ch?y cell c?u h?nh, n?p model, t?i m?u ho?c ch?y pipeline. N?u c?c bi?n c?u h?nh ?? c? th? d?ng ???ng d?n c?a ch?ng; c? th? s?a tr?c ti?p `EVAL_PREDICTIONS`/`EVAL_ANNOTATIONS` trong cell. Cell ?nh ? m?c 8 ch? c?n ch?y sau m?c 7.

Cell b?o c?o c? bi?u ?? ngang hai panel: Precision/Recall/F1 strict theo t?ng l?p v? Strict F1 so v?i Loose F1 (ch? ??nh v?), c?ng micro P/R/F1 v? accuracy c? ?i?u ki?n. Bi?u ?? hi?n th? inline v? l?u `layout_metrics.png`/`.svg`. L?p kh?ng c? GT/prediction ghi N/A, gi? ri?ng Unmapped khi c?. Loose theo c?ng protocol evaluator d?ng chung: gh?p kh?ng r?ng bu?c nh?n, TP/FN nh?m theo l?p GT, FP ch?a gh?p nh?m theo l?p d? ?o?n.

M?c 9 l? cell ??c l?p t?nh COCO mAP@[.50:.95], AP50/AP75 v? strict Micro-F1 b?ng ??ng `src/common/metrics.py`. Baseline d?ng score PP-DocLayoutV3 th?t; final k? th?a min score c?c ngu?n layout, ho?c min score detector d?ng cho residual (proxy kh?ng hi?u chu?n, kh?ng coi l? confidence GLM). Thi?u score th?t th? AP N/A, kh?ng ??t score m?c ??nh. Unmapped b? lo?i kh?i COCO AP v? ???c th?ng k?; strict F1 v?n t?nh ch?ng l? FP. Cell ??c snapshot ?? c? ho?c t?o snapshot m?i, kh?ng ch?y l?i model v? kh?ng ghi ?? prediction. Dependency `pycocotools` ?? ???c th?m v?o requirements notebook.
