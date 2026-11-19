# Hậu xử lý MinerU v2.1: attribution dòng OCR và hàng xóm

Hai notebook S/V3 dùng chung `src/cascaded/attribution.py`. Tầng PP/YOLO/recovery,
model, prompt layout-only, kích thước crop và ngưỡng semantic/Q được giữ như v2.
`fusion.finish_layout` giữ lại làm tham chiếu v2; runner gọi hàm mới.

## Bốn bước quyết định

1. Đổi bbox MinerU từ crop sang trang gốc, không clip theo target và không đo tỷ lệ
   diện tích bbox nằm trong target. Bbox không giao polygon dòng mục tiêu là
   `PURE_CONTEXT_BOX`. Thiếu mọi bbox liên quan: `STEP1_NO_BOX_TOUCHED_TARGET`.
2. Giữ cố định ownership trước khi xét bất kỳ target nào. Tách dòng mục tiêu, dòng
   đã có chủ, orphan khác và vùng đã chốt ngoài tập fallback. Đo nhiễm bằng diện tích
   hợp polygon OCR giao bbox, không tính khoảng trắng. Candidate độc lập cần nhiễm
   <=0.10; vùng 0.10–0.15 không được tự cho qua. Vùng phi chữ bảo vệ theo
   `intersection/min(area_candidate, area_protected)` <=0.15, không có ngoại lệ theo nhãn.
3. Mỗi bbox cần ít nhất một dòng mục tiêu hợp lệ. Tìm partition không trùng dòng,
   IoU từng cặp <=0.30, tối đa24 ứng viên và beam32, chọn Q cao nhất trong các phương án
   khả thi của beam. Đây là tìm kiếm có giới hạn, không bảo đảm tối ưu toàn cục.
   Orphan cần >=80% **số dòng**; REPLACE cần đủ tất cả dòng mục tiêu và mọi dòng fallback.
   Dòng orphan còn thiếu được giữ ở review, GT thiếu vẫn tính FN.
4. Chỉ tính Q khi có partition khả thi. Giữ công thức v2:
   `Q = 0.45*coverage_area - 0.30*split_penalty + 0.25*semantic`.
   Mỗi bbox semantic>=0.60 và Q mới phải >Q KEEP+0.10. Q coverage dùng diện tích,
   điều kiện bước3 dùng số dòng; log lưu cả hai. Không có ứng viên thì không ghi lỗi Q.

## Snap và chia dòng

Dòng có polygon coverage>=0.80 được nhận bình thường. Nếu chưa đủ, chỉ snap cạnh
trên/dưới khi CovX>=0.85, CovY>=0.50, tâm dọc cách hai mép bbox ít nhất0.10 chiều cao
dòng và dòng gần ngang (góc cạnh dài của hình chữ nhật xoay nhỏ nhất <=10 độ).
Snap chỉ nới trục Y, giữ nguyên trục X. Tính lại coverage/nhiễm/overlap sau snap.
Dòng xoay chỉ dùng coverage polygon, không suy diễn từ envelope để snap.

Chia ngang không được chấp nhận chỉ dựa trên hai bbox bổ sung nhau. Ghi
`HORIZONTAL_SPLIT_UNVERIFIED`; chưa có tọa độ từ/pixel khoảng trắng để chứng minh.
Cắt dọc qua thân dòng không đủ bằng chứng ghi `LINE_BISECTION`.

## ABSORB một hàng xóm Text

Bbox có dòng của hàng xóm được xét ở nhánh ABSORB, kể cả khi tỷ lệ nhiễm thấp:
không để bbox mới lấy mất dòng đã có chủ. Chỉ nhận đúng một hàng xóm label=`text`,
MinerU cũng dự đoán Text, không lẫn orphan khác và không đang thay fallback conflict.
Cùng cột, lệch lề trái <=một chiều cao dòng, chiều cao lệch <20%, dòng nối liên tục;
chặn dấu hiệu caption/list/heading. Hàng xóm được sửa tối đa một lần mỗi trang.

Bbox mới là envelope(bbox Text cũ + polygon dòng mục tiêu được nhận), không phải bbox
rộng của MinerU. Giữ ID, nhãn, nguồn và toàn bộ dòng cũ; bổ sung lineage MinerU,
stage=`mineru_absorb`, bỏ score detector vì geometry mới không còn score đo được.
Envelope không được quét thêm orphan, owner khác hoặc mở rộng qua vùng khác.
So KEEP/ABSORB trên hợp dòng cũ + dòng mục tiêu. Cập nhật nguyên khối hoặc hoàn tác;
không gán lại ownership toàn trang sau commit.

## Nhật ký và kết quả

- Lỗi model/cắt output/OOM/hết ngân sách: `MODEL_*`, dừng trước Q.
- Candidate: `VERTICAL_SNAP_APPLIED`, `FOREIGN_CONTAMINATION`,
  `PROTECTED_NONTEXT_OVERLAP`, `STEP2_VALID_CANDIDATE` và lý do cụ thể khác.
- Partition thiếu: `STEP3_INCOMPLETE_TARGET_SET_COVERAGE`, `NO_VALID_PARTITION`.
- Quyết định: `STEP4_ACCEPTED_MINERU`, `ACCEPTED_NEIGHBOR_MERGE`,
  `STEP4_LOST_TO_KEEP_BASELINE`, `ATOMIC_ROLLBACK`.

Mỗi bbox lưu tọa độ thô/trang/trước và sau snap, CovX/CovY/polygon, line IDs,
nhiễm, hàng xóm, semantic, việc được chọn vào partition và mã loại. Partition được
chọn vẫn có thể thua KEEP; màu magenta trong overlay không có nghĩa đã được chấp nhận.

Output mới: `outputs/cascaded_layout/<s|v3>/<real|mock>/v2.1/<hash>/`.
Không ghi đè cache hay kết quả v2. Mỗi ảnh có `result.json`, `mineru.json`, crop;
báo cáo có thêm `mineru_boxes.csv`, `decisions.csv`, checkpoint summary trong metrics.
Chỉ hai báo cáo cùng schema, cấu hình, ảnh/GT/taxonomy mới được so sánh.

## Chạy trên Windows

Restart Kernel để nạp mã mới. Giữ `MOCK_MODE=False`, chọn đúng môi trường đã cài model,
rồi Run All từng notebook, đóng kernel trước khi chạy notebook kia. Giữ mặc định
`SPLIT='valid'`, `RUN_COUNT=20`, `SMOKE_FIRST=True`; dữ liệu vẫn ở `datasets/` thư mục gốc.
Không cần cài thêm dependency cho thay đổi hậu xử lý này. Không chạy song song hai notebook
trên GPU4GB. Chưa có kết luận cải thiện trước khi bạn chạy lại cả hai trên cùng20 ảnh.

Lần triển khai này chỉ chạy một ảnh thật theo yêu cầu, không chạy lại batch20 ảnh.
Kiểm thử hình học và notebook MOCK chỉ xác nhận hành vi/luồng, không chứng minh chất lượng model.
Kết quả smoke thực tế và giới hạn đã kiểm chứng:
[cascaded_layout_v2_1_validation.md](cascaded_layout_v2_1_validation.md).
