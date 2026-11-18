"""Deterministically generate the two sibling notebooks. No model loading."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text, id):
    value = {'cell_type': kind, 'id': id, 'metadata': {}, 'source': text.strip()+'\n'}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def notebook(variant):
    model = 'PP-DocLayout-S' if variant == 's' else 'PP-DocLayoutV3'
    cells = []
    def md(text, id):
        cells.append(cell('markdown', text, id))
    def code(text, id):
        cells.append(cell('code', text, id))
    md(f'''# Cascaded Layout v2.1 — {model} + YOLO nhỏ + MinerU
Windows / GPU 4 GB. Layout-only, không dùng GLM hoặc sinh lại nội dung OCR.
PP giữ baseline; YOLO chỉ bổ sung bbox nhỏ; MinerU qua cổng KEEP/REPLACE/ABSORB.
Bbox MinerU được kiểm chứng theo dòng OCR và hàng xóm, không theo diện tích target.
Vùng chưa giải quyết vào review riêng, không xuất detection unknown.
Không cam kết 30–50 ms/trang. MOCK chỉ kiểm thử luồng, không phải chất lượng model.
Notebook cũ và kết quả Dual-Skip v4 được giữ riêng. Xem `docs/cascaded_layout.md`
và `docs/cascaded_layout_v2_1.md`. Output mới nằm trong nhánh `v2.1/` riêng.
''', 'intro')
    code('''import sys
from pathlib import Path
roots = [p for base in (Path.cwd(), *Path.cwd().parents) for p in (base, base/'method2')]
METHOD2_ROOT = next(p.resolve() for p in roots if (p/'src/cascaded/runner.py').is_file())
existing = sys.modules.get('src')
if existing is not None and (not getattr(existing, '__file__', None) or Path(existing.__file__).resolve().parent != METHOD2_ROOT/'src'):
    raise RuntimeError('Kernel đã nạp package src khác. Restart Kernel.')
sys.path.insert(0, str(METHOD2_ROOT))
print('Kernel:', sys.executable)
print('Method 2:', METHOD2_ROOT)
''', 'paths')
    md('''## 1. Môi trường riêng
Cài PyTorch + torchvision CUDA phù hợp trước, rồi cài requirements riêng.
Cài `paddlepaddle-gpu` theo hướng dẫn Windows của Paddle phù hợp CUDA/driver.
Không tự gỡ hoặc thay bản PyTorch đang hoạt động. Cell dưới mặc định tắt.
Sau khi cài thư viện, Restart Kernel. Không dùng vLLM trên Windows native.
''', 'dependencies-info')
    code('''INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(METHOD2_ROOT/'requirements-cascaded.txt')])
import importlib.metadata as metadata
for name in ['torch', 'torchvision', 'paddlepaddle-gpu', 'paddleocr', 'doclayout-yolo', 'mineru-vl-utils', 'transformers']:
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, 'CHƯA CÀI')
''', 'dependencies')
    md('''## 2. Cấu hình chung
Chạy một ảnh thật trước (`SMOKE_FIRST=True`), rồi chạy đủ 20 ảnh ở cell kế tiếp.
Chỉ chỉnh ngưỡng trên validation; khi dùng test, giữ cấu hình đã chốt.
Đổi `MOCK_MODE=True` để thử luồng mà không tải trọng số.
Hai notebook phải dùng cùng danh sách ảnh và cấu hình ngoài biến thể detector.
''', 'config-info')
    code(f'''from src.cascaded.types import Config
MOCK_MODE = False
SPLIT = 'valid'  # validation để so sánh phương án; test chỉ báo cáo cấu hình đã chốt
RUN_COUNT = 20
SMOKE_FIRST = True
from src.cascaded.dataset import doclaynet_processed_dir
DATASET_DIR = doclaynet_processed_dir(METHOD2_ROOT, SPLIT)
INPUT_PATH = ''  # Đường dẫn ảnh/PDF riêng; để trống dùng DocLayNet
PAGES = None
PDF_DPI = 150
CONFIG = Config(variant='{variant}', device='cuda', mock=MOCK_MODE,
    yolo_max_area=0.02, yolo_max_height=0.08,
    enable_yolo=True, enable_recovery=True, enable_mineru=True,
    enable_wbf=False,  # WBF: ablation riêng, mặc định tắt
    orphan_line_fraction=0.80, max_foreign_ink=0.10,
    max_nontext_overlap=0.15, partition_candidates=24, partition_beam=32)
OUTPUT_ROOT = METHOD2_ROOT/'outputs/cascaded_layout'
if SPLIT == 'test':
    print('TEST: không chọn ngưỡng theo kết quả này.')
print(CONFIG)
print('Dataset:', DATASET_DIR)
''', 'config')
    md('''## 3. Chuẩn bị ảnh
Dùng dữ liệu chung tại `datasets/doclaynet*/processed/` theo split của dự án.
Khi chạy cell này, nếu thiếu ảnh hoặc GT thì tự gọi script tải ở thư mục gốc.
Nếu đã đủ ảnh thì không tải lại. Không dùng ảnh preview hoặc lưu dataset vào outputs.
''', 'data-info')
    code('''from src.cascaded.runner import prepare_records, Experiment
records, annotation = prepare_records(METHOD2_ROOT, DATASET_DIR, SPLIT, RUN_COUNT,
    INPUT_PATH, PAGES, PDF_DPI, mock=MOCK_MODE)
print('Ảnh:', len(records), '| GT:', annotation)
from src.cascaded.runner import check_runtime
readiness = check_runtime(CONFIG)
print(readiness)
if not readiness['ready']:
    raise RuntimeError('Runtime chưa sẵn sàng. Cài dependency trong đúng kernel; không tự chuyển sang MOCK.')
''', 'data')
    md('''## 4. Thử một ảnh trước
Mỗi pha đóng model trước pha tiếp theo. Lỗi nạp model dừng rõ ràng.
Lỗi crop MinerU giữ KEEP baseline; orphan không có baseline vào review_evidence.
Cache model và prediction đều được tái sử dụng khi cấu hình/ảnh không đổi.
''', 'smoke-info')
    code('''if SMOKE_FIRST:
    smoke = Experiment(CONFIG, records[:1], OUTPUT_ROOT/'smoke')
    for phase in ('paddle', 'yolo', 'mineru'):
        print(smoke.run_phase(phase))
    from src.cascaded.report import preview
    preview(smoke.results()[0], annotation=annotation)
''', 'smoke')
    md('## 5. Chạy theo pha trên toàn tập ảnh', 'phases-info')
    code('''experiment = Experiment(CONFIG, records, OUTPUT_ROOT)
print('Thư mục:', experiment.root)
''', 'experiment')
    for phase in ('paddle', 'yolo', 'mineru'):
        code(f"print(experiment.run_phase('{phase}'))", phase)
    md('''## 6. Xem kết quả và bằng chứng
Panel cuối màu vàng là review_evidence, không nằm trong final_detections.
Cam: score thấp/không có score detector; xanh: proposal mạnh.
Màu chỉ dùng hiển thị, không vẽ vào ảnh gửi MinerU.
Trong crop: đỏ = dòng mục tiêu, cyan = vùng hàng xóm đã chốt,
cam nét đứt = bbox MinerU, magenta = partition được chọn để so Q (chưa chắc được nhận).
Bảng từng bbox cho biết snap, nhiễm dòng ngoài, hàng xóm và mã bị loại.
''', 'preview-info')
    code('''from src.cascaded.report import preview, timing_report, review_summary, attribution_rows
from IPython.display import display
import pandas as pd
results = experiment.results()
for result in results[:3]:
    preview(result, annotation=annotation)
    display(pd.DataFrame(result['evidence']))
    display(pd.DataFrame(attribution_rows([result])))
    display(pd.DataFrame(result['decisions']))
display(pd.json_normalize(timing_report(results)))
display(pd.json_normalize(review_summary(results)))
print('Thời gian tải model/cache, wall time và VRAM toàn GPU: manifest.json')
''', 'preview')
    md('''## 7. Đánh giá layout đã lưu
So sánh bốn mức trên đúng cùng ảnh. GT bị bỏ sót vẫn tính FN dù có evidence review.
Literal giữ 11 lớp, generic title là Unmapped.
Merged-heading là phân tích bổ sung 10 lớp. Accuracy có điều kiện trên bbox đã khớp.
Final AP là N/A: không tạo confidence giả cho MinerU.
Cell này chạy riêng được sau khi mở lại notebook: tự đọc báo cáo đã lưu gần nhất.
Không cần Run All model. Bảng COCO bổ sung cũng tự tính ngay trong cell này.
''', 'evaluation-info')
    code('''from pathlib import Path
import sys
if 'METHOD2_ROOT' not in globals():
    roots = [p for base in (Path.cwd(), *Path.cwd().parents) for p in (base, base/'method2')]
    METHOD2_ROOT = next(p.resolve() for p in roots if (p/'src/cascaded/runner.py').is_file())
    sys.path.insert(0, str(METHOD2_ROOT))
from src.cascaded.report import evaluate
from src.cascaded.runner import read_json
from IPython.display import display
import pandas as pd
MERGED_HEADINGS = False
report = None
if globals().get('MOCK_MODE', False):
    print('MOCK: bỏ qua báo cáo chất lượng thật.')
else:
    # Điền thư mục report cụ thể nếu muốn xem một lượt khác; trống = lượt gần nhất.
    SAVED_REPORT_DIR = ''
    if SAVED_REPORT_DIR:
        REPORT_DIR = Path(SAVED_REPORT_DIR)
        report = read_json(REPORT_DIR/'metrics.json')
    elif 'experiment' in globals() and globals().get('annotation') is not None:
        results = experiment.results()
        if len(results) != len(experiment.records):
            raise RuntimeError('Lượt chạy chưa đủ prediction; hoàn tất các pha trước khi đánh giá.')
        REPORT_DIR = experiment.root/('report_merged' if MERGED_HEADINGS else 'report_literal')
        report = evaluate(results, annotation, REPORT_DIR, merged=MERGED_HEADINGS, run_manifest=experiment.manifest_path)
    elif 'experiment' in globals() and globals().get('annotation') is None:
        print('Ảnh/PDF riêng chưa có GT: chỉ xem kết quả và timing.')
    else:
        base = METHOD2_ROOT/'outputs/cascaded_layout/VARIANT/real/v2.1'
        name = 'report_merged' if MERGED_HEADINGS else 'report_literal'
        choices = sorted(base.glob(f'*/{name}/metrics.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        if not choices:
            raise FileNotFoundError('Chưa có báo cáo đã lưu. Chạy các pha model một lần trước.')
        REPORT_DIR = choices[0].parent
        report = read_json(choices[0])
    if report is not None:
        display(pd.DataFrame(report['rows']))
        print('Báo cáo:', REPORT_DIR)
        for stage, detail in report['details'].items():
            print(stage, 'AP:', detail['ap_status'])
''', 'evaluation')
    cells[-1]['source'] = cells[-1]['source'].replace('/VARIANT/', f'/{variant}/')
    md('''## 8. So sánh S và V3
Sau khi chạy cả hai notebook, điền đúng hai file metrics.json (S trước, V3 sau).
Không tự chọn báo cáo cũ có điểm tốt. Hàm kiểm tra ảnh, hash dữ liệu và taxonomy.
''', 'compare-info')
    code('''from src.cascaded.report import compare
S_REPORT = ''
V3_REPORT = ''
if S_REPORT and V3_REPORT:
    display(compare([Path(S_REPORT), Path(V3_REPORT)], OUTPUT_ROOT/'comparison_v2_1'))
else:
    print('Chạy cả hai notebook rồi điền S_REPORT / V3_REPORT để so sánh.')
''', 'compare')
    md('## 9. Giải phóng (an toàn khi chạy lại)', 'close-info')
    code('''if 'experiment' in globals():
    experiment.close()
if 'smoke' in globals():
    smoke.close()
print('Đã đóng model của các pha.')
''', 'close')
    if variant in ('s', 'v3'):
        md('''## 10. COCO mAP@0.50:0.95 bổ sung từ prediction đã lưu
Hàm dùng chung: `src/common/metrics.py::coco_map`. Cột `map` trước đó đã là
trung bình AP ở 10 ngưỡng IoU 0.50, 0.55, …, 0.95.
`detector_score` dùng score thật, chỉ áp dụng cho baseline có đủ score.
`uniform_score_1` đặt **mọi bbox của mọi stage cùng score=1.0** để đánh giá bổ sung
cả recovery/final. Đây là quy ước đầu vào COCOeval, không phải confidence model.
Các bbox bằng điểm được sắp cố định theo tọa độ/nhãn/ID; AP vẫn có thể phụ thuộc
thứ tự khi bằng điểm và giới hạn COCO maxDets=100 mỗi ảnh/lớp.
Chỉ so các hàng cùng score_policy. Unmapped bị loại khỏi COCO AP và được đếm riêng;
strict F1 ở bảng trước vẫn phạt các prediction này.
Cell chỉ đọc snapshot và GT, không nạp model, không ghi đè metrics.json cũ.
Nếu đã tắt kernel, chạy cell đường dẫn đầu notebook rồi chạy riêng cell này;
đổi COCO_REPORT_DIR nếu muốn chọn lượt chạy khác.
''', 'coco-extra-info')
        code('''from src.cascaded.runner import read_json, write_json
from src.cascaded.report import shared_metrics, convert
from src.cascaded.dataset import doclaynet_processed_dir
from pathlib import Path
import pandas as pd
from IPython.display import display

if 'REPORT_DIR' in globals():
    COCO_REPORT_DIR = Path(REPORT_DIR)
else:
    candidates = sorted((METHOD2_ROOT/'outputs/cascaded_layout/VARIANT/real/v2.1').glob(
        '*/report_literal/metrics.json'), key=lambda p:p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError('Chưa có báo cáo cho biến thể này. Chạy các pha và cell 7 trước.')
    COCO_REPORT_DIR = candidates[0].parent
saved_report = read_json(COCO_REPORT_DIR/'metrics.json')
if saved_report['variant'] != 'VARIANT':
    raise ValueError('Báo cáo không thuộc biến thể của notebook này.')
saved_results = [read_json(COCO_REPORT_DIR/'snapshot'/f'{id}.json')
                 for id in saved_report['image_ids']]
if any(r['mode'] != 'REAL' for r in saved_results):
    raise ValueError('Không đánh giá chất lượng bằng MOCK.')
# Có thể đổi đường dẫn GT nếu dùng dữ liệu riêng.
COCO_GT_PATH = Path(globals().get('annotation') or
    doclaynet_processed_dir(METHOD2_ROOT, 'valid')/'annotations/validation.json')
import hashlib
if hashlib.sha256(COCO_GT_PATH.read_bytes()).hexdigest() != saved_report['annotation_sha256']:
    matches = [p for p in (METHOD2_ROOT.parent/'datasets').glob('doclaynet*/processed/annotations/*.json')
               if hashlib.sha256(p.read_bytes()).hexdigest() == saved_report['annotation_sha256']]
    if len(matches) != 1:
        raise ValueError('Không tìm thấy GT khớp báo cáo; chọn đúng COCO_GT_PATH.')
    COCO_GT_PATH = matches[0]
gt = read_json(COCO_GT_PATH)
ids = set(saved_report['image_ids'])
merged = saved_report['merged_headings']
def mapped(cid):
    return 8 if merged and cid == 11 else cid
categories = {mapped(c['id']):dict(c, id=mapped(c['id'])) for c in gt['categories']}
if merged and 8 in categories:
    categories[8]['name'] = 'Heading'
coco_gt = {'images':[im for im in gt['images'] if im['id'] in ids],
    'annotations':[dict(a, category_id=mapped(a['category_id']))
                   for a in gt['annotations'] if a['image_id'] in ids],
    'categories':list(categories.values())}
rows, details = [], {}
for stage in ('paddle','ensemble','recovery','final'):
    predictions, excluded = [], 0
    for result in saved_results:
        regions = sorted(result['stages'][stage], key=lambda r:(*r['bbox'],r['label'],r['id']))
        for item in convert(regions, merged):
            if item['category_id'] not in categories:
                excluded += 1
                continue
            predictions.append(dict(item, image_id=result['record']['image_id']))
    policies = ['uniform_score_1']
    if stage in ('paddle','ensemble') and all(p.get('score') is not None for p in predictions):
        policies.insert(0, 'detector_score')
    for policy in policies:
        inputs = [dict(p, score=1.0) if policy == 'uniform_score_1' else dict(p) for p in predictions]
        ap = shared_metrics().coco_map(coco_gt, inputs)
        details[stage+':'+policy] = ap
        rows.append({'stage':stage, 'score_policy':policy,
            'coco_map_50_95':ap['overall']['map'], 'coco_map_50':ap['overall']['map50'],
            'coco_map_75':ap['overall']['map75'], 'unmapped_excluded':excluded})
table = pd.DataFrame(rows)
display(table)
table.to_csv(COCO_REPORT_DIR/'coco_map_extra.csv', index=False, encoding='utf-8-sig')
write_json(COCO_REPORT_DIR/'coco_map_extra.json', {'source_report':str(COCO_REPORT_DIR/'metrics.json'),
    'note':'uniform_score_1 is a tied-score diagnostic, not model confidence; maxDets=100.',
    'rows':rows, 'details':details})
print('Đã lưu:', COCO_REPORT_DIR/'coco_map_extra.csv')
''', 'coco-extra')
        cells[-1]['source'] = cells[-1]['source'].replace('VARIANT', variant)
        import textwrap
        cells[-1]['source'] = ("if globals().get('MOCK_MODE', False):\n"
            "    print('MOCK: bỏ qua COCO AP thật.')\nelse:\n"+textwrap.indent(cells[-1]['source'], '    '))
        evaluation_cell = next(c for c in cells if c['id'] == 'evaluation')
        evaluation_cell['source'] += ("\n# COCO bổ sung tự chạy cùng cell 7.\n"
            "if report is not None:\n"+textwrap.indent(cells[-1]['source'], '    '))
    md('''## 11. So sánh trực quan 20 ảnh: prediction cuối và ground truth
Lưu tối đa20 ảnh ghép: GT bên trái, prediction cuối bên phải; chỉ hiển thị5 ảnh đầu.
Cùng lớp dùng cùng màu, nhãn Unmapped màu xám; vùng review không tính là detection.
Đọc snapshot đã đánh giá, không chạy lại model. Folder `visual_comparison_final/`
nằm trong thư mục báo cáo, kèm `index.csv` ghi TP/FP/FN tại strict IoU0.50.
Nếu bộ kết quả có ít hơn20 ảnh thì lưu đúng số ảnh hiện có, không tạo thêm ảnh.
''', 'visual-comparison-info')
    code('''from pathlib import Path
import sys, importlib.util
if not globals().get('MOCK_MODE', False):
    if 'METHOD2_ROOT' not in globals():
        roots = [p for base in (Path.cwd(), *Path.cwd().parents) for p in (base, base/'method2')]
        METHOD2_ROOT = next(p.resolve() for p in roots if (p/'src/cascaded/runner.py').is_file())
        sys.path.insert(0, str(METHOD2_ROOT))
    # Điền thư mục report cụ thể nếu muốn chọn lượt khác.
    VISUAL_REPORT_DIR = ''
    if VISUAL_REPORT_DIR:
        visual_report = Path(VISUAL_REPORT_DIR)
    elif 'REPORT_DIR' in globals():
        visual_report = Path(REPORT_DIR)
    else:
        options = sorted((METHOD2_ROOT/'outputs/cascaded_layout/VARIANT/real/v2.1').glob(
            '*/report_literal/metrics.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        if not options:
            raise FileNotFoundError('Chưa có báo cáo đã lưu; chạy cell 7 trước.')
        visual_report = options[0].parent
    from src.cascaded.runner import read_json
    if read_json(visual_report/'metrics.json')['variant'] != 'VARIANT':
        raise ValueError('Báo cáo không thuộc biến thể notebook này.')
    spec = importlib.util.spec_from_file_location('cascaded_visual_comparison',
        METHOD2_ROOT/'scripts/cascaded_visual_comparison.py')
    visual_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visual_module)
    visual_rows = visual_module.export_comparisons(visual_report, limit=20, display_count=5)
else:
    print('MOCK: bỏ qua so sánh chất lượng thật.')
''', 'visual-comparison')
    cells[-1]['source'] = cells[-1]['source'].replace('VARIANT', variant)
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python', 'version': '3.12'}},
            'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    for variant, name in [('s', 'S'), ('v3', 'V3')]:
        path = ROOT/f'Cascaded_Layout_{name}_YOLO_MinerU.ipynb'
        path.write_text(json.dumps(notebook(variant), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
        print(path.name)
