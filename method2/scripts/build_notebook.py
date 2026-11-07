"""Rebuild the checked-in notebook (authoring only; not needed to run it)."""
from pathlib import Path
import textwrap

import nbformat as nbf


def code(source, *, tags=()):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + '\n', metadata={'tags': list(tags)})


def markdown(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.12'}}
    nb.cells = [
        markdown('''
        # Dual-Skip v4 — chạy trực tiếp và xem kết quả

        Chạy lần lượt các cell trong Jupyter hoặc VS Code. Notebook sử dụng các module trong
        **method2/src**: giữ cả thư mục `method2` bên cạnh notebook. Không cần API/server.

        **Chạy thật là mặc định.** Mặc định lấy mẫu DocLayNet 50 ảnh và chạy 20 ảnh đầu.
        Để dùng ảnh/PDF riêng, đặt `USE_DOCLAYNET_SAMPLE=False` và điền `INPUT_PATH`.
        Lần đầu cần tải trọng số
        PaddleOCR và GLM-OCR, hoặc trỏ tới model đã lưu cục bộ. Paddle chạy GPU 0 (cần paddlepaddle-gpu),
        GLM xử lý từng crop bằng CUDA FP16; có thể đổi `GLM_DEVICE="cpu"` nếu thiếu VRAM.
        ''') ,
        markdown('## 1. Xác định thư mục và chuẩn bị thư viện'),
        code('''
        import sys
        from pathlib import Path

        candidates = [p for base in (Path.cwd(), *Path.cwd().parents)
                      for p in (base, base / "method2")]
        METHOD2_ROOT = next((p.resolve() for p in candidates
                            if (p / "src/pipeline.py").is_file()
                            and (p / "configs/default.yaml").is_file()), None)
        if METHOD2_ROOT is None:
            raise RuntimeError("Mở notebook từ thư mục repository hoặc method2.")
        existing_src = sys.modules.get("src")
        if existing_src is not None:
            origin = getattr(existing_src, "__file__", None)
            if origin is None or Path(origin).resolve().parent != METHOD2_ROOT / "src":
                raise RuntimeError("Kernel đã import package src khác. Restart Kernel rồi chạy lại.")
        if str(METHOD2_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD2_ROOT))
        print("method2:", METHOD2_ROOT)
        print("Python kernel:", sys.executable)
        ''') ,
        code('''
        # Chỉ bật nếu cần cài các thư viện còn thiếu. Không cài/thay bản torch CUDA ở đây.
        INSTALL_DEPENDENCIES = False
        if INSTALL_DEPENDENCIES:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                                   str(METHOD2_ROOT / "requirements-notebook.txt")])
            print("Cài xong. Nếu kernel đã import phiên bản cũ, Restart Kernel trước khi chạy tiếp.")
        ''', tags=['installation']),
        markdown('### Cài Paddle GPU (chạy một lần khi chuyển từ CPU)\n\nChạy cell dependency chung trước, sau đó bật `INSTALL_PADDLE_GPU=True` ở cell dưới.\nCell dùng đúng Python của kernel, gỡ bản Paddle CPU/GPU cũ rồi cài bản GPU CUDA 12.6;\nkhông thay PyTorch. Nếu đã nạp Paddle, Restart Kernel trước khi cài. Cài xong phải\nRestart Kernel và đặt lại cờ thành `False` trước khi Run All.\n\nLệnh theo [hướng dẫn Windows chính thức của Paddle](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/windows-pip_en.md).\nGPU 4 GB có thể thiếu VRAM khi chạy đồng thời Paddle và GLM.\n'),
        code('# Chạy sau cell cài dependency chung; chỉ bật khi cần chuyển Paddle CPU sang GPU.\n# Restart Kernel trước nếu đã nạp Paddle, và Restart Kernel sau khi cài xong.\nINSTALL_PADDLE_GPU = False\nPADDLE_GPU_VERSION = "3.2.2"\nPADDLE_GPU_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"\nif INSTALL_PADDLE_GPU:\n    import subprocess\n    import sys\n    if "paddle" in sys.modules:\n        raise RuntimeError("Paddle đã được nạp. Restart Kernel rồi chạy cell này trước khi nạp mô hình.")\n    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y",\n                           "paddlepaddle", "paddlepaddle-gpu"])\n    subprocess.check_call([sys.executable, "-m", "pip", "install",\n                           f"paddlepaddle-gpu=={PADDLE_GPU_VERSION}", "-i", PADDLE_GPU_INDEX])\n    print("Đã cài Paddle GPU. Restart Kernel, giữ INSTALL_PADDLE_GPU=False khi chạy lại.")\nelse:\n    print("Bỏ qua cài Paddle GPU. Đổi INSTALL_PADDLE_GPU=True nếu đang dùng bản CPU.")\n', tags=['installation', 'paddle-gpu-installation']),
        code('''
        import importlib.metadata as metadata
        from IPython.display import display, Markdown, HTML
        import pandas as pd
        import torch

        packages = ["torch", "transformers", "paddlepaddle", "paddlepaddle-gpu", "paddleocr", "Pillow",
                    "matplotlib", "pandas", "PyMuPDF", "ipykernel"]
        versions = []
        for package in packages:
            try:
                version = metadata.version(package)
            except metadata.PackageNotFoundError:
                version = "CHƯA CÀI — bật cell cài thư viện ở trên"
            versions.append({"thư viện": package, "phiên bản": version})
        display(pd.DataFrame(versions))
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"GPU: {torch.cuda.get_device_name(0)} | trống {free / 2**30:.2f}/{total / 2**30:.2f} GiB")
        else:
            print("CUDA không khả dụng: GLM auto sẽ chạy CPU, có thể chậm hơn.")
        ''', tags=['environment']),
        markdown('''
        ## 2. Cấu hình

        `PAGES=None` chạy tất cả các trang PDF; `[1, 3]` chạy trang 1 và 3. Ảnh chỉ có trang 1.
        Đường dẫn tương đối được tính từ thư mục làm việc của kernel; nên dùng đường dẫn tuyệt đối.
        `MAX_REGION_PREVIEWS=None` để xem tất cả crop. Thay model/device thì chạy lại cell khởi tạo.
        ''') ,
        code('''
        USE_DOCLAYNET_SAMPLE = True
        DOCLAYNET_DOWNLOAD_COUNT = 50
        DOCLAYNET_RUN_COUNT = 20
        DOCLAYNET_SPLIT = "test"
        DOCLAYNET_SAMPLE_DIR = METHOD2_ROOT / "outputs" / f"doclaynet_sample_{DOCLAYNET_SPLIT}"
        INPUT_PATH = r""  # Ví dụ: r"D:\\Tai lieu\\trang.png" hoặc r"D:\\Tai lieu\\bao cao.pdf"
        PAGES = None
        PDF_DPI = 150
        GLM_MODEL = "zai-org/GLM-OCR"  # Hoặc đường dẫn thư mục model đã tải
        GLM_DEVICE = "auto"          # "auto", "cuda", "cpu"
        GLM_MAX_CROP_EDGE = 1280
        GLM_MAX_NEW_TOKENS = 1024
        MAX_REGION_PREVIEWS = 20
        SAVE_OUTPUTS = True
        OUTPUT_ROOT = METHOD2_ROOT / "outputs/notebook_residual_glm"
        MOCK_MODE = False            # Chỉ bật để kiểm tra luồng; không OCR nội dung thật
        CLASSIFY_UNLABELED_REGIONS = True  # GLM phân loại residual với ngữ cảnh trang
        ''', tags=['parameters']),
        markdown('''
        ### Tải mẫu DocLayNet nhẹ: 50 ảnh, chạy thử 20 ảnh

        Cell này gọi script `scripts/download_datasets/download_doclaynet.py` của repository ở chế độ
        `--lightweight`: chỉ tải ảnh và annotation được chọn, không tải cả shard parquet.
        Dữ liệu đã có được tái sử dụng. Mặc định lấy 50 ảnh đầu của split test và chạy 20 ảnh đầu,
        phù hợp kiểm tra luồng; đây không phải mẫu ngẫu nhiên đại diện để kết luận benchmark.
        ''') ,
        code('''
        if USE_DOCLAYNET_SAMPLE:
            import json
            import subprocess

            DOCLAYNET_SAMPLE_DIR = Path(DOCLAYNET_SAMPLE_DIR).expanduser().resolve()
            if not 1 <= DOCLAYNET_RUN_COUNT <= DOCLAYNET_DOWNLOAD_COUNT <= 1000:
                raise ValueError("Cần 1 <= số ảnh chạy <= số ảnh tải <= 1000.")
            download_script = METHOD2_ROOT.parent / "scripts/download_datasets/download_doclaynet.py"
            if not download_script.is_file():
                raise FileNotFoundError(f"Thiếu script tải DocLayNet: {download_script}")
            subprocess.run([sys.executable, str(download_script), "--lightweight", "--splits", DOCLAYNET_SPLIT,
                "--max-images", str(DOCLAYNET_DOWNLOAD_COUNT), "--sample-output-dir", str(DOCLAYNET_SAMPLE_DIR)],
                check=True)
            sample_manifest = json.loads((DOCLAYNET_SAMPLE_DIR / "annotations" / f"{DOCLAYNET_SPLIT}.json")
                                         .read_text(encoding="utf-8"))
            sample_records = sample_manifest["images"][:DOCLAYNET_RUN_COUNT]
            if len(sample_records) != DOCLAYNET_RUN_COUNT:
                raise ValueError("Chưa đủ ảnh mẫu; chạy lại cell tải dữ liệu.")
            doclaynet_run_paths = [DOCLAYNET_SAMPLE_DIR / "images" / record["file_name"] for record in sample_records]
            if not all(path.is_file() for path in doclaynet_run_paths):
                raise FileNotFoundError("Thiếu ảnh mẫu; chạy lại cell tải dữ liệu để bổ sung.")
            print(f"Đã chuẩn bị {len(sample_manifest['images'])} ảnh; sẽ chạy {len(doclaynet_run_paths)} ảnh.")
            display(pd.DataFrame(sample_records)[["id", "file_name", "width", "height"]])
        else:
            print("Dùng INPUT_PATH; bỏ qua tải DocLayNet.")
        ''', tags=['doclaynet-download']),
        markdown('## 3. Nạp ảnh/PDF và xem trang đầu đã chọn'),
        code('''
        import itertools
        import cv2
        from PIL import Image
        from src.document_io import iter_document_pages

        if not USE_DOCLAYNET_SAMPLE and not str(INPUT_PATH).strip():
            raise ValueError("Hãy điền INPUT_PATH trong cell cấu hình trước.")
        if "page_iterator" in globals():
            page_iterator.close()
        input_paths = doclaynet_run_paths if USE_DOCLAYNET_SAMPLE else [Path(INPUT_PATH).expanduser().resolve()]
        def selected_document_pages():
            for input_path in input_paths:
                yield from iter_document_pages(input_path, None if USE_DOCLAYNET_SAMPLE else PAGES, dpi=PDF_DPI)
        page_iterator = selected_document_pages()
        first_page = next(page_iterator)
        document_path = first_page.source_path
        document_pages = itertools.chain([first_page], page_iterator)
        print(f"Tài liệu: {document_path.name} | trang đầu được chọn: {first_page.page_number}")
        preview = Image.fromarray(cv2.cvtColor(first_page.image, cv2.COLOR_BGR2RGB))
        preview.thumbnail((900, 1000))
        display(preview)
        ''') ,
        markdown('''
        ## 4. Khởi tạo mô hình — nạp một lần, tái sử dụng cho mọi trang

        PaddleOCR chạy GPU 0 cùng GLM, cần cài paddlepaddle-gpu. GPU 4 GB có thể thiếu VRAM. Bộ nhớ GPU còn phụ thuộc kích thước crop và
        lượng token sinh. Nếu OOM, đóng mô hình ở cell cuối, đổi sang CPU hoặc giảm hai giới hạn
        GLM rồi chạy lại cell này. Không tự chuyển sang mock khi mô hình lỗi.
        ''') ,
        code('''
        import asyncio
        import json
        import gc
        from src.pipeline import load_config, parse_page
        from src.runtime import create_adapters, close_adapters

        config = load_config({
            "models": {"semantic_parser": GLM_MODEL},
            "routing": {"classify_unlabeled_regions": CLASSIFY_UNLABELED_REGIONS},
            "inference": {"device": GLM_DEVICE, "max_crop_edge": GLM_MAX_CROP_EDGE,
                          "max_new_tokens": GLM_MAX_NEW_TOKENS, "max_in_flight": 1},
        })
        new_key = json.dumps({"models": config["models"], "inference": config["inference"],
                              "mock": MOCK_MODE}, sort_keys=True)
        if globals().get("adapters_key") != new_key or "adapters" not in globals():
            if "adapters" in globals():
                close_adapters(adapters)
                del adapters
                gc.collect()
            adapters = create_adapters(config, mock_mode=MOCK_MODE)
            adapters_key = new_key
        # Windows: Paddle runs in a child process to isolate cuDNN DLLs from PyTorch.
        if not MOCK_MODE and hasattr(adapters[0], "initialize"):
            await asyncio.to_thread(adapters[0].initialize)
        try:
            await asyncio.to_thread(adapters[-1].initialize)
        except Exception:
            display(Markdown("**Nạp GLM thất bại.** Xem lỗi bên dưới. Nếu thiếu VRAM, đổi "
                             '`GLM_DEVICE="cpu"` rồi chạy lại cell khởi tạo.'))
            raise
        print("GLM:", adapters[-1].device, "| PaddleOCR: GPU 0 (gpu:0)")
        if MOCK_MODE:
            display(Markdown("### MOCK — mọi nội dung nhận dạng là dữ liệu giả lập, không phải kết quả thật."))
        ''', tags=['models']),
        markdown('### Thử một crop residual trước khi chạy batch\n\nSau khi nạp GLM, bật `RUN_TARGET_PROBE=True` để thử riêng `recovered:1` của `1.png`.\nCell hiển thị chính ảnh gửi cho GLM, OCR theo prompt chuẩn và JSON phân loại; dùng hai\nlượt suy luận riêng để chẩn đoán, không đổi các kết quả đã lưu hay tự chấp nhận nhãn.\nChỉnh đường dẫn nếu dùng tài liệu khác. Mặc định tắt khi Run All.\n'),
        code('# Thử đúng một vùng trước khi chạy cả batch; dùng GLM đã nạp, không nạp model thứ hai.\nRUN_TARGET_PROBE = False\nPROBE_RESULT = METHOD2_ROOT / "outputs/notebook_residual_glm/1/page_0001/result.json"\nPROBE_IMAGE = METHOD2_ROOT / "outputs/doclaynet_sample_test/images/1.png"\nPROBE_REGION = "recovered:1"\nif RUN_TARGET_PROBE:\n    import cv2\n    import numpy as np\n    from PIL import Image\n    from src.schemas.region import LayoutRegion\n    from src.semantic.residual_labeling import build_target_crop\n    from src.semantic.residual_prompt import build_classification_prompt, parse_classification_with_note\n    from src.semantic.probe_scheduler import ProbeScheduler\n    from src.semantic.evidence_features import calculate_content_consistency\n    saved = json.loads(PROBE_RESULT.read_text(encoding="utf-8"))\n    record = next(r for r in saved["final_regions"] if r["id"] == PROBE_REGION)\n    image = cv2.imdecode(np.fromfile(str(PROBE_IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR)\n    if image is None:\n        raise ValueError(f"Không đọc được ảnh: {PROBE_IMAGE}")\n    region = LayoutRegion(id=PROBE_REGION, bbox=record["bbox_final"], label="residual")\n    target = build_target_crop(image, region.bbox)\n    display(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB)))\n    display(Markdown("**OCR đang có:** " + record["content"]))\n    prompt = build_classification_prompt(region, (image.shape[1], image.shape[0]), record["content"])\n    probe_evidence = await ProbeScheduler(2, 1).run(adapters[-1], [\n        ("probe:target_ocr", target, "Text Recognition:"),\n        ("classify:probe_target", target, prompt),\n    ])\n    probe_report = {"region": PROBE_REGION, "mode": "MOCK" if MOCK_MODE else "REAL",\n                    "scope": "diagnostic only; no relabel acceptance or saved results modified",\n                    "reference_ocr": record["content"], "evidence": [e.model_dump() for e in probe_evidence]}\n    if probe_evidence[1].status == "ok":\n        try:\n            classified, note = parse_classification_with_note(probe_evidence[1].content)\n            probe_report.update(proposed_label=classified.label, target_text=classified.text, normalization=note,\n                transcript_agreement=calculate_content_consistency(classified.text, [record["content"]]))\n        except ValueError as exc:\n            probe_report["parse_error"] = str(exc)\n    display(HTML("<details open><summary>Đầu ra thô và đối chiếu crop</summary><pre>" +\n                 __import__("html").escape(json.dumps(probe_report, ensure_ascii=False, indent=2)) + "</pre></details>"))\n    probe_dir = METHOD2_ROOT / "outputs/residual_target_probe"\n    probe_dir.mkdir(parents=True, exist_ok=True)\n    cv2.imencode(".png", target)[1].tofile(str(probe_dir / "target.png"))\n    (probe_dir / "probe.json").write_text(json.dumps(probe_report, ensure_ascii=False, indent=2), encoding="utf-8")\n', tags=['target-probe']),
        markdown('''
        ## 5. Chạy Dual-Skip và hiển thị đầy đủ từng trang

        Mỗi trang được hiển thị và xuất file ngay khi hoàn tất. JSON giữ đầy đủ bằng chứng và nguồn gốc.
        Trang lỗi có thông báo và dòng trạng thái riêng. Để chạy lại tài liệu, chạy lại cell nạp ảnh/PDF trước.
        ''') ,
        code('''
        import time
        from html import escape
        from src.notebook_visuals import display_page_result, export_page_result
        from src.adapters.paddle_worker import PaddleEnvironmentError

        results, summary_records, stage_events, exported_paths = [], [], [], []
        processed_pages = 0
        try:
            for page in document_pages:
                document_path = page.source_path
                processed_pages += 1
                started = time.perf_counter()
                status = display(Markdown(f"**{escape(document_path.name)} · trang {page.page_number}: bắt đầu…**"), display_id=True)
                def on_progress(stage, message):
                    elapsed = time.perf_counter() - started
                    stage_events.append({"tài liệu": document_path.name, "trang": page.page_number, "giai đoạn": stage,
                                         "giây từ đầu trang": round(elapsed, 3), "thông báo": message})
                    status.update(Markdown(f"**{escape(document_path.name)} · trang {page.page_number} · {elapsed:.1f}s · {escape(message)}**"))
                try:
                    page_config = config | {"document_id": document_path.stem, "page_index": page.page_number - 1}
                    result = await parse_page(page.image, page_config, *adapters, progress_callback=on_progress)
                    elapsed = time.perf_counter() - started
                    results.append(result)
                    display_page_result(page.image, result, max_region_previews=MAX_REGION_PREVIEWS,
                                        mock_mode=MOCK_MODE)
                    output = export_page_result(page.image, result, OUTPUT_ROOT, document_path.stem) if SAVE_OUTPUTS else None
                    if output is not None:
                        exported_paths.append(str(output))
                        print("Đã lưu:", output)
                    summary_records.append({"tài liệu": document_path.name, "trang": page.page_number, "chế độ": "MOCK" if MOCK_MODE else "REAL",
                        "trạng thái": "có cảnh báo" if result.warnings else "hoàn tất", "giây suy luận": round(elapsed, 2),
                        "vùng": len(result.final_regions), "lượt GLM": result.glm_requests,
                        "residual còn lại": sum(r.label_final in ("residual", "unknown", "unlabeled", "") for r in result.final_regions),
                        "đổi nhãn được chấp nhận": sum(r.action == "RELABEL" and r.status == "accepted" for r in result.revisions),
                        "cảnh báo": len(result.warnings), "thư mục": str(output or "không lưu")})
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    status.update(Markdown(f"**{escape(document_path.name)} · trang {page.page_number} lỗi: {escape(str(exc))}**"))
                    summary_records.append({"tài liệu": document_path.name, "trang": page.page_number, "chế độ": "MOCK" if MOCK_MODE else "REAL",
                        "trạng thái": "LỖI", "giây suy luận": round(elapsed, 2), "lỗi": str(exc)})
                    if isinstance(exc, PaddleEnvironmentError):
                        raise  # DLL/import failure affects every page; preserve the first traceback.
        finally:
            page_iterator.close()
        if not processed_pages:
            raise RuntimeError("Danh sách trang đã chạy hết. Chạy lại cell nạp ảnh/PDF rồi chạy cell này.")
        ''', tags=['run']),
        markdown('## 6. Tổng hợp và vị trí kết quả'),
        code('''
        display(pd.DataFrame(summary_records))
        display(Markdown("### Nhật ký tiến độ từng giai đoạn"))
        display(pd.DataFrame(stage_events))
        if exported_paths:
            print("Các thư mục đã xuất:")
            print("\\n".join(exported_paths))
        print("results chứa các PageParseResult để tiếp tục phân tích.")
        ''', tags=['summary']),
        markdown('''
        ## 7. Báo cáo định lượng trên DocLayNet

        Cell chỉ đọc prediction đã lưu, không chạy lại mô hình. So sánh baseline và final
        trên cùng snapshot; mặc định gộp Title + Section-header ở cả hai phía (10 lớp).
        Accuracy là accuracy phân loại trên bbox đã ghép, không phải accuracy OCR.
        Chỉ số thuộc tập ảnh đang có; chưa thể so trực tiếp với benchmark Docling/M2Doc
        khác dataset, split hoặc protocol. Không tự tạo mAP khi thiếu confidence phù hợp.
        ''') ,
        code('''
        # Cell độc lập: chạy được trên kernel mới, không cần nạp mô hình.
        import sys
        from pathlib import Path
        from datetime import datetime, timezone, timedelta
        from IPython.display import display, Markdown, HTML
        import pandas as pd

        report_candidates = [p for base in (Path.cwd(), *Path.cwd().parents)
                             for p in (base, base / "method2")]
        METHOD2_ROOT = next((p.resolve() for p in report_candidates
                            if (p / "src/evaluation/notebook_report.py").is_file()
                            and (p / "configs/default.yaml").is_file()), None)
        if METHOD2_ROOT is None:
            raise RuntimeError("Mở notebook từ thư mục repository hoặc method2 để tìm kết quả đã lưu.")
        if str(METHOD2_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD2_ROOT))
        report_src = sys.modules.get("src")
        if report_src is not None and (getattr(report_src, "__file__", None) is None or
                Path(report_src.__file__).resolve().parent != METHOD2_ROOT / "src"):
            raise RuntimeError("Kernel đã nạp package src khác. Restart Kernel rồi chạy riêng cell báo cáo này.")
        from importlib import reload
        from src.evaluation import notebook_report as layout_report
        layout_report = reload(layout_report)  # Chỉ nạp lại helper báo cáo, không nạp lại torch/model.
        evaluate_saved_notebook = layout_report.evaluate_saved_notebook

        RUN_LAYOUT_EVALUATION = True
        EVAL_MERGED_HEADINGS = True
        EVAL_IOU = 0.5  # Bảng tổng hợp có cả 0.5 và 0.75; bảng từng lớp dùng giá trị này.
        EVAL_SAMPLE_DIR = Path(globals().get("DOCLAYNET_SAMPLE_DIR", METHOD2_ROOT / "outputs/doclaynet_sample_test"))
        EVAL_SPLIT = globals().get("DOCLAYNET_SPLIT", "test")
        EVAL_ANNOTATIONS = EVAL_SAMPLE_DIR / "annotations" / f"{EVAL_SPLIT}.json"
        EVAL_IMAGES = EVAL_SAMPLE_DIR / "images"
        EVAL_PREDICTIONS = Path(globals().get("OUTPUT_ROOT", METHOD2_ROOT / "outputs/notebook_residual_glm"))
        EVAL_REPORT = None
        if globals().get("MOCK_MODE", False):
            display(Markdown("**MOCK: bỏ qua báo cáo chất lượng thật.**"))
        elif not RUN_LAYOUT_EVALUATION or not globals().get("USE_DOCLAYNET_SAMPLE", True):
            display(Markdown("Bỏ qua đánh giá. Cần prediction và GT DocLayNet cùng tập ảnh."))
        elif not EVAL_ANNOTATIONS.is_file():
            display(Markdown("Chưa có annotation GT. Chạy cell tải mẫu DocLayNet trước."))
        elif not EVAL_PREDICTIONS.is_dir() or not any(EVAL_PREDICTIONS.glob("*/page_*/result.json")):
            display(Markdown(f"Chưa có kết quả đã lưu tại `{EVAL_PREDICTIONS}`. Sửa EVAL_PREDICTIONS đến thư mục kết quả cần xem."))
        else:
            EVAL_OUTPUT = METHOD2_ROOT / "outputs/evaluation" / ("notebook_" +
                datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f"))
            EVAL_REPORT, metric_table, class_table = evaluate_saved_notebook(
                EVAL_ANNOTATIONS, EVAL_PREDICTIONS, EVAL_OUTPUT,
                merged_headings=EVAL_MERGED_HEADINGS, iou=EVAL_IOU)
            print(f"Đã đánh giá {EVAL_REPORT['image_count']} ảnh thật; GT: {EVAL_REPORT['ground_truth_boxes']} vùng; "
                  f"final: {EVAL_REPORT['final_boxes']} vùng.")
            print("Taxonomy:", "10 lớp (gộp Heading)" if EVAL_MERGED_HEADINGS else "11 lớp DocLayNet gốc")
            import matplotlib.pyplot as plt
            metric_figure = layout_report.layout_metrics_figure(
                EVAL_REPORT, merged_headings=EVAL_MERGED_HEADINGS, iou=EVAL_IOU)
            try:
                metric_figure.savefig(EVAL_OUTPUT / "layout_metrics.png", dpi=160, bbox_inches="tight")
                metric_figure.savefig(EVAL_OUTPUT / "layout_metrics.svg", bbox_inches="tight")
                display(metric_figure)
            finally:
                plt.close(metric_figure)
            display(metric_table.style.format({c: "{:.2f}" for c in metric_table if "(%)" in c}, na_rep="—"))
            display(Markdown(f"**Theo từng lớp — final, IoU ≥ {EVAL_IOU}:**"))
            display(class_table.style.format({c: "{:.2f}" for c in class_table if "(%)" in c}))
            display(pd.DataFrame(EVAL_REPORT["metrics"][str(EVAL_IOU)][
                "final_headings_merged" if EVAL_MERGED_HEADINGS else "final_literal"]["top_confusions"]))
            print("Nhãn chưa mapping (vẫn tính FP):", EVAL_REPORT["unmapped_labels"])
            print("Ảnh/kết quả bị loại khỏi snapshot:", len(EVAL_REPORT["excluded"]))
            print("Đã lưu báo cáo JSON, Markdown và CSV tại:", EVAL_OUTPUT)
        ''', tags=['layout-evaluation']),
        markdown('''
        ## 8. So sánh ảnh final với nhãn thật

        Chạy sau cell báo cáo. Mặc định xem tối đa 3 ảnh từ đúng snapshot đã đánh giá;
        đổi `GT_COMPARE_DOCUMENTS` thành `["1", "5", "15"]` để chọn ảnh cụ thể.
        Bên trái là dự đoán cuối, bên phải là annotation GT. `Text [residual]` vẫn là
        nhãn residual được mapping khi đánh giá, không phải một RELABEL đã chấp nhận.
        ''') ,
        code('''
        from IPython.display import display, Markdown

        GT_COMPARE_MAX_IMAGES = 3
        GT_COMPARE_DOCUMENTS = None  # Ví dụ: ["1", "5", "15"]
        if globals().get("EVAL_REPORT") is None:
            display(Markdown("Chạy cell báo cáo trên kết quả thật trước để có snapshot và GT."))
        else:
            from src.evaluation.notebook_report import final_gt_figures
            import matplotlib.pyplot as plt
            comparison_dir = EVAL_OUTPUT / "gt_comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            for document, figure in final_gt_figures(
                    EVAL_REPORT, EVAL_IMAGES, EVAL_OUTPUT / "snapshot",
                    merged_headings=EVAL_MERGED_HEADINGS,
                    max_images=GT_COMPARE_MAX_IMAGES, documents=GT_COMPARE_DOCUMENTS):
                try:
                    figure.savefig(comparison_dir / f"{document}_final_vs_gt.png", dpi=120, bbox_inches="tight")
                    display(figure)
                finally:
                    plt.close(figure)
            print("Đã lưu ảnh so sánh tại:", comparison_dir)
        ''', tags=['final-vs-gt']),
        markdown('''
        ## 9. COCO mAP, AP50 và strict Micro-F1 từ src/common

        Cell chạy riêng được, chỉ đọc snapshot/kết quả đã lưu. Gọi trực tiếp
        `src/common/metrics.py::coco_map` và `evaluate(require_same_category=True)`.
        COCO mAP trung bình IoU 0.50:0.95; AP50 ở IoU 0.50; strict Micro-F1 dùng
        `COCO_STRICT_IOU`. Cả baseline/final dùng cùng tập ảnh và taxonomy.

        **Score final là proxy kế thừa**, không phải xác suất GLM: lấy min score layout
        của các vùng nguồn; với residual dùng min score detector dòng chữ. Không dùng
        score mặc định của recovered, không tạo score 1.0. Thiếu score đo được thì AP
        ghi N/A. Nhãn Unmapped bị loại khỏi COCO AP theo yêu cầu hàm, nhưng vẫn tính FP
        trong strict Micro-F1. Vì chính sách score khác nhau, không coi AP final là
        confidence đã hiệu chuẩn hoặc benchmark trực tiếp với Docling/M2Doc.
        ''') ,
        code('''
        import sys
        from pathlib import Path
        from datetime import datetime, timezone, timedelta
        from importlib import reload
        from IPython.display import display, Markdown
        import pandas as pd

        coco_candidates = [p for base in (Path.cwd(), *Path.cwd().parents)
                           for p in (base, base / "method2")]
        COCO_ROOT = next((p.resolve() for p in coco_candidates
                          if (p / "src/evaluation/coco_report.py").is_file()
                          and (p / "configs/default.yaml").is_file()), None)
        if COCO_ROOT is None:
            raise RuntimeError("Mở notebook từ repository hoặc method2.")
        if str(COCO_ROOT) not in sys.path:
            sys.path.insert(0, str(COCO_ROOT))
        coco_src = sys.modules.get("src")
        if coco_src is not None and (getattr(coco_src, "__file__", None) is None or
                Path(coco_src.__file__).resolve().parent != COCO_ROOT / "src"):
            raise RuntimeError("Kernel nạp package src khác; Restart Kernel rồi chạy riêng cell này.")
        from src.evaluation import coco_report as coco_helpers
        from src.evaluation.notebook_report import evaluate_saved_notebook
        coco_helpers = reload(coco_helpers)

        RUN_COCO_EVALUATION = True
        COCO_MERGED_HEADINGS = globals().get("EVAL_MERGED_HEADINGS", True)
        COCO_STRICT_IOU = 0.5
        COCO_ANNOTATIONS = Path(globals().get("EVAL_ANNOTATIONS", COCO_ROOT / "outputs/doclaynet_sample_test/annotations/test.json"))
        COCO_PREDICTIONS = Path(globals().get("EVAL_PREDICTIONS", globals().get("OUTPUT_ROOT", COCO_ROOT / "outputs/notebook_residual_glm")))
        COCO_REPORT = None
        if globals().get("MOCK_MODE", False):
            display(Markdown("**MOCK: bỏ qua COCO AP chất lượng thật.**"))
        elif not RUN_COCO_EVALUATION:
            display(Markdown("Đã tắt cell COCO evaluation."))
        elif not COCO_ANNOTATIONS.is_file() or not COCO_PREDICTIONS.is_dir():
            display(Markdown("Thiếu GT/kết quả. Sửa COCO_ANNOTATIONS và COCO_PREDICTIONS đến dữ liệu đã lưu."))
        else:
            COCO_OUTPUT = COCO_ROOT / "outputs/evaluation" / ("coco_notebook_" +
                datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f"))
            if globals().get("EVAL_REPORT") is not None and "EVAL_OUTPUT" in globals():
                COCO_INPUT_REPORT = EVAL_REPORT
                COCO_SNAPSHOT = Path(EVAL_OUTPUT) / "snapshot"
            else:
                COCO_INPUT_REPORT, _, _ = evaluate_saved_notebook(
                    COCO_ANNOTATIONS, COCO_PREDICTIONS, COCO_OUTPUT,
                    merged_headings=COCO_MERGED_HEADINGS, iou=0.5)
                COCO_SNAPSHOT = COCO_OUTPUT / "snapshot"
            COCO_REPORT, coco_table, coco_class_table = coco_helpers.evaluate_coco_snapshot(
                COCO_INPUT_REPORT, COCO_SNAPSHOT, COCO_OUTPUT,
                merged_headings=COCO_MERGED_HEADINGS, strict_iou=COCO_STRICT_IOU)
            print("Ảnh thật:", COCO_REPORT["image_count"], "|", COCO_REPORT["taxonomy"])
            display(coco_table.style.format({c: "{:.2f}" for c in coco_table if "(%)" in c}, na_rep="N/A"))
            display(Markdown("**AP theo lớp (N/A nếu không có GT):**"))
            display(coco_class_table.style.format({c: "{:.2f}" for c in coco_class_table if "(%)" in c}, na_rep="N/A"))
            for method, detail in COCO_REPORT["modes"].items():
                print(method, "| score:", detail["score_policy"], "| AP loại Unmapped:", detail["ap_excluded_unmapped"])
                if detail["ap_error"]:
                    print("AP chưa tính được:", detail["ap_error"])
            print("Nếu thiếu pycocotools, chạy riêng: %pip install pycocotools")
            print("Đã lưu COCO JSON và CSV:", COCO_OUTPUT)
        ''', tags=['coco-evaluation']),
        markdown('''
        ### Giải phóng mô hình (tùy chọn)

        Chỉ bật khi đã chạy xong hoặc muốn đổi model/device. Sau khi giải phóng, chạy lại cell khởi tạo
        trước khi chạy tài liệu khác. Không cần Restart Kernel nếu các thư viện không thay đổi.
        ''') ,
        code('''
        RELEASE_MODELS = False
        if RELEASE_MODELS and "adapters" in globals():
            close_adapters(adapters)
            del adapters
            adapters_key = None
            gc.collect()
            print("Đã giải phóng mô hình.")
        ''', tags=['cleanup']),
    ]
    for index, cell in enumerate(nb.cells):
        cell.id = f'dual-skip-{index:02d}'
    nbf.validate(nb)
    return nb


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / 'Dual_Skip_v4.ipynb'
    nbf.write(build(), target)
