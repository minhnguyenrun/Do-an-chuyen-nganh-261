"""Do thoi gian xu ly moi trang va peak RAM/VRAM o lop chay model.

Dung chung cho moi method:

    monitor = ResourceMonitor()
    with monitor.time_page() as timer:
        prediction = run_model(image)
    prediction["inference_time_s"] = timer.seconds
    ...
    save_runtime_summary(method, dataset, monitor.summary(...))

Quy uoc:
  - Thoi gian moi trang = end-to-end mot lan goi model (doc anh, tien xu ly,
    model, hau xu ly), KHONG tinh thoi gian load model (do rieng).
  - Neu chay tren GPU (torch), dong bo GPU truoc khi dung dong ho vi kernel
    CUDA chay bat dong bo.
  - Peak RAM = peak working set / max RSS cua ca process (gom ca model).
  - Peak VRAM = torch.cuda.max_memory_allocated (chi tinh bo nho torch cap phat).
"""

from contextlib import contextmanager
from datetime import datetime
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT_DIR / "outputs" / "runtime"


def _torch_cuda():
    """Tra ve module torch neu da duoc import va co CUDA, nguoc lai None.

    Chi dung torch khi model da import no, tranh tu import torch (cham, ton RAM)
    cho cac method khong dung torch.
    """
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def sync_device() -> None:
    torch = _torch_cuda()
    if torch is not None:
        torch.cuda.synchronize()


def peak_ram_mb() -> float | None:
    """Peak RAM cua process hien tai (MB), None neu khong do duoc."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is not None:
            return peak / 2**20
    except ImportError:
        pass
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux tra ve KB, macOS tra ve byte.
        return maxrss / 2**20 if sys.platform == "darwin" else maxrss / 2**10
    except ImportError:
        return None


def peak_vram_mb() -> float | None:
    torch = _torch_cuda()
    if torch is None:
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def device_name() -> str:
    torch = _torch_cuda()
    if torch is not None:
        return f"cuda:{torch.cuda.get_device_name()}"
    return f"cpu:{platform.processor() or platform.machine()}"


class _Timer:
    seconds: float = 0.0


class ResourceMonitor:
    def __init__(self) -> None:
        self.page_seconds: list[float] = []
        self.model_load_seconds: float | None = None

    @contextmanager
    def time_page(self):
        timer = _Timer()
        sync_device()
        started = time.perf_counter()
        try:
            yield timer
        finally:
            sync_device()
            timer.seconds = time.perf_counter() - started
            self.page_seconds.append(timer.seconds)

    @contextmanager
    def time_model_load(self):
        started = time.perf_counter()
        try:
            yield
        finally:
            sync_device()
            self.model_load_seconds = time.perf_counter() - started

    def summary(self, **extra) -> dict:
        return {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "device": device_name(),
            "cpu_count": os.cpu_count(),
            "pages": len(self.page_seconds),
            "model_load_s": self.model_load_seconds,
            **page_time_stats(self.page_seconds),
            "peak_ram_mb": peak_ram_mb(),
            "peak_vram_mb": peak_vram_mb(),
            **extra,
        }


def page_time_stats(seconds: list[float]) -> dict:
    if not seconds:
        return {"mean_s_per_page": None, "median_s_per_page": None, "p95_s_per_page": None}
    ordered = sorted(seconds)
    return {
        "mean_s_per_page": statistics.fmean(ordered),
        "median_s_per_page": statistics.median(ordered),
        "p95_s_per_page": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def runtime_path(method: str, dataset: str) -> Path:
    return RUNTIME_DIR / method / f"{dataset}.json"


def save_runtime_summary(method: str, dataset: str, summary: dict) -> Path:
    """Ghi them 1 lan chay vao outputs/runtime/<method>/<dataset>.json.

    Giu lich su moi lan chay (inference co the bi ngat va chay tiep bang
    --skip-existing), peak RAM/VRAM bao cao = max qua cac lan chay.
    """
    path = runtime_path(method, dataset)
    runs = load_runtime_runs(method, dataset)
    runs.append(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_runtime_runs(method: str, dataset: str) -> list[dict]:
    path = runtime_path(method, dataset)
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("runs", [])
