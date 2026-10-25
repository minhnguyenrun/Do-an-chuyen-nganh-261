"""Lazy image/PDF loading with Unicode paths and one-based page selection."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    image: np.ndarray
    source_path: Path | None = None


def _selected_pages(pages, count):
    selected = list(range(1, count + 1)) if pages is None else list(pages)
    if not selected or any(type(n) is not int or n < 1 or n > count for n in selected):
        raise ValueError(f'PAGES must contain page numbers between 1 and {count}')
    if len(selected) != len(set(selected)):
        raise ValueError('PAGES contains duplicate page numbers')
    return selected


def iter_document_pages(path, pages=None, *, dpi=150, max_pixels=40_000_000):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy tài liệu: {path}')
    if type(dpi) is not int or dpi < 36 or dpi > 600:
        raise ValueError('PDF DPI must be an integer between 36 and 600')
    if path.suffix.lower() == '.pdf':
        import pymupdf

        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError('PDF có mật khẩu; cần mở khóa trước khi xử lý.')
            for number in _selected_pages(pages, len(document)):
                page = document[number - 1]
                if (page.rect.width * dpi / 72) * (page.rect.height * dpi / 72) > max_pixels:
                    raise ValueError('PDF page too large; lower PDF_DPI')
                pixels = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                rgb = np.frombuffer(pixels.samples, dtype=np.uint8).reshape(pixels.height, pixels.width, 3)
                yield DocumentPage(number, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), path)
    else:
        _selected_pages(pages, 1)
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise ValueError(f'Không đọc được ảnh: {path}') from exc
        if image is None:
            raise ValueError(f'Không đọc được ảnh: {path}')
        if image.shape[0] * image.shape[1] > max_pixels:
            raise ValueError('Image too large; resize below 40 million pixels')
        yield DocumentPage(1, image, path)
