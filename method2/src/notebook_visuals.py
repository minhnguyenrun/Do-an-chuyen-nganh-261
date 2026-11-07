"""Notebook display and exported overlays; no model inference in visualization."""
import colorsys
import hashlib
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def label_color(label):
    fixed = {'text': '#1677b8', 'title': '#8c3bbd', 'table': '#d16c00',
             'formula': '#d12c69', 'figure': '#009879', 'residual': '#797979'}
    if label in fixed:
        return fixed[label]
    hue = int(hashlib.sha256(label.encode('utf-8')).hexdigest()[:8], 16) / 0xffffffff
    rgb = colorsys.hsv_to_rgb(hue, .75, .7)
    return '#' + ''.join(f'{round(c * 255):02x}' for c in rgb)


def _canvas(image):
    from matplotlib.font_manager import FontProperties, findfont

    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font_size = max(12, min(30, image.shape[1] // 85))
    font = ImageFont.truetype(findfont(FontProperties(family='DejaVu Sans')), font_size)
    return canvas, draw, font


def draw_layout_overlay(image, regions, *, final=False):
    canvas, draw, font = _canvas(image)
    width = max(2, image.shape[1] // 500)
    for region in regions:
        bbox = region.bbox_final if final else region.bbox
        label = region.label_final if final else region.label
        color = label_color(label)
        text = f'#{region.order_final + 1} {label}' if final else f'{region.id} {label}'
        draw.rectangle(bbox, outline=color, width=width)
        x, y = bbox[:2]
        bounds = draw.textbbox((0, 0), text, font=font)
        height = bounds[3] - bounds[1] + 6
        top = max(0, y - height)
        draw.rectangle((x, top, min(image.shape[1], x + bounds[2] + 6), top + height), fill=color)
        draw.text((x + 3, top - bounds[1] + 2), text, fill='white', font=font)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def draw_line_overlay(image, lines):
    canvas, draw, _ = _canvas(image)
    for line in lines:
        points = list(line.polygon)
        draw.line(points + points[:1], fill='#e02929', width=max(2, image.shape[1] // 700))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def comparison_figure(image, result):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    overlays = (
        draw_layout_overlay(image, result.original_regions),
        draw_line_overlay(image, result.original_lines),
        draw_layout_overlay(image, result.final_regions, final=True),
    )
    fig, axes = plt.subplots(1, 3, figsize=(19, 9))
    titles = ('Skip-H: layout ban đầu', 'Skip-L: polygon dòng chữ', 'Kết quả cuối và thứ tự đọc')
    for ax, overlay, title in zip(axes, overlays, titles):
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    labels = sorted({r.label for r in result.original_regions} | {r.label_final for r in result.final_regions})
    if labels:
        fig.legend(handles=[Patch(color=label_color(label), label=label) for label in labels],
                   loc='lower center', ncol=min(7, len(labels)))
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig, overlays


def region_records(result):
    return [dict(order=r.order_final + 1, id=r.id, label=r.label_final, bbox=r.bbox_final,
                 sources=', '.join(r.source_region_ids), lines=len(r.source_line_ids),
                 needs_review=r.needs_review, content=r.content[:180]) for r in result.final_regions]


def revision_records(result):
    return [dict(action=r.action.value, status=r.status.value if hasattr(r.status, 'value') else r.status,
                 sources=', '.join(r.source_region_ids), target=r.target_region_id,
                 label=r.new_label, baseline_score=round(r.baseline_score, 4), score=round(r.score, 4),
                 gain=round(r.score - r.baseline_score, 4), reasons='; '.join(r.reasons)) for r in result.revisions]


def safe_markup(content):
    import bleach

    return bleach.clean(content, tags={'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
        'caption', 'br', 'b', 'strong', 'em', 'i', 'p', 'ul', 'ol', 'li'},
        attributes={'td': ['colspan', 'rowspan'], 'th': ['colspan', 'rowspan']}, strip=True)


def display_page_result(image, result, *, max_region_previews=20, mock_mode=False):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import HTML, Markdown, display

    if max_region_previews is not None and (type(max_region_previews) is not int or max_region_previews < 0):
        raise ValueError('max_region_previews must be None or a nonnegative integer')
    marker = ' — MOCK: không phải nhận dạng thật' if mock_mode else ''
    display(Markdown(f'## Trang {result.page_index + 1}{marker}'))
    display(Markdown(f'**{len(result.final_regions)} vùng · {len(result.original_lines)} dòng · '
                     f'{result.glm_requests} lượt suy luận GLM**'))
    fig, _ = comparison_figure(image, result)
    display(fig)
    plt.close(fig)
    display(Markdown('### Thông tin vùng'))
    display(pd.DataFrame(region_records(result)))
    display(Markdown('### Các thay đổi và quyết định'))
    display(pd.DataFrame(revision_records(result)) if result.revisions else Markdown('Không có đề xuất thay đổi.'))
    classifications = [dict(region=e.hypothesis_id.removeprefix('classify:').removesuffix(':retry'),
                            attempt=2 if e.hypothesis_id.endswith(':retry') else 1,
                            normalization=e.error if e.status == 'ok' and e.error.startswith('normalized_json:') else '',
                            proposed_label=e.classification.label if e.classification else '',
                            status=e.status, text=e.classification.text if e.classification else '',
                            error=e.error if e.status != 'ok' else '')
                       for e in result.evidence if e.hypothesis_id.startswith('classify:')]
    if classifications:
        display(Markdown('### Nhãn GLM đề xuất cho vùng chưa có nhãn'))
        display(pd.DataFrame(classifications))
    if result.warnings:
        display(Markdown('### Cảnh báo\n' + '\n'.join('- ' + escape(w) for w in result.warnings)))
    failed = [e.model_dump() for e in result.evidence if e.status != 'ok']
    if failed:
        display(pd.DataFrame(failed)[['hypothesis_id', 'status', 'error']])
    display(Markdown('### Nội dung tài liệu'))
    display(Markdown(safe_markup(result.markdown)))
    display(Markdown('### Crop và nội dung từng vùng'))
    selected = result.final_regions if max_region_previews is None else result.final_regions[:max_region_previews]
    for region in selected:
        display(Markdown(f'**{region.order_final + 1}. {escape(region.id)} — {escape(region.label_final)}**'))
        x1, y1, x2, y2 = region.bbox_final
        display(Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
        if region.content_format == 'html':
            display(HTML(safe_markup(region.content)))
        elif region.content_format == 'latex':
            display(Markdown('$$\n' + safe_markup(region.content) + '\n$$'))
        else:
            display(Markdown(safe_markup(region.content)))
    if len(selected) < len(result.final_regions):
        display(Markdown(f'Đang xem {len(selected)}/{len(result.final_regions)} crop. '
                         'Đặt MAX_REGION_PREVIEWS=None để xem toàn bộ.'))
    display(HTML('<details><summary>JSON và provenance đầy đủ</summary><pre>' +
                 escape(result.model_dump_json(indent=2)) + '</pre></details>'))


def export_page_result(image, result, output_root, document_name):
    import matplotlib.pyplot as plt

    # Use a leaf directory even if document_name was supplied as a path.
    leaf = Path(str(document_name)).stem
    if leaf in ('', '.', '..'):
        raise ValueError('invalid document name')
    folder = Path(output_root) / leaf / f'page_{result.page_index + 1:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'result.json').write_text(result.model_dump_json(indent=2), encoding='utf-8')
    (folder / 'result.md').write_text(result.markdown, encoding='utf-8')
    fig, overlays = comparison_figure(image, result)
    try:
        for name, overlay in zip(('layout_original.png', 'lines.png', 'layout_final.png'), overlays):
            ok, encoded = cv2.imencode('.png', overlay)
            if not ok:
                raise ValueError(f'cannot encode {name}')
            encoded.tofile(folder / name)
        fig.savefig(folder / 'comparison.png', dpi=130)
    finally:
        plt.close(fig)
    return folder
