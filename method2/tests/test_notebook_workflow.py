import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.document_io import iter_document_pages
from src.notebook_visuals import (comparison_figure, display_page_result, export_page_result,
                                  label_color, safe_markup)
from src.pipeline import load_config, parse_page
from src.runtime import create_adapters


def write_image(path):
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.putText(image, 'Test page', (8, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.imencode('.png', image)[1].tofile(path)
    return image


def test_unicode_image_path_and_page_validation(tmp_path):
    path = tmp_path / 'ảnh tiếng Việt.png'
    image = write_image(path)
    pages = list(iter_document_pages(path, [1]))
    assert pages[0].page_number == 1
    assert np.array_equal(pages[0].image, image)
    for selection in ([0], [2], [], [1, 1]):
        with pytest.raises(ValueError):
            list(iter_document_pages(path, selection))
    with pytest.raises(FileNotFoundError):
        list(iter_document_pages(tmp_path / 'missing.png'))


def test_pdf_selection_order_dpi_and_limit(tmp_path):
    pymupdf = pytest.importorskip('pymupdf')
    path = tmp_path / 'tài liệu.pdf'
    with pymupdf.open() as document:
        for number in range(3):
            page = document.new_page(width=144, height=144)
            page.insert_text((20, 40), f'Page {number + 1}')
        document.save(path)
    pages = list(iter_document_pages(path, [3, 1], dpi=150))
    assert [p.page_number for p in pages] == [3, 1]
    assert pages[0].image.shape == (300, 300, 3)
    with pytest.raises(ValueError):
        list(iter_document_pages(path, [4]))
    with pytest.raises(ValueError, match='too large'):
        list(iter_document_pages(path, max_pixels=5))


def test_inline_visuals_exports_and_schema_roundtrip(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import IPython.display

    image = write_image(tmp_path / 'ảnh.png')
    config = load_config()
    result = asyncio.run(parse_page(image, config | {'page_index': 2}, *create_adapters(config, mock_mode=True)))
    fig, overlays = comparison_figure(image, result)
    assert len(overlays) == 3 and all(o.shape == image.shape for o in overlays)
    plt.close(fig)
    folder = export_page_result(image, result, tmp_path / 'đầu ra', 'tài liệu')
    assert folder.name == 'page_0003'
    assert {p.name for p in folder.iterdir()} == {'result.json', 'result.md', 'comparison.png',
        'layout_original.png', 'lines.png', 'layout_final.png'}
    assert result.model_validate_json((folder / 'result.json').read_text(encoding='utf-8')) == result
    emitted = []
    monkeypatch.setattr(IPython.display, 'display', lambda value: emitted.append(value))
    display_page_result(image, result, max_region_previews=1, mock_mode=True)
    assert emitted and any('MOCK' in getattr(value, 'data', '') for value in emitted if isinstance(getattr(value, 'data', ''), str))
    assert not plt.get_fignums()
    assert label_color('custom') == label_color('custom')
    assert '<script' not in safe_markup('<script>alert(1)</script><table><tr><td>A</td></tr></table>')
    assert '<table>' in safe_markup('<table onclick="bad()"><tr><td>A</td></tr></table>')


@pytest.mark.parametrize('sample_mode', [False, True], ids=['single-image', 'doclaynet-20-images'])
def test_notebook_in_fresh_kernel_with_mock_adapters(tmp_path, sample_mode):
    nbf = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / 'ảnh notebook.png'
    write_image(path)
    sample_dir = tmp_path / 'sample'
    if sample_mode:
        (sample_dir / 'images').mkdir(parents=True)
        (sample_dir / 'annotations').mkdir()
        records = []
        for index in range(50):
            file_name = f'{index}.png'
            write_image(sample_dir / 'images' / file_name)
            records.append(dict(id=index, file_name=file_name, width=160, height=120))
        manifest = dict(info=dict(source='docling-project/DocLayNet-v1.2', mode='lightweight', split='test',
                                 offset=0, requested_images=50), images=records, annotations=[], categories=[])
        (sample_dir / 'annotations/test.json').write_text(json.dumps(manifest), encoding='utf-8')
    notebook = nbf.read(root / 'Dual_Skip_v4.ipynb', as_version=4)
    nbf.validate(notebook)
    for cell in notebook.cells:
        if 'parameters' in cell.metadata.get('tags', []):
            cell.source += '\n' + '\n'.join([
                f'INPUT_PATH = {str(path)!r}', 'MOCK_MODE = True', f'USE_DOCLAYNET_SAMPLE = {sample_mode!r}',
                f'DOCLAYNET_SAMPLE_DIR = Path({str(sample_dir)!r})',
                f'OUTPUT_ROOT = Path({str(tmp_path / "exports")!r})', 'MAX_REGION_PREVIEWS = 1'])
    nbclient.NotebookClient(notebook, timeout=180, kernel_name='python3',
        resources={'metadata': {'path': str(root)}}).execute()
    assert all(output.output_type != 'error' for cell in notebook.cells if cell.cell_type == 'code'
               for output in cell.get('outputs', []))
    assert any('image/png' in output.get('data', {}) for cell in notebook.cells
               for output in cell.get('outputs', []))
    assert len(list((tmp_path / 'exports').rglob('result.json'))) == (20 if sample_mode else 1)
    # Preserve an executed, explicitly MOCK preview beside other ignored notebook outputs.
    target = root / 'outputs/notebook_validation'
    target.mkdir(parents=True, exist_ok=True)
    file_name = 'Dual_Skip_v4_mock_sample20_executed.ipynb' if sample_mode else 'Dual_Skip_v4_mock_executed.ipynb'
    nbf.write(notebook, target / file_name)


def test_html_tables_and_latex_are_rendered_inline(monkeypatch):
    import IPython.display
    from src.schemas.region import LayoutRegion, FinalRegion
    from src.schemas.output import PageParseResult

    image = np.full((100, 120, 3), 255, np.uint8)
    regions = [LayoutRegion(id='t', bbox=(5, 5, 100, 40), label='table'),
               LayoutRegion(id='f', bbox=(5, 50, 100, 90), label='formula')]
    table = '<table><tr><td>A</td></tr></table>'
    finals = [FinalRegion(id='t', bbox_final=regions[0].bbox, label_final='table', order_final=0,
                         content=table, content_format='html', source_region_ids=['t']),
              FinalRegion(id='f', bbox_final=regions[1].bbox, label_final='formula', order_final=1,
                         content=r'x^2 = y', content_format='latex', source_region_ids=['f'])]
    result = PageParseResult(original_regions=regions, final_regions=finals, markdown=table + '\n\n$$x^2 = y$$')
    emitted = []
    monkeypatch.setattr(IPython.display, 'display', lambda value: emitted.append(value))
    display_page_result(image, result, max_region_previews=None)
    assert any(isinstance(value, IPython.display.HTML) and '<table>' in value.data for value in emitted)
    assert any(isinstance(value, IPython.display.Markdown) and '$$\n' in value.data for value in emitted)
