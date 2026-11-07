import json

import cv2
import numpy as np
import pytest

from src.evaluation.notebook_report import evaluate_saved_notebook, final_gt_figures, layout_metrics_figure


def test_report_compares_same_taxonomy_and_visualizes_frozen_gt(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = tmp_path / 'ảnh'
    images.mkdir()
    cv2.imencode('.png', np.full((100, 100, 3), 255, np.uint8))[1].tofile(str(images / '1.png'))
    annotation = tmp_path / 'gt.json'
    annotation.write_text(json.dumps(dict(
        images=[dict(id=1, file_name='1.png', width=100, height=100)],
        categories=[dict(id=8, name='Section-header'), dict(id=11, name='Title')],
        annotations=[dict(image_id=1, category_id=8, bbox=[10, 10, 50, 20])])), encoding='utf-8')
    predictions = tmp_path / 'predictions'
    folder = predictions / '1'
    folder.mkdir(parents=True)
    result = dict(document_id='1', page_index=0,
        original_regions=[dict(id='r', bbox=[10, 10, 60, 30], label='title')],
        final_regions=[dict(id='r', bbox_final=[10, 10, 60, 30], label_final='title')],
        evidence=[dict(parser='glm_local')], revisions=[])
    path = folder / 'result.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    output = tmp_path / 'evaluation'
    report, summary, categories = evaluate_saved_notebook(annotation, predictions, output)
    assert len(summary) == 4
    assert set(summary['Micro-F1 (%)']) == {100.}
    assert set(summary['Classification accuracy (%)']) == {100.}
    assert 'Heading' in categories['Class'].tolist()
    assert (output / 'notebook_summary.csv').is_file()
    assert '10 classes' in (output / 'notebook_report.md').read_text(encoding='utf-8')
    chart = layout_metrics_figure(json.loads(json.dumps(report)))
    assert len(chart.axes) == 2
    assert len(chart.axes[0].patches) == 30 and len(chart.axes[1].patches) == 20
    assert 'Heading' in [tick.get_text() for tick in chart.axes[0].get_yticklabels()]
    assert any('N/A' in text.get_text() for text in chart.axes[0].texts)
    assert max(patch.get_width() for patch in chart.axes[0].patches) == 1
    plt.close(chart)
    # A later live output change must not change the evaluated snapshot's figure.
    result['final_regions'][0]['label_final'] = 'residual'
    path.write_text(json.dumps(result), encoding='utf-8')
    figures = list(final_gt_figures(report, images, output / 'snapshot', documents=['1']))
    assert len(figures) == 1 and len(figures[0][1].axes) == 2
    assert 'Final prediction' in figures[0][1].axes[0].get_title()
    plt.close(figures[0][1])
    _, literal, _ = evaluate_saved_notebook(annotation, predictions, tmp_path / 'literal', merged_headings=False)
    assert literal.iloc[0]['Micro-F1 (%)'] == 0
    with pytest.raises(ValueError, match='outside'):
        evaluate_saved_notebook(annotation, predictions, predictions / 'evaluation')
    with pytest.raises(ValueError, match='IoU'):
        evaluate_saved_notebook(annotation, predictions, output, iou=.1)


def test_ground_truth_preview_rejects_mismatched_image_dimensions(tmp_path):
    annotation = tmp_path / 'gt.json'
    annotation.write_text(json.dumps(dict(images=[dict(id=1, file_name='1.png', width=100, height=100)],
        annotations=[], categories=[])))
    cv2.imencode('.png', np.full((50, 50, 3), 255, np.uint8))[1].tofile(str(tmp_path / '1.png'))
    report = dict(annotation_file=str(annotation), sources=[dict(image_id=1, file_name='1.png')])
    with pytest.raises(ValueError, match='coordinate frame'):
        list(final_gt_figures(report, tmp_path, tmp_path))


def test_report_cells_need_no_previous_path_configuration_or_model_imports():
    from pathlib import Path
    nbf = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    root = Path(__file__).resolve().parents[1]
    original = nbf.read(root / 'Dual_Skip_v4.ipynb', as_version=4)
    # Mock flag prevents evaluating any real user outputs; no path/configuration
    # variables are supplied, which is the case that previously raised NameError.
    cells = [nbf.v4.new_code_cell('MOCK_MODE = True')]
    cells.extend(nbf.v4.new_code_cell(c.source) for c in original.cells
                 if set(c.metadata.get('tags', [])) & {'layout-evaluation', 'final-vs-gt'})
    cells.append(nbf.v4.new_code_cell("assert 'DOCLAYNET_SAMPLE_DIR' not in globals()\n"
        "assert 'OUTPUT_ROOT' not in globals()\nassert 'torch' not in sys.modules\n"
        "assert 'paddle' not in sys.modules\nassert EVAL_PREDICTIONS.is_absolute()"))
    notebook = nbf.v4.new_notebook(cells=cells)
    nbclient.NotebookClient(notebook, timeout=90, kernel_name='python3',
        resources={'metadata': {'path': str(root)}}).execute()
