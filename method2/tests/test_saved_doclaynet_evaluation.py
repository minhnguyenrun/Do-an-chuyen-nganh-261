import importlib.util
import json
from pathlib import Path

import pytest


def test_saved_snapshot_matching_unknown_classes_and_heading_sensitivity(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_saved_doclaynet.py'
    spec = importlib.util.spec_from_file_location('saved_doclaynet_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gt = dict(images=[dict(id=i, file_name=f'{i}.png', width=100, height=100) for i in (1, 2, 3, 4)],
              categories=[dict(id=8, name='Section-header'), dict(id=10, name='Text'), dict(id=11, name='Title')],
              annotations=[dict(image_id=1, category_id=10, bbox=[0, 0, 20, 20]),
                           dict(image_id=2, category_id=8, bbox=[0, 0, 20, 20]),
                           dict(image_id=4, category_id=10, bbox=[0, 0, 20, 20])])
    annotation_path = tmp_path / 'test.json'
    annotation_path.write_text(json.dumps(gt))
    predictions = tmp_path / 'predictions'
    for image_id, label in [(1, 'text'), (2, 'title'), (3, 'text')]:
        final = [dict(id='b', bbox_final=[0, 0, 20, 20], label_final=label, content='real content')]
        if image_id == 1:
            final.append(dict(id='unknown', bbox_final=[50, 50, 60, 60], label_final='number', content='1'))
        result = dict(document_id=str(image_id), page_index=0, final_regions=final,
                      original_regions=[dict(id='b', bbox=[0, 0, 20, 20], label=label)],
                      evidence=[dict(parser='glm_mock' if image_id == 3 else 'glm_local')], revisions=[])
        folder = predictions / str(image_id)
        folder.mkdir(parents=True)
        (folder / 'result.json').write_text(json.dumps(result))
    output = tmp_path / 'report'
    report = module.evaluate_saved(annotation_path, predictions, output)
    assert report['image_count'] == 2 and report['ground_truth_boxes'] == 2
    primary = report['metrics']['0.5']['final_literal']
    assert primary['strict']['overall']['f1'] == pytest.approx(.4)
    assert primary['classification_on_localized_matches']['accuracy'] == .5
    assert primary['localization_only']['overall']['f1'] == pytest.approx(.8)
    assert report['metrics']['0.5']['final_title_as_section']['strict']['overall']['f1'] == pytest.approx(.8)
    assert report['unmapped_labels'] == {'number': 1}
    assert len(list((output / 'snapshot').rglob('result.json'))) == 2
    module.write_markdown_report(report, output)
    assert (output / 'report.md').is_file()
