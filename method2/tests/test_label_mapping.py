import pytest

from src.label_mapping import (GLM_SDK_LAYOUT_LABELS, GLM_SDK_TO_PIPELINE,
    GLM_SDK_OCR_TASKS, DOCLAYNET_CLASSES, UNMAPPED_REASONS,
    normalize_layout_label, doclaynet_category, mapping_records)


def test_all_25_official_sdk_labels_have_auditable_two_stage_mapping():
    assert len(set(GLM_SDK_LAYOUT_LABELS)) == 25
    assert set(GLM_SDK_LAYOUT_LABELS) == set(GLM_SDK_TO_PIPELINE) == set(GLM_SDK_OCR_TASKS)
    rows = mapping_records()
    assert len(rows) == 25
    assert {r['pipeline_label'] for r in rows if not r['doclaynet_id']} == set(UNMAPPED_REASONS)
    assert all(r['note'] for r in rows if not r['doclaynet_id'])


def test_heading_collapse_is_explicit_and_merged_mode_is_consistent():
    for source in ('doc_title', 'paragraph_title'):
        label = normalize_layout_label(source)
        assert label == 'title'
        assert doclaynet_category(label) == 11
        assert doclaynet_category(label, heading_mode='merged') == 8
    assert doclaynet_category('paragraph_title') == 8
    assert doclaynet_category('doc_title', heading_mode='merged') == 8
    assert len(DOCLAYNET_CLASSES) == 11
    with pytest.raises(ValueError):
        doclaynet_category('title', heading_mode='guess')


def test_formula_variants_merge_but_equation_number_is_not_formula():
    for source in ('display_formula', 'inline_formula'):
        assert doclaynet_category(normalize_layout_label(source)) == 3
    assert doclaynet_category(normalize_layout_label('formula_number')) == 0


def test_sdk_abandon_and_skip_do_not_drop_our_predictions():
    assert GLM_SDK_OCR_TASKS['header'] == 'abandon'
    assert doclaynet_category(normalize_layout_label('header')) == 6
    assert GLM_SDK_OCR_TASKS['chart'] == 'skip'
    assert doclaynet_category(normalize_layout_label('chart')) == 7
    assert doclaynet_category('list_item') == 4
    assert doclaynet_category('residual') == 10
    assert normalize_layout_label('future_unknown') == 'future_unknown'
    assert doclaynet_category('future_unknown') == 0
