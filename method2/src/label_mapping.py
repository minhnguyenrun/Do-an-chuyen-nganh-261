"""GLM-OCR SDK (PP-DocLayoutV3) -> Dual-Skip -> DocLayNet.

Sources checked 2026-10-24:
https://github.com/zai-org/GLM-OCR/blob/main/glmocr/config.yaml
https://github.com/zai-org/GLM-OCR/blob/main/glmocr/layout/layout_detector.py
https://github.com/DS4SD/DocLayNet

The 25 labels belong to the SDK's layout detector, NOT to an independent
classification head of the GLM-OCR recognition model. Cross-dataset mappings
below are our evaluation policy, not an official GLM/DocLayNet crosswalk.
"""

GLM_SDK_LAYOUT_LABELS = (
    'abstract', 'algorithm', 'aside_text', 'chart', 'content', 'display_formula',
    'doc_title', 'figure_title', 'footer', 'footer_image', 'footnote',
    'formula_number', 'header', 'header_image', 'image', 'inline_formula',
    'number', 'paragraph_title', 'reference', 'reference_content', 'seal',
    'table', 'text', 'vertical_text', 'vision_footnote',
)
GLM_SDK_TO_PIPELINE = {label: label for label in GLM_SDK_LAYOUT_LABELS} | {
    'doc_title': 'title', 'paragraph_title': 'title',
    'display_formula': 'formula', 'inline_formula': 'formula',
}
GLM_SDK_OCR_TASKS = {
    **{label: 'text' for label in (
        'abstract', 'algorithm', 'content', 'doc_title', 'figure_title',
        'paragraph_title', 'reference_content', 'text', 'vertical_text',
        'vision_footnote', 'seal', 'formula_number')},
    'table': 'table', 'display_formula': 'formula', 'inline_formula': 'formula',
    'chart': 'skip', 'image': 'skip',
    **{label: 'abandon' for label in (
        'header', 'footer', 'number', 'footnote', 'aside_text', 'reference',
        'footer_image', 'header_image')},
}
DOCLAYNET_CLASSES = {
    1: 'Caption', 2: 'Footnote', 3: 'Formula', 4: 'List-item',
    5: 'Page-footer', 6: 'Page-header', 7: 'Picture', 8: 'Section-header',
    9: 'Table', 10: 'Text', 11: 'Title',
}
# Keep this literal policy compatible with previously saved evaluation reports.
PIPELINE_TO_DOCLAYNET = {
    'figure_title': 1, 'table_title': 1, 'chart_title': 1, 'caption': 1,
    'footnote': 2, 'formula': 3, 'display_formula': 3, 'inline_formula': 3,
    'list': 4, 'list_item': 4, 'list-item': 4,
    'footer': 5, 'footer_image': 5, 'header': 6, 'header_image': 6,
    'image': 7, 'figure': 7, 'chart': 7, 'picture': 7,
    'paragraph_title': 8, 'section_header': 8, 'table': 9,
    'text': 10, 'residual': 10, 'abstract': 10, 'content': 10,
    'reference': 10, 'reference_content': 10, 'aside_text': 10,
    'algorithm': 10, 'vertical_text': 10,
    'title': 11, 'doc_title': 11,
}
UNMAPPED_REASONS = {
    'number': 'Page number needs page context to distinguish header/footer.',
    'formula_number': 'An equation identifier is not a complete Formula region.',
    'seal': 'No explicit seal class; Picture is not assumed automatically.',
    'vision_footnote': 'Visual note may be Caption/Footnote/Text; role needs verification.',
}


def normalize_layout_label(label):
    """Preserve unknown labels so they remain visible for review/evaluation."""
    return GLM_SDK_TO_PIPELINE.get(label, label)


def doclaynet_category(label, *, heading_mode='literal'):
    if heading_mode not in ('literal', 'section', 'merged'):
        raise ValueError('heading_mode must be literal, section or merged')
    category = PIPELINE_TO_DOCLAYNET.get(label, 0)
    if heading_mode in ('section', 'merged') and label == 'title':
        category = 8
    if heading_mode == 'merged' and category == 11:
        category = 8
    return category


def mapping_records():
    """One auditable record for every official SDK layout label."""
    records = []
    for source in GLM_SDK_LAYOUT_LABELS:
        pipeline = normalize_layout_label(source)
        category = doclaynet_category(pipeline)
        records.append(dict(source_label=source, pipeline_label=pipeline,
            sdk_task=GLM_SDK_OCR_TASKS[source], doclaynet_id=category,
            doclaynet_label=DOCLAYNET_CLASSES.get(category, 'Unmapped'),
            merged_heading_id=doclaynet_category(pipeline, heading_mode='merged'),
            note=('Title/Section-header distinction was collapsed; use merged-heading evaluation.'
                  if pipeline == 'title' else UNMAPPED_REASONS.get(pipeline, ''))))
    return records
