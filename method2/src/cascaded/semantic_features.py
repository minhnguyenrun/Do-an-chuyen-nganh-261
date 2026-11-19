"""Shared label-free features for PDF-cell training and measured OCR inference."""
import hashlib
import json
import math
import re
import unicodedata


EMBEDDING_DIM = 256
MIN_OCR_CONFIDENCE = .75
MAX_TOKENS = 512
PREFIXES = {
    'caption_prefix': r'^(?:fig(?:ure)?|table|chart|plate|exhibit)\b',
    'note_prefix': r'^(?:note|source|footnote)\b',
    'list_prefix': r'^(?:[•◦▪‣–—-]\s|\(?\d+[.)]\s|\(?[a-z][.)]\s)',
}
FEATURE_NAMES = (
    [f'embedding_{index:03d}' for index in range(EMBEDDING_DIM)]
    + ['left', 'top', 'width', 'height', 'area', 'line_count_log', 'char_count_log',
       'word_count_log', 'uppercase_fraction', 'digit_fraction', 'punctuation_fraction']
    + list(PREFIXES)
)
FEATURE_SCHEMA = {
    'schema_version': 1,
    'embedding_dim': EMBEDDING_DIM,
    'embedding_max_tokens': MAX_TOKENS,
    'minimum_ocr_rec_score': MIN_OCR_CONFIDENCE,
    'pdf_cell_weighting': 'uniform',
    'ocr_line_weighting': 'measured_rec_score',
    'feature_names': FEATURE_NAMES,
    'prefix_patterns': PREFIXES,
}


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


FEATURE_FINGERPRINT = hashlib.sha256(canonical_bytes(FEATURE_SCHEMA)).hexdigest()


def region_features(texts, bbox, width, height, encode, *, rec_scores=None):
    """Return one finite vector. A missing weight vector denotes PDF-cell input."""
    import numpy as np

    texts = [text.strip() for text in texts]
    if not texts or not all(texts):
        raise ValueError('semantic features require nonempty text')
    x0, y0, x1, y1 = bbox
    if (not all(math.isfinite(value) for value in (*bbox, width, height))
            or not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height):
        raise ValueError('invalid semantic geometry')
    weights = None
    if rec_scores is not None:
        weights = np.asarray(rec_scores, dtype=np.float64)
        if (weights.shape != (len(texts),) or not np.isfinite(weights).all()
                or ((weights < MIN_OCR_CONFIDENCE) | (weights > 1)).any()):
            raise ValueError('semantic OCR confidence below threshold')
    vectors = np.asarray(encode(texts), dtype=np.float64)
    if vectors.shape != (len(texts), EMBEDDING_DIM) or not np.isfinite(vectors).all():
        raise ValueError('invalid Model2Vec feature matrix')
    text = ' '.join(texts)
    letters = sum(char.isalpha() for char in text)
    visible = sum(not char.isspace() for char in text)
    w, h = (x1 - x0) / width, (y1 - y0) / height
    numeric = [
        x0 / width, y0 / height, w, h, w * h,
        math.log1p(len(texts)), math.log1p(len(text)), math.log1p(len(text.split())),
        sum(char.isupper() for char in text) / max(1, letters),
        sum(char.isdigit() for char in text) / visible,
        sum(unicodedata.category(char).startswith('P') for char in text) / visible,
        *(float(bool(re.match(pattern, texts[0], re.IGNORECASE))) for pattern in PREFIXES.values()),
    ]
    return np.concatenate((np.average(vectors, axis=0, weights=weights), numeric)).astype(np.float32)
