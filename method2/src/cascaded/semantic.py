"""Offline native semantic artifact and optional calibrated OCR prediction."""
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import process_time

from .semantic_features import (EMBEDDING_DIM, FEATURE_FINGERPRINT, FEATURE_NAMES,
                                FEATURE_SCHEMA, MAX_TOKENS, MIN_OCR_CONFIDENCE,
                                canonical_bytes, region_features)
from .types import CATEGORIES


MODEL_SCHEMA = 1
ENCODER_REPO = 'minishlab/potion-base-8M'
ENCODER_REVISION = 'bf8b056651a2c21b8d2565580b8569da283cab23'
ENCODER_FILES = {'config.json', 'tokenizer.json', 'model.safetensors'}
PACKAGE_VERSIONS = {'model2vec': '0.10.0', 'lightgbm': '4.7.0'}
CORPUS_FILES = {'corpus.jsonl', 'corpus_manifest.json', 'split_manifest.json',
                'embedding_manifest.json', 'package_manifest.json', 'source_manifest.json'}
ARTIFACT_FILES = CORPUS_FILES | {
    'feature_schema.json', 'classifier.txt', 'calibration.json', 'train_metrics.json',
    *(f'encoder/{name}' for name in ENCODER_FILES),
}
TEXT_CLASSES = frozenset(CATEGORIES) - {'image', 'chart'}
MIN_FIT = 5
MIN_CALIBRATION = 3
MIN_CALIBRATION_TOTAL = 30


def fingerprint(value):
    return sha256(canonical_bytes(value)).hexdigest()


def file_record(root, name):
    path = root / name
    digest = sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(name), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def verify_file(root, record, expected=None):
    name = record['path']
    relative = Path(name)
    path = (root / relative).resolve()
    if (not isinstance(name, str) or relative.anchor or '..' in relative.parts
            or not path.is_relative_to(root.resolve()) or (expected is not None and name != expected)):
        raise ValueError('invalid manifest path')
    actual = file_record(root, name)
    if actual != {key: record[key] for key in actual}:
        raise ValueError(f'file size or SHA256 mismatch: {name}')
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_bytes(canonical_bytes(value) + b'\n')


def verify_files(root, files, expected):
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError('manifest files do not match schema')
    for name, record in files.items():
        verify_file(root, record, name)


def verify_encoder(root, manifest):
    if (manifest['schema_version'] != 1 or manifest['repo_id'] != ENCODER_REPO
            or manifest['revision'] != ENCODER_REVISION or manifest['status'] != 'acquired'
            or manifest['config']['hidden_dim'] != EMBEDDING_DIM):
        raise ValueError('unexpected Model2Vec source or schema')
    files = manifest['files']
    if len(files) != len(ENCODER_FILES) or {item['name'] for item in files} != ENCODER_FILES:
        raise ValueError('incomplete encoder manifest')
    paths = []
    for item in files:
        path = verify_file(root, item)
        if path.name != item['name']:
            raise ValueError('unexpected encoder filename')
        paths.append(path)
    directory = paths[0].parent
    if any(path.parent != directory for path in paths):
        raise ValueError('encoder files are in different directories')
    if read_json(directory / 'config.json') != manifest['config']:
        raise ValueError('encoder config disagrees with manifest')
    return directory


def load_static_encoder(directory):
    from model2vec import StaticModel

    if not directory.is_dir() or not all((directory / name).is_file() for name in ENCODER_FILES):
        raise ValueError('encoder must be a complete local Model2Vec directory')
    # StaticModel's local path branch uses only these files.
    model = StaticModel.from_pretrained(directory, max_length=MAX_TOKENS)
    if model.dim != EMBEDDING_DIM:
        raise ValueError('unexpected Model2Vec width')
    return model


def encode_lines(model, texts):
    return model.encode(texts, show_progress_bar=False, use_multiprocessing=False)


def temperature_probabilities(logits, temperature):
    import numpy as np

    raw = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(raw).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('nonfinite logits or invalid temperature')
    logits = (raw - raw.max(axis=-1, keepdims=True)) / temperature
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def supported_classes(class_support):
    return sorted(label for label, counts in class_support.items()
                  if counts['fit'] >= MIN_FIT and counts['calibration'] >= MIN_CALIBRATION)


def validated_head(manifest, calibration):
    classes, mapping, support = (manifest[key] for key in ('classes', 'class_mapping', 'class_support'))
    if (classes != sorted(set(classes)) or len(classes) < 2 or not set(classes) <= TEXT_CLASSES
            or mapping != {label: index for index, label in enumerate(classes)}
            or not set(classes) <= set(support) or set(support) != TEXT_CLASSES):
        raise ValueError('invalid classifier classes')
    for label, counts in support.items():
        if (set(counts) != {'fit', 'calibration'}
                or any(type(value) is not int or value < 0 for value in counts.values())
                or (counts['fit'] > 0) != (label in mapping)):
            raise ValueError('invalid class support')
    eligible = supported_classes(support)
    total = sum(counts['calibration'] for counts in support.values())
    known = sum(support[label]['calibration'] for label in classes)
    if (manifest['supported_classes'] != eligible or calibration['supported_classes'] != eligible
            or calibration['calibration_total'] != total or calibration['calibration_known'] != known):
        raise ValueError('invalid calibration support')
    status = calibration['status']
    temperature = calibration['temperature']
    if (calibration['schema_version'] != 1 or calibration['method'] != 'multiclass_temperature'
            or status != manifest['calibration_status'] or status not in {'calibrated', 'uncalibrated'}
            or (status == 'calibrated' and
                (known < MIN_CALIBRATION_TOTAL or len(eligible) < 2 or type(temperature) not in (int, float)
                 or not math.isfinite(temperature) or not .05 <= temperature <= 20))
            or (status == 'uncalibrated' and temperature is not None)):
        raise ValueError('invalid calibration metadata')
    return classes, eligible, temperature


class SemanticArbitrator:
    """One optional resolver per experiment, with a cache scoped to the active page."""

    def __init__(self, reason):
        self.available = False
        self.abstain_reason = reason
        self.calibration_status = 'unavailable'
        self.model_fingerprint = None
        self.feature_fingerprint = None
        self.supported_classes = ()
        self._page_lines = None
        self._line_vectors = {}

    def status(self):
        return {'available': self.available, 'abstain_reason': self.abstain_reason,
                'calibration_status': self.calibration_status,
                'supported_classes': list(self.supported_classes),
                'model_fingerprint': self.model_fingerprint,
                'feature_fingerprint': self.feature_fingerprint}

    @classmethod
    def from_artifact(cls, path):
        if path is None:
            return cls('artifact_not_configured')
        try:
            root = Path(path)
            manifest_path = root / 'model_manifest.json'
            if not manifest_path.is_file():
                return cls('artifact_missing')
            manifest = read_json(manifest_path)
            model_fingerprint = manifest.pop('model_fingerprint')
            if (manifest['schema_version'] != MODEL_SCHEMA or manifest['model_kind'] != 'Model2Vec+LightGBM'
                    or fingerprint(manifest) != model_fingerprint
                    or manifest['feature_fingerprint'] != FEATURE_FINGERPRINT
                    or manifest['package_versions'] != PACKAGE_VERSIONS):
                return cls('artifact_schema_mismatch')
            verify_files(root, manifest['files'], ARTIFACT_FILES)
            if read_json(root / 'feature_schema.json') != FEATURE_SCHEMA:
                return cls('feature_schema_mismatch')
            embedding = read_json(root / 'embedding_manifest.json')
            directory = verify_encoder(root, embedding)
            if directory != (root / 'encoder').resolve():
                return cls('artifact_encoder_path_mismatch')
            calibration = read_json(root / 'calibration.json')
            classes, supported, temperature = validated_head(manifest, calibration)
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return cls('artifact_integrity_error')

        try:
            if any(version(package) != pin for package, pin in PACKAGE_VERSIONS.items()):
                return cls('dependency_version_mismatch')
            import lightgbm as lgb
            import numpy as np

            model = load_static_encoder(directory)
        except (ImportError, OSError):
            return cls('dependency_unavailable')
        except Exception:
            return cls('encoder_load_error')
        try:
            classifier = lgb.Booster(model_str=(root / 'classifier.txt').read_text(encoding='utf-8'))
            if (classifier.feature_name() != FEATURE_NAMES or classifier.num_model_per_iteration() != len(classes)
                    or not classifier.dump_model()['objective'].startswith('multiclass ')):
                return cls('classifier_schema_mismatch')
            logits = classifier.predict(np.zeros((1, len(FEATURE_NAMES))), raw_score=True, num_threads=1)
            if logits.shape != (1, len(classes)) or not np.isfinite(logits).all():
                return cls('classifier_schema_mismatch')
        except Exception:
            return cls('classifier_load_error')
        resolver = cls(None)
        resolver.available = True
        resolver.calibration_status = calibration['status']
        resolver.model_fingerprint = model_fingerprint
        resolver.feature_fingerprint = FEATURE_FINGERPRINT
        resolver.supported_classes = tuple(supported)
        resolver._classes = classes
        resolver._temperature = temperature
        resolver._encoder = model
        resolver._classifier = classifier
        return resolver

    def _embed_page(self, texts, page_lines):
        import numpy as np

        if self._page_lines is not page_lines:
            self._page_lines = page_lines
            self._line_vectors = {}
        missing = list(dict.fromkeys(text for text in texts if text not in self._line_vectors))
        if missing:
            vectors = encode_lines(self._encoder, missing)
            if vectors.shape != (len(missing), EMBEDDING_DIM) or not np.isfinite(vectors).all():
                raise ValueError('invalid cached line vectors')
            self._line_vectors.update(zip(missing, vectors))
        return np.asarray([self._line_vectors[text] for text in texts])

    def predict(self, label, lines, page_lines, bbox, width, height):
        started = process_time()
        result = dict(self.status(), probabilities={})
        if self.available:
            if label not in self.supported_classes:
                result['abstain_reason'] = 'unsupported_class'
            elif self.calibration_status != 'calibrated':
                result['abstain_reason'] = 'uncalibrated'
            elif (not lines or any(not line.text.strip() or line.rec_score is None
                                   or not MIN_OCR_CONFIDENCE <= line.rec_score <= 1 for line in lines)):
                result['abstain_reason'] = 'insufficient_ocr_confidence'
            else:
                try:
                    features = region_features([line.text for line in lines], bbox, width, height,
                                               lambda texts: self._embed_page(texts, page_lines),
                                               rec_scores=[line.rec_score for line in lines])
                    raw = self._classifier.predict(features[None, :], raw_score=True, num_threads=1)
                    if raw.shape != (1, len(self._classes)):
                        raise ValueError('invalid classifier output shape')
                    probs = temperature_probabilities(raw[0], self._temperature)
                    result['probabilities'] = dict(zip(self._classes, map(float, probs)))
                except Exception:
                    result['abstain_reason'] = 'inference_error'
        result['cpu_seconds'] = process_time() - started
        return result
