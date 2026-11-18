"""Fit a native LightGBM head and temperature on separate true-train documents."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import distributions
import math
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from time import process_time

from prepare_cascaded_semantic import (DATASET, PAGE_LIMIT, ROOT, SPLIT, grouped_split,
                                      require, source_examples, verified_handoff)
from src.cascaded.semantic import (
    CORPUS_FILES, MODEL_SCHEMA, PACKAGE_VERSIONS, TEXT_CLASSES, MIN_CALIBRATION_TOTAL,
    encode_lines, file_record, fingerprint, load_static_encoder, read_json, supported_classes,
    temperature_probabilities, verify_encoder, verify_file, verify_files, write_json,
)
from src.cascaded.semantic_features import FEATURE_FINGERPRINT, FEATURE_NAMES, FEATURE_SCHEMA, canonical_bytes, region_features


def validate_corpus(root):
    manifest = read_json(root / 'corpus_manifest.json')
    require(manifest['schema_version'] == 1 and manifest['repo_id'] == DATASET
            and manifest['config'] == 'default' and manifest['split'] == SPLIT
            and manifest['text_source'] == 'pdf_cells' and manifest['weighting'] == 'uniform'
            and manifest['ocr_used'] is False, 'unexpected semantic corpus provenance')
    verify_files(root, manifest['files'], CORPUS_FILES - {'corpus_manifest.json'})
    assets = Path(manifest['assets_root']).resolve()
    verify_file(assets, manifest['assets_manifest'], 'assets_manifest.json')
    _, packages, embedding, source = verified_handoff(assets)
    require(all(read_json(root / name) == old for name, old in (
        ('package_manifest.json', packages), ('embedding_manifest.json', embedding),
        ('source_manifest.json', source))), 'corpus provenance differs from reviewed handoff')
    pages, rows = source_examples(assets, source, PAGE_LIMIT)
    split = read_json(root / 'split_manifest.json')
    require(split == grouped_split(pages, rows, split['validation_disjointness'])
            and manifest['page_count'] == len(pages) and manifest['example_count'] == len(rows),
            'corpus grouping or counts differ from reviewed source')
    digest = sha256()
    for row in rows:
        digest.update(canonical_bytes(row) + b'\n')
    require(digest.hexdigest() == manifest['files']['corpus.jsonl']['sha256'],
            'corpus bytes differ from reviewed train rows')
    return rows, manifest, split, assets, packages, embedding, source


def use_isolated_target(packages):
    target = Path(packages['install_target']).resolve()

    def normalized(name):
        return re.sub(r'[-_.]+', '-', name).lower()

    actual = {normalized(dist.metadata['Name']): dist.version for dist in distributions(path=[str(target)])}
    expected = {normalized(name): pin for name, pin in packages['installed_versions'].items()}
    require(actual == expected, 'isolated dependency target differs from its package manifest')
    sys.path.insert(0, str(target))
    return target


def calibration_metrics(probs, labels):
    import numpy as np

    if len(labels) == 0:
        return None
    correct_prob = probs[np.arange(len(labels)), labels]
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = (confidence >= low) & ((confidence < high) if index < 9 else (confidence <= high))
        count = int(selected.sum())
        bins.append({'lower': low, 'upper': high, 'count': count,
                     'mean_confidence': float(confidence[selected].mean()) if count else None,
                     'accuracy': float(correct[selected].mean()) if count else None})
    return {
        'examples': len(labels),
        'log_loss': float(-np.log(np.clip(correct_prob, np.finfo(float).tiny, 1.)).mean()),
        'brier_multiclass': float((1 - 2 * correct_prob + np.square(probs).sum(axis=1)).mean()),
        'accuracy': float(correct.mean()),
        'expected_calibration_error': sum(
            entry['count'] * abs(entry['accuracy'] - entry['mean_confidence']) / len(labels)
            for entry in bins if entry['count']
        ),
        'reliability_bins': bins,
    }


def fit_temperature(logits, labels):
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    def nll(log_temperature):
        scaled = logits / math.exp(log_temperature)
        return float((logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]).mean())

    limit = math.log(20)
    result = minimize_scalar(nll, bounds=(-limit, limit), method='bounded', options={'xatol': 1e-5})
    if not result.success or not math.isfinite(result.fun):
        raise ValueError('temperature optimization did not converge')
    return math.exp(result.x) if result.fun < nll(0.) else 1.


def train(root, output, seed):
    started = process_time()
    rows, corpus, split, assets, packages, embedding, source = validate_corpus(root)
    target = use_isolated_target(packages)
    import lightgbm as lgb
    import model2vec
    import numpy as np

    require(all(Path(module.__file__).resolve().is_relative_to(target)
                for module in (lgb, model2vec, np)), 'learned dependencies were not imported from the isolated target')
    fit = [row for row in rows if row['partition'] == 'fit']
    calibration = [row for row in rows if row['partition'] == 'calibration']
    fit_counts = Counter(row['label'] for row in fit)
    calibration_counts = Counter(row['label'] for row in calibration)
    classes = sorted(fit_counts)
    require(len(classes) >= 2 and calibration, 'insufficient train fit classes or calibration examples')
    mapping = {label: index for index, label in enumerate(classes)}
    known = [row for row in calibration if row['label'] in mapping]
    support = {label: {'fit': fit_counts[label], 'calibration': calibration_counts[label]}
               for label in sorted(TEXT_CLASSES)}
    eligible = supported_classes(support)
    directory = verify_encoder(assets, embedding)
    encoder = load_static_encoder(directory)

    def matrix(selected):
        return np.stack([
            region_features([cell['text'] for cell in row['cells']], row['bbox'], row['width'], row['height'],
                            lambda texts: encode_lines(encoder, texts))
            for row in selected
        ])

    def labels(selected):
        return np.array([mapping[row['label']] for row in selected], dtype=np.int32)

    params = {
        'objective': 'multiclass', 'num_class': len(classes), 'seed': seed, 'device_type': 'cpu',
        'num_threads': 1, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1,
        'learning_rate': .07, 'num_leaves': 15, 'max_depth': 4, 'min_data_in_leaf': 5,
    }
    rounds = 100
    classifier = lgb.train(params, lgb.Dataset(matrix(fit), label=labels(fit), feature_name=FEATURE_NAMES),
                           num_boost_round=rounds)
    require(classifier.num_trees() > len(classes), 'LightGBM did not learn any tree splits')
    temperatures_eligible = len(known) >= MIN_CALIBRATION_TOTAL and len(eligible) >= 2
    status = 'calibrated' if temperatures_eligible else 'uncalibrated'
    logits = (classifier.predict(matrix(known), raw_score=True, num_threads=1) if known
              else np.empty((0, len(classes)), dtype=np.float64))
    require(logits.shape == (len(known), len(classes)) and np.isfinite(logits).all(),
            'invalid LightGBM calibration logits')
    y_cal = labels(known)
    temperature = fit_temperature(logits, y_cal) if temperatures_eligible else None
    cal_manifest = {
        'schema_version': 1, 'method': 'multiclass_temperature', 'status': status,
        'temperature': temperature, 'supported_classes': eligible, 'calibration_total': len(calibration),
        'calibration_known': len(known),
    }
    if temperature is None:
        cal_manifest['reason'] = 'insufficient_known_calibration_examples_or_supported_classes'
    supported_examples = sum(row['label'] in eligible for row in calibration)
    train_metrics = {
        'fit_examples': len(fit), 'calibration_examples': len(calibration),
        'calibration_unknown_examples': len(calibration) - len(known),
        'calibration_supported_examples': supported_examples,
        'calibration_supported_coverage': supported_examples / len(calibration),
        'calibration_uncalibrated': calibration_metrics(temperature_probabilities(logits, 1.), y_cal),
        'calibration_calibrated': (calibration_metrics(temperature_probabilities(logits, temperature), y_cal)
                                   if temperature is not None else None),
        'cpu_seconds': process_time() - started,
        'iterations': classifier.current_iteration(), 'trees': classifier.num_trees(),
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.semantic-artifact-', dir=output) as temporary:
        artifact = Path(temporary) / 'artifact'
        (artifact / 'encoder').mkdir(parents=True)
        for name in CORPUS_FILES - {'embedding_manifest.json', 'corpus_manifest.json'}:
            shutil.copyfile(root / name, artifact / name)
        local_files = []
        for record in embedding['files']:
            name = f"encoder/{record['name']}"
            shutil.copyfile(assets / record['path'], artifact / name)
            local_files.append(dict(record, path=name, source_asset_path=record['path']))
        write_json(artifact / 'embedding_manifest.json',
                   dict(embedding, local_model_directory='encoder', files=local_files))
        artifact_corpus_files = dict(corpus['files'])
        artifact_corpus_files['embedding_manifest.json'] = file_record(artifact, 'embedding_manifest.json')
        write_json(artifact / 'corpus_manifest.json', dict(corpus, files=artifact_corpus_files))
        write_json(artifact / 'feature_schema.json', FEATURE_SCHEMA)
        write_json(artifact / 'calibration.json', cal_manifest)
        write_json(artifact / 'train_metrics.json', train_metrics)
        (artifact / 'classifier.txt').write_bytes(classifier.model_to_string().encode('utf-8'))
        manifest = {
            'schema_version': MODEL_SCHEMA, 'model_kind': 'Model2Vec+LightGBM',
            'feature_fingerprint': FEATURE_FINGERPRINT, 'package_versions': PACKAGE_VERSIONS,
            'encoder_repo_id': embedding['repo_id'], 'encoder_revision': embedding['revision'],
            'classes': classes, 'class_mapping': mapping, 'class_support': support,
            'supported_classes': eligible, 'calibration_status': status,
            'source_split': SPLIT, 'source_repo_id': DATASET,
            'observed_source_revision': source['observed_repo_sha_before'],
            'api_revision_verified': source['api_revision_verified'],
            'training_text_source': corpus['text_source'], 'training_ocr_used': corpus['ocr_used'],
            'training_domain_shift': corpus['domain_shift'],
            'input_corpus_manifest': file_record(root, 'corpus_manifest.json'),
            'fit_page_count': split['fit']['page_count'], 'calibration_page_count': split['calibration']['page_count'],
            'lightgbm_parameters': params, 'num_boost_round': rounds,
            'files': {path.relative_to(artifact).as_posix(): file_record(artifact, path.relative_to(artifact).as_posix())
                      for path in sorted(artifact.rglob('*')) if path.is_file()},
        }
        model_fingerprint = fingerprint(manifest)
        write_json(artifact / 'model_manifest.json', dict(manifest, model_fingerprint=model_fingerprint))
        destination = output / model_fingerprint
        require(not destination.exists(), 'semantic artifact output already exists')
        artifact.rename(destination)
    return destination, cal_manifest


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    try:
        destination, calibration = train(args.corpus.resolve(), args.output.resolve(), args.seed)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        parser.exit(1, f'Training failed: {error}\n')
    print(f'Calibration status: {calibration["status"]}')
    print(f'Supported classes: {", ".join(calibration["supported_classes"])}')
    print(f'Artifact: {destination}')


if __name__ == '__main__':
    main()
