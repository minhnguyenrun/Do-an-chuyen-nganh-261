"""Build an immutable PDF-cell corpus from the reviewed true-train handoff."""
import argparse
from hashlib import sha1, sha256
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascaded.semantic import (PACKAGE_VERSIONS, TEXT_CLASSES, file_record, read_json,
                                   verify_encoder, verify_file, verify_files, write_json)
from src.cascaded.semantic_features import canonical_bytes
from src.cascaded.types import CATEGORIES


DATASET = 'docling-project/DocLayNet-v1.2'
SPLIT = 'train'
ASSET_ROOT = ROOT.parent / 'scratch' / 'cascaded_phase2' / 'prepared_assets'
VALID_IDENTITIES = ASSET_ROOT.parent / 'valid50' / 'manifest.json'
PAGE_LIMIT = 64
BLOCK_LIMIT = 25 * 1024**2
TOTAL_LIMIT = 100 * 1024**2
CATEGORY_LABEL = {CATEGORIES[label]: label for label in TEXT_CLASSES}
IDENTITY_KEYS = ('page_hash', 'original_filename', 'page_no')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hex(value, width):
    return isinstance(value, str) and len(value) == width and all(c in '0123456789abcdef' for c in value)


def verify_packages(root, manifest):
    require(manifest['schema_version'] == 1 and manifest['status'] == 'installed_isolated'
            and manifest['requested_versions'] == PACKAGE_VERSIONS and manifest['extras'] == []
            and manifest['source'] == 'https://pypi.org/simple', 'unexpected isolated package handoff')
    target = Path(manifest['install_target']).resolve()
    require(target.is_dir() and target.is_relative_to(root) and target != root,
            'package target must be inside the reviewed asset root')
    installed = manifest['installed_versions']
    require(all(installed.get(package) == pin for package, pin in PACKAGE_VERSIONS.items()),
            'semantic dependency versions do not match')
    wheels = manifest['wheels']
    require(len(wheels) == len({record['path'] for record in wheels})
            and sum(record['bytes'] for record in wheels) <= 250 * 1024**2, 'invalid wheel inventory')
    names = set()
    for record in wheels:
        path = verify_file(root, record)
        require(path.parent == root / 'wheels' and path.suffix == '.whl', 'unexpected wheel path')
        names.add(path.name)
    require(all(any(name.startswith(f'{package}-{pin}-') for name in names)
                for package, pin in PACKAGE_VERSIONS.items()), 'missing pinned direct wheel')
    return target


def verified_handoff(root):
    manifest = read_json(root / 'assets_manifest.json')
    paths = {'package_manifest_path': 'packages.json',
             'embedding_manifest_path': 'embedding/manifest.json',
             'train_manifest_path': 'train/manifest.json'}
    require(manifest['schema_version'] == 1
            and all(manifest[key] == name for key, name in paths.items()), 'unknown prepared asset schema')
    verify_files(root, manifest['hashes'], set(paths.values()))
    packages, embedding, source = (read_json(root / name) for name in paths.values())
    verify_packages(root, packages)
    directory = verify_encoder(root, embedding)
    for item in embedding['files']:
        if item['name'] == 'model.safetensors':
            oid = item['sha256']
        else:
            content = (directory / item['name']).read_bytes()
            oid = sha1(f'blob {len(content)}\0'.encode('ascii') + content).hexdigest()
        require(item['upstream_oid'] == oid, 'encoder disagrees with upstream object digest')
    return manifest, packages, embedding, source


def bbox_xyxy(box, width, height):
    x, y, w, h = box
    if not all(math.isfinite(value) for value in box) or w <= 0 or h <= 0:
        return None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


def page_examples(row, identity):
    meta = row['metadata']
    width, height = meta['coco_width'], meta['coco_height']
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            'invalid canonical page dimensions')
    matched = row['bboxes'], row['category_id'], row['pdf_cells']
    require(len({len(column) for column in matched}) == 1, 'mismatched PDF-cell annotation arrays')
    for index, (raw_box, category, cells) in enumerate(zip(*matched)):
        require(type(category) is int and category in CATEGORIES.values(), 'unknown DocLayNet category')
        if category not in CATEGORY_LABEL:
            continue
        box = bbox_xyxy(raw_box, width, height)
        if box is None:
            continue
        text_cells = []
        for cell in cells:
            text = cell['text'].strip()
            cell_box = bbox_xyxy(cell['bbox'], width, height)
            if text and cell_box is not None:
                text_cells.append({'text': text, 'bbox': cell_box})
        if text_cells:
            yield dict(identity, region_idx=index, label=CATEGORY_LABEL[category], bbox=box,
                       width=width, height=height, cells=text_cells, text_source='pdf_cells')


def source_examples(root, source, limit):
    revision = source['observed_repo_sha_before']
    require(source['schema_version'] == 1 and source['repo_id'] == DATASET
            and source['config'] == 'default' and source['split'] == SPLIT
            and source['status'] == 'acquired_immutable_payloads_unpinned_api'
            and source['api_revision_verified'] is False and source['errors'] == []
            and is_hex(revision, 40) and revision == source['observed_repo_sha_after'],
            'expected bounded true-train viewer payloads with stable observed revision')
    total_rows = source['train_num_rows']
    offsets = source['selected_offsets']
    payloads = source['raw_payloads']
    limit_record = source['limits']
    require(type(total_rows) is int and total_rows >= PAGE_LIMIT
            and offsets == [round(index * (total_rows - 4) / 15) for index in range(16)]
            and len(payloads) == len(offsets) and limit_record['pages'] == PAGE_LIMIT
            and limit_record['response_bytes'] == BLOCK_LIMIT and limit_record['total_bytes'] == TOTAL_LIMIT
            and all(0 < record['bytes'] <= BLOCK_LIMIT for record in payloads)
            and sum(record['bytes'] for record in payloads) <= TOTAL_LIMIT,
            'invalid prospective train sample or raw byte cap')
    rows = source['rows']
    require(len(rows) == PAGE_LIMIT and len(rows) <= limit,
            'reviewed handoff exceeds requested page cap; input rows will not be reselected')
    pages, examples, observed, page_hashes, document_pages = [], [], [], set(), set()
    for offset, record in zip(offsets, payloads):
        path = verify_file(root, record, f'train/raw_rows/block_{offset}.json')
        block = read_json(path)
        require(block['num_rows_total'] == total_rows and block['partial'] is False
                and [item['row_idx'] for item in block['rows']] == list(range(offset, offset + 4)),
                'viewer block pagination disagrees with train acquisition')
        for item in block['rows']:
            require(item['truncated_cells'] == [], 'viewer row has truncated cells')
            row = item['row']
            identity = {key: row['metadata'][key] for key in IDENTITY_KEYS}
            page_hash, document, page = (identity[key] for key in IDENTITY_KEYS)
            require(is_hex(page_hash, 64) and isinstance(document, str) and bool(document.strip())
                    and type(page) is int and page >= 0 and page_hash not in page_hashes
                    and (document, page) not in document_pages, 'invalid or duplicate source page')
            page_hashes.add(page_hash)
            document_pages.add((document, page))
            observed.append(dict(identity, row_idx=item['row_idx']))
            identity = dict(observed[-1], repo_id=DATASET, config='default', split=SPLIT,
                            qualified_row_id=f'{DATASET}/default/{SPLIT}/{item["row_idx"]}')
            pages.append(identity)
            examples.extend(page_examples(row, identity))
    require(observed == rows, 'raw source row identities disagree with train manifest')
    return pages, examples


def verify_valid_disjoint(pages, validation):
    if not validation.is_file():
        return {'status': 'validation_identity_manifest_unavailable', 'test_accessed': False}
    manifest = read_json(validation)
    require(manifest['split'] == 'validation' and manifest['status'] == 'ready',
            'unexpected validation identity manifest')
    images = manifest['images']
    require(images and all(is_hex(row['page_hash'], 64) and bool(row['original_filename']) for row in images),
            'incomplete validation page identities')
    require(not ({row['page_hash'] for row in images} & {row['page_hash'] for row in pages})
            and not ({row['original_filename'] for row in images} & {row['original_filename'] for row in pages}),
            'train and validation identities collide')
    return {'status': 'disjoint', 'validation_pages_checked': len(images), 'test_accessed': False,
            'validation_manifest_sha256': file_record(validation.parent, validation.name)['sha256']}


def grouped_split(pages, examples, valid_check):
    docs = {page['original_filename'] for page in pages}
    ordered = sorted(docs, key=lambda doc: (sha256(doc.encode('utf-8')).hexdigest(), doc))
    require(len(ordered) >= 2, 'fit and calibration require separate source documents')
    count = max(1, min(len(ordered) - 1, round(.25 * len(ordered))))
    calibration = set(ordered[:count])
    for row in [*pages, *examples]:
        row['partition'] = 'calibration' if row['original_filename'] in calibration else 'fit'
    result = {'schema_version': 1, 'source_split': SPLIT, 'group_by': 'original_filename',
              'ordering': 'ascending SHA256 UTF-8 original_filename', 'validation_disjointness': valid_check}
    for partition in ('fit', 'calibration'):
        selected = [page for page in pages if page['partition'] == partition]
        documents = sorted({page['original_filename'] for page in selected})
        n_examples = sum(row['partition'] == partition for row in examples)
        result[partition] = {'documents': documents, 'document_count': len(documents),
                             'pages': selected, 'page_count': len(selected), 'example_count': n_examples}
    return result


def prepare(assets, output, limit, validation):
    require(1 <= limit <= PAGE_LIMIT, '--max-pages must be in 1..64')
    require(not output.exists(), 'corpus output already exists')
    handoff, _, _, source = verified_handoff(assets)
    pages, examples = source_examples(assets, source, limit)
    split = grouped_split(pages, examples, verify_valid_disjoint(pages, validation))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.semantic-corpus-', dir=output.parent) as temporary:
        corpus = Path(temporary) / 'corpus'
        corpus.mkdir()
        with (corpus / 'corpus.jsonl').open('wb') as file:
            for row in examples:
                file.write(canonical_bytes(row) + b'\n')
        write_json(corpus / 'split_manifest.json', split)
        for key, name in (('package_manifest_path', 'package_manifest.json'),
                          ('embedding_manifest_path', 'embedding_manifest.json'),
                          ('train_manifest_path', 'source_manifest.json')):
            shutil.copyfile(assets / handoff[key], corpus / name)
        names = ('corpus.jsonl', 'split_manifest.json', 'package_manifest.json',
                 'embedding_manifest.json', 'source_manifest.json')
        write_json(corpus / 'corpus_manifest.json', {
            'schema_version': 1, 'repo_id': DATASET, 'config': 'default', 'split': SPLIT,
            'text_source': 'pdf_cells', 'weighting': 'uniform', 'ocr_used': False,
            'domain_shift': 'Paired PDF cells differ from OCR transcripts and line segmentation.',
            'assets_root': str(assets), 'assets_manifest': file_record(assets, 'assets_manifest.json'),
            'page_count': len(pages), 'example_count': len(examples),
            'files': {name: file_record(corpus, name) for name in names},
        })
        corpus.rename(output)
    return split


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=ASSET_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-pages', type=int, default=PAGE_LIMIT)
    parser.add_argument('--prepare-only', action='store_true', required=True)
    parser.add_argument('--validation-manifest', type=Path, default=VALID_IDENTITIES)
    args = parser.parse_args()
    try:
        split = prepare(args.assets_root.resolve(), args.output.resolve(), args.max_pages,
                        args.validation_manifest.resolve())
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f'Preparation failed: {error}\n')
    for role in ('fit', 'calibration'):
        group = split[role]
        print(f"{role}: {group['document_count']} documents, {group['page_count']} pages, "
              f"{group['example_count']} examples")
    print(f'Corpus: {args.output.resolve()}')


if __name__ == '__main__':
    main()
