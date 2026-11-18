"""Canonical split paths and automatic download/repair, with no network or weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from src.cascaded.dataset import doclaynet_processed_dir, ensure_doclaynet
from src.cascaded.runner import prepare_records, write_json


def data(count):
    return {'images': [{'id': i, 'file_name': f'{i}.png', 'width': 20, 'height': 20} for i in range(count)],
            'annotations': [{'id': i+1, 'image_id': i, 'category_id': 10, 'bbox': [0,0,10,10]} for i in range(count)],
            'categories': [{'id': 10, 'name': 'Text'}]}


def populate(directory, count, images=True):
    value = data(count)
    write_json(directory/'annotations/test.json', value)
    if images:
        for im in value['images']:
            path = directory/'images'/im['file_name']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original-image')
    return value


@pytest.mark.parametrize('split,folder,ann', [('train','doclaynet','train'),
    ('val','doclaynet_val','validation'), ('validation','doclaynet_val','validation'),
    ('valid','doclaynet_val','validation'), ('test','doclaynet_test','test')])
def test_shared_split_mapping(tmp_path, split, folder, ann):
    from src.cascaded.dataset import _splits
    assert doclaynet_processed_dir(tmp_path/'method2', split) == tmp_path/'datasets'/folder/'processed'
    assert _splits().resolve(split).annotation == ann


def test_existing_images_no_downloader_and_gt_paths(tmp_path, monkeypatch):
    root = tmp_path/'method2'
    directory = doclaynet_processed_dir(root, 'test')
    populate(directory, 3)
    monkeypatch.setattr('subprocess.check_call', lambda *a, **k: pytest.fail('must reuse dataset'))
    records, annotation = prepare_records(root, split='test', count=2)
    assert annotation == directory/'annotations/test.json'
    assert len(records) == 2 and Path(records[0]['image']).parent == directory/'images'


@pytest.mark.parametrize('missing', ['all', 'image', 'annotation', 'count'])
def test_missing_data_calls_root_script(tmp_path, monkeypatch, missing):
    root = tmp_path/'method2'
    directory = doclaynet_processed_dir(root, 'test')
    if missing != 'all':
        populate(directory, 1 if missing == 'count' else 2)
        if missing == 'image':
            (directory/'images/0.png').unlink()
        if missing == 'annotation':
            (directory/'annotations/test.json').unlink()
    calls = []
    def download(args):
        calls.append(args)
        populate(directory, 2)
    monkeypatch.setattr('subprocess.check_call', download)
    ensure_doclaynet(root, 'test', 2)
    assert calls == [[sys.executable, str(root.parent/'scripts/download_datasets/download_doclaynet.py'),
                      '--ensure','--splits','test','--max-images','2']]
    ensure_doclaynet(root, 'test', 2)
    assert len(calls) == 1


def test_rejects_previews_and_incorrect_split_directory(tmp_path):
    root = tmp_path/'method2'
    directory = doclaynet_processed_dir(root, 'test')
    value = populate(directory, 1)
    write_json(directory/'annotations/test.json', dict(value, info={'mode':'lightweight'}))
    with pytest.raises(ValueError, match='viewer'):
        ensure_doclaynet(root, 'test', 1)
    with pytest.raises(ValueError, match='shared dataset'):
        prepare_records(root, tmp_path/'method2/outputs/sample', 'test', 1)


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    # Offline stubs only for optional transport imports; real disk/merge code is exercised.
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda:None, hf_hub_download=None))
    monkeypatch.setitem(sys.modules, 'tqdm', SimpleNamespace(tqdm=lambda x, **k:x))
    path = Path(__file__).resolve().parents[2]/'scripts/download_datasets/download_doclaynet.py'
    spec = importlib.util.spec_from_file_location('ensure_downloader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'DATASETS_DIR', tmp_path/'datasets')
    monkeypatch.setattr(module, 'RAW_DIR', tmp_path/'datasets/doclaynet/raw')
    return module


def test_downloader_existing_dataset_does_not_access_network(downloader, monkeypatch):
    directory = downloader.processed_dir('test')
    populate(directory, 3)
    monkeypatch.setattr(downloader, 'list_shards', lambda *a:pytest.fail('no network'))
    assert downloader.ensure_processed_images('test', 2) == directory/'annotations/test.json'


def test_downloader_rebuilds_annotation_from_shards_without_network(downloader, monkeypatch):
    directory = downloader.processed_dir('test')
    value = populate(directory, 3)
    write_json(directory/'annotations/_shards/test-00000.json', dict(value, complete=True))
    (directory/'annotations/test.json').unlink()
    monkeypatch.setattr(downloader, 'list_shards', lambda *a:pytest.fail('no network'))
    result = downloader.ensure_processed_images('test', 2)
    assert len(json.loads(result.read_text())['images']) == 3


@pytest.mark.parametrize('existing,missing,count,expected_limit', [(3,False,5,5), (5,True,2,None), (0,False,2,2)])
def test_downloader_downloads_extends_and_repairs_without_shrinking(downloader, monkeypatch, existing, missing, count, expected_limit):
    directory = downloader.processed_dir('test')
    filename = 'data/test-00000.parquet'
    if existing:
        value = populate(directory, existing)
        write_json(downloader.shard_record_path('test', filename), dict(value, complete=missing))
        if missing:
            (directory/'images/0.png').unlink()
    calls = []
    monkeypatch.setattr(downloader, 'list_shards', lambda *a:[('test',filename,100)])
    monkeypatch.setattr(downloader, 'download_shard', lambda name: calls.append(name) or Path('fake.parquet'))
    def extract(split, name, path, max_images):
        assert max_images == expected_limit
        value = populate(directory, max(existing,count))
        write_json(downloader.shard_record_path(split,name), dict(value,complete=True))
    monkeypatch.setattr(downloader,'extract_shard',extract)
    manifest = downloader.ensure_processed_images('test',count)
    assert calls == [filename]
    assert len(json.loads(manifest.read_text())['images']) == max(existing,count)


def test_both_notebooks_use_shared_dataset_and_auto_download():
    root = Path(__file__).resolve().parents[1]
    for name in ('S','V3'):
        nb = json.loads((root/f'Cascaded_Layout_{name}_YOLO_MinerU.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']: ''.join(c['source']) for c in nb['cells']}
        assert 'DATASET_DIR = doclaynet_processed_dir(METHOD2_ROOT, SPLIT)' in cells['config']
        assert 'prepare_records(METHOD2_ROOT, DATASET_DIR' in cells['data']
        assert 'DOWNLOAD_SAMPLE' not in cells['data'] and '--lightweight' not in cells['data']


def test_merge_repair_preserves_imported_images_outside_shard_records(downloader):
    directory = downloader.processed_dir('test')
    value = populate(directory, 5)
    partial = data(2)
    write_json(directory/'annotations/_shards/test-00000.json', dict(partial,complete=False))
    downloader.merge_split('test', preserve_existing=True)
    result = json.loads((directory/'annotations/test.json').read_text())
    assert result['images'] == value['images']
    assert {ann['image_id'] for ann in result['annotations']} == set(range(5))
    assert len({ann['id'] for ann in result['annotations']}) == 5
