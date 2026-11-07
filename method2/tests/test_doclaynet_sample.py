import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from PIL import Image


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[2] / 'scripts/download_datasets/download_doclaynet.py'
    spec = importlib.util.spec_from_file_location('doclaynet_sample_downloader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(index):
    return {'row_idx': index, 'truncated_cells': [], 'row': {
        'image': {'src': f'https://example.test/image/{index}'},
        'metadata': {'image_id': index + 1, 'coco_width': 20, 'coco_height': 40, 'doc_category': 'test'},
        'bboxes': [[2, 4, 10, 20]], 'category_id': [9]}}


def fake_transport(log):
    image = io.BytesIO()
    Image.new('RGB', (10, 20), 'white').save(image, format='JPEG')
    def fetch(url):
        log.append(url)
        if urlparse(url).path == '/rows':
            query = parse_qs(urlparse(url).query)
            start, length = int(query['offset'][0]), int(query['length'][0])
            return json.dumps({'rows': [row(i) for i in range(start, start + length)]}).encode()
        return image.getvalue()
    return fetch


def test_lightweight_pagination_scaling_and_no_redownload(downloader, tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(downloader, 'read_sample_url', fake_transport(log))
    path = downloader.download_lightweight_sample('test', 7, tmp_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert len(data['images']) == 7 and len(data['annotations']) == 7
    assert data['annotations'][0]['bbox'] == [1, 2, 5, 10]
    assert data['images'][0]['width'] == 10
    assert sum('/rows?' in url for url in log) == 2
    assert not list(tmp_path.rglob('*.parquet'))
    log.clear()
    downloader.download_lightweight_sample('test', 7, tmp_path)
    assert log == []
    with pytest.raises(ValueError, match='another split'):
        downloader.download_lightweight_sample('train', 7, tmp_path / '..' / tmp_path.name)


def test_interrupted_sample_resumes_without_duplicate_annotations(downloader, tmp_path, monkeypatch):
    log = []
    fetch = fake_transport(log)
    def interrupt(url):
        if url.endswith('/image/1'):
            raise RuntimeError('interrupted')
        return fetch(url)
    monkeypatch.setattr(downloader, 'read_sample_url', interrupt)
    with pytest.raises(RuntimeError):
        downloader.download_lightweight_sample('test', 3, tmp_path)
    monkeypatch.setattr(downloader, 'read_sample_url', fetch)
    path = downloader.download_lightweight_sample('test', 3, tmp_path)
    data = json.loads(path.read_text())
    assert len(data['images']) == 3 and len(data['annotations']) == 3
    assert len({ann['id'] for ann in data['annotations']}) == 3
    assert sum(url.endswith('/image/0') for url in log) == 1


def test_truncated_metadata_and_invalid_limits_are_rejected(downloader, tmp_path, monkeypatch):
    bad = row(0)
    bad['truncated_cells'] = ['metadata']
    monkeypatch.setattr(downloader, 'read_sample_url', lambda url: json.dumps({'rows': [bad]}).encode())
    with pytest.raises(ValueError, match='truncated'):
        downloader.download_lightweight_sample('test', 1, tmp_path)
    with pytest.raises(ValueError):
        downloader.download_lightweight_sample('test', 0, tmp_path)
