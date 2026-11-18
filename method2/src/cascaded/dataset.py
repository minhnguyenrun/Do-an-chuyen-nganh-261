"""Canonical dataset locations use the repository's shared split definitions."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import subprocess
import sys
import json


@lru_cache(maxsize=1)
def _splits():
    path = Path(__file__).resolve().parents[3] / 'src/common/splits.py'
    spec = importlib.util.spec_from_file_location('_cascaded_repository_splits', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def doclaynet_processed_dir(method_root, split):
    info = _splits().resolve(split)
    return Path(method_root).resolve().parent / 'datasets' / info.dataset / 'processed'


def ensure_doclaynet(method_root, split, count):
    if type(count) is not int or count < 1:
        raise ValueError('Image count must be a positive integer')
    info = _splits().resolve(split)
    directory = doclaynet_processed_dir(method_root, split)
    annotation = directory / 'annotations' / f'{info.annotation}.json'

    def ready():
        if not annotation.is_file():
            return False
        data = json.loads(annotation.read_text(encoding='utf-8'))
        if data.get('info', {}).get('mode') == 'lightweight':
            raise ValueError('Canonical dataset contains viewer previews; original images/GT required')
        selected = data.get('images', [])[:count]
        return len(selected) == count and 'annotations' in data and bool(data.get('categories')) and all(
            (directory / 'images' / im['file_name']).is_file()
            and (directory / 'images' / im['file_name']).stat().st_size > 0 for im in selected)

    if not ready():
        script = Path(method_root).resolve().parent / 'scripts/download_datasets/download_doclaynet.py'
        print(f'Thieu anh/GT: tu dong tai bo sung vao {directory}', flush=True)
        subprocess.check_call([sys.executable, str(script), '--ensure', '--splits', info.annotation,
                               '--max-images', str(count)])
        if not ready():
            raise ValueError(f'Downloader finished without {count} complete images + GT: {annotation}')
    return directory, annotation
