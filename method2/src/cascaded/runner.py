"""Resumable model phases. Only this module writes experiment outputs."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time
import sys
import numpy as np
from .types import Config, Proposal, Line, Result, serialize_proposal
from .fusion import prepare_layout
from .inference_policy import build_inference_request
from .attribution import finish_layout
from .adapters import make_adapter
from .telemetry import DeviceMemory
from .v3_features import missing_layout_features

ALGORITHM_VERSION = 'cascade-2.1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def versions():
    packages = ['numpy', 'opencv-python', 'shapely', 'paddlepaddle-gpu', 'paddleocr', 'paddlex',
                'torch', 'transformers', 'doclayout-yolo', 'mineru-vl-utils']
    result = {'python': sys.version, 'algorithm': ALGORITHM_VERSION}
    result['source_sha256'] = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def check_runtime(cfg):
    """Report readiness without loading or downloading weights."""
    import importlib.util
    modules = ['cv2', 'shapely', 'pandas', 'matplotlib']
    if not cfg.mock:
        modules += ['torch', 'torchvision', 'paddle', 'paddleocr', 'doclayout_yolo', 'mineru_vl_utils', 'transformers']
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    info = {'mode': 'MOCK' if cfg.mock else 'REAL', 'missing_modules': missing, 'versions': versions()}
    if not missing and not cfg.mock:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if cfg.device == 'cuda' and not info['cuda_available']:
            info['error'] = 'CUDA unavailable; explicitly choose CPU or install the matching torch build'
    info['ready'] = not missing and 'error' not in info
    return info


def image_read(path):
    import cv2
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def cuda_measure(cfg, reset=False):
    if cfg.mock or cfg.device != 'cuda':
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {'allocated_mb': torch.cuda.max_memory_allocated()/2**20,
            'reserved_mb': torch.cuda.max_memory_reserved()/2**20,
            'scope': 'PyTorch allocator only; Paddle worker is not included'}


class Experiment:
    def __init__(self, cfg, records, output_root, adapter_factory=make_adapter):
        self.cfg, self.records, self.factory = cfg, list(records), adapter_factory
        if not self.records or len({r['key'] for r in records}) != len(records):
            raise ValueError('Need nonempty records with unique keys')
        if any(Path(r['key']).name != r['key'] or r['key'] in ('.', '..') for r in records):
            raise ValueError('Record keys must be safe directory names')
        self.root = Path(output_root)/cfg.variant/('mock' if cfg.mock else 'real')
        self.environment = versions()
        signature_inputs = {'config': asdict(cfg), 'environment': self.environment}
        self.semantic = None
        if cfg.semantic_enabled:
            from .semantic import SemanticArbitrator

            self.semantic = SemanticArbitrator.from_artifact(cfg.semantic_artifact_path)
            signature_inputs['semantic'] = self.semantic.status()
        self.signature = digest(signature_inputs)
        self.root = self.root/'v2.1'/self.signature[:12]
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root/'manifest.json'
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {}
        self.manifest.update(config=asdict(cfg), environment=self.environment, signature=self.signature,
                             records=self.records, mode='MOCK' if cfg.mock else 'REAL')
        if self.semantic is not None:
            self.manifest['semantic'] = self.semantic.status()
        self.active = None
        self._save()

    def _save(self):
        write_json(self.manifest_path, self.manifest)

    def folder(self, record):
        return self.root/record['key']

    def fingerprint(self, record):
        return hashlib.sha256(Path(record['image']).read_bytes()).hexdigest()

    def cached(self, record, phase):
        path = self.folder(record)/f'{phase}.json'
        if not path.exists():
            return None
        try:
            data = read_json(path)
        except (ValueError, OSError):
            return None
        if phase == 'mineru':
            for dependency in ('paddle', 'yolo'):
                source = self.folder(record)/f'{dependency}.json'
                if (not source.exists() or data.get('dependencies', {}).get(dependency) !=
                        hashlib.sha256(source.read_bytes()).hexdigest()):
                    return None
        if data['signature'] == self.signature and data['image_sha256'] == self.fingerprint(record):
            if phase == 'paddle' and not data['output'].get('layout_features'):
                data['output']['layout_features'] = missing_layout_features()
            return data
        return None

    def run_phase(self, phase, progress=print):
        if phase not in ('paddle', 'yolo', 'mineru'):
            raise ValueError('Unknown phase')
        started = time.perf_counter()
        semantic_args = {'semantic': self.semantic} if self.semantic is not None else {}
        todo = [r for r in self.records if self.cached(r, phase) is None]
        phase_info = {'cache_hits': len(self.records)-len(todo), 'computed': len(todo)}
        memory = DeviceMemory(enabled=not self.cfg.mock)
        memory.__enter__()
        try:
            if phase != 'mineru' and todo and not (phase == 'yolo' and not self.cfg.enable_yolo):
                self.active = self.factory(self.cfg, phase)
                loaded = time.perf_counter()
                self.active.initialize()
                phase_info['load_seconds'] = time.perf_counter()-loaded
                phase_info['model'] = self.active.metadata
            for record in todo:
                progress(f'{phase}: {record["key"]}')
                image = image_read(record['image'])
                self.folder(record).mkdir(parents=True, exist_ok=True)
                cuda_measure(self.cfg, reset=True) if phase != 'paddle' else None
                begin = time.perf_counter()
                if phase == 'mineru':
                    paddle, yolo = self.cached(record, 'paddle'), self.cached(record, 'yolo')
                    if paddle is None or yolo is None:
                        raise ValueError('Run Paddle and YOLO phases before MinerU')
                    proposals = [Proposal(**p) for p in paddle['output']['proposals']+yolo['output']['proposals']]
                    lines = [Line(**l) for l in paddle['output']['lines']]
                    state = prepare_layout(proposals, lines, self.cfg, image.shape[1], image.shape[0], **semantic_args)
                    targets = state['targets']
                    predictions = {}
                    calls = 0
                    for target in targets:
                        if not self.cfg.enable_mineru:
                            predictions[target['id']] = {'status': 'disabled', 'blocks': [], 'tokens': 0}
                            continue
                        if not target['line_ids']:
                            predictions[target['id']] = {'status': 'no_target_lines', 'blocks': [], 'tokens': 0}
                            continue
                        if calls >= self.cfg.max_crops:
                            continue
                        calls += 1
                        try:
                            request = build_inference_request(target, lines, image.shape[1], image.shape[0], self.cfg)
                        except (ValueError, TypeError, OverflowError) as exc:
                            predictions[target['id']] = {'status': 'invalid_request', 'error': str(exc),
                                'tokens': None, 'blocks': [], 'adaptive_inference_requested': self.cfg.enable_adaptive_inference}
                            continue
                        requested_request = request
                        if self.active is None:
                            self.active = self.factory(self.cfg, phase)
                            load_start = time.perf_counter()
                            self.active.initialize()
                            phase_info['load_seconds'] = time.perf_counter()-load_start
                            phase_info['model'] = self.active.metadata
                            # Model load is not charged to this page's warm inference timer.
                            begin += phase_info['load_seconds']
                        adaptive_effective = self.cfg.enable_adaptive_inference
                        fallback_reason = None
                        capability = getattr(self.active, 'inference_override_capability', None)
                        if adaptive_effective and capability is not None:
                            supported, fallback_reason = capability()
                            if not supported:
                                # Use the complete legacy request, including its crop geometry.
                                request = build_inference_request(target, lines, image.shape[1], image.shape[0],
                                    replace(self.cfg, enable_adaptive_inference=False))
                                adaptive_effective = False
                        crop_bbox = request['crop_bbox']
                        a, b, c, d = crop_bbox
                        crop = image[b:d, a:c].copy()
                        import cv2
                        if not adaptive_effective:
                            scale = min(1., self.cfg.crop_edge/max(crop.shape[:2]))
                            if scale < 1:
                                crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
                        cv2.imencode('.png', crop)[1].tofile(self.folder(record)/f'{target["id"].replace(":", "_")}.png')
                        crop_start = time.perf_counter()
                        try:
                            if adaptive_effective:
                                response = self.active.predict(crop, max_new_tokens=request['max_new_tokens'],
                                                               layout_image_size=request['layout_size'])
                            else:
                                response = self.active.predict(crop)
                        except Exception as exc:
                            response = {'status': 'error', 'error': str(exc), 'tokens': None, 'blocks': []}
                        predictions[target['id']] = dict(response, crop_bbox=crop_bbox,
                            seconds=time.perf_counter()-crop_start, inference_request=request,
                            requested_inference_request=requested_request,
                            adaptive_inference_requested=self.cfg.enable_adaptive_inference,
                            inference_fallback_reason=fallback_reason if not adaptive_effective else None,
                            inference_effective=response.get('inference_override', {
                                'applied': False if not adaptive_effective else None,
                                'reason': fallback_reason or ('legacy_request' if not adaptive_effective else 'adapter_did_not_report_effective_settings'),
                                'effective_max_new_tokens': request['max_new_tokens'] if not adaptive_effective else None,
                                'effective_layout_image_size': request['layout_size'] if not adaptive_effective else None}))
                    final, evidence, reviews, decisions = finish_layout(state, lines, predictions, self.cfg,
                                                                       image.shape[1], image.shape[0], **semantic_args)
                    output = {'tier1': [serialize_proposal(p) for p in state['ensemble']],
                              'tier2': [serialize_proposal(p) for p in state['recovery']],
                              'final': [serialize_proposal(p) for p in final], 'targets': targets, 'evidence': evidence,
                              'review_evidence': reviews, 'decisions': decisions}
                elif phase == 'yolo' and not self.cfg.enable_yolo:
                    output = {'proposals': []}
                else:
                    output = self.active.predict(image)
                if phase == 'paddle' and not output.get('layout_features'):
                    output['layout_features'] = missing_layout_features()
                peak = cuda_measure(self.cfg) if phase != 'paddle' else None
                data = {'signature': self.signature, 'image_sha256': self.fingerprint(record),
                        'output': output, 'inference_seconds': time.perf_counter()-begin, 'peak_memory': peak}
                if phase == 'mineru':
                    data['dependencies'] = {name: hashlib.sha256((self.folder(record)/f'{name}.json').read_bytes()).hexdigest()
                                            for name in ('paddle', 'yolo')}
                write_json(self.folder(record)/f'{phase}.json', data)
                # Refresh the page result only once all three cache records are valid.
                if phase == 'mineru':
                    self._result(record, image, data)
        except Exception as exc:
            phase_info['error'] = str(exc)
            raise
        finally:
            self.close()
            memory.__exit__()
            phase_info['device_memory'] = memory.report()
            phase_info['wall_seconds'] = time.perf_counter()-started
            self.manifest.setdefault('phase_runs', {}).setdefault(phase, []).append(phase_info)
            self._save()
        return phase_info

    def _result(self, record, image, mineru):
        paddle, yolo = self.cached(record, 'paddle'), self.cached(record, 'yolo')
        proposals = paddle['output']['proposals']+yolo['output']['proposals']
        output = mineru['output']
        warnings = [f"{e['target']['id']}:{e['status']}" for e in output['evidence'] if e['status'] != 'ok']
        result = Result(record['document_id'], record['page_index'], image.shape[1], image.shape[0],
            proposals, paddle['output']['lines'],
            {'paddle': [p for p in proposals if p['source'] == 'paddle' and p['score'] >= self.cfg.high],
             'ensemble': output['tier1'], 'recovery': output['tier2'], 'final': output['final']},
            output['evidence'], warnings,
            {name: {'inference_seconds': data['inference_seconds'], 'peak_memory': data['peak_memory']}
             for name, data in [('paddle', paddle), ('yolo', yolo), ('mineru', mineru)]},
            'MOCK' if self.cfg.mock else 'REAL').to_dict()
        result.update(image_sha256=mineru['image_sha256'], record=record, config=asdict(self.cfg))
        result['layout_features'] = paddle['output']['layout_features']
        result['final_detections'] = output['final']
        result['review_evidence'] = output['review_evidence']
        result['decisions'] = output['decisions']
        if self.semantic is not None:
            result['semantic'] = self.semantic.status()
        write_json(self.folder(record)/'result.json', result)

    def results(self):
        results = []
        for record in self.records:
            if self.cached(record, 'mineru') is None:
                continue
            path = self.folder(record)/'result.json'
            if path.exists():
                results.append(read_json(path))
        return results

    def close(self):
        if self.active is not None:
            try:
                self.active.close()
            finally:
                self.active = None


def prepare_records(method_root, dataset_dir=None, split='test', count=20, input_path='', pages=None, dpi=150,
                    mock=False):
    """Use shared datasets; automatically call the root downloader if incomplete."""
    import cv2
    root = Path(method_root)
    if mock:
        path = root/'outputs/cascaded_layout/mock_input.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', np.full((400, 300, 3), 255, np.uint8))[1].tofile(path)
        return [{'key': 'mock_page', 'document_id': 'mock', 'page_index': 0, 'image': str(path.resolve()),
                 'image_id': None}], None
    if input_path:
        from ..document_io import iter_document_pages
        records = []
        source = Path(input_path).resolve()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        for page in iter_document_pages(source, pages, dpi=dpi):
            key = f'{source.stem}_{source_hash}_page_{page.page_number:04d}'
            path = root/'outputs/cascaded_layout/inputs'/f'{key}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.png', page.image)[1].tofile(path)
            records.append({'key': key, 'document_id': source.stem, 'page_index': page.page_number-1,
                            'image': str(path.resolve()), 'image_id': None})
        return records, None
    from .dataset import doclaynet_processed_dir, ensure_doclaynet
    canonical = doclaynet_processed_dir(root, split)
    if dataset_dir is not None and Path(dataset_dir).resolve() != canonical:
        raise ValueError(f'Use the shared dataset directory for this split: {canonical}')
    dataset_dir, annotation = ensure_doclaynet(root, split, count)
    data = read_json(annotation)
    images = data['images'][:count]
    if len(images) != count:
        raise ValueError(f'Need {count} images in the shared dataset')
    records = [{'key': f'image_{im["id"]}', 'document_id': Path(im['file_name']).stem,
                'page_index': 0, 'image': str((dataset_dir/'images'/im['file_name']).resolve()),
                'image_id': im['id']} for im in images]
    for record in records:
        if not Path(record['image']).is_file():
            raise FileNotFoundError(record['image'])
    return records, annotation
