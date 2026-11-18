"""Lazy real adapters and explicit synthetic fixtures for offline validation."""
import gc
import inspect
import logging
import multiprocessing as mp
import traceback
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
from .types import Proposal, Line, label_for, serialize_proposal
from .v3_features import layout_feature_audit, normalize_paddle_box


def _layout_metadata(layout, cfg):
    predictor = getattr(layout, 'paddlex_predictor', None)
    backend = getattr(predictor, 'runner', None)
    if backend is None:
        backend = predictor
    backend_type = type(backend) if backend is not None else None
    shape_mode_default = None
    process = getattr(predictor, 'process', None)
    if process is not None:
        try:
            parameter = inspect.signature(process).parameters.get('layout_shape_mode')
            if parameter is not None and isinstance(parameter.default, str):
                shape_mode_default = parameter.default
        except (ValueError, TypeError):
            pass
    return {'layout_model': cfg.paddle_model, 'layout_shape_mode_default': shape_mode_default,
            'layout_backend': (f'{backend_type.__module__}.{backend_type.__qualname__}'
                               if backend_type else None)}


def _paddle_child(pipe, cfg):
    """No torch imports here: CUDA DLLs belong to this child only."""
    try:
        from ..adapters.paddle_worker import _prepare_paddle_process
        _prepare_paddle_process()
        from ..adapters.common import unwrap
        import paddle
        import paddleocr
        layout = paddleocr.LayoutDetection(model_name=cfg.paddle_model, device=cfg.paddle_device,
                                           threshold=cfg.low, enable_mkldnn=False)
        detector = paddleocr.TextDetection(model_name='PP-OCRv5_mobile_det', device=cfg.paddle_device, enable_mkldnn=False)
        recognizer = paddleocr.TextRecognition(model_name='PP-OCRv5_mobile_rec', device=cfg.paddle_device, enable_mkldnn=False)
        try:
            paddlex_version = version('paddlex')
        except PackageNotFoundError:
            paddlex_version = None
        metadata = _layout_metadata(layout, cfg)
        metadata.update(paddle=paddle.__version__, paddleocr=getattr(paddleocr, '__version__', None),
                        paddlex=paddlex_version)
        pipe.send(('ready', dict(metadata, detector='PP-OCRv5_mobile_det', recognizer='PP-OCRv5_mobile_rec')))
        while True:
            image = pipe.recv()
            if image is None:
                break
            h, w = image.shape[:2]
            proposals, lines, crops = [], [], []
            layout_records = []
            for output in layout.predict(image, threshold=cfg.low):
                for raw in unwrap(output)['boxes']:
                    proposal, record = normalize_paddle_box(
                        raw, width=w, height=h, variant=cfg.variant, proposal_id=f'paddle:{len(proposals)}')
                    record['raw_index'] = len(layout_records)
                    layout_records.append(record)
                    if proposal is not None:
                        proposals.append(proposal)
            for output in detector.predict(image):
                data = unwrap(output)
                if len(data['dt_polys']) != len(data['dt_scores']):
                    raise ValueError('text detector response length mismatch')
                for points, score in zip(data['dt_polys'], data['dt_scores']):
                    points = [[max(0, min(w, int(x))), max(0, min(h, int(y)))] for x, y in points]
                    from shapely.geometry import Polygon
                    if not Polygon(points).is_valid or Polygon(points).area <= 0:
                        continue
                    x1, y1 = np.min(points, axis=0)
                    x2, y2 = np.max(points, axis=0)
                    lines.append(Line(f'line:{len(lines)}', points, '', float(score)))
                    crops.append(image[y1:y2, x1:x2].copy())
            if crops:
                recognized = [unwrap(r) for r in recognizer.predict(crops)]
                texts = [r['rec_text'] for r in recognized]
                if len(texts) != len(lines):
                    raise ValueError('recognizer response length mismatch')
                for line, text, raw in zip(lines, texts, recognized):
                    line.text = text
                    score = raw.get('rec_score')
                    line.rec_score = float(score) if score is not None else None
            pipe.send(('ok', {'proposals': [serialize_proposal(p) for p in proposals],
                              'lines': [asdict(l) for l in lines],
                              'layout_features': layout_feature_audit(layout_records, metadata=metadata)}))
    except BaseException:
        pipe.send(('error', traceback.format_exc()))
    finally:
        pipe.close()


class PaddleAdapter:
    def __init__(self, cfg):
        self.cfg, self.process, self.pipe = cfg, None, None
        self.metadata = {}

    def initialize(self):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_paddle_child, args=(child, self.cfg), daemon=True)
        self.process.start()
        child.close()
        self.metadata = self._receive('ready')

    def _receive(self, expected):
        if not self.pipe.poll(600):
            raise TimeoutError('Paddle worker did not respond within 600 seconds')
        status, payload = self.pipe.recv()
        if status != expected:
            raise RuntimeError(f'Paddle worker failed: {payload}')
        return payload

    def predict(self, image):
        self.pipe.send(image)
        return self._receive('ok')

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.pipe.send(None)
                except (EOFError, OSError):
                    pass
                self.process.join(3)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(3)
            self.pipe.close()
            self.process.close()
            self.process = self.pipe = None


def release_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class YoloAdapter:
    def __init__(self, cfg):
        self.cfg, self.model = cfg, None
        self.metadata = {}

    def initialize(self):
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10
        path = self.cfg.yolo_model
        if not Path(path).is_file():
            path = hf_hub_download(path, self.cfg.yolo_file)
        self.model = YOLOv10(path)
        self.metadata = {'weights': str(path), 'parameters': sum(p.numel() for p in self.model.model.parameters())}

    def predict(self, image):
        results = self.model.predict(image, imgsz=1024, conf=self.cfg.low,
                                     device='cuda:0' if self.cfg.device == 'cuda' else 'cpu', verbose=False)
        proposals = []
        for result in results:
            for b, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                     result.boxes.cls.cpu().tolist()):
                raw = result.names[int(cls)]
                proposals.append(serialize_proposal(Proposal(f'yolo:{len(proposals)}', b, label_for(raw, 'yolo'),
                                                             'yolo', raw, score)))
        return {'proposals': proposals}

    def close(self):
        self.model = None
        release_cuda()


class MinerUAdapter:
    def __init__(self, cfg):
        self.cfg, self.client, self.model, self.processor = cfg, None, None, None
        self.metadata = {}

    def initialize(self):
        import torch
        from mineru_vl_utils import MinerUClient
        from mineru_vl_utils.mineru_client import MinerUSamplingParams
        from mineru_vl_utils.transformers_loading import load_transformers_model, load_transformers_processor
        if self.cfg.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Select device="cpu" explicitly.')
        self.model = load_transformers_model(self.cfg.mineru_model, device_map=self.cfg.device)
        self.model.set_attn_implementation('sdpa')
        self.model.to(dtype=torch.float16 if self.cfg.device == 'cuda' else torch.float32)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = load_transformers_processor(self.cfg.mineru_model)
        self.client = MinerUClient(backend='transformers', model=self.model, processor=self.processor,
            sampling_params={'[layout]': MinerUSamplingParams(max_new_tokens=self.cfg.max_tokens)},
            layout_image_size=(self.cfg.crop_edge, self.cfg.crop_edge), batch_size=1, use_tqdm=False)
        self.metadata = {'model': self.cfg.mineru_model,
                         'parameters': sum(p.numel() for p in self.model.parameters()),
                         'revision': getattr(self.model.config, '_commit_hash', None), 'backend': 'transformers'}

    def inference_override_capability(self):
        """Capability verified against mineru-vl-utils 2.0.8's sequential API."""
        sampling = getattr(self.client, 'sampling_params', None)
        helper = getattr(self.client, 'helper', None)
        if not isinstance(sampling, dict) or helper is None or not hasattr(helper, 'layout_image_size'):
            return False, 'unsupported_client_settings; legacy inference fallback'
        effective = sampling.get('[layout]') or sampling.get('[default]')
        if effective is not None and not hasattr(effective, 'max_new_tokens'):
            return False, 'unsupported_sampling_params; legacy inference fallback'
        return True, 'sequential_official_helper_settings'

    def predict(self, crop, *, max_new_tokens=None, layout_image_size=None):
        """Use the official layout API; instrument generate to detect truncation/token count."""
        import torch
        import cv2
        from PIL import Image
        requested = max_new_tokens is not None or layout_image_size is not None
        if max_new_tokens is not None and (isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int) or max_new_tokens < 1):
            raise ValueError('max_new_tokens must be a positive integer')
        if layout_image_size is not None and (not isinstance(layout_image_size, (list, tuple))
                or len(layout_image_size) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in layout_image_size)):
            raise ValueError('layout_image_size must contain two positive integers')
        effective_cap = self.cfg.max_tokens
        override = None
        sampling = helper = None
        if requested:
            supported, reason = self.inference_override_capability()
            override = {'applied': False, 'reason': reason,
                        'requested_max_new_tokens': max_new_tokens,
                        'requested_layout_image_size': layout_image_size,
                        'effective_max_new_tokens': effective_cap,
                        'effective_layout_image_size': (self.cfg.crop_edge, self.cfg.crop_edge)}
            if supported:
                sampling, helper = self.client.sampling_params, self.client.helper
                had_layout = '[layout]' in sampling
                original_sampling = sampling.get('[layout]')
                original_size = helper.layout_image_size
                active_sampling = sampling.get('[layout]') or sampling.get('[default]')
                if active_sampling is None:
                    from mineru_vl_utils.mineru_client import MinerUSamplingParams
                    active_sampling = MinerUSamplingParams(max_new_tokens=self.cfg.max_tokens)
                # Do not mutate even a single field on the original sampling object.
                active_sampling = deepcopy(active_sampling)
                if max_new_tokens is not None:
                    active_sampling.max_new_tokens = max_new_tokens
                effective_cap = active_sampling.max_new_tokens
                if effective_cap is None:
                    effective_cap = self.cfg.max_tokens
                override.update(applied=True, effective_max_new_tokens=effective_cap,
                                effective_layout_image_size=tuple(layout_image_size) if layout_image_size is not None else original_size)
            else:
                logging.getLogger(__name__).warning(reason)
        original = self.model.generate
        generated = []

        def measured_generate(*args, **kwargs):
            output = original(*args, **kwargs)
            sequences = output.sequences if hasattr(output, 'sequences') else output
            n_input = kwargs['input_ids'].shape[-1]
            generated.append(sequences[0, n_input:].detach().cpu())
            return output

        error = None
        blocks = []
        try:
            if sampling is not None:
                sampling['[layout]'] = active_sampling
                if layout_image_size is not None:
                    helper.layout_image_size = tuple(layout_image_size)
            self.model.generate = measured_generate
            with torch.inference_mode():
                blocks = self.client.layout_detect(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        except Exception as exc:
            error = str(exc)
        finally:
            self.model.generate = original
            if sampling is not None:
                helper.layout_image_size = original_size
                if had_layout:
                    sampling['[layout]'] = original_sampling
                else:
                    sampling.pop('[layout]', None)
        metadata = {'inference_override': override} if requested else {}
        if len(generated) != 1:
            return {'status': 'error', 'error': error or 'Cannot verify MinerU generation completion/token count',
                    'tokens': None, 'blocks': [], **metadata}
        tokens = generated[0].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        incomplete = len(tokens) >= effective_cap and (not tokens or tokens[-1] not in eos)
        raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
        # Parser can silently drop malformed boxes. Reject the whole output when it does.
        malformed = raw.count('<|box_start|>') != len(blocks) or not blocks
        return {'status': 'error' if requested and error else 'incomplete' if incomplete else 'invalid_output' if malformed else 'ok',
                'tokens': len(tokens), 'raw': raw, 'error': error,
                'blocks': [{'bbox': list(b.bbox), 'label': label_for(b.type, 'mineru'), 'raw_label': b.type}
                           for b in blocks], **metadata}

    def close(self):
        self.client = self.model = self.processor = None
        release_cuda()


class MockAdapter:
    """Synthetic fixtures, never a fallback for a real model error."""
    def __init__(self, cfg, role):
        self.role, self.metadata = role, {'mode': 'MOCK', 'role': role}

    def initialize(self):
        pass

    def close(self):
        pass

    def predict(self, image, *, max_new_tokens=None, layout_image_size=None):
        if self.role == 'mineru':
            return {'status': 'ok', 'tokens': 12, 'raw': 'MOCK',
                    'blocks': [{'bbox': [0, 0, 1, 1], 'label': 'text', 'raw_label': 'text'}],
                    **({'inference_override': {'applied': True, 'reason': 'synthetic_mock',
                        'effective_max_new_tokens': max_new_tokens,
                        'effective_layout_image_size': layout_image_size}}
                       if max_new_tokens is not None or layout_image_size is not None else {})}
        if self.role == 'yolo':
            return {'proposals': []}
        h, w = image.shape[:2]
        p = Proposal('paddle:0', [w*.1, h*.1, w*.9, h*.35], 'doc_title', 'paddle', 'doc_title', .9)
        lines = [Line('line:0', [[w*.12,h*.12],[w*.8,h*.12],[w*.8,h*.2],[w*.12,h*.2]], 'MOCK title', .9, .9),
                 Line('line:1', [[w*.1,h*.6],[w*.8,h*.6],[w*.8,h*.65],[w*.1,h*.65]], 'MOCK orphan', .9, .9)]
        return {'proposals': [serialize_proposal(p)], 'lines': [asdict(l) for l in lines]}


def make_adapter(cfg, role):
    if cfg.mock:
        return MockAdapter(cfg, role)
    return {'paddle': PaddleAdapter, 'yolo': YoloAdapter, 'mineru': MinerUAdapter}[role](cfg)
