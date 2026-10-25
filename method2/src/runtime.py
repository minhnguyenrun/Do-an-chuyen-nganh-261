"""Shared notebook/CLI adapter construction. Paddle uses GPU 0."""
from .adapters.layout_ppdocv3 import LayoutDetectorAdapter
from .adapters.text_detector_ppocr import TextDetectorAdapter
from .adapters.light_ocr_ppocr import LightOcrAdapter
from .adapters.glm_ocr_local import LocalGlmOcrAdapter
from .adapters.paddle_worker import PaddleWorker, PaddleProxy
import sys


def create_adapters(config, *, mock_mode=False):
    models, inference = config['models'], config['inference']
    paddle_adapters = (
        LayoutDetectorAdapter(mock_mode=mock_mode, model_name=models['layout'], device='gpu:0', enable_mkldnn=False),
        TextDetectorAdapter(mock_mode=mock_mode, model_name=models['text_detector'], device='gpu:0', enable_mkldnn=False),
        LightOcrAdapter(mock_mode=mock_mode, model_name=models['light_ocr'], device='gpu:0', enable_mkldnn=False),
    )
    if sys.platform == 'win32' and not mock_mode:
        worker = PaddleWorker(models)
        paddle_adapters = tuple(PaddleProxy(worker, role) for role in ('layout', 'text_detector', 'light_ocr'))
    return (*paddle_adapters,
        LocalGlmOcrAdapter(model=models['semantic_parser'], device=inference['device'],
            max_new_tokens=inference['max_new_tokens'], max_crop_edge=inference['max_crop_edge'], mock_mode=mock_mode),
    )


def close_adapters(adapters):
    """Release both the shared Paddle child and GLM weights."""
    for adapter in adapters:
        close = getattr(adapter, 'close', None)
        if close is not None:
            close()
