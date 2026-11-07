import numpy as np
import pytest

from src.adapters.paddle_worker import PaddleWorker, PaddleProxy
from src.pipeline import load_config
from src.runtime import close_adapters


def test_shared_worker_roundtrip_errors_and_cleanup():
    worker = PaddleWorker(load_config()['models'], mock_mode=True)
    proxies = [PaddleProxy(worker, role) for role in ('layout', 'text_detector', 'light_ocr')]
    image = np.full((32, 48, 3), 255, dtype=np.uint8)
    try:
        assert proxies[0].initialize() == 'mock'
        process = worker._process
        assert proxies[0].detect(image)[0].bbox == (0, 0, 48, 32)
        assert proxies[1].detect_lines(image)[0].line_id == 'L0'
        assert proxies[2].recognize([image, image]) == ['Mock text', 'Mock text']
        assert worker._process is process
        with pytest.raises(RuntimeError, match='AttributeError'):
            worker.call(('layout', 'invalid_method'), image)
        assert proxies[2].recognize([]) == []
    finally:
        close_adapters(proxies)
    assert worker._process is None
    close_adapters(proxies)
