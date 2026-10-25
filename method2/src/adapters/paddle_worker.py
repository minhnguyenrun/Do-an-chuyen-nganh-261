"""Run Paddle in a local child process to isolate its CUDA DLLs from PyTorch."""
import atexit
import multiprocessing
import threading
import traceback


class PaddleEnvironmentError(RuntimeError):
    """An environment failure that should stop the document batch."""


class _NoTorchImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('Torch is disabled in the Paddle-only worker.', name=fullname)
        return None


def _prepare_paddle_process():
    """Keep optional ModelScope/Transformers Torch discovery out of this worker."""
    import os
    import sys
    import importlib.util
    if 'torch' in sys.modules and sys.modules['torch'] is not None:
        raise RuntimeError('Paddle child already imported Torch; restart the kernel.')
    # ModelScope probes Torch even for downloads of Paddle-only models. Hide it
    # from optional discovery and reject actual imports in this child only.
    # Keep sys.modules free of placeholders: SciPy inspects its values as modules.
    original_find_spec = importlib.util.find_spec
    def find_spec(name, package=None):
        if name == 'torch' or name.startswith('torch.'):
            return None
        return original_find_spec(name, package)
    importlib.util.find_spec = find_spec
    sys.meta_path.insert(0, _NoTorchImports())
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    import paddle
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError('Paddle GPU requires paddlepaddle-gpu; CPU build is installed.')
    # Validate the full import chain now, rather than failing on every page.
    from paddleocr import LayoutDetection, TextDetection, TextRecognition
    return paddle.__version__


def _serve(connection, models, mock_mode):
    from .layout_ppdocv3 import LayoutDetectorAdapter
    from .text_detector_ppocr import TextDetectorAdapter
    from .light_ocr_ppocr import LightOcrAdapter

    options = dict(mock_mode=mock_mode, device='gpu:0', enable_mkldnn=False)
    adapters = {
        'layout': LayoutDetectorAdapter(model_name=models['layout'], **options),
        'text_detector': TextDetectorAdapter(model_name=models['text_detector'], **options),
        'light_ocr': LightOcrAdapter(model_name=models['light_ocr'], **options),
    }
    initialized = False
    initialization_error = None
    version = 'mock'
    try:
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            try:
                if initialization_error is not None:
                    raise RuntimeError('Paddle initialization previously failed; restart kernel.\n' + initialization_error)
                if not initialized:
                    try:
                        if not mock_mode:
                            version = _prepare_paddle_process()
                        initialized = True
                    except Exception:
                        initialization_error = traceback.format_exc()
                        raise
                if operation == 'initialize':
                    result = version
                else:
                    role, method = operation
                    result = getattr(adapters[role], method)(payload)
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class PaddleWorker:
    def __init__(self, models, *, mock_mode=False):
        self.models = dict(models)
        self.mock_mode = mock_mode
        self._process = self._connection = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def call(self, operation, payload=None):
        with self._lock:
            if self._process is None:
                context = multiprocessing.get_context('spawn')
                self._connection, child = context.Pipe()
                self._process = context.Process(target=_serve, args=(child, self.models, self.mock_mode), daemon=True)
                self._process.start()
                child.close()
            if not self._process.is_alive():
                raise RuntimeError('Paddle worker stopped; release models and initialize again.')
            try:
                self._connection.send((operation, payload))
                while not self._connection.poll(.25):
                    if not self._process.is_alive():
                        raise RuntimeError(f'Paddle worker exited (code {self._process.exitcode}).')
                success, result = self._connection.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise RuntimeError('Paddle worker connection failed; initialize again.') from exc
            if not success:
                if any(marker in result for marker in ('WinError', 'partially initialized module',
                                                       'Paddle initialization previously failed')):
                    raise PaddleEnvironmentError('Paddle environment error:\n' + result)
                raise RuntimeError('Paddle worker error:\n' + result)
            return result

    def initialize(self):
        return self.call('initialize')

    def close(self):
        with self._lock:
            if self._process is not None:
                if self._process.is_alive():
                    try:
                        self._connection.send(('close', None))
                    except (EOFError, BrokenPipeError, OSError):
                        pass
                    self._process.join(timeout=3)
                    if self._process.is_alive():
                        self._process.terminate()
                        self._process.join(timeout=3)
                self._connection.close()
                self._process.close()
                self._process = self._connection = None


class PaddleProxy:
    def __init__(self, worker, role):
        self.worker, self.role = worker, role
        self.options = {'device': 'gpu:0', 'enable_mkldnn': False}

    def initialize(self):
        return self.worker.initialize()

    def detect(self, image):
        return self.worker.call((self.role, 'detect'), image)

    def detect_lines(self, image):
        return self.worker.call((self.role, 'detect_lines'), image)

    def recognize(self, crops):
        return self.worker.call((self.role, 'recognize'), crops)

    def close(self):
        self.worker.close()
