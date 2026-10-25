import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.adapters.glm_ocr_local import LocalGlmOcrAdapter


class Processor:
    def __init__(self):
        self.images, self.decoded = [], []
        # Minimal tokenizer for testing the constrained prefix without a model download.
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())
            def __len__(self):
                return 256
        self.tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        self.images.append(messages[0]['content'][0]['image'])
        assert kwargs['tokenize'] and kwargs['add_generation_prompt']
        return {'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.ones((1, 3), dtype=torch.int64),
                'pixel_values': torch.ones((1, 3, 8, 8))}

    def decode(self, tokens, **kwargs):
        self.decoded.append(tokens.tolist())
        return 'recognized'


class Model(torch.nn.Module):
    def __init__(self, tokens=(7, 99), error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.generation_config = SimpleNamespace(eos_token_id=[99])
        self.tokens, self.error = tokens, error
        self.moves = 0

    def to(self, *args, **kwargs):
        self.moves += 1
        return super().to(*args, **kwargs)

    def generate(self, **kwargs):
        self.last_generation = kwargs
        assert torch.is_inference_mode_enabled()
        assert not self.training and not self.weight.requires_grad
        assert kwargs['do_sample'] is False
        assert kwargs['input_ids'].dtype == torch.int64
        assert kwargs['pixel_values'].dtype == self.weight.dtype
        if self.error:
            raise RuntimeError(self.error)
        return torch.cat((kwargs['input_ids'], torch.tensor([self.tokens], device=self.weight.device)), dim=1)


def test_rgb_resize_token_trimming_freeze_and_model_reuse():
    processor, model = Processor(), Model()
    adapter = LocalGlmOcrAdapter(device='cpu', max_crop_edge=64, processor=processor, model_instance=model)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    async def run():
        return await adapter.recognize_batch([('h1', image, 'Text Recognition:'), ('h2', image, 'Text Recognition:')])
    result = asyncio.run(run())
    assert all(r.status == 'ok' and r.parser == 'glm_local' for r in result)
    assert processor.images[0].size == (64, 32)
    assert processor.images[0].getpixel((0, 0)) == (0, 0, 255)
    assert processor.decoded == [[7, 99], [7, 99]]
    assert model.moves == 1
    assert image.shape == (100, 200, 3)


def test_truncation_and_out_of_memory_are_visible_evidence():
    async def run(adapter):
        return (await adapter.recognize_batch([('h', np.zeros((10, 10, 3), np.uint8), 'Text Recognition:')]))[0]
    truncated = LocalGlmOcrAdapter(device='cpu', max_new_tokens=1, processor=Processor(), model_instance=Model(tokens=(7,)))
    assert asyncio.run(run(truncated)).status == 'incomplete'
    failing = LocalGlmOcrAdapter(device='cpu', processor=Processor(), model_instance=Model(error='CUDA out of memory'))
    result = asyncio.run(run(failing))
    assert result.status == 'error' and 'GLM_DEVICE="cpu"' in result.error
    assert 'mock' not in result.parser


def test_unavailable_cuda_and_invalid_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        LocalGlmOcrAdapter(device='cuda', processor=Processor(), model_instance=Model()).initialize()
    with pytest.raises(ValueError):
        LocalGlmOcrAdapter(device='unknown')


def test_structured_classification_task_uses_same_model_with_bounded_token_limit():
    from src.schemas.region import LayoutRegion
    from src.semantic.residual_prompt import build_classification_prompt, parse_classification

    class JsonProcessor(Processor):
        def decode(self, tokens, **kwargs):
            return '{"label":"header","text":"Annual report"}'
    processor, model = JsonProcessor(), Model()
    adapter = LocalGlmOcrAdapter(device='cpu', max_new_tokens=1024, processor=processor, model_instance=model)
    region = LayoutRegion(id='r', bbox=(0, 0, 100, 20), label='residual')
    prompt = build_classification_prompt(region, (100, 200), 'Annual report')
    async def run():
        return (await adapter.recognize_batch([('classify:r', np.zeros((200, 100, 3), np.uint8), prompt)]))[0]
    result = asyncio.run(run())
    assert result.status == 'ok' and parse_classification(result.content).label == 'header'
    assert model.last_generation['max_new_tokens'] == 768 and model.moves == 1
    assert callable(model.last_generation['prefix_allowed_tokens_fn'])
    assert model.last_generation['stopping_criteria'][0](torch.tensor([[1, 2, 3, 7]]), None)


def test_classification_prefix_allows_all_labels_but_never_schema_keys():
    from src.semantic.residual_prompt import classification_prefix_constraint, LABELS
    tokenizer = Processor().tokenizer
    allowed = classification_prefix_constraint(tokenizer, 3)
    assert allowed(0, torch.tensor([1, 2, 3])) == [ord('{')]
    for label in LABELS:
        prefix = '{"label":"' + label + '","text":"'
        tokens = [1, 2, 3]
        for token in prefix.encode():
            assert token in allowed(0, torch.tensor(tokens))
            tokens.append(token)
        assert len(allowed(0, torch.tensor(tokens))) == len(tokenizer)
    assert ord('p') not in allowed(0, torch.tensor([1, 2, 3, ord('{'), ord('"')]))
