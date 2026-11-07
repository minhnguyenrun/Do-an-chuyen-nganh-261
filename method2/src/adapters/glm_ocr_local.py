"""GLM-OCR in the notebook process: no HTTP transport or serving runtime."""
import asyncio
import threading

import cv2
import numpy as np
from PIL import Image

from ..schemas.evidence import ParseEvidence
from ..semantic.prompt_router import PROMPTS
from ..semantic.residual_prompt import (is_classification_prompt, classification_instruction,
    classification_prefix_constraint, parse_classification)


class LocalGlmOcrAdapter:
    def __init__(self, model='zai-org/GLM-OCR', device='auto', max_new_tokens=1024,
                 max_crop_edge=1280, mock_mode=False, *, processor=None, model_instance=None):
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        if max_new_tokens < 1 or max_crop_edge < 28:
            raise ValueError('invalid generation or crop limit')
        if (processor is None) != (model_instance is None):
            raise ValueError('inject both processor and model_instance')
        self.model_source = str(model)
        self.requested_device = device
        self.device = None
        self.dtype = None
        self.max_new_tokens = max_new_tokens
        self.max_crop_edge = max_crop_edge
        self.mock_mode = mock_mode
        self.processor = processor
        self.model = model_instance
        self._initialized = False
        self._lock = threading.RLock()

    def initialize(self):
        """Load once; invoke explicitly before processing pages to surface load errors."""
        with self._lock:
            if self._initialized:
                return self
            if self.mock_mode:
                self.device = 'mock'
                self._initialized = True
                return self
            import torch

            self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.requested_device == 'auto' else self.requested_device
            if self.device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable. Set GLM_DEVICE="cpu" and initialize again.')
            self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
            if self.model is None:
                from transformers import AutoProcessor, GlmOcrForConditionalGeneration

                processor = AutoProcessor.from_pretrained(self.model_source)
                model = GlmOcrForConditionalGeneration.from_pretrained(
                    self.model_source, dtype=self.dtype, attn_implementation='sdpa')
            else:
                processor, model = self.processor, self.model
            model = model.to(device=self.device, dtype=self.dtype)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.processor, self.model = processor, model
            self._initialized = True
            return self

    def _prepare_image(self, crop):
        if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or not crop.size:
            raise ValueError('expected a nonempty uint8 BGR crop')
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image.thumbnail((self.max_crop_edge, self.max_crop_edge), Image.Resampling.LANCZOS)
        return image

    def _recognize_one(self, request):
        hypothesis_id, crop, prompt = request
        try:
            with self._lock:
                self.initialize()
                if self.mock_mode:
                    samples = {'Text Recognition:': 'Mock text',
                               'Table Recognition:': '<table><tr><td>Mock</td></tr></table>',
                               'Formula Recognition:': r'x^2 + y^2 = z^2'}
                    content = '{"label":"unknown","text":""}' if is_classification_prompt(prompt) else samples[prompt]
                    return ParseEvidence(hypothesis_id=hypothesis_id, parser='glm_mock', prompt=prompt,
                                         content=content, status='ok')
                import torch

                image = self._prepare_image(crop)
                classification_task = is_classification_prompt(prompt)
                model_prompt = classification_instruction(prompt) if classification_task else prompt
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': model_prompt}]}]
                inputs = self.processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt')
                inputs = {key: value.to(device=self.device, dtype=self.dtype) if torch.is_floating_point(value)
                          else value.to(self.device) for key, value in inputs.items()}
                input_length = inputs['input_ids'].shape[-1]
                token_limit = min(self.max_new_tokens, 768) if classification_task else self.max_new_tokens
                generation_options = {}
                if classification_task:
                    from transformers import StoppingCriteriaList
                    generation_options['prefix_allowed_tokens_fn'] = classification_prefix_constraint(
                        self.processor.tokenizer, input_length)
                    def complete_json(input_ids, scores, **kwargs):
                        text = self.processor.decode(input_ids[0, input_length:], skip_special_tokens=True)
                        try:
                            parse_classification(text)
                            return True
                        except ValueError:
                            return False
                    generation_options['stopping_criteria'] = StoppingCriteriaList([complete_json])
                with torch.inference_mode():
                    output = self.model.generate(**inputs, max_new_tokens=token_limit,
                                                 do_sample=False, use_cache=True, return_dict_in_generate=False,
                                                 **generation_options)
                generated = output[0, input_length:]
                content = self.processor.decode(generated, skip_special_tokens=True).strip()
                eos = self.model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                truncated = len(generated) >= token_limit and int(generated[-1]) not in eos_ids
                if classification_task and truncated:
                    try:
                        parse_classification(content)
                        truncated = False  # Complete valid JSON does not require a following EOS.
                    except ValueError:
                        pass
                limit_error = (f'Classification reached {token_limit} tokens; keep the region for review.'
                               if classification_task else
                               'Generation reached max_new_tokens; increase the limit if memory permits.')
                return ParseEvidence(hypothesis_id=hypothesis_id, parser='glm_local', prompt=prompt,
                    content=content, status='incomplete' if truncated or not content else 'ok',
                    error=limit_error if truncated else '')
        except Exception as exc:
            message = str(exc)
            if 'out of memory' in message.lower():
                message += ' GLM ran out of memory. Release models, set GLM_DEVICE="cpu" or reduce GLM_MAX_CROP_EDGE / GLM_MAX_NEW_TOKENS, then rerun.'
            return ParseEvidence(hypothesis_id=hypothesis_id, parser='glm_local', prompt=prompt,
                                 content='', status='error', error=message)

    async def recognize_batch(self, requests):
        requests = list(requests)
        for _, _, prompt in requests:
            if prompt not in PROMPTS and not is_classification_prompt(prompt):
                raise ValueError(f'unsupported GLM prompt: {prompt}')
        return [await asyncio.to_thread(self._recognize_one, request) for request in requests]

    def close(self):
        """Release references before switching device or model in a notebook."""
        with self._lock:
            self.model = self.processor = None
            self._initialized = False
            if not self.mock_mode:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
