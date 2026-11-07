import asyncio
import numpy as np
import pytest
from src.adapters.layout_ppdocv3 import LayoutDetectorAdapter
from src.adapters.text_detector_ppocr import TextDetectorAdapter
from src.adapters.light_ocr_ppocr import LightOcrAdapter
from src.adapters.glm_ocr_local import LocalGlmOcrAdapter


class Engine:
    def __init__(self,result):
        self.result = result
    def predict(self,image):
        yield {'res':self.result}


def test_paddle_result_normalization_without_model_downloads():
    image = np.zeros((100,100,3),np.uint8)
    layout = LayoutDetectorAdapter(engine=Engine({'boxes':[{'coordinate':[-2.5,1,90.2,101],
        'label':'display_formula','score':.8,'order':4}]}))
    result = layout.detect(image)[0]
    assert result.bbox == (0,1,91,100) and result.label == 'formula' and result.order == 4
    detector = TextDetectorAdapter(engine=Engine({'dt_polys':np.array([[[0,0],[20,0],[20,10],[0,10]]]),'dt_scores':[.9]}))
    assert detector.detect_lines(image)[0].polygon[2] == (20,10)
    assert LightOcrAdapter(engine=Engine({'rec_text':'hello'})).recognize([image]) == ['hello']


def test_local_mock_and_prompt_registry():
    async def run():
        client = LocalGlmOcrAdapter(mock_mode=True)
        result = await client.recognize_batch([('h',np.zeros((10,10,3),np.uint8),'Text Recognition:')])
        assert result[0].status == 'ok' and result[0].parser == 'glm_mock'
        with pytest.raises(ValueError):
            await client.recognize_batch([('h',np.zeros((10,10,3),np.uint8),'invented prompt')])
    asyncio.run(run())


def test_layout_null_missing_and_zero_order_preserve_stable_fallbacks():
    boxes = [{'coordinate': [1, 1 + i * 15, 40, 10 + i * 15], 'label': 'text', 'score': .9}
             for i in range(4)]
    boxes[0]['order'] = None
    boxes[1]['order'] = 0
    boxes[3]['order'] = 7
    layout = LayoutDetectorAdapter(engine=Engine({'boxes': boxes}))
    regions = layout.detect(np.zeros((100, 100, 3), np.uint8))
    assert [r.order for r in regions] == [0, 0, 2, 7]
    assert [r.id for r in regions] == ['B0', 'B1', 'B2', 'B3']
