from .common import unwrap,freeze_engine


class LightOcrAdapter:
    def __init__(self,engine=None,mock_mode=False,model_name='PP-OCRv5_server_rec',**kwargs):
        self.engine,self.mock_mode,self.model_name,self.options = engine,mock_mode,model_name,kwargs
        if engine is not None:
            freeze_engine(engine)

    def recognize(self,crops):
        if not crops:
            return []
        if self.mock_mode:
            return ['Mock text' for _ in crops]
        if self.engine is None:
            from paddleocr import TextRecognition
            self.engine = TextRecognition(model_name=self.model_name,**self.options)
            freeze_engine(self.engine)
        result = [unwrap(r)['rec_text'] for r in self.engine.predict(crops)]
        if len(result) != len(crops):
            raise ValueError('recognizer results do not match crops')
        return result
