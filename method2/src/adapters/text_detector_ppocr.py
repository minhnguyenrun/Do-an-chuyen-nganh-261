from ..schemas.region import TextLineEvidence
from .common import unwrap,freeze_engine


class TextDetectorAdapter:
    def __init__(self,engine=None,mock_mode=False,model_name='PP-OCRv5_server_det',**kwargs):
        self.engine,self.mock_mode,self.model_name,self.options = engine,mock_mode,model_name,kwargs
        if engine is not None:
            freeze_engine(engine)

    def detect_lines(self,image):
        h,w = image.shape[:2]
        if self.mock_mode:
            return [TextLineEvidence(line_id='L0',polygon=[(0,0),(w,0),(w,h),(0,h)])]
        if self.engine is None:
            from paddleocr import TextDetection
            self.engine = TextDetection(model_name=self.model_name,**self.options)
            freeze_engine(self.engine)
        lines = []
        for result in self.engine.predict(image):
            data = unwrap(result)
            polygons,scores = data['dt_polys'],data['dt_scores']
            if len(polygons) != len(scores):
                raise ValueError('detector polygons/scores length mismatch')
            for polygon,score in zip(polygons,scores):
                points = tuple((max(0,min(w,int(round(x)))),max(0,min(h,int(round(y))))) for x,y in polygon)
                lines.append(TextLineEvidence(line_id=f'L{len(lines)}',polygon=points,score=float(score)))
        return lines
