from ..schemas.region import LayoutRegion
from ..spatial.coordinates import clamp_bbox
from .common import unwrap,freeze_engine
from ..label_mapping import normalize_layout_label


class LayoutDetectorAdapter:
    def __init__(self,engine=None,mock_mode=False,model_name='PP-DocLayoutV3',**kwargs):
        self.engine,self.mock_mode,self.model_name,self.options = engine,mock_mode,model_name,kwargs
        if engine is not None:
            freeze_engine(engine)

    def detect(self,image):
        h,w = image.shape[:2]
        if self.mock_mode:
            return [LayoutRegion(id='B0',bbox=(0,0,w,h),label='text',score=.95)]
        if self.engine is None:
            from paddleocr import LayoutDetection
            self.engine = LayoutDetection(model_name=self.model_name,**self.options)
            freeze_engine(self.engine)
        regions = []
        for result in self.engine.predict(image):
            for b in unwrap(result)['boxes']:
                bbox = clamp_bbox(b['coordinate'],w,h)
                label = normalize_layout_label(b['label'])
                raw_polygon = b.get('polygon_points',b.get('polygon',()))
                polygon = tuple((max(0,min(w,int(round(x)))),max(0,min(h,int(round(y))))) for x,y in raw_polygon) if raw_polygon is not None else ()
                # Paddle may include the key with None, so dict.get's default is not enough.
                # Preserve an explicit zero and let the geometry DAG determine final order.
                raw_order = b.get('order')
                order = len(regions) if raw_order is None else raw_order
                regions.append(LayoutRegion(id=f'B{len(regions)}',bbox=bbox,polygon=polygon,
                    label=label,order=order,score=float(b['score'])))
        return regions
