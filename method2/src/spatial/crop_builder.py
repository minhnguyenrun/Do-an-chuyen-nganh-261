import cv2
import numpy as np
from shapely.geometry import Polygon, box
from .coordinates import expand_bbox, union_bboxes, clamp_bbox


def build_crop(image, bbox, polygon=None, mode='masked_whiteout', *, line_polygons=(), mask_certain=True):
    if mode not in ('raw', 'masked_whiteout'):
        raise ValueError('unknown crop mode')
    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape[1], image.shape[0])
    crop = image[y1:y2, x1:x2].copy()
    if mode == 'raw' or not polygon or not mask_certain:
        return crop
    shape = Polygon(polygon)
    if not shape.is_valid or shape.area <= 0:
        return crop
    # Never let a stale macro polygon erase constituent text after a merge/expansion.
    shapes = [shape]
    for line in line_polygons:
        line_shape = Polygon(line)
        if not line_shape.is_valid or not box(x1,y1,x2,y2).covers(line_shape):
            return crop
        shapes.append(line_shape)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for shape in shapes:
        points = np.rint(np.asarray(shape.exterior.coords)-[x1,y1]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    crop[mask == 0] = 255
    return crop
