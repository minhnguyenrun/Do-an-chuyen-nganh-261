import math


def union_bboxes(bboxes):
    boxes = list(bboxes)
    if not boxes:
        raise ValueError('cannot union empty boxes')
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def polygon_bbox(polygon):
    return (math.floor(min(p[0] for p in polygon)), math.floor(min(p[1] for p in polygon)),
            math.ceil(max(p[0] for p in polygon)), math.ceil(max(p[1] for p in polygon)))


def clamp_bbox(bbox, max_w, max_h):
    x1, y1, x2, y2 = bbox
    result = max(0, math.floor(x1)), max(0, math.floor(y1)), min(max_w, math.ceil(x2)), min(max_h, math.ceil(y2))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError('bbox outside image or empty')
    return result


def normalize_bbox(bbox, source_size, target_size):
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    return clamp_bbox((bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy), *target_size)


def expand_bbox(bbox, margin_ratio, max_w, max_h):
    if not 0 <= margin_ratio <= 1:
        raise ValueError('margin ratio must be in [0,1]')
    x1, y1, x2, y2 = bbox
    dx, dy = (x2-x1)*margin_ratio, (y2-y1)*margin_ratio
    return clamp_bbox((x1-dx, y1-dy, x2+dx, y2+dy), max_w, max_h)


def intersection_area(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])
