"""Geometry policy for the measured Paddle V3 proposal features."""
from math import inf, isfinite
from numbers import Real

from shapely.errors import ShapelyError
from shapely.geometry import Polygon, box


def v3_polygons_enabled(cfg):
    return cfg.variant == 'v3' and cfg.enable_v3_polygons


def measured_shape(p, *, use_v3_polygon=False):
    points = p.polygon
    if (not use_v3_polygon or p.source != 'paddle'
            or not isinstance(points, (list, tuple)) or len(points) < 3):
        return None
    try:
        if any(not isinstance(point, (list, tuple)) or len(point) != 2
               or any(isinstance(v, bool) or not isinstance(v, Real)
                      or not isfinite(v) or v < 0 for v in point) for point in points):
            return None
        shape = Polygon(points)
        if shape.is_valid and 0 < shape.area < inf:
            return shape
    except (ValueError, TypeError, OverflowError, ShapelyError):
        return None
    return None


def proposal_shape(p, *, use_v3_polygon=False):
    """Use validated measured geometry only for eligible Paddle proposals."""
    shape = measured_shape(p, use_v3_polygon=use_v3_polygon)
    return box(*p.bbox) if shape is None else shape


def line_shape(line):
    shape = Polygon(line.polygon)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError(f'invalid OCR polygon: {line.id}')
    return shape


def proposal_line_coverage(line, p, *, use_v3_polygon=False):
    shape = line_shape(line)
    return shape.intersection(proposal_shape(p, use_v3_polygon=use_v3_polygon)).area / shape.area


def _intersection_area(candidate_bbox, p, shape):
    if shape is not None:
        return box(*candidate_bbox).intersection(shape).area
    a, b, c, d = candidate_bbox
    x1, y1, x2, y2 = p.bbox
    return max(0, min(c, x2)-max(a, x1)) * max(0, min(d, y2)-max(b, y1))


def proposal_intersection_area(candidate_bbox, p, *, use_v3_polygon=False):
    return _intersection_area(candidate_bbox, p, measured_shape(p, use_v3_polygon=use_v3_polygon))


def protected_overlap(candidate_bbox, p, *, use_v3_polygon=False, area_floor=1e-9):
    """Keep the legacy min-area denominator, using actual shape area when available."""
    shape = measured_shape(p, use_v3_polygon=use_v3_polygon)
    a, b, c, d = candidate_bbox
    x1, y1, x2, y2 = p.bbox
    candidate_area = max(0, c-a) * max(0, d-b)
    protected_area = (x2-x1) * (y2-y1) if shape is None else shape.area
    return _intersection_area(candidate_bbox, p, shape) / max(area_floor, min(candidate_area, protected_area))


def set_proposal_bbox(p, bbox):
    """A geometry edit invalidates the detector polygon for this proposal."""
    if tuple(p.bbox) != tuple(bbox):
        p.polygon = None
    p.bbox = bbox
