"""Measured public Paddle layout features, independent of model imports.

Public polygons are page-pixel output, not a claim of access to raw masks.
Reading order is accepted only from V3's positive integer public order.
"""
import math
from numbers import Integral, Real
from shapely.errors import ShapelyError
from shapely.geometry import Polygon, box
from .types import Proposal, label_for


FEATURE_SCHEMA_VERSION = 'paddle-layout-public-v1'


def json_safe(value, reasons=None, path='raw'):
    """Copy public/numpy output to strict JSON; explain nonfinite/unknown data."""
    reasons = reasons if reasons is not None else []
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if math.isfinite(value):
            return float(value)
        reasons.append(f'{path}:nonfinite')
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item, reasons, f'{path}.{key}') for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, reasons, f'{path}[{i}]') for i, item in enumerate(value)]
    # numpy arrays/scalars, without importing a model or requiring numpy here.
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist(), reasons, path)
    reasons.append(f'{path}:unsupported_type:{type(value).__name__}')
    return str(value)


def _polygon(points, width, height):
    if points is None:
        return None, 'missing', 'missing_polygon_points'
    try:
        if hasattr(points, 'tolist'):
            points = points.tolist()
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            raise ValueError('insufficient_points')
        coordinates = []
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, Real)
                           or not math.isfinite(v) for v in point)):
                raise ValueError('invalid_or_nonfinite_points')
            coordinates.append([float(v) for v in point])
        if len(set(map(tuple, coordinates))) < 3:
            raise ValueError('insufficient_distinct_points')
        shape = Polygon(coordinates)
        if not shape.is_valid or shape.is_empty or not 0 < shape.area < math.inf:
            raise ValueError('invalid_polygon')
        page = box(0, 0, width, height)
        if page.covers(shape):
            return coordinates, 'valid', None
        clipped = shape.intersection(page)
        if (clipped.geom_type != 'Polygon' or clipped.is_empty or not clipped.is_valid
                or not 0 < clipped.area < math.inf or clipped.interiors):
            raise ValueError('invalid_or_multicomponent_clipping')
        return [list(point) for point in clipped.exterior.coords[:-1]], 'clipped', None
    except (ValueError, TypeError, OverflowError, ShapelyError) as exc:
        return None, 'invalid', str(exc)


def normalize_paddle_box(raw: dict, *, width: int, height: int, variant: str,
                         proposal_id: str) -> tuple[Proposal | None, dict]:
    """Normalize one public box without sorting or fabricating optional evidence."""
    reasons = []
    order = raw.get('order') if isinstance(raw, dict) else None
    audit = {'proposal_id': proposal_id, 'raw_box': json_safe(raw, reasons),
             'raw_order': json_safe(order, reasons, 'order'),
             'order_status': 'unavailable', 'polygon_status': 'unavailable',
             'reasons': reasons}
    try:
        if not isinstance(raw, dict):
            raise ValueError('expected_box_mapping')
        a, b, c, d = raw['coordinate']
        bbox = [max(0, int(a)), max(0, int(b)),
                min(width, math.ceil(c)), min(height, math.ceil(d))]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError('empty_bbox')
        label = raw['label']
        if not isinstance(label, str):
            raise ValueError('invalid_label')
        proposal = Proposal(proposal_id, bbox, label_for(label, 'paddle'),
                            'paddle', label, float(raw['score']))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        audit['proposal_id'] = None
        audit['reasons'].append(f'invalid_detection:{exc}')
        return None, audit
    if variant != 'v3':
        audit.update(order_status='disabled_variant', polygon_status='disabled_variant')
        reasons.append('features_unavailable_for_variant')
        return proposal, audit
    if order is None:
        audit['order_status'] = 'missing'
        reasons.append('missing_public_order')
    elif isinstance(order, Integral) and not isinstance(order, bool) and order > 0:
        proposal.reading_order_rank = int(order) - 1
        audit['order_status'] = 'valid'
    else:
        audit['order_status'] = 'invalid'
        reasons.append('invalid_public_order:expected_positive_integer')
    proposal.polygon, audit['polygon_status'], reason = _polygon(raw.get('polygon_points'), width, height)
    if reason:
        reasons.append(reason)
    return proposal, audit


def layout_feature_audit(records, *, metadata=None, status=None):
    """Summarize extraction records without mistaking absent evidence for rank0."""
    rank_count = sum(record['order_status'] == 'valid' for record in records)
    polygon_count = sum(record['polygon_status'] in ('valid', 'clipped') for record in records)
    result = {'schema_version': FEATURE_SCHEMA_VERSION,
              'status': status or ('available' if rank_count or polygon_count else 'unavailable'),
              'raw_boxes': [record['raw_box'] for record in records],
              'records': [{key: value for key, value in record.items() if key != 'raw_box'}
                          for record in records],
              'rank_count': rank_count, 'polygon_count': polygon_count,
              'order_status_counts': {}, 'polygon_status_counts': {},
              'reasons': sorted({reason for record in records for reason in record['reasons']}),
              'provenance': 'paddle_public_boxes',
              'polygon_representation': 'public_polygon_points'}
    for feature in ('order', 'polygon'):
        for record in records:
            value = record[f'{feature}_status']
            counts = result[f'{feature}_status_counts']
            counts[value] = counts.get(value, 0) + 1
    if not records:
        result['reasons'].append('no_public_layout_boxes')
    result.update(json_safe(metadata or {}, result['reasons'], 'metadata'))
    return result


def missing_layout_features():
    """Historical cache capability, annotated in memory without changing files."""
    result = layout_feature_audit([], status='unavailable')
    result['reasons'] = ['missing_layout_features']
    result['provenance'] = 'unavailable'
    result['polygon_representation'] = None
    return result
