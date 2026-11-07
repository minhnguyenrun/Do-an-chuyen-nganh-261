from .coordinates import polygon_bbox, union_bboxes
from ..schemas.region import LayoutRegion


def detect_residual_lines(lines, assigned_line_ids):
    return [line for line in lines if line.line_id not in assigned_line_ids]


def cluster_residual_lines(residual_lines, max_dist=25):
    pending = list(residual_lines)
    groups = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            for line in pending[:]:
                b = polygon_bbox(line.polygon)
                if any(_near(b, polygon_bbox(other.polygon), max_dist) for other in group):
                    pending.remove(line)
                    group.append(line)
                    changed = True
        groups.append(group)
    return [LayoutRegion(id=f'residual:{i}', bbox=union_bboxes(polygon_bbox(l.polygon) for l in group),
                         label='residual', order=i, score=min(l.score for l in group))
            for i, group in enumerate(groups)]


def _near(a, b, distance):
    return min(a[2], b[2]) > max(a[0], b[0]) and max(a[1]-b[3], b[1]-a[3], 0) <= distance
