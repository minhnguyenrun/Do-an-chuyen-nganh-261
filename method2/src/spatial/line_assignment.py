from shapely.geometry import Polygon, box
from .coordinates import polygon_bbox


def assign_lines_to_regions(regions, lines, iou_threshold=0.3):
    ownership = {r.id: [] for r in regions}
    for line in lines:
        polygon = Polygon(line.polygon)
        if not polygon.is_valid or polygon.area <= 0:
            continue
        candidates = []
        for region in regions:
            coverage = polygon.intersection(box(*region.bbox)).area / polygon.area
            if coverage >= iou_threshold:
                candidates.append((coverage, -box(*region.bbox).area, -region.order, region.id))
        if candidates:
            ownership[max(candidates)[-1]].append(line.line_id)
    by_id = {line.line_id:line for line in lines}
    for ids in ownership.values():
        ids.sort(key=lambda id:(polygon_bbox(by_id[id].polygon)[1],polygon_bbox(by_id[id].polygon)[0],id))
    return ownership
