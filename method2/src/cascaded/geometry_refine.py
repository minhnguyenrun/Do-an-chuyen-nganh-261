"""Conservative OCR-supported edits, independent of label and quality scoring."""
from math import atan2, degrees, isfinite

from shapely.errors import GEOSException
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .core import VISUAL, area, cluster, envelope, intersect
from .v3_geometry import proposal_shape, protected_overlap, v3_polygons_enabled


TEXT_ROLES = {'text', 'title', 'doc_title', 'paragraph_title', 'caption',
              'footnote', 'list_item'}
EPS = 1e-8


def _valid_box(bbox):
    return (len(bbox) == 4 and all(isfinite(v) for v in bbox)
            and min(bbox) >= 0 and bbox[0] < bbox[2] and bbox[1] < bbox[3])


def _horizontal(poly):
    points = list(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = list(zip(points, points[1:]+points[:1]))
    a, b = max(edges, key=lambda pair: (pair[1][0]-pair[0][0])**2
               +(pair[1][1]-pair[0][1])**2)
    angle = abs(degrees(atan2(b[1]-a[1], b[0]-a[0]))) % 180
    return min(angle, 180-angle) <= 10


def refine_candidate_bbox(bbox, label, target_lines, lines, owners, confirmed, cfg):
    """Return an atomic edit or the original geometry with a rejection reason.

    Core membership is measured once on the original box. A later envelope
    cannot turn fragments into evidence or silently discard another owner.
    All page-line/semantic/partition guards remain the caller's responsibility.
    """
    original = list(bbox)
    result = dict(bbox=original[:], core_line_ids=[], excluded_line_ids=[],
                  codes=[], changed=False, valid=True)
    if not cfg.enable_geometry_refinement:
        return result

    def reject(reason):
        return dict(result, bbox=original[:], changed=False, valid=False,
                    codes=['GEOMETRY_REFINEMENT_REJECTED', reason])

    try:
        if not _valid_box(original):
            return reject('INVALID_GEOMETRY')
        if len({line.id for line in lines}) != len(lines):
            return reject('DUPLICATE_OCR_LINE_IDS')
        polygons = {}
        for line in lines:
            if not all(len(p) == 2 and all(isfinite(v) for v in p) for p in line.polygon):
                return reject('INVALID_OCR_POLYGON')
            poly = Polygon(line.polygon)
            if not poly.is_valid or poly.area <= EPS:
                return reject('INVALID_OCR_POLYGON')
            polygons[line.id] = poly
        target_ids = {line.id for line in target_lines}
        if not target_ids <= polygons.keys() or len(target_ids) != len(target_lines):
            return reject('INVALID_TARGET_LINES')
        window = box(*original)
        initial = {lid: p.intersection(window).area/p.area for lid, p in polygons.items()}
        core = [line for line in target_lines if initial[line.id] >= .80]
        result['core_line_ids'] = [line.id for line in core]
        if not core:
            return reject('NO_ORIGINAL_CORE_LINES')
        if any(line.id in owners for line in core):
            return reject('CORE_HAS_FROZEN_OWNER')
        if len(cluster(core)) != 1:
            return reject('cross_column_or_disconnected_lines')
        threshold = max(.80, cfg.line_coverage)

        def preserves(candidate):
            return (_valid_box(candidate) and all(
                polygons[line.id].intersection(box(*candidate)).area/polygons[line.id].area
                >= threshold-EPS for line in core))

        # Non-text boxes keep their visual extent; OCR never defines their edge.
        if label not in TEXT_ROLES:
            return result
        current, codes, excluded = original[:], [], set()
        evidence = list(core)
        core_ids = {line.id for line in core}
        for line in lines:
            lid, poly = line.id, polygons[line.id]
            if initial[lid] <= EPS or lid in core_ids:
                continue
            a, b, c, d = poly.bounds
            cross = intersect([a, b, c, d], original)
            cov_x = max(0, cross[2]-cross[0])/(c-a)
            cov_y = max(0, cross[3]-cross[1])/(d-b)
            if not _horizontal(poly):
                return reject('ROTATED_PARTIAL_LINE')
            choices = []
            # A grazing line must cross the outside edge, never the interior.
            if 0 < cov_y < .45:
                if b < original[1] < d:
                    choices.append([current[0], max(current[1], d), current[2], current[3]])
                if b < original[3] < d:
                    choices.append([current[0], current[1], current[2], min(current[3], b)])
            if 0 < cov_x < .30:
                if a < original[0] < c:
                    choices.append([max(current[0], c), current[1], current[2], current[3]])
                if a < original[2] < c:
                    choices.append([current[0], current[1], min(current[2], a), current[3]])
            choices = [v for v in choices if preserves(v)
                       and poly.intersection(box(*v)).area <= EPS]
            if choices:
                current = max(choices, key=lambda v: (area(v), tuple(v)))
                excluded.add(lid)
                if 'ACTIVE_SHRINK_APPLIED' not in codes:
                    codes.append('ACTIVE_SHRINK_APPLIED')
                continue
            if lid in target_ids and initial[lid] < .80:
                margin = min((b+d)/2-original[1], original[3]-(b+d)/2)/(d-b)
                # Keep exactly the legacy bounded vertical snap eligibility.
                if (cov_x < cfg.snap_cov_x or cov_y < cfg.snap_cov_y
                        or margin < cfg.snap_center_margin):
                    return reject('INNER_LINE_BISECTION')
                evidence.append(line)

        protected = [p for p in confirmed if p.label in VISUAL]
        use_v3_polygon = v3_polygons_enabled(cfg)
        for region in protected:
            if not _valid_box(region.bbox):
                return reject('INVALID_PROTECTED_GEOMETRY')
            overlap = protected_overlap(current, region, use_v3_polygon=use_v3_polygon, area_floor=EPS)
            if overlap <= cfg.max_nontext_overlap:
                continue
            shape = proposal_shape(region, use_v3_polygon=use_v3_polygon)
            ink = unary_union([polygons[line.id] for line in evidence])
            if ink.intersection(shape).area/max(EPS, ink.area) > .05:
                return reject('PROTECTED_NONTEXT_INK_OVERLAP')
            a, b, c, d = shape.bounds if use_v3_polygon else region.bbox
            choices = [[current[0], current[1], min(current[2], a), current[3]],
                       [max(current[0], c), current[1], current[2], current[3]],
                       [current[0], current[1], current[2], min(current[3], b)],
                       [current[0], max(current[1], d), current[2], current[3]]]
            choices = [v for v in choices if preserves(v)]
            if not choices:
                return reject('PROTECTED_NONTEXT_UNCARVABLE')
            current = max(choices, key=lambda v: (area(v), tuple(v)))
            if 'NON_TEXT_CARVING_APPLIED' not in codes:
                codes.append('NON_TEXT_CARVING_APPLIED')

        # Refit only text evidence. Intersect with earlier cuts so an envelope
        # cannot reintroduce a grazing line overlapping a retained core polygon.
        fitted = envelope([list(polygons[line.id].bounds) for line in evidence])
        # Permit only the bounded vertical snap above to extend the original box.
        limits = [current[0], min(current[1], fitted[1]), current[2], max(current[3], fitted[3])]
        if excluded or 'NON_TEXT_CARVING_APPLIED' in codes:
            limits = current
        fitted = intersect(fitted, limits)
        if not preserves(fitted):
            return reject('CORE_COVERAGE_LOSS')
        if fitted != current:
            codes.append('OCR_ENVELOPE_REFIT_APPLIED')
        current = fitted
        final_window = box(*current)
        for line in lines:
            lid = line.id
            after = polygons[lid].intersection(final_window).area/polygons[lid].area
            if lid in excluded and after > EPS:
                return reject('EXCLUDED_LINE_REINTRODUCED')
            if lid in target_ids and lid not in excluded and initial[lid] > EPS and after < threshold-EPS:
                return reject('TARGET_LINE_COVERAGE_LOSS')
            # Fully or substantially covered foreign ink cannot disappear just
            # because a text envelope happens to be smaller than the model box.
            if lid not in target_ids and lid not in excluded and after+EPS < initial[lid]:
                return reject('FOREIGN_INK_HIDDEN_BY_REFIT')
        if any(protected_overlap(current, p, use_v3_polygon=use_v3_polygon, area_floor=EPS)
               > cfg.max_nontext_overlap for p in protected):
            return reject('PROTECTED_NONTEXT_OVERLAP')
        return dict(result, bbox=current, excluded_line_ids=[line.id for line in lines if line.id in excluded],
                    codes=codes, changed=current != original)
    except (ValueError, TypeError, KeyError, IndexError, GEOSException):
        return reject('INVALID_GEOMETRY')
