from dataclasses import dataclass, field
from .line_assignment import assign_lines_to_regions


@dataclass
class RegionGraph:
    nodes: dict
    columns: dict
    page_w: int
    page_h: int
    edges: set = field(default_factory=set)
    neighbors: set = field(default_factory=set)
    lines: tuple = ()
    ownership: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    applied_revision_ids: set = field(default_factory=set)
    rolled_back_revision_ids: set = field(default_factory=set)

    def get_column_id(self, region_id):
        return self.columns[region_id]

    def is_cross_column(self, a, b):
        return self.columns[a] != self.columns[b]


def build_region_graph(regions, page_w, page_h, lines=(), coverage_threshold=.3):
    if page_w <= 0 or page_h <= 0:
        raise ValueError('invalid page dimensions')
    if len({r.id for r in regions}) != len(regions):
        raise ValueError('duplicate region IDs')
    columns, clusters = {}, []
    # Wide headings must not bridge and collapse separate body columns.
    narrow = [r for r in regions if r.bbox[2]-r.bbox[0] < page_w*.7]
    for r in sorted(narrow, key=lambda r: (r.bbox[0],r.bbox[1])):
        left,right = r.bbox[0],r.bbox[2]
        matches = [i for i,(x1,x2) in enumerate(clusters)
                   if max(left-x2,x1-right,0) <= max(page_w*.04,min(right-left,x2-x1)*.3)]
        if matches:
            i = matches[0]
            clusters[i] = (min(left,clusters[i][0]),max(right,clusters[i][1]))
        else:
            i = len(clusters)
            clusters.append((left,right))
        columns[r.id] = i
    for r in regions:
        columns.setdefault(r.id,-1)
    graph = RegionGraph({r.id:r for r in regions},columns,page_w,page_h,lines=tuple(lines),
                        ownership=assign_lines_to_regions(regions,lines,coverage_threshold),
                        lineage={r.id:(r.id,) for r in regions})
    # Wide blocks partition the page into vertical bands; columns flow within a band.
    wide = sorted((r for r in regions if columns[r.id] == -1),key=lambda r:r.bbox[1])
    def rank(r):
        band = sum(w.bbox[1] <= r.bbox[1] for w in wide)
        return band, columns[r.id], r.bbox[1],r.bbox[0],r.order,r.id
    ordered = sorted(regions,key=rank)
    graph.edges = {(a.id,b.id) for a,b in zip(ordered,ordered[1:])}
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            if columns[a.id] == columns[b.id]:
                gap = max(a.bbox[1]-b.bbox[3],b.bbox[1]-a.bbox[3],0)
                if gap <= page_h*.08:
                    graph.neighbors.add(tuple(sorted((a.id,b.id))))
    return graph
