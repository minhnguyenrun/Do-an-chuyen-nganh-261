from dataclasses import dataclass
from ..spatial.region_graph import RegionGraph


@dataclass(frozen=True, init=False)
class HighSpatialSkip:
    regions: tuple
    _nodes: tuple
    _columns: tuple
    _size: tuple
    _edges: frozenset
    _neighbors: frozenset
    _lines: tuple
    _ownership: tuple
    _lineage: tuple

    def __init__(self, regions, graph):
        object.__setattr__(self,'regions',tuple(regions))
        object.__setattr__(self,'_nodes',tuple(graph.nodes.items()))
        object.__setattr__(self,'_columns',tuple(graph.columns.items()))
        object.__setattr__(self,'_size',(graph.page_w,graph.page_h))
        object.__setattr__(self,'_edges',frozenset(graph.edges))
        object.__setattr__(self,'_neighbors',frozenset(graph.neighbors))
        object.__setattr__(self,'_lines',tuple(graph.lines))
        object.__setattr__(self,'_ownership',tuple((id,tuple(lines)) for id,lines in graph.ownership.items()))
        object.__setattr__(self,'_lineage',tuple((id,tuple(sources)) for id,sources in graph.lineage.items()))

    @property
    def graph(self):
        return RegionGraph(dict(self._nodes),dict(self._columns),*self._size,edges=set(self._edges),
            neighbors=set(self._neighbors),lines=self._lines,ownership={id:list(lines) for id,lines in self._ownership},
            lineage=dict(self._lineage))
