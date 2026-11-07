import heapq


def reconcile_order_as_dag(graph):
    edges = {(a,b) for a,b in graph.edges if a in graph.nodes and b in graph.nodes and a != b}
    remaining = set(graph.nodes)
    result = []
    def rank(id):
        r = graph.nodes[id]
        return r.order,r.bbox[1],r.bbox[0],id
    while remaining:
        ready = [id for id in remaining if not any(b == id and a in remaining for a,b in edges)]
        if not ready:
            # Deterministic recovery; keep a trace instead of returning a truncated page.
            first = min(remaining,key=rank)
            edges = {(a,b) for a,b in edges if b != first}
            graph.warnings.append(f'reading_order_cycle_broken:{first}')
            ready = [first]
        first = min(ready,key=rank)
        result.append(first)
        remaining.remove(first)
    return result
