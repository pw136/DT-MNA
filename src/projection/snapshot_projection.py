
import itertools
from collections import Counter

def hypergraph_to_simplicial_snapshot(hyperedges, kappa):

    if kappa < 1:
        raise ValueError("kappa must be at least one")
    target_size = kappa + 1
    cells = []
    for hyperedge in hyperedges:
        if len(hyperedge) <= target_size:
            cells.append(tuple(hyperedge))
        else:
            cells.extend(tuple(cell) for cell in itertools.combinations(hyperedge, target_size))
    return tuple(cells)

def groups_to_multigraph_snapshot(groups):
    weights = Counter()
    for group in groups:
        for left, right in itertools.combinations(sorted(group), 2):
            weights[(left, right)] += 1
    return dict(weights)

def groups_to_simplegraph_snapshot(groups):
    return tuple(sorted(groups_to_multigraph_snapshot(groups).keys()))

