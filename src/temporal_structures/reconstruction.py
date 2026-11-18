
def _adjacency(edges):
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency

def _connected_components(adjacency):
    unseen = set(adjacency)
    components = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = {start}
        while stack:
            node = stack.pop()
            new_nodes = adjacency[node] & unseen
            unseen -= new_nodes
            component |= new_nodes
            stack.extend(new_nodes)
        components.append(tuple(sorted(component)))
    return components

def _maximal_cliques(adjacency):
    cliques = []

    def bron_kerbosch(current, candidates, excluded):
        if not candidates and not excluded:
            if len(current) >= 2:
                cliques.append(tuple(sorted(current)))
            return
        pivot_pool = candidates | excluded
        pivot = max(pivot_pool, key=lambda node: len(adjacency[node])) if pivot_pool else None
        extension = candidates - (adjacency[pivot] if pivot is not None else set())
        for node in list(extension):
            bron_kerbosch(
                current | {node},
                candidates & adjacency[node],
                excluded & adjacency[node],
            )
            candidates.remove(node)
            excluded.add(node)

    bron_kerbosch(set(), set(adjacency), set())
    return sorted(set(cliques))

def _density(group, adjacency):
    if len(group) < 2:
        return 0.0
    edges = sum(1 for left in group for right in group if left < right and right in adjacency[left])
    possible = len(group) * (len(group) - 1) / 2.0
    return edges / possible

def reconstruct_hyperedges_from_pairwise_snapshot(
    edges,
    method="maximal_cliques",
    density_threshold=0.7,
):

    adjacency = _adjacency(edges)
    if method == "connected_components":
        return tuple(_connected_components(adjacency))
    cliques = _maximal_cliques(adjacency)
    if method == "maximal_cliques":
        return tuple(cliques)
    if method == "dense_subgraphs":
        dense = [
            component
            for component in _connected_components(adjacency)
            if _density(component, adjacency) >= density_threshold
        ]
        dense.extend(clique for clique in cliques if clique not in dense)
        return tuple(sorted(set(dense)))
    raise ValueError("unknown reconstruction method: %s" % method)

