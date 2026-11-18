
import numpy as np

from .temporal_hypergraph import TemporalHypergraph

def random_temporal_hypergraph(
    node_count,
    time_count,
    events_per_snapshot,
    hyperedge_sizes=(2, 3),
    persistence=0.0,
    seed=0,
):
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence must be in [0, 1]")
    rng = np.random.RandomState(seed)
    nodes = tuple(range(node_count))
    snapshots = {}
    previous = []
    for time_point in range(time_count):
        retained = [edge for edge in previous if rng.random_sample() < persistence]
        current = list(retained[:events_per_snapshot])
        attempts = 0
        while len(current) < events_per_snapshot:
            size = int(rng.choice(hyperedge_sizes))
            if size > node_count:
                raise ValueError("hyperedge size exceeds node count")
            edge = tuple(sorted(int(value) for value in rng.choice(nodes, size=size, replace=False)))
            if edge not in current:
                current.append(edge)
            attempts += 1
            if attempts > events_per_snapshot * 1000:
                raise RuntimeError("unable to generate enough unique hyperedges")
        snapshots[time_point] = tuple(current)
        previous = current
    return TemporalHypergraph(nodes, snapshots)

