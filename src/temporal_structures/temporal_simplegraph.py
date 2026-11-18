
from ._utils import normalize_nodes, require_time

class TemporalSimpleGraph(object):
    def __init__(self, nodes, snapshots):
        self.nodes = normalize_nodes(nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        normalized = {}
        for time_point in sorted(snapshots):
            edges = set()
            for left, right in snapshots[time_point]:
                if left == right:
                    raise ValueError("self-loops are not supported")
                if left not in self.node_to_index or right not in self.node_to_index:
                    raise ValueError("edge contains an unknown node")
                edges.add(tuple(sorted((left, right), key=self.node_to_index.get)))
            normalized[time_point] = tuple(sorted(edges, key=lambda e: (
                self.node_to_index[e[0]], self.node_to_index[e[1]]
            )))
        self.snapshots = normalized
        self.time_points = tuple(normalized.keys())
        self._neighbors = {}
        for time_point, edges in self.snapshots.items():
            adjacency = {node: [] for node in self.nodes}
            for left, right in edges:
                adjacency[left].append(right)
                adjacency[right].append(left)
            self._neighbors[time_point] = {
                node: tuple(neighbors)
                for node, neighbors in adjacency.items()
            }

    def __len__(self):
        return len(self.nodes)

    def active_edges(self, time_point):
        require_time(self.snapshots, time_point)
        return self.snapshots[time_point]

    def neighbors(self, node, time_point):
        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        require_time(self.snapshots, time_point)
        return self._neighbors[time_point][node]
