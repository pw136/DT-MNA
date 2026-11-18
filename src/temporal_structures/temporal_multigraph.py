
import itertools
from collections import Counter

from ._utils import normalize_nodes, normalize_snapshots, require_time

class TemporalMultiGraph(object):

    def __init__(self, nodes, contact_groups, source_blocks=None):
        self.nodes = normalize_nodes(nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.contact_groups = normalize_snapshots(self.nodes, contact_groups)
        self.snapshots = self.contact_groups
        self.time_points = tuple(self.contact_groups.keys())
        if source_blocks is None:
            self.source_blocks = {
                time_point: tuple((group,) for group in groups)
                for time_point, groups in self.contact_groups.items()
            }
        else:
            self.source_blocks = {}
            for time_point in self.time_points:
                blocks = tuple(
                    tuple(tuple(group) for group in block)
                    for block in source_blocks[time_point]
                )
                flattened = tuple(group for block in blocks for group in block)
                if flattened != self.contact_groups[time_point]:
                    raise ValueError("source_blocks must flatten to contact_groups")
                self.source_blocks[time_point] = blocks
        self._weights = {}
        self._weighted_neighbors = {}
        for time_point, groups in self.contact_groups.items():
            weights = Counter()
            for group in groups:
                for left, right in itertools.combinations(group, 2):
                    edge = tuple(sorted((left, right), key=self.node_to_index.get))
                    weights[edge] += 1
            self._weights[time_point] = weights
            adjacency = {node: [] for node in self.nodes}
            for (left, right), weight in weights.items():
                adjacency[left].append((right, int(weight)))
                adjacency[right].append((left, int(weight)))
            self._weighted_neighbors[time_point] = {
                node: tuple(neighbors)
                for node, neighbors in adjacency.items()
            }
        self._incident_source_blocks = {}
        for time_point, blocks in self.source_blocks.items():
            incidence = {node: [] for node in self.nodes}
            for block in blocks:
                block_nodes = {
                    node for group in block for node in group
                }
                for node in block_nodes:
                    incidence[node].append(block)
            self._incident_source_blocks[time_point] = {
                node: tuple(items) for node, items in incidence.items()
            }

    def __len__(self):
        return len(self.nodes)

    def active_contact_groups(self, time_point):
        require_time(self.contact_groups, time_point)
        return self.contact_groups[time_point]

    def incident_contact_groups(self, node, time_point):
        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        return tuple(
            group
            for group in self.active_contact_groups(time_point)
            if node in group
        )

    def active_source_blocks(self, time_point):
        require_time(self.contact_groups, time_point)
        return self.source_blocks[time_point]

    def incident_source_blocks(self, node, time_point):
        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        require_time(self.contact_groups, time_point)
        return self._incident_source_blocks[time_point][node]

    def edge_weights(self, time_point):
        require_time(self.contact_groups, time_point)
        return dict(self._weights[time_point])

    def edge_weight(self, left, right, time_point):
        edge = tuple(sorted((left, right), key=self.node_to_index.get))
        return int(self._weights[time_point].get(edge, 0))

    def weighted_neighbors(self, node, time_point):
        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        require_time(self.contact_groups, time_point)
        return self._weighted_neighbors[time_point][node]

    def active_edges(self, time_point, repeated=True):
        weights = self._weights[time_point]
        if not repeated:
            return tuple(weights.keys())
        return tuple(edge for edge, weight in weights.items() for _ in range(weight))
