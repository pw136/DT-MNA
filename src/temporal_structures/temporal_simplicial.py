
import itertools

from ._utils import normalize_nodes, normalize_snapshots, require_time

class TemporalSimplicialComplex(object):

    def __init__(
        self,
        nodes,
        snapshots,
        max_order=None,
        source_groups=None,
        source_events=None,
    ):
        self.nodes = normalize_nodes(nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.snapshots = normalize_snapshots(self.nodes, snapshots)
        self.time_points = tuple(self.snapshots.keys())
        self._incident = {}
        for time_point, cells in self.snapshots.items():
            incidence = {node: [] for node in self.nodes}
            for cell in cells:
                for node in cell:
                    incidence[node].append(cell)
            self._incident[time_point] = {
                node: tuple(groups) for node, groups in incidence.items()
            }
        if source_groups is None:
            self.source_groups = {
                time_point: tuple((cell,) for cell in cells)
                for time_point, cells in self.snapshots.items()
            }
        else:
            self.source_groups = {}
            for time_point in self.time_points:
                blocks = []
                for block in source_groups[time_point]:
                    blocks.append(tuple(tuple(cell) for cell in block))
                flattened = tuple(cell for block in blocks for cell in block)
                if flattened != self.snapshots[time_point]:
                    raise ValueError("source_groups must flatten to the stored snapshots")
                self.source_groups[time_point] = tuple(blocks)
        if source_events is None:
            self.source_events = {}
            for time_point, blocks in self.source_groups.items():
                events = []
                for block in blocks:
                    event_nodes = {
                        node for cell in block for node in cell
                    }
                    events.append(
                        tuple(
                            sorted(
                                event_nodes,
                                key=self.node_to_index.get,
                            )
                        )
                    )
                self.source_events[time_point] = tuple(events)
        else:
            self.source_events = {}
            for time_point in self.time_points:
                events = tuple(
                    tuple(event)
                    for event in source_events[time_point]
                )
                if len(events) != len(self.source_groups[time_point]):
                    raise ValueError(
                        "source_events must align one-to-one with source_groups"
                    )
                for event in events:
                    if not event or any(
                        node not in self.node_to_index for node in event
                    ):
                        raise ValueError(
                            "source_events contain an unknown or empty hyperedge"
                        )
                self.source_events[time_point] = events
        self._incident_source_groups = {}
        for time_point, blocks in self.source_groups.items():
            incidence = {node: [] for node in self.nodes}
            for block in blocks:
                block_nodes = {
                    node for cell in block for node in cell
                }
                for node in block_nodes:
                    incidence[node].append(block)
            self._incident_source_groups[time_point] = {
                node: tuple(groups) for node, groups in incidence.items()
            }
        observed_order = max(
            [len(cell) - 1 for cells in self.snapshots.values() for cell in cells] or [1]
        )
        self.max_order = observed_order if max_order is None else int(max_order)
        if self.max_order < observed_order:
            raise ValueError("max_order is smaller than an observed simplex order")

    def __len__(self):
        return len(self.nodes)

    def active_maximal_simplices(self, time_point):
        require_time(self.snapshots, time_point)
        return self.snapshots[time_point]

    def active_generator_multiset(self, time_point):

        return self.active_maximal_simplices(time_point)

    def incident_maximal_simplices(self, node, time_point):
        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        require_time(self.snapshots, time_point)
        return self._incident[time_point][node]

    def active_source_groups(self, time_point):
        require_time(self.snapshots, time_point)
        return self.source_groups[time_point]

    def active_source_events(self, time_point):

        require_time(self.snapshots, time_point)
        return self.source_events[time_point]

    def active_generator_records(self, time_point):

        records = []
        for source_event, block in zip(
            self.active_source_events(time_point),
            self.active_source_groups(time_point),
        ):
            records.extend(
                (source_event, cell)
                for cell in block
            )
        return tuple(records)

    def incident_source_groups(self, node, time_point):
        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        require_time(self.snapshots, time_point)
        return self._incident_source_groups[time_point][node]

    def simplices(self, time_point, order=None):

        result = []
        for cell in self.active_maximal_simplices(time_point):
            orders = range(1, len(cell)) if order is None else (order,)
            for current_order in orders:
                face_size = current_order + 1
                if face_size <= len(cell):
                    result.extend(itertools.combinations(cell, face_size))
        return tuple(result)

    def topological_simplices(self, time_point, order=None):

        unique = set(self.simplices(time_point, order=order))
        return tuple(
            sorted(
                unique,
                key=lambda cell: (
                    len(cell),
                    tuple(self.node_to_index[node] for node in cell),
                ),
            )
        )
