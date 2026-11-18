
from collections import Counter, defaultdict

from ._utils import normalize_nodes, normalize_snapshots, require_time

class TemporalHypergraph(object):

    def __init__(self, nodes, snapshots):
        self.nodes = normalize_nodes(nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.snapshots = normalize_snapshots(self.nodes, snapshots)
        self.time_points = tuple(self.snapshots.keys())
        self._incident = {}
        for time_point, hyperedges in self.snapshots.items():
            incidence = {node: [] for node in self.nodes}
            for hyperedge in hyperedges:
                for node in hyperedge:
                    incidence[node].append(hyperedge)
            self._incident[time_point] = {
                node: tuple(groups) for node, groups in incidence.items()
            }

    def __len__(self):
        return len(self.nodes)

    def active_hyperedges(self, time_point):

        require_time(self.snapshots, time_point)
        return self.snapshots[time_point]

    def event_multiplicities(self, time_point):

        return dict(Counter(self.active_hyperedges(time_point)))

    def incident_hyperedges(self, node, time_point):

        if node not in self.node_to_index:
            raise KeyError("unknown node: %r" % (node,))
        require_time(self.snapshots, time_point)
        return self._incident[time_point][node]

    @staticmethod
    def hyperedge_size(hyperedge):
        return len(hyperedge)

    def hyperedge_sizes(self, time_point=None):
        if time_point is not None:
            return tuple(len(h) for h in self.active_hyperedges(time_point))
        return tuple(
            len(h)
            for t in self.time_points
            for h in self.active_hyperedges(t)
        )

    @classmethod
    def from_events(cls, nodes, events):

        snapshots = defaultdict(list)
        for time_point, hyperedge in events:
            snapshots[time_point].append(tuple(hyperedge))
        return cls(nodes, dict(snapshots))

    @classmethod
    def from_benson_files(
        cls,
        nverts_path,
        simplices_path,
        times_path,
        window_size=1,
        max_events=None,
    ):

        if window_size <= 0:
            raise ValueError("window_size must be positive")

        with open(nverts_path, "r") as handle:
            sizes = [int(line.strip()) for line in handle if line.strip()]
        with open(times_path, "r") as handle:
            times = [int(line.strip()) for line in handle if line.strip()]
        if len(sizes) != len(times):
            raise ValueError("nverts and times must contain the same event count")
        event_count = len(sizes) if max_events is None else min(len(sizes), max_events)

        with open(simplices_path, "r") as handle:
            simplex_nodes = (int(line.strip()) for line in handle if line.strip())
            events = []
            first_time = min(times[:event_count]) if event_count else 0
            for event_idx in range(event_count):
                group = tuple(next(simplex_nodes) for _ in range(sizes[event_idx]))
                if len(group) >= 2:
                    binned_time = int((times[event_idx] - first_time) // window_size)
                    events.append((binned_time, group))

        nodes = sorted({node for _, group in events for node in group})
        return cls.from_events(nodes, events)
