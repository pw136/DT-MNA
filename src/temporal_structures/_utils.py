
from collections import OrderedDict

def normalize_nodes(nodes):
    result = tuple(nodes)
    if len(result) != len(set(result)):
        raise ValueError("nodes must be unique")
    if not result:
        raise ValueError("at least one node is required")
    return result

def normalize_group(group, node_set, minimum_size=2):
    normalized = tuple(group)
    if len(normalized) < minimum_size:
        raise ValueError("an interaction must contain at least two nodes")
    if len(set(normalized)) != len(normalized):
        raise ValueError("an interaction cannot repeat a node")
    unknown = set(normalized) - node_set
    if unknown:
        raise ValueError("interaction contains unknown nodes: %r" % sorted(unknown))
    return normalized

def normalize_snapshots(nodes, snapshots, minimum_size=2):
    node_set = set(nodes)
    normalized = OrderedDict()
    for time_point in sorted(snapshots):
        normalized[time_point] = tuple(
            normalize_group(group, node_set, minimum_size=minimum_size)
            for group in snapshots[time_point]
        )
    return normalized

def require_time(snapshots, time_point):
    if time_point not in snapshots:
        raise KeyError("unknown time point: %r" % (time_point,))

