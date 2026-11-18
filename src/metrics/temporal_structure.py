
from collections import Counter

import numpy as np

def _canonical_group(group):
    return tuple(sorted(group))

def snapshot_signature(groups):

    counts = Counter(_canonical_group(group) for group in groups)
    return tuple(sorted(counts.items()))

def multiset_jaccard_distance(groups_left, groups_right):

    left = Counter(_canonical_group(group) for group in groups_left)
    right = Counter(_canonical_group(group) for group in groups_right)
    keys = set(left) | set(right)
    union = sum(max(left[key], right[key]) for key in keys)
    if union == 0:
        return 0.0
    intersection = sum(min(left[key], right[key]) for key in keys)
    return 1.0 - intersection / float(union)

def _snapshot_groups(temporal_network, time_point):
    if hasattr(temporal_network, "active_hyperedges"):
        return temporal_network.active_hyperedges(time_point)
    if hasattr(temporal_network, "active_maximal_simplices"):
        return temporal_network.active_maximal_simplices(time_point)
    if hasattr(temporal_network, "active_contact_groups"):
        return temporal_network.active_contact_groups(time_point)
    if hasattr(temporal_network, "active_edges"):
        try:
            return temporal_network.active_edges(time_point, repeated=True)
        except TypeError:
            return temporal_network.active_edges(time_point)
    raise TypeError("unsupported temporal network type")

def temporal_variation_metrics(temporal_network):

    snapshots = [
        tuple(_snapshot_groups(temporal_network, time_point))
        for time_point in temporal_network.time_points
    ]
    signatures = [snapshot_signature(groups) for groups in snapshots]
    turnover = [
        multiset_jaccard_distance(left, right)
        for left, right in zip(snapshots[:-1], snapshots[1:])
    ]
    active_nodes = [
        len({node for group in groups for node in group})
        for groups in snapshots
    ]
    event_counts = [len(groups) for groups in snapshots]
    return {
        "time_points": list(temporal_network.time_points),
        "snapshot_count": len(snapshots),
        "unique_snapshot_count": len(set(signatures)),
        "consecutive_jaccard_distance": turnover,
        "mean_consecutive_jaccard_distance": float(np.mean(turnover)) if turnover else 0.0,
        "max_consecutive_jaccard_distance": float(max(turnover)) if turnover else 0.0,
        "active_node_count": active_nodes,
        "event_count": event_counts,
        "is_structurally_temporal": len(set(signatures)) > 1,
    }

