
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from ..temporal_structures import TemporalHypergraph
from ._common import project_root

PAPER_SSC_PROFILES = OrderedDict(
    [
        (
            "SSC1",
            {
                "nodes": 100,
                "interactions": 208,
                "mean_degree_star": 8.0,
                "mean_degree": 10.0,
                "mean_triangle_degree": 3.0,
                "clustering_1": 0.2325,
                "clustering_2": 0.2020,
            },
        ),
        (
            "SSC2",
            {
                "nodes": 100,
                "interactions": 256,
                "mean_degree_star": 10.0,
                "mean_degree": 12.0,
                "mean_triangle_degree": 4.0,
                "clustering_1": 0.1847,
                "clustering_2": 0.1762,
            },
        ),
        (
            "SSC3",
            {
                "nodes": 500,
                "interactions": 2170,
                "mean_degree_star": 15.0,
                "mean_degree": 10.0,
                "mean_triangle_degree": 5.0,
                "clustering_1": 0.2009,
                "clustering_2": 0.2020,
            },
        ),
        (
            "SSC4",
            {
                "nodes": 2000,
                "interactions": 9403,
                "mean_degree_star": 20.0,
                "mean_degree": 4.0,
                "mean_triangle_degree": 8.0,
                "clustering_1": 0.0552,
                "clustering_2": 0.0516,
            },
        ),
    ]
)

REAL_PROFILES = OrderedDict(
    [
        (
            "Email-Enron",
            {
                "folder": "email-Enron",
                "published_nodes": 143,
                "published_interactions": 1542,
                "published_size3_interactions": 6577,
            },
        ),
        (
            "NDC-classes-full",
            {
                "folder": "NDC-classes-full",
                "published_nodes": 1161,
                "published_interactions": 1224,
                "published_size3_interactions": 37810,
            },
        ),
        (
            "Primary-School",
            {
                "folder": "contact-primary-school",
                "published_nodes": 242,
                "published_interactions": 12799,
                "published_size3_interactions": 5138,
            },
        ),
        (
            "High-School",
            {
                "folder": "contact-high-school",
                "published_nodes": 327,
                "published_interactions": 7937,
                "published_size3_interactions": 2369,
            },
        ),
    ]
)

def _unique_random_groups(node_count, count, size, rng, occupied):
    groups = []
    attempts = 0
    while len(groups) < count:
        group = tuple(
            sorted(
                int(value)
                for value in rng.choice(node_count, size=size, replace=False)
            )
        )
        if group not in occupied:
            occupied.add(group)
            groups.append(group)
        attempts += 1
        if attempts > max(10000, count * 100):
            raise RuntimeError("unable to generate enough unique synthetic groups")
    return groups

def build_paper_ssc_temporal(
    name,
    time_count=12,
    persistence=0.25,
    seed=0,
):

    if name not in PAPER_SSC_PROFILES:
        raise KeyError("unknown SSC profile: %s" % name)
    if time_count < 2:
        raise ValueError("time_count must be at least 2")
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence must be in [0, 1]")

    profile = dict(PAPER_SSC_PROFILES[name])
    rng = np.random.RandomState(seed)
    node_count = int(profile["nodes"])
    interaction_count = int(profile["interactions"])
    triangle_count = min(
        interaction_count,
        int(round(node_count * profile["mean_triangle_degree"] / 3.0)),
    )
    pair_count = interaction_count - triangle_count
    occupied = set()
    groups = _unique_random_groups(
        node_count, triangle_count, 3, rng, occupied
    )
    groups.extend(
        _unique_random_groups(node_count, pair_count, 2, rng, occupied)
    )
    rng.shuffle(groups)

    snapshots = {time_point: [] for time_point in range(time_count)}
    for index, group in enumerate(groups):
        start = index % time_count
        snapshots[start].append(group)
        if start + 1 < time_count and rng.random_sample() < persistence:
            snapshots[start + 1].append(group)

    temporal = TemporalHypergraph(
        range(node_count),
        {time_point: tuple(items) for time_point, items in snapshots.items()},
    )
    event_count = sum(
        len(temporal.active_hyperedges(time_point))
        for time_point in temporal.time_points
    )
    metadata = {
        "name": name,
        "kind": "synthetic",
        "nodes": node_count,
        "time_points": time_count,
        "events": event_count,
        "unique_aggregate_interactions": interaction_count,
        "pair_interactions": pair_count,
        "triangle_interactions": triangle_count,
        "published_profile": profile,
        "construction": (
            "deterministic configuration-style regeneration from published "
            "N, |E| and mean triangle degree; aggregate interactions assigned "
            "to temporal snapshots with one-step persistence"
        ),
        "exact_original_realization_available": False,
    }
    return temporal, metadata

def _read_ints(path):
    with open(str(path), "r") as handle:
        return [int(line.strip()) for line in handle if line.strip()]

def load_real_temporal(
    name,
    max_events=600,
    time_count=12,
    max_hyperedge_size=6,
):

    if name not in REAL_PROFILES:
        raise KeyError("unknown real-data profile: %s" % name)
    profile = dict(REAL_PROFILES[name])
    folder = project_root() / "data" / profile["folder"]
    prefix = profile["folder"]
    sizes = _read_ints(folder / (prefix + "-nverts.txt"))
    times = _read_ints(folder / (prefix + "-times.txt"))
    simplex_nodes = _read_ints(folder / (prefix + "-simplices.txt"))
    if len(sizes) != len(times):
        raise ValueError("nverts and times length mismatch for %s" % name)

    accepted = []
    cursor = 0
    for size, timestamp in zip(sizes, times):
        group = tuple(simplex_nodes[cursor : cursor + size])
        cursor += size
        if 2 <= size <= max_hyperedge_size:
            accepted.append((int(timestamp), group))
            if len(accepted) >= max_events:
                break
    if not accepted:
        raise ValueError("no events remain after filtering %s" % name)

    accepted.sort(key=lambda item: item[0])
    first_time = accepted[0][0]
    last_time = accepted[-1][0]
    span = max(1, last_time - first_time + 1)
    snapshots = {time_point: [] for time_point in range(time_count)}
    for timestamp, group in accepted:
        time_point = min(
            time_count - 1,
            int((timestamp - first_time) * time_count // span),
        )
        snapshots[time_point].append(group)
    nodes = sorted({node for _, group in accepted for node in group})
    temporal = TemporalHypergraph(
        nodes,
        {time_point: tuple(items) for time_point, items in snapshots.items()},
    )
    counts = Counter(
        tuple(sorted(group))
        for time_point in temporal.time_points
        for group in temporal.active_hyperedges(time_point)
    )
    metadata = {
        "name": name,
        "kind": "real",
        "nodes": len(nodes),
        "time_points": time_count,
        "events": len(accepted),
        "unique_aggregate_interactions": len(counts),
        "duplicate_event_count": int(sum(counts.values()) - len(counts)),
        "source_timestamp_min": first_time,
        "source_timestamp_max": last_time,
        "max_hyperedge_size": max_hyperedge_size,
        "published_profile": {
            key: value
            for key, value in profile.items()
            if key != "folder"
        },
        "construction": (
            "chronological timestamped event sample, filtered by maximum "
            "hyperedge size and rebinned into equal-duration temporal snapshots"
        ),
        "exact_original_realization_available": True,
    }
    return temporal, metadata

def load_all_validation_datasets(config):
    datasets = OrderedDict()
    base_seed = int(config["seed"])
    for offset, name in enumerate(PAPER_SSC_PROFILES):
        datasets[name] = build_paper_ssc_temporal(
            name,
            time_count=int(config["time_count"]),
            persistence=float(config["synthetic_persistence"]),
            seed=base_seed + offset,
        )
    for name in REAL_PROFILES:
        datasets[name] = load_real_temporal(
            name,
            max_events=int(config["real_max_events"]),
            time_count=int(config["time_count"]),
            max_hyperedge_size=int(config["real_max_hyperedge_size"]),
        )
    return datasets
