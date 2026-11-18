
from pathlib import Path
from collections import Counter

import numpy as np

from ..dynamics import (
    ExposureSemantics,
    HypergraphSIParameters,
    SimplicialSIParameters,
    simulate_temporal_si,
)
from ..projection import project_temporal_hypergraph_to_simplicial
from ..temporal_structures import TemporalHypergraph
from ..theory import build_temporal_c1_lambda_rates
from ._common import common_parser, project_root, write_result

DATASETS = {
    "high-school": "contact-high-school",
    "primary-school": "contact-primary-school",
    "email-enron": "email-Enron",
    "ndc": "NDC-classes-full",
}

def _paths(dataset):
    folder_name = DATASETS[dataset]
    folder = project_root() / "data" / folder_name
    prefix = folder_name
    return (
        folder / (prefix + "-nverts.txt"),
        folder / (prefix + "-simplices.txt"),
        folder / (prefix + "-times.txt"),
    )

def _filter_max_size(temporal, max_hyperedge_size):
    snapshots = {
        time_point: tuple(
            edge
            for edge in temporal.active_hyperedges(time_point)
            if len(edge) <= max_hyperedge_size
        )
        for time_point in temporal.time_points
    }
    snapshots = {time: edges for time, edges in snapshots.items() if edges}
    nodes = sorted({node for edges in snapshots.values() for edge in edges for node in edge})
    return TemporalHypergraph(nodes, snapshots)

def run_experiment(
    dataset="high-school",
    max_events=300,
    window_size=20,
    max_hyperedge_size=8,
    seed=0,
):
    nverts, simplices, times = _paths(dataset)
    temporal = TemporalHypergraph.from_benson_files(
        str(nverts),
        str(simplices),
        str(times),
        window_size=window_size,
        max_events=max_events,
    )
    temporal = _filter_max_size(temporal, max_hyperedge_size)
    if not temporal.time_points:
        raise ValueError("no events remain after filtering")
    kappa = 2
    projected = project_temporal_hypergraph_to_simplicial(temporal, kappa)
    beta = {}
    for time_point in temporal.time_points:
        beta[(time_point, 1)] = 0.002
        beta[(time_point, 2)] = 0.001
    rates = build_temporal_c1_lambda_rates(
        temporal,
        kappa,
        2.0,
        beta,
        exposure_semantics=ExposureSemantics.INDEPENDENT_EXPOSURES,
    )
    params_h = HypergraphSIParameters(rates, 2.0)
    params_c = SimplicialSIParameters(beta, ExposureSemantics.INDEPENDENT_EXPOSURES)
    initial = np.zeros(len(temporal.nodes), dtype=np.int8)
    activity = Counter(
        node
        for time_point in temporal.time_points
        for edge in temporal.active_hyperedges(time_point)
        for node in edge
    )
    seed_node = max(temporal.nodes, key=lambda node: (activity[node], -temporal.node_to_index[node]))
    initial[temporal.node_to_index[seed_node]] = 1
    rng = np.random.RandomState(seed)
    uniforms = rng.random_sample((len(temporal.time_points), len(temporal.nodes)))
    result_h = simulate_temporal_si(temporal, params_h, initial, uniforms=uniforms)
    result_c = simulate_temporal_si(projected, params_c, initial, uniforms=uniforms)
    max_probability_error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(
            result_h.conditional_probabilities, result_c.conditional_probabilities
        )
    )
    return {
        "experiment": "E9",
        "dataset": dataset,
        "nodes": len(temporal.nodes),
        "time_points": len(temporal.time_points),
        "events": sum(len(temporal.active_hyperedges(t)) for t in temporal.time_points),
        "initial_seed_node": seed_node,
        "max_conditional_probability_error": max_probability_error,
        "sample_paths_identical_under_common_random_numbers": bool(
            np.array_equal(result_h.states, result_c.states)
        ),
        "hypergraph_density": result_h.infection_density,
        "simplicial_density": result_c.infection_density,
    }

def main():
    parser = common_parser(__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="high-school")
    parser.add_argument("--max-events", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--max-hyperedge-size", type=int, default=8)
    args = parser.parse_args()
    result = run_experiment(
        args.dataset,
        args.max_events,
        args.window_size,
        args.max_hyperedge_size,
        args.seed,
    )
    path = write_result("e9_real_data_" + args.dataset, result, args.output_dir)
    print(path)

if __name__ == "__main__":
    main()
