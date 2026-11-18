
from collections import Counter, OrderedDict

import numpy as np

from ..dynamics import (
    ExposureSemantics,
    HypergraphSIParameters,
    PairwiseSIParameters,
    SimplicialSIParameters,
    infection_probabilities,
    simulate_temporal_si,
)
from ..metrics import max_kernel_tv
from ..metrics.trajectory_distance import path_distribution_tv
from ..projection import (
    groups_to_multigraph_snapshot,
    groups_to_simplegraph_snapshot,
    hypergraph_to_simplicial_snapshot,
    project_temporal_hypergraph_to_multigraph,
    project_temporal_hypergraph_to_simplegraph,
    project_temporal_hypergraph_to_simplicial,
    project_temporal_simplicial_to_multigraph,
)
from ..temporal_structures import TemporalHypergraph, TemporalSimplicialComplex
from ..theory import (
    C3ProjectionRoute,
    all_binary_states,
    build_exact_transition_kernel,
    build_temporal_c1_lambda_rates,
    build_temporal_c3_lambda_rates,
    trajectory_distribution,
)
from ._common import initial_distribution
from .exp_e5_window_size import run_experiment as run_window_counterexample

def build_representations(temporal_hypergraph, kappa):
    simplicial = project_temporal_hypergraph_to_simplicial(
        temporal_hypergraph, kappa
    )
    multigraph_direct = project_temporal_hypergraph_to_multigraph(
        temporal_hypergraph
    )
    multigraph_via_simplicial = project_temporal_simplicial_to_multigraph(
        simplicial
    )
    simplegraph = project_temporal_hypergraph_to_simplegraph(
        temporal_hypergraph
    )
    return {
        "H": temporal_hypergraph,
        "C": simplicial,
        "MG_direct": multigraph_direct,
        "MG": multigraph_via_simplicial,
        "G": simplegraph,
    }

def _counter_error(left, right):
    left = Counter(tuple(item) for item in left)
    right = Counter(tuple(item) for item in right)
    return int(sum(abs(left[key] - right[key]) for key in set(left) | set(right)))

def _weight_error(left, right):
    keys = set(left) | set(right)
    return int(max([abs(left.get(key, 0) - right.get(key, 0)) for key in keys] or [0]))

def validate_structural_conversion(temporal_hypergraph, kappa):
    representations = build_representations(temporal_hypergraph, kappa)
    simplicial = representations["C"]
    multigraph_direct = representations["MG_direct"]
    multigraph_via = representations["MG"]
    simplegraph = representations["G"]
    cell_errors = []
    direct_weight_errors = []
    via_weight_errors = []
    simple_errors = []
    route_support_errors = []
    provenance_errors = []

    for time_point in temporal_hypergraph.time_points:
        groups = temporal_hypergraph.active_hyperedges(time_point)
        expected_cells = hypergraph_to_simplicial_snapshot(groups, kappa)
        actual_cells = simplicial.active_maximal_simplices(time_point)
        cell_errors.append(_counter_error(expected_cells, actual_cells))

        expected_direct_weights = groups_to_multigraph_snapshot(groups)
        direct_weight_errors.append(
            _weight_error(
                expected_direct_weights,
                multigraph_direct.edge_weights(time_point),
            )
        )
        expected_via_weights = groups_to_multigraph_snapshot(actual_cells)
        via_weight_errors.append(
            _weight_error(
                expected_via_weights,
                multigraph_via.edge_weights(time_point),
            )
        )

        expected_simple = set(groups_to_simplegraph_snapshot(groups))
        actual_simple = set(simplegraph.active_edges(time_point))
        simple_errors.append(len(expected_simple.symmetric_difference(actual_simple)))
        via_support = set(multigraph_via.active_edges(time_point, repeated=False))
        route_support_errors.append(
            len(expected_simple.symmetric_difference(via_support))
        )

        blocks = simplicial.active_source_groups(time_point)
        if len(blocks) != len(groups):
            provenance_errors.append(abs(len(blocks) - len(groups)))
        for hyperedge, block in zip(groups, blocks):
            expected_block = hypergraph_to_simplicial_snapshot(
                (hyperedge,), kappa
            )
            provenance_errors.append(_counter_error(expected_block, block))

    time_alignment = (
        temporal_hypergraph.time_points
        == simplicial.time_points
        == multigraph_direct.time_points
        == multigraph_via.time_points
        == simplegraph.time_points
    )
    node_alignment = all(
        tuple(network.nodes) == tuple(temporal_hypergraph.nodes)
        for network in representations.values()
    )
    result = {
        "snapshot_count": len(temporal_hypergraph.time_points),
        "hypergraph_event_count": int(
            sum(
                len(temporal_hypergraph.active_hyperedges(time_point))
                for time_point in temporal_hypergraph.time_points
            )
        ),
        "simplicial_cell_count": int(
            sum(
                len(simplicial.active_maximal_simplices(time_point))
                for time_point in simplicial.time_points
            )
        ),
        "direct_multigraph_edge_count_with_multiplicity": int(
            sum(
                len(multigraph_direct.active_edges(time_point, repeated=True))
                for time_point in multigraph_direct.time_points
            )
        ),
        "via_simplicial_multigraph_edge_count_with_multiplicity": int(
            sum(
                len(multigraph_via.active_edges(time_point, repeated=True))
                for time_point in multigraph_via.time_points
            )
        ),
        "simplegraph_edge_count": int(
            sum(
                len(simplegraph.active_edges(time_point))
                for time_point in simplegraph.time_points
            )
        ),
        "max_simplicial_multiset_error": max(cell_errors or [0]),
        "max_direct_multigraph_weight_error": max(direct_weight_errors or [0]),
        "max_via_simplicial_multigraph_weight_error": max(via_weight_errors or [0]),
        "max_simplegraph_support_error": max(simple_errors or [0]),
        "max_direct_indirect_simple_support_error": max(route_support_errors or [0]),
        "max_provenance_error": max(provenance_errors or [0]),
        "time_alignment": bool(time_alignment),
        "node_alignment": bool(node_alignment),
    }
    result["passed"] = bool(
        time_alignment
        and node_alignment
        and all(
            result[key] == 0
            for key in (
                "max_simplicial_multiset_error",
                "max_direct_multigraph_weight_error",
                "max_via_simplicial_multigraph_weight_error",
                "max_simplegraph_support_error",
                "max_direct_indirect_simple_support_error",
                "max_provenance_error",
            )
        )
    )
    return result, representations

def _activity_seed(temporal_hypergraph):
    activity = Counter(
        node
        for time_point in temporal_hypergraph.time_points
        for group in temporal_hypergraph.active_hyperedges(time_point)
        for node in group
    )
    return max(
        temporal_hypergraph.nodes,
        key=lambda node: (
            activity[node],
            -temporal_hypergraph.node_to_index[node],
        ),
    )

def representative_states(temporal_hypergraph, seed):
    rng = np.random.RandomState(seed)
    node_count = len(temporal_hypergraph.nodes)
    states = []
    first = np.zeros(node_count, dtype=np.int8)
    first[temporal_hypergraph.node_to_index[_activity_seed(temporal_hypergraph)]] = 1
    states.append(first)
    for fraction in (0.10, 0.35, 0.60):
        state = np.zeros(node_count, dtype=np.int8)
        count = max(1, min(node_count - 1, int(round(node_count * fraction))))
        indices = rng.choice(node_count, size=count, replace=False)
        state[indices] = 1
        states.append(state)
    return tuple(states)

def _beta_rates(time_points, beta_pairwise, beta_higher):
    rates = {}
    for time_point in time_points:
        rates[(time_point, 1)] = float(beta_pairwise)
        rates[(time_point, 2)] = float(beta_higher)
    return rates

def _pair_beta(time_points, beta_pairwise):
    return {time_point: float(beta_pairwise) for time_point in time_points}

def build_positive_parameters(representations, config):
    hypergraph = representations["H"]
    kappa = int(config["kappa"])
    nu = float(config["nu"])
    beta = float(config["beta_pairwise"])
    higher = float(config["beta_higher_order"])
    beta_c1 = _beta_rates(hypergraph.time_points, beta, higher)
    c1_rates = build_temporal_c1_lambda_rates(
        hypergraph,
        kappa,
        nu,
        beta_c1,
        exposure_semantics=ExposureSemantics.INDEPENDENT_EXPOSURES,
    )
    beta_c2 = _beta_rates(hypergraph.time_points, beta, 0.0)
    pair_beta = _pair_beta(hypergraph.time_points, beta)
    c3_via_rates = build_temporal_c3_lambda_rates(
        hypergraph,
        kappa,
        nu,
        pair_beta,
        exposure_semantics=ExposureSemantics.INDEPENDENT_EXPOSURES,
        route=C3ProjectionRoute.VIA_SIMPLICIAL,
    )
    c3_direct_rates = build_temporal_c3_lambda_rates(
        hypergraph,
        kappa,
        nu,
        pair_beta,
        exposure_semantics=ExposureSemantics.INDEPENDENT_EXPOSURES,
        route=C3ProjectionRoute.DIRECT,
    )
    return {
        "H_C1": HypergraphSIParameters(c1_rates, nu),
        "C_C1": SimplicialSIParameters(
            beta_c1, ExposureSemantics.INDEPENDENT_EXPOSURES
        ),
        "C_C2": SimplicialSIParameters(
            beta_c2, ExposureSemantics.INDEPENDENT_EXPOSURES
        ),
        "MG": PairwiseSIParameters(
            pair_beta, ExposureSemantics.INDEPENDENT_EXPOSURES
        ),
        "H_C3_via": HypergraphSIParameters(c3_via_rates, nu),
        "H_C3_direct": HypergraphSIParameters(c3_direct_rates, nu),
        "beta_c1": beta_c1,
        "pair_beta": pair_beta,
    }

def max_probability_error(
    network_left,
    params_left,
    network_right,
    params_right,
    states,
):
    if network_left.time_points != network_right.time_points:
        raise ValueError("time points must be aligned")
    maximum = 0.0
    for time_point in network_left.time_points:
        for state in states:
            left = infection_probabilities(
                state, network_left, time_point, params_left
            )
            right = infection_probabilities(
                state, network_right, time_point, params_right
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(left - right))) if len(left) else 0.0,
            )
    return maximum

def validate_dataset_dynamics(representations, config, seed):
    hypergraph = representations["H"]
    simplicial = representations["C"]
    multigraph = representations["MG"]
    multigraph_direct = representations["MG_direct"]
    parameters = build_positive_parameters(representations, config)
    states = representative_states(hypergraph, seed)
    errors = {
        "H_to_C_C1": max_probability_error(
            hypergraph,
            parameters["H_C1"],
            simplicial,
            parameters["C_C1"],
            states,
        ),
        "C_to_MG_C2": max_probability_error(
            simplicial,
            parameters["C_C2"],
            multigraph,
            parameters["MG"],
            states,
        ),
        "H_to_C_to_MG_C3": max_probability_error(
            hypergraph,
            parameters["H_C3_via"],
            multigraph,
            parameters["MG"],
            states,
        ),
        "H_to_MG_direct_C3": max_probability_error(
            hypergraph,
            parameters["H_C3_direct"],
            multigraph_direct,
            parameters["MG"],
            states,
        ),
    }

    initial = states[0]
    rng = np.random.RandomState(seed + 1000)
    uniforms = rng.random_sample(
        (len(hypergraph.time_points), len(hypergraph.nodes))
    )
    pairs = OrderedDict(
        [
            (
                "H_to_C_C1",
                (
                    hypergraph,
                    parameters["H_C1"],
                    simplicial,
                    parameters["C_C1"],
                ),
            ),
            (
                "C_to_MG_C2",
                (
                    simplicial,
                    parameters["C_C2"],
                    multigraph,
                    parameters["MG"],
                ),
            ),
            (
                "H_to_C_to_MG_C3",
                (
                    hypergraph,
                    parameters["H_C3_via"],
                    multigraph,
                    parameters["MG"],
                ),
            ),
            (
                "H_to_MG_direct_C3",
                (
                    hypergraph,
                    parameters["H_C3_direct"],
                    multigraph_direct,
                    parameters["MG"],
                ),
            ),
        ]
    )
    path_matches = {}
    endpoint_densities = {}
    for label, (left_net, left_params, right_net, right_params) in pairs.items():
        left_result = simulate_temporal_si(
            left_net, left_params, initial, uniforms=uniforms
        )
        right_result = simulate_temporal_si(
            right_net, right_params, initial, uniforms=uniforms
        )
        path_matches[label] = bool(
            np.array_equal(left_result.states, right_result.states)
        )
        endpoint_densities[label] = {
            "source": float(left_result.infection_density[-1]),
            "target": float(right_result.infection_density[-1]),
        }
    tolerance = float(config["exact_tolerance"])
    return {
        "max_conditional_probability_error": errors,
        "common_random_path_match": path_matches,
        "endpoint_density": endpoint_densities,
        "representative_state_count": len(states),
        "passed": bool(
            max(errors.values()) < tolerance
            and all(path_matches.values())
        ),
    }

def _kernel_sequence(network, parameters):
    return [
        build_exact_transition_kernel(network, time_point, parameters)
        for time_point in network.time_points
    ]

def _kernel_and_path_error(
    initial,
    kernels_left,
    kernels_right,
):
    kernel_error = max(
        max_kernel_tv(left, right)
        for left, right in zip(kernels_left, kernels_right)
    )
    paths_left = trajectory_distribution(initial, kernels_left)
    paths_right = trajectory_distribution(initial, kernels_right)
    return {
        "max_kernel_tv": float(kernel_error),
        "path_distribution_tv": float(
            path_distribution_tv(paths_left, paths_right)
        ),
    }

def run_exact_exhaustive_validation(config):
    hypergraph = TemporalHypergraph(
        range(5),
        {
            0: [(0, 1, 2, 3)],
            1: [(1, 2, 4), (0, 3)],
            2: [(0, 2, 3), (1, 4)],
        },
    )
    representations = build_representations(hypergraph, int(config["kappa"]))
    parameters = build_positive_parameters(representations, config)
    initial = initial_distribution(5, (0,))
    sequences = {
        "H_C1": _kernel_sequence(hypergraph, parameters["H_C1"]),
        "C_C1": _kernel_sequence(
            representations["C"], parameters["C_C1"]
        ),
        "C_C2": _kernel_sequence(
            representations["C"], parameters["C_C2"]
        ),
        "MG": _kernel_sequence(
            representations["MG"], parameters["MG"]
        ),
        "H_C3_via": _kernel_sequence(
            hypergraph, parameters["H_C3_via"]
        ),
        "H_C3_direct": _kernel_sequence(
            hypergraph, parameters["H_C3_direct"]
        ),
        "MG_direct": _kernel_sequence(
            representations["MG_direct"], parameters["MG"]
        ),
    }
    conversions = {
        "H_to_C_C1": _kernel_and_path_error(
            initial, sequences["H_C1"], sequences["C_C1"]
        ),
        "C_to_MG_C2": _kernel_and_path_error(
            initial, sequences["C_C2"], sequences["MG"]
        ),
        "H_to_C_to_MG_C3": _kernel_and_path_error(
            initial, sequences["H_C3_via"], sequences["MG"]
        ),
        "H_to_MG_direct_C3": _kernel_and_path_error(
            initial, sequences["H_C3_direct"], sequences["MG_direct"]
        ),
    }

    pairwise = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1), (2, 3)],
            1: [(1, 2)],
        },
    )
    pair_multi = project_temporal_hypergraph_to_multigraph(pairwise)
    pair_simple = project_temporal_hypergraph_to_simplegraph(pairwise)
    pair_beta = _pair_beta(pairwise.time_points, config["beta_pairwise"])
    pair_params = PairwiseSIParameters(
        pair_beta, ExposureSemantics.INDEPENDENT_EXPOSURES
    )
    lossless_error = max(
        max_kernel_tv(left, right)
        for left, right in zip(
            _kernel_sequence(pair_multi, pair_params),
            _kernel_sequence(pair_simple, pair_params),
        )
    )

    lossy = TemporalHypergraph(
        range(4),
        {0: [(0, 1, 2), (0, 1, 3)]},
    )
    lossy_multi = project_temporal_hypergraph_to_multigraph(lossy)
    lossy_simple = project_temporal_hypergraph_to_simplegraph(lossy)
    lossy_params = PairwiseSIParameters(
        {0: float(config["beta_pairwise"])},
        ExposureSemantics.INDEPENDENT_EXPOSURES,
    )
    lossy_error = max_kernel_tv(
        build_exact_transition_kernel(lossy_multi, 0, lossy_params),
        build_exact_transition_kernel(lossy_simple, 0, lossy_params),
    )
    tolerance = float(config["exact_tolerance"])
    result = {
        "node_count": 5,
        "state_count": len(all_binary_states(5)),
        "time_points": 3,
        "conversions": conversions,
        "MG_to_G_no_repeated_edges_kernel_tv": float(lossless_error),
        "MG_to_G_repeated_edge_loss_kernel_tv": float(lossy_error),
    }
    result["passed"] = bool(
        max(
            metric
            for values in conversions.values()
            for metric in values.values()
        )
        < tolerance
        and lossless_error < tolerance
        and lossy_error > tolerance
    )
    return result

def _shift_simplicial(temporal_simplicial):
    points = temporal_simplicial.time_points
    snapshots = {}
    sources = {}
    source_events = {}
    for index, time_point in enumerate(points):
        source_time = points[(index + 1) % len(points)]
        snapshots[time_point] = temporal_simplicial.active_maximal_simplices(
            source_time
        )
        sources[time_point] = temporal_simplicial.active_source_groups(
            source_time
        )
        source_events[time_point] = temporal_simplicial.active_source_events(
            source_time
        )
    return TemporalSimplicialComplex(
        temporal_simplicial.nodes,
        snapshots,
        max_order=temporal_simplicial.max_order,
        source_groups=sources,
        source_events=source_events,
    )

def validate_negative_controls(representations, config, seed):
    hypergraph = representations["H"]
    simplicial = representations["C"]
    multigraph = representations["MG"]
    simplegraph = representations["G"]
    parameters = build_positive_parameters(representations, config)
    states = representative_states(hypergraph, seed)

    wrong_rates = {
        key: float(value) * float(config["wrong_c1_scale"])
        for key, value in parameters["H_C1"].lambda_rates.items()
    }
    wrong_h = HypergraphSIParameters(wrong_rates, float(config["nu"]))
    c1_error = max_probability_error(
        hypergraph,
        wrong_h,
        simplicial,
        parameters["C_C1"],
        states,
    )

    wrong_c2_rates = _beta_rates(
        hypergraph.time_points,
        config["beta_pairwise"],
        config["negative_triangle_beta"],
    )
    wrong_c = SimplicialSIParameters(
        wrong_c2_rates, ExposureSemantics.INDEPENDENT_EXPOSURES
    )
    c2_error = max_probability_error(
        simplicial,
        wrong_c,
        multigraph,
        parameters["MG"],
        states,
    )

    shifted = _shift_simplicial(simplicial)
    shift_error = max_probability_error(
        hypergraph,
        parameters["H_C1"],
        shifted,
        parameters["C_C1"],
        states,
    )

    simple_params = PairwiseSIParameters(
        parameters["pair_beta"],
        ExposureSemantics.INDEPENDENT_EXPOSURES,
    )
    multiplicity_error = max_probability_error(
        representations["MG_direct"],
        simple_params,
        simplegraph,
        simple_params,
        states,
    )
    tolerance = float(config["exact_tolerance"])
    return {
        "wrong_C1_probability_error": float(c1_error),
        "nonzero_higher_order_C2_error": float(c2_error),
        "one_step_time_shift_error": float(shift_error),
        "MG_to_G_multiplicity_loss_error": float(multiplicity_error),
        "detected_control_count": int(
            sum(
                value > tolerance
                for value in (
                    c1_error,
                    c2_error,
                    shift_error,
                    multiplicity_error,
                )
            )
        ),
        "passed": bool(c1_error > tolerance and c2_error > tolerance),
    }

def _reorder_hypergraph(temporal_hypergraph, permutation):
    points = temporal_hypergraph.time_points
    snapshots = {
        time_point: temporal_hypergraph.active_hyperedges(points[source_index])
        for time_point, source_index in zip(points, permutation)
    }
    return TemporalHypergraph(temporal_hypergraph.nodes, snapshots)

def validate_temporal_reordering(
    temporal_hypergraph,
    original_representations,
    config,
    seed,
):
    rng = np.random.RandomState(seed)
    states = representative_states(temporal_hypergraph, seed + 50)
    original_parameters = build_positive_parameters(
        original_representations, config
    )
    aligned_errors = []
    misaligned_errors = []
    source_order_effects = []
    time_count = len(temporal_hypergraph.time_points)
    for _ in range(int(config["reorder_count"])):
        permutation = tuple(
            int(value) for value in rng.permutation(time_count)
        )
        reordered = _reorder_hypergraph(temporal_hypergraph, permutation)
        reordered_representations = build_representations(
            reordered, int(config["kappa"])
        )
        reordered_parameters = build_positive_parameters(
            reordered_representations, config
        )
        aligned_errors.append(
            max_probability_error(
                reordered,
                reordered_parameters["H_C1"],
                reordered_representations["C"],
                reordered_parameters["C_C1"],
                states,
            )
        )
        misaligned_errors.append(
            max_probability_error(
                reordered,
                reordered_parameters["H_C1"],
                original_representations["C"],
                original_parameters["C_C1"],
                states,
            )
        )
        source_order_effects.append(
            max_probability_error(
                reordered,
                reordered_parameters["H_C1"],
                temporal_hypergraph,
                original_parameters["H_C1"],
                states,
            )
        )
    tolerance = float(config["exact_tolerance"])
    return {
        "reorder_count": int(config["reorder_count"]),
        "aligned_conversion_max_error": float(max(aligned_errors or [0.0])),
        "misaligned_conversion_mean_error": float(
            np.mean(misaligned_errors)
        ),
        "misaligned_conversion_max_error": float(
            max(misaligned_errors or [0.0])
        ),
        "source_order_effect_mean": float(np.mean(source_order_effects)),
        "source_order_effect_max": float(
            max(source_order_effects or [0.0])
        ),
        "passed": bool(
            max(aligned_errors or [0.0]) < tolerance
            and max(misaligned_errors or [0.0]) > tolerance
        ),
    }

def aggregate_temporal_hypergraph(temporal_hypergraph, window_size):
    points = temporal_hypergraph.time_points
    snapshots = {}
    for start in range(0, len(points), window_size):
        events = []
        for time_point in points[start : start + window_size]:
            events.extend(temporal_hypergraph.active_hyperedges(time_point))
        snapshots[start // window_size] = tuple(events)
    return TemporalHypergraph(temporal_hypergraph.nodes, snapshots)

def _mean_endpoint_density(
    temporal_hypergraph,
    config,
    simulations,
    seed,
):
    representations = build_representations(
        temporal_hypergraph, int(config["kappa"])
    )
    parameters = build_positive_parameters(representations, config)
    initial = representative_states(temporal_hypergraph, seed)[0]
    densities = []
    for replicate in range(simulations):
        result = simulate_temporal_si(
            temporal_hypergraph,
            parameters["H_C1"],
            initial,
            seed=seed + 10000 + replicate,
        )
        densities.append(float(result.infection_density[-1]))
    return float(np.mean(densities)), representations, parameters, initial

def validate_window_sensitivity(temporal_hypergraph, config, seed):
    rows = []
    baseline_density = None
    for window_size in config["window_sizes"]:
        aggregated = aggregate_temporal_hypergraph(
            temporal_hypergraph, int(window_size)
        )
        mean_density, representations, parameters, initial = (
            _mean_endpoint_density(
                aggregated,
                config,
                int(config["window_simulations"]),
                seed,
            )
        )
        if baseline_density is None:
            baseline_density = mean_density
        states = representative_states(aggregated, seed + int(window_size))
        conversion_error = max_probability_error(
            aggregated,
            parameters["H_C1"],
            representations["C"],
            parameters["C_C1"],
            states,
        )
        rng = np.random.RandomState(seed + 5000 + int(window_size))
        uniforms = rng.random_sample(
            (len(aggregated.time_points), len(aggregated.nodes))
        )
        source_path = simulate_temporal_si(
            aggregated,
            parameters["H_C1"],
            initial,
            uniforms=uniforms,
        )
        target_path = simulate_temporal_si(
            representations["C"],
            parameters["C_C1"],
            initial,
            uniforms=uniforms,
        )
        event_counts = [
            len(aggregated.active_hyperedges(time_point))
            for time_point in aggregated.time_points
        ]
        rows.append(
            {
                "window_size": int(window_size),
                "snapshot_count": len(aggregated.time_points),
                "mean_events_per_snapshot": float(np.mean(event_counts)),
                "max_events_per_snapshot": int(max(event_counts or [0])),
                "mean_endpoint_density": mean_density,
                "endpoint_density_change_from_window1": float(
                    mean_density - baseline_density
                ),
                "H_to_C_probability_error": float(conversion_error),
                "common_random_path_match": bool(
                    np.array_equal(source_path.states, target_path.states)
                ),
            }
        )
    tolerance = float(config["exact_tolerance"])
    return {
        "rows": rows,
        "exact_small_network_counterexample": run_window_counterexample(),
        "passed": bool(
            max(row["H_to_C_probability_error"] for row in rows)
            < tolerance
            and all(row["common_random_path_match"] for row in rows)
        ),
    }
