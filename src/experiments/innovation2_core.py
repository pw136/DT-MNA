
from __future__ import division

import math
import time
from collections import Counter, OrderedDict

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ..approximate import (
    adaptive_model_sequence,
    build_error_certificate,
    check_scalar_qstar_feasibility,
    cvar,
    fit_dynamics_aware_pairwise,
    fit_nonnegative_mapping,
    fit_pairwise_hazard,
    fit_piecewise_scalar,
    fit_temporal_scalar,
    pairwise_effect,
    pareto_front,
    projection_collision_lower_bound,
    required_pairwise_beta,
    select_minimal_surrogate,
)
from ..dynamics import (
    ExposureSemantics,
    HypergraphSIParameters,
    PairwiseSIParameters,
    SimplicialSIParameters,
    infection_probabilities,
    simulate_temporal_si,
)
from ..dynamics.sis_temporal import SISParameters
from ..metrics import distribution_tv, kernel_row_tv
from ..projection import (
    project_temporal_hypergraph_to_multigraph,
    project_temporal_hypergraph_to_simplegraph,
    project_temporal_hypergraph_to_simplicial,
)
from ..temporal_structures import TemporalHypergraph
from ..theory import all_binary_states, build_exact_transition_kernel, propagate_distribution
from ._common import initial_distribution
from .conversion_validation_data import load_all_validation_datasets

INDEPENDENT = ExposureSemantics.INDEPENDENT_EXPOSURES

def _json_dataclass(value):
    return {
        key: item
        for key, item in value.__dict__.items()
    }

def _channel_direction(channels):
    if not channels:
        return np.asarray([], dtype=float)
    if len(channels) == 1:
        return np.asarray([0.0], dtype=float)
    raw = np.asarray(
        [
            0.71 * (size - 2)
            + 0.43 * (infected - 1)
            + 0.29 * (1 if (size + infected) % 2 else -1)
            for size, infected in channels
        ],
        dtype=float,
    )
    raw -= raw.mean()
    deviation = raw.std()
    if deviation <= 1e-15:
        raw = np.linspace(-1.0, 1.0, len(channels))
        deviation = raw.std()
    if deviation <= 1e-15:
        return np.zeros(len(channels), dtype=float)
    return raw / deviation

def _all_channels(temporal_hypergraph, time_point):
    sizes = sorted(
        set(
            len(group)
            for group in temporal_hypergraph.active_hyperedges(time_point)
        )
    )
    return tuple(
        (size, infected)
        for size in sizes
        for infected in range(1, size)
    )

def build_conflicting_lambda_rates(
    temporal_hypergraph,
    nu,
    base_beta,
    conflict_strength,
    time_drift=0.0,
):

    rates = {}
    required = {}
    time_count = max(1, len(temporal_hypergraph.time_points))
    for position, time_point in enumerate(temporal_hypergraph.time_points):
        channels = _all_channels(temporal_hypergraph, time_point)
        directions = _channel_direction(channels)
        temporal_shift = float(time_drift) * math.sin(
            2.0 * math.pi * position / float(time_count)
        )
        for channel_index, (size, infected) in enumerate(channels):
            beta = (
                float(base_beta)
                + temporal_shift
                + float(conflict_strength) * directions[channel_index]
            )
            beta = min(0.35, max(0.002, beta))
            effect = float(pairwise_effect(beta, [infected])[0])
            rates[(time_point, size, infected)] = effect / float(infected ** nu)
            required[(time_point, size, infected)] = beta
    return rates, required

def _structural_channel_data(
    temporal_hypergraph,
    rates,
    nu,
    infection_fraction_by_time=None,
):
    result = OrderedDict()
    for position, time_point in enumerate(temporal_hypergraph.time_points):
        if infection_fraction_by_time is None:
            fraction = min(0.45, 0.06 + 0.03 * position)
        else:
            fraction = float(infection_fraction_by_time[position])
        size_counts = Counter(
            len(group)
            for group in temporal_hypergraph.active_hyperedges(time_point)
        )
        effects = []
        exposures = []
        weights = []
        channels = []
        for size in sorted(size_counts):
            group_count = size_counts[size]
            for infected in range(1, size):
                combination = (
                    math.factorial(size - 1)
                    // (
                        math.factorial(infected)
                        * math.factorial(size - 1 - infected)
                    )
                )
                probability = (
                    combination
                    * (fraction ** infected)
                    * ((1.0 - fraction) ** (size - 1 - infected))
                )
                weight = max(
                    1e-12,
                    group_count * size * (1.0 - fraction) * probability,
                )
                effects.append(
                    rates[(time_point, size, infected)]
                    * float(infected ** nu)
                )
                exposures.append(float(infected))
                weights.append(weight)
                channels.append((size, infected))
        if not effects:
            effects = [0.0]
            exposures = [0.0]
            weights = [1e-12]
            channels = [(0, 0)]
        result[time_point] = {
            "effects": np.asarray(effects, dtype=float),
            "exposures": np.asarray(exposures, dtype=float),
            "weights": np.asarray(weights, dtype=float),
            "channels": tuple(channels),
        }
    return result

def _fit_structural_betas(channel_data, mode="time", smoothness=0.0):
    points = tuple(channel_data.keys())
    if mode == "global":
        effects = np.concatenate([channel_data[t]["effects"] for t in points])
        exposures = np.concatenate([channel_data[t]["exposures"] for t in points])
        weights = np.concatenate([channel_data[t]["weights"] for t in points])
        active = exposures > 0.0
        fit = fit_pairwise_hazard(
            effects[active], exposures[active], weights[active]
        )
        return {time_point: float(fit.parameters) for time_point in points}
    hazards = [
        -np.log1p(-np.clip(channel_data[t]["effects"], 0.0, 1.0 - 1e-15))
        for t in points
    ]
    designs = [channel_data[t]["exposures"] for t in points]
    weights = [channel_data[t]["weights"] for t in points]
    if mode == "piecewise":
        parameters, _, _ = fit_piecewise_scalar(
            hazards,
            designs,
            weights,
            switch_penalty=float(smoothness),
            bounds=(0.0, -math.log(0.05)),
        )
    else:
        parameters = fit_temporal_scalar(
            hazards,
            designs,
            weights,
            smoothness=float(smoothness),
            bounds=(0.0, -math.log(0.05)),
        )
    betas = 1.0 - np.exp(-parameters)
    return {
        time_point: float(betas[position])
        for position, time_point in enumerate(points)
    }

def _small_temporal_hypergraph():
    return TemporalHypergraph(
        range(7),
        {
            0: [(0, 1, 2), (0, 1, 2), (2, 3, 4, 5)],
            1: [(0, 2, 3), (1, 3, 4, 5), (0, 5, 6)],
            2: [(0, 1, 4, 6), (1, 2, 3), (2, 3, 4)],
            3: [(0, 3, 5), (0, 3, 5), (1, 2, 4, 6)],
            4: [(0, 1, 2, 5), (2, 4, 6), (1, 3, 5)],
        },
    )

def run_a0_exact_recovery(nu=2.0):
    hypergraph = TemporalHypergraph(
        range(5),
        {
            0: [(0, 1, 2, 3), (1, 4)],
            1: [(0, 2, 4), (1, 2, 3)],
            2: [(0, 1, 3, 4), (2, 4)],
        },
    )
    multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
    rates = {}
    beta_by_time = {}
    feasibility = {}
    fitted = {}
    for position, time_point in enumerate(hypergraph.time_points):
        beta = 0.035 + 0.012 * position
        beta_by_time[time_point] = beta
        channels = _all_channels(hypergraph, time_point)
        effects = []
        exposures = []
        required = []
        for size, infected in channels:
            effect = float(pairwise_effect(beta, [infected])[0])
            rates[(time_point, size, infected)] = effect / float(infected ** nu)
            effects.append(effect)
            exposures.append(infected)
            required.append(beta)
        check = check_scalar_qstar_feasibility(required, np.ones(len(required)))
        fit = fit_pairwise_hazard(effects, exposures)
        feasibility[time_point] = _json_dataclass(check)
        fitted[time_point] = float(fit.parameters)
    parameters_h = HypergraphSIParameters(rates, nu=nu)
    parameters_m = PairwiseSIParameters(fitted, INDEPENDENT)
    kernels_h = [
        build_exact_transition_kernel(hypergraph, t, parameters_h)
        for t in hypergraph.time_points
    ]
    kernels_m = [
        build_exact_transition_kernel(multigraph, t, parameters_m)
        for t in hypergraph.time_points
    ]
    initial = initial_distribution(len(hypergraph.nodes), (0,))
    distributions_h = propagate_distribution(initial, kernels_h)
    distributions_m = propagate_distribution(initial, kernels_m)
    kernel_errors = [
        float(kernel_row_tv(left, right).max())
        for left, right in zip(kernels_h, kernels_m)
    ]
    endpoint_error = distribution_tv(distributions_h[-1], distributions_m[-1])
    parameter_error = max(
        abs(beta_by_time[t] - fitted[t])
        for t in hypergraph.time_points
    )
    return {
        "experiment": "A0",
        "beta_true": beta_by_time,
        "beta_fitted": fitted,
        "max_parameter_error": float(parameter_error),
        "per_time_max_kernel_tv": kernel_errors,
        "max_kernel_tv": float(max(kernel_errors)),
        "endpoint_tv": float(endpoint_error),
        "feasibility": feasibility,
        "passed": bool(
            parameter_error < 1e-10
            and max(kernel_errors) < 1e-10
            and endpoint_error < 1e-10
        ),
    }

def _effect_metrics(beta, effects, exposures, weights):
    predictions = pairwise_effect(beta, exposures)
    residuals = np.abs(predictions - effects)
    normalized = weights / weights.sum()
    return {
        "mae": float(np.dot(normalized, residuals)),
        "rmse": float(np.sqrt(np.dot(normalized, residuals ** 2))),
        "worst": float(residuals.max()),
    }

def run_a1_conflict_scan(conflict_levels, replicates=30, seed=0):
    rng = np.random.RandomState(seed)
    methods = (
        "Direct copy",
        "Arithmetic mean",
        "Frequency weighted",
        "Hazard projection",
        "Dynamics aware",
    )
    rows = []
    for conflict in conflict_levels:
        for replicate in range(int(replicates)):
            channel_count = 18
            exposures = rng.randint(1, 5, size=channel_count).astype(float)
            weights = np.exp(rng.normal(0.0, 0.9, size=channel_count))
            direction = rng.normal(size=channel_count)
            direction = (direction - direction.mean()) / direction.std()
            required = np.clip(0.06 + float(conflict) * direction, 0.002, 0.38)
            effects = pairwise_effect(required, exposures)
            feasibility = check_scalar_qstar_feasibility(
                required, np.ones(channel_count)
            )
            collision_lb, _ = projection_collision_lower_bound(
                effects,
                tuple(int(value) for value in exposures),
            )
            beta_values = OrderedDict()
            beta_values["Direct copy"] = float(required[0])
            beta_values["Arithmetic mean"] = float(required.mean())
            beta_values["Frequency weighted"] = float(np.average(required, weights=weights))
            beta_values["Hazard projection"] = float(
                fit_pairwise_hazard(effects, exposures, weights).parameters
            )
            beta_values["Dynamics aware"] = float(
                fit_dynamics_aware_pairwise(
                    effects,
                    exposures,
                    weights,
                    tail_weight=0.35,
                    tail_quantile=0.90,
                ).parameters
            )
            for method in methods:
                metrics = _effect_metrics(
                    beta_values[method], effects, exposures, weights
                )
                rows.append(
                    {
                        "conflict_target": float(conflict),
                        "conflict_observed": float(required.std()),
                        "replicate": int(replicate),
                        "method": method,
                        "beta": beta_values[method],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "worst": metrics["worst"],
                        "collision_lower_bound": float(collision_lb),
                        "qstar_feasible": bool(feasibility.feasible),
                    }
                )
    summary = []
    for conflict in conflict_levels:
        for method in methods:
            subset = [
                row
                for row in rows
                if row["conflict_target"] == float(conflict)
                and row["method"] == method
            ]
            summary.append(
                {
                    "conflict": float(conflict),
                    "method": method,
                    "mean_mae": float(np.mean([row["mae"] for row in subset])),
                    "sd_mae": float(np.std([row["mae"] for row in subset])),
                    "mean_worst": float(np.mean([row["worst"] for row in subset])),
                    "mean_lower_bound": float(
                        np.mean([row["collision_lower_bound"] for row in subset])
                    ),
                    "qstar_feasible_rate": float(
                        np.mean([row["qstar_feasible"] for row in subset])
                    ),
                }
            )
    return {
        "experiment": "A1",
        "methods": methods,
        "replicates": int(replicates),
        "rows": rows,
        "summary": summary,
    }

def _source_probability_cache(hypergraph, parameters, states):
    return {
        time_point: np.asarray(
            [
                infection_probabilities(state, hypergraph, time_point, parameters)
                for state in states
            ]
        )
        for time_point in hypergraph.time_points
    }

def _fit_pairwise_to_probability_cache(
    network,
    states,
    source_cache,
    time_points,
    mode,
):
    def time_objective(beta, time_point):
        params = PairwiseSIParameters({time_point: float(beta)}, INDEPENDENT)
        residuals = []
        for state_index, state in enumerate(states):
            predicted = infection_probabilities(state, network, time_point, params)
            susceptible = state == 0
            residuals.extend(
                (predicted[susceptible] - source_cache[time_point][state_index, susceptible]).tolist()
            )
        return float(np.mean(np.asarray(residuals) ** 2))

    if mode == "time":
        return {
            time_point: float(
                minimize_scalar(
                    lambda beta: time_objective(beta, time_point),
                    bounds=(0.0, 0.35),
                    method="bounded",
                    options={"xatol": 1e-10},
                ).x
            )
            for time_point in time_points
        }

    def global_objective(beta):
        return float(
            np.mean([time_objective(beta, time_point) for time_point in time_points])
        )

    beta = float(
        minimize_scalar(
            global_objective,
            bounds=(0.0, 0.35),
            method="bounded",
            options={"xatol": 1e-10},
        ).x
    )
    return {time_point: beta for time_point in time_points}

def _fit_simplicial_to_probability_cache(
    simplicial,
    states,
    source_cache,
    time_points,
    initial_beta_by_time,
):
    rates = {}
    for time_point in time_points:
        def objective(values):
            params = SimplicialSIParameters(
                {
                    (time_point, 1): float(values[0]),
                    (time_point, 2): float(values[1]),
                },
                INDEPENDENT,
            )
            residuals = []
            for state_index, state in enumerate(states):
                predicted = infection_probabilities(
                    state, simplicial, time_point, params
                )
                susceptible = state == 0
                residuals.extend(
                    (
                        predicted[susceptible]
                        - source_cache[time_point][state_index, susceptible]
                    ).tolist()
                )
            return float(np.mean(np.asarray(residuals) ** 2))

        result = minimize(
            objective,
            [initial_beta_by_time[time_point], 0.005],
            method="L-BFGS-B",
            bounds=[(0.0, 0.35), (0.0, 0.35)],
            options={"maxiter": 250, "ftol": 1e-14},
        )
        rates[(time_point, 1)] = float(result.x[0])
        rates[(time_point, 2)] = float(result.x[1])
    return rates

def _interaction_count(network):
    total = 0
    for time_point in network.time_points:
        if hasattr(network, "active_hyperedges"):
            total += len(network.active_hyperedges(time_point))
        elif hasattr(network, "active_maximal_simplices"):
            total += len(network.active_maximal_simplices(time_point))
        elif hasattr(network, "active_edges"):
            try:
                total += len(network.active_edges(time_point, repeated=True))
            except TypeError:
                total += len(network.active_edges(time_point))
    return int(total)

def _evaluate_small_candidate(
    name,
    network,
    parameters,
    source_kernels,
    initial,
    parameter_count,
):
    start = time.perf_counter()
    kernels = [
        build_exact_transition_kernel(network, t, parameters)
        for t in network.time_points
    ]
    elapsed = time.perf_counter() - start
    source_distribution = propagate_distribution(initial, source_kernels)
    candidate_distribution = propagate_distribution(initial, kernels)
    per_time = [
        float(kernel_row_tv(left, right).max())
        for left, right in zip(source_kernels, kernels)
    ]
    endpoint = distribution_tv(source_distribution[-1], candidate_distribution[-1])
    interactions = _interaction_count(network)
    return {
        "name": name,
        "network": network,
        "parameters": parameters,
        "kernels": kernels,
        "parameter_count": int(parameter_count),
        "interaction_count": interactions,
        "complexity": float(interactions + 10 * parameter_count),
        "runtime_seconds": float(elapsed),
        "max_kernel_tv": float(max(per_time)),
        "mean_kernel_tv": float(np.mean(per_time)),
        "endpoint_tv": float(endpoint),
        "density_trajectory": [
            float(np.dot(distribution, all_binary_states(len(network.nodes)).mean(axis=1)))
            for distribution in candidate_distribution
        ],
    }

def run_a2_surrogate_hierarchy(nu=2.0, error_budget=0.025):
    hypergraph = _small_temporal_hypergraph()
    multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
    simplegraph = project_temporal_hypergraph_to_simplegraph(hypergraph)
    simplicial = project_temporal_hypergraph_to_simplicial(hypergraph, 2)
    rates, required = build_conflicting_lambda_rates(
        hypergraph, nu, 0.055, 0.026, time_drift=0.014
    )
    parameters_h = HypergraphSIParameters(rates, nu=nu)
    states = all_binary_states(len(hypergraph.nodes))
    source_cache = _source_probability_cache(hypergraph, parameters_h, states)
    time_points = hypergraph.time_points
    beta_g_global = _fit_pairwise_to_probability_cache(
        simplegraph, states, source_cache, time_points, "global"
    )
    beta_g_time = _fit_pairwise_to_probability_cache(
        simplegraph, states, source_cache, time_points, "time"
    )
    beta_m_global = _fit_pairwise_to_probability_cache(
        multigraph, states, source_cache, time_points, "global"
    )
    beta_m_time = _fit_pairwise_to_probability_cache(
        multigraph, states, source_cache, time_points, "time"
    )
    channel_data = _structural_channel_data(hypergraph, rates, nu)
    beta_m_smooth = _fit_structural_betas(
        channel_data, mode="time", smoothness=40.0
    )
    beta_c_time = _fit_simplicial_to_probability_cache(
        simplicial, states, source_cache, time_points, beta_m_time
    )
    source_kernels = [
        build_exact_transition_kernel(hypergraph, t, parameters_h)
        for t in time_points
    ]
    initial = initial_distribution(len(hypergraph.nodes), (0,))
    candidates = []
    candidates.append(
        _evaluate_small_candidate(
            "G-global",
            simplegraph,
            PairwiseSIParameters(beta_g_global, INDEPENDENT),
            source_kernels,
            initial,
            1,
        )
    )
    candidates.append(
        _evaluate_small_candidate(
            "G-time",
            simplegraph,
            PairwiseSIParameters(beta_g_time, INDEPENDENT),
            source_kernels,
            initial,
            len(time_points),
        )
    )
    candidates.append(
        _evaluate_small_candidate(
            "MG-global",
            multigraph,
            PairwiseSIParameters(beta_m_global, INDEPENDENT),
            source_kernels,
            initial,
            1,
        )
    )
    candidates.append(
        _evaluate_small_candidate(
            "MG-smooth",
            multigraph,
            PairwiseSIParameters(beta_m_smooth, INDEPENDENT),
            source_kernels,
            initial,
            len(time_points),
        )
    )
    candidates.append(
        _evaluate_small_candidate(
            "MG-time",
            multigraph,
            PairwiseSIParameters(beta_m_time, INDEPENDENT),
            source_kernels,
            initial,
            len(time_points),
        )
    )
    candidates.append(
        _evaluate_small_candidate(
            "SC2-time",
            simplicial,
            SimplicialSIParameters(beta_c_time, INDEPENDENT),
            source_kernels,
            initial,
            2 * len(time_points),
        )
    )
    candidates.append(
        _evaluate_small_candidate(
            "H-original",
            hypergraph,
            parameters_h,
            source_kernels,
            initial,
            len(rates),
        )
    )
    public = []
    for candidate in candidates:
        public.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in ("network", "parameters", "kernels")
            }
        )
    selected = select_minimal_surrogate(
        public,
        error_budget,
        error_key="endpoint_tv",
        complexity_key="complexity",
    )
    front = pareto_front(public, error_key="endpoint_tv", cost_key="complexity")
    context = {
        "hypergraph": hypergraph,
        "multigraph": multigraph,
        "parameters_h": parameters_h,
        "beta_m_smooth": beta_m_smooth,
        "source_kernels": source_kernels,
        "initial": initial,
        "required": required,
    }
    return {
        "experiment": "A2",
        "error_budget": float(error_budget),
        "candidates": public,
        "selected": selected,
        "pareto_names": [item["name"] for item in front],
    }, context

def run_a3_error_certificate(context, recovery=0.32):
    hypergraph = context["hypergraph"]
    multigraph = context["multigraph"]
    source_parameters = SISParameters(context["parameters_h"], recovery)
    surrogate_parameters = SISParameters(
        PairwiseSIParameters(context["beta_m_smooth"], INDEPENDENT),
        recovery,
    )
    source_kernels = [
        build_exact_transition_kernel(
            hypergraph, t, source_parameters, update_rule="sis"
        )
        for t in hypergraph.time_points
    ]
    surrogate_kernels = [
        build_exact_transition_kernel(
            multigraph, t, surrogate_parameters, update_rule="sis"
        )
        for t in hypergraph.time_points
    ]
    certificate = build_error_certificate(
        context["initial"], source_kernels, surrogate_kernels
    )
    effects = []
    features = []
    for (time_point, size, infected), beta in context["required"].items():
        effects.append(float(pairwise_effect(beta, [infected])[0]))
        features.append((time_point, infected))
    lower_bound, collision_groups = projection_collision_lower_bound(
        effects, features
    )
    result = _json_dataclass(certificate)
    result.update(
        {
            "experiment": "A3",
            "irreducible_collision_lower_bound": float(lower_bound),
            "collision_group_count": int(len(collision_groups)),
            "passed": bool(
                certificate.covered_by_basic
                and certificate.covered_by_expected
                and certificate.covered_by_contraction
            ),
        }
    )
    return result

def _temporal_pattern(name, length):
    x = np.arange(length, dtype=float)
    if name == "constant":
        return np.full(length, 0.065)
    if name == "smooth":
        return 0.065 + 0.025 * np.sin(2.0 * math.pi * x / length)
    if name == "jump":
        return np.where(x < length // 2, 0.04, 0.105)
    if name == "multistage":
        return np.asarray(
            [0.035] * (length // 4)
            + [0.075] * (length // 4)
            + [0.125] * (length // 4)
            + [0.055] * (length - 3 * (length // 4)),
            dtype=float,
        )
    raise KeyError(name)

def run_a4_temporal_drift(seed=0, length=30):
    rng = np.random.RandomState(seed)
    scenarios = OrderedDict()
    for scenario in ("constant", "smooth", "jump", "multistage"):
        truth = _temporal_pattern(scenario, length)
        targets = []
        designs = []
        weights = []
        for time_index in range(length):
            exposure = np.arange(1, 5, dtype=float)
            direction = np.asarray([-1.1, -0.2, 0.35, 0.95])
            required = np.clip(
                truth[time_index]
                + 0.008 * direction
                + rng.normal(0.0, 0.0015, size=4),
                0.002,
                0.3,
            )
            effects = pairwise_effect(required, exposure)
            targets.append(-np.log1p(-effects))
            designs.append(exposure)
            weights.append(np.asarray([4.0, 3.0, 2.0, 1.0]))
        all_targets = np.concatenate(targets)
        all_designs = np.concatenate(designs)
        all_weights = np.concatenate(weights)
        global_hazard = float(
            np.dot(all_weights, all_designs * all_targets)
            / np.dot(all_weights, all_designs ** 2)
        )
        global_beta = np.full(length, 1.0 - np.exp(-global_hazard))
        time_hazard = fit_temporal_scalar(
            targets, designs, weights, smoothness=0.0, bounds=(0.0, 1.0)
        )
        smooth_hazard = fit_temporal_scalar(
            targets, designs, weights, smoothness=120.0, bounds=(0.0, 1.0)
        )
        piece_hazard, change_points, _ = fit_piecewise_scalar(
            targets,
            designs,
            weights,
            switch_penalty=0.025,
            bounds=(0.0, 1.0),
        )
        estimates = OrderedDict(
            [
                ("Global", global_beta),
                ("Per-time", 1.0 - np.exp(-time_hazard)),
                ("Smooth", 1.0 - np.exp(-smooth_hazard)),
                ("Piecewise", 1.0 - np.exp(-piece_hazard)),
            ]
        )
        scenarios[scenario] = {
            "truth": truth,
            "estimates": estimates,
            "rmse": {
                method: float(np.sqrt(np.mean((values - truth) ** 2)))
                for method, values in estimates.items()
            },
            "change_points": tuple(int(value) for value in change_points),
        }
    return {"experiment": "A4", "scenarios": scenarios}

def run_a5_generalization(nu=2.0):
    hypergraph = _small_temporal_hypergraph()
    multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
    rates, _ = build_conflicting_lambda_rates(
        hypergraph, nu, 0.055, 0.032, time_drift=0.015
    )
    parameters_h = HypergraphSIParameters(rates, nu)
    states = all_binary_states(len(hypergraph.nodes))
    train_indices = [
        index
        for index, state in enumerate(states)
        if 1 <= int(state.sum()) <= 2
    ]
    test_indices = [
        index
        for index, state in enumerate(states)
        if 3 <= int(state.sum()) <= 5
    ]
    train_times = hypergraph.time_points[:3]
    test_times = hypergraph.time_points[3:]
    source_cache = _source_probability_cache(hypergraph, parameters_h, states)

    def residual_vector(beta, state_indices, time_points):
        params = PairwiseSIParameters(float(beta), INDEPENDENT)
        residuals = []
        for time_point in time_points:
            for state_index in state_indices:
                state = states[state_index]
                susceptible = state == 0
                predicted = infection_probabilities(
                    state, multigraph, time_point, params
                )
                residuals.extend(
                    np.abs(
                        predicted[susceptible]
                        - source_cache[time_point][state_index, susceptible]
                    ).tolist()
                )
        return np.asarray(residuals, dtype=float)

    channel_data = _structural_channel_data(hypergraph, rates, nu)
    analytic = list(_fit_structural_betas(channel_data, mode="global").values())[0]
    mean_fit = minimize_scalar(
        lambda beta: float(np.mean(residual_vector(beta, train_indices, train_times) ** 2)),
        bounds=(0.0, 0.35),
        method="bounded",
    )
    robust_fit = minimize_scalar(
        lambda beta: float(
            np.mean(residual_vector(beta, train_indices, train_times) ** 2)
            + 0.45 * cvar(
                residual_vector(beta, train_indices, train_times), 0.90
            ) ** 2
        ),
        bounds=(0.0, 0.35),
        method="bounded",
    )
    methods = OrderedDict(
        [
            ("Direct copy", 0.055),
            ("Analytic projection", float(analytic)),
            ("Dynamics mean", float(mean_fit.x)),
            ("Dynamics robust", float(robust_fit.x)),
        ]
    )
    rows = []
    for method, beta in methods.items():
        train = residual_vector(beta, train_indices, train_times)
        test = residual_vector(beta, test_indices, test_times)
        rows.append(
            {
                "method": method,
                "beta": float(beta),
                "train_mae": float(train.mean()),
                "test_mae": float(test.mean()),
                "generalization_gap": float(test.mean() - train.mean()),
                "test_worst": float(test.max()),
            }
        )
    return {
        "experiment": "A5",
        "train_state_count": len(train_indices),
        "test_state_count": len(test_indices),
        "rows": rows,
    }

def run_a6_ablation(seed=0):
    rng = np.random.RandomState(seed)
    exposures = rng.randint(1, 6, size=40).astype(float)
    weights = np.exp(rng.normal(0.0, 1.0, size=40))
    direction = rng.normal(size=40)
    direction = (direction - direction.mean()) / direction.std()
    required = np.clip(0.065 + 0.075 * direction, 0.002, 0.35)
    effects = pairwise_effect(required, exposures)
    fits = OrderedDict()
    fits["Full"] = fit_dynamics_aware_pairwise(
        effects, exposures, weights, tail_weight=0.40, tail_quantile=0.90
    )
    fits["No frequency weights"] = fit_dynamics_aware_pairwise(
        effects, exposures, np.ones(len(effects)), tail_weight=0.40, tail_quantile=0.90
    )
    fits["No tail loss"] = fit_dynamics_aware_pairwise(
        effects, exposures, weights, tail_weight=0.0, tail_quantile=0.90
    )
    fits["Hazard only"] = fit_pairwise_hazard(effects, exposures, weights)
    rows = []
    for name, fit in fits.items():
        rows.append(
            {
                "method": name,
                "beta": float(fit.parameters),
                "weighted_mae": float(fit.mean_absolute_error),
                "weighted_rmse": float(fit.weighted_rmse),
                "worst_error": float(fit.worst_error),
            }
        )
    return {"experiment": "A6", "rows": rows}

def run_a7_adaptive_reduction(seed=0, error_budget=0.025):
    rng = np.random.RandomState(seed)
    conflict_schedule = np.asarray(
        [0.008] * 4
        + [0.028] * 4
        + [0.075] * 5
        + [0.035] * 4
        + [0.010] * 4,
        dtype=float,
    )
    names = ("G", "MG", "SC2", "H")
    complexities = np.asarray([1.0, 2.0, 4.5, 10.0])
    error_matrix = []
    for conflict in conflict_schedule:
        infected = np.tile(np.arange(1, 5, dtype=float), 5)
        if conflict <= 0.012:
            multiplicity = np.ones(len(infected), dtype=float)
        else:
            multiplicity = rng.randint(1, 3, size=len(infected)).astype(float)
        synergy = 0.5 * infected * (infected - 1.0)
        noise = rng.normal(size=len(infected))
        noise = (noise - noise.mean()) / noise.std()
        noise_scale = 0.08 if conflict < 0.06 else 0.85
        hazard = (
            -math.log(1.0 - 0.055) * infected * multiplicity
            + conflict * (0.75 * synergy + noise_scale * noise)
        )
        hazard = np.clip(hazard, 1e-6, 3.0)
        effects = 1.0 - np.exp(-hazard)
        weights = np.exp(rng.normal(0.0, 0.5, size=len(infected)))
        graph_fit = fit_dynamics_aware_pairwise(
            effects,
            np.maximum(1.0, infected),
            weights,
            tail_weight=0.25,
            tail_quantile=0.90,
        )
        multi_fit = fit_dynamics_aware_pairwise(
            effects,
            infected * multiplicity,
            weights,
            tail_weight=0.25,
            tail_quantile=0.90,
        )
        hazard_targets = -np.log1p(-effects)
        design = np.column_stack(
            [
                infected * multiplicity,
                0.5 * infected * (infected - 1.0),
            ]
        )
        simplex_fit = fit_nonnegative_mapping(
            design,
            hazard_targets,
            weights,
            l1_penalty=0.002,
            upper_bound=0.8,
        )
        simplex_probability = 1.0 - np.exp(-simplex_fit.predictions)
        simplex_error = float(
            np.average(np.abs(simplex_probability - effects), weights=weights)
        )
        error_matrix.append(
            [
                graph_fit.mean_absolute_error,
                multi_fit.mean_absolute_error,
                simplex_error,
                0.0,
            ]
        )
    error_matrix = np.asarray(error_matrix, dtype=float)
    adaptive = adaptive_model_sequence(
        error_matrix,
        complexities,
        names,
        error_budget=error_budget,
        complexity_weight=0.012,
        switch_cost=0.006,
    )
    no_switch = adaptive_model_sequence(
        error_matrix,
        complexities,
        names,
        error_budget=error_budget,
        complexity_weight=0.012,
        switch_cost=0.0,
    )
    return {
        "experiment": "A7",
        "conflict_schedule": conflict_schedule,
        "model_names": names,
        "complexities": complexities,
        "error_matrix": error_matrix,
        "error_budget": float(error_budget),
        "adaptive": _json_dataclass(adaptive),
        "without_switch_cost": _json_dataclass(no_switch),
        "mean_selected_complexity": float(
            np.mean([complexities[index] for index in adaptive.selected_indices])
        ) if adaptive.feasible else float("inf"),
    }

def _time_mean_required(required, time_points):
    result = {}
    for time_point in time_points:
        values = [
            value
            for (time_value, _, _), value in required.items()
            if time_value == time_point
        ]
        result[time_point] = float(np.mean(values)) if values else 0.0
    return result

def _evaluate_probability_method(
    source_probabilities,
    states_by_time,
    target_network,
    target_parameters,
):
    residuals = []
    worst = 0.0
    start = time.perf_counter()
    for position, time_point in enumerate(target_network.time_points):
        for sample_index, state in enumerate(states_by_time[position]):
            predicted = infection_probabilities(
                state, target_network, time_point, target_parameters
            )
            susceptible = state == 0
            values = np.abs(
                predicted[susceptible]
                - source_probabilities[position][sample_index][susceptible]
            )
            residuals.extend(values.tolist())
            if len(values):
                worst = max(worst, float(values.max()))
    elapsed = time.perf_counter() - start
    return float(np.mean(residuals)), float(worst), float(elapsed)

def _fit_dataset_dynamics_betas(
    hypergraph,
    multigraph,
    parameters_h,
    sample_count,
    rng,
):
    beta_by_time = {}
    for position, time_point in enumerate(hypergraph.time_points):
        effects = []
        exposures = []
        fraction = min(0.42, 0.04 + 0.025 * position)
        for _ in range(int(sample_count)):
            state = (rng.random_sample(len(hypergraph.nodes)) < fraction).astype(np.int8)
            if not state.any():
                state[rng.randint(len(state))] = 1
            source_probability = infection_probabilities(
                state, hypergraph, time_point, parameters_h
            )
            state_by_node = {
                node: int(state[index])
                for index, node in enumerate(multigraph.nodes)
            }
            for node_index, node in enumerate(multigraph.nodes):
                if state[node_index] == 1:
                    continue
                exposure_count = sum(
                    weight
                    for other, weight in multigraph.weighted_neighbors(
                        node, time_point
                    )
                    if state_by_node[other] == 1
                )
                if exposure_count > 0:
                    effects.append(float(source_probability[node_index]))
                    exposures.append(float(exposure_count))
        if effects:
            fit = fit_dynamics_aware_pairwise(
                effects,
                exposures,
                np.ones(len(effects)),
                tail_weight=0.20,
                tail_quantile=0.95,
            )
            beta_by_time[time_point] = float(fit.parameters)
        else:
            beta_by_time[time_point] = 0.0
    return beta_by_time

def run_a8_dataset_suite(config):
    data_config = {
        "seed": int(config["seed"]),
        "time_count": int(config["time_count"]),
        "synthetic_persistence": float(config["synthetic_persistence"]),
        "real_max_events": int(config["real_max_events"]),
        "real_max_hyperedge_size": int(config["real_max_hyperedge_size"]),
    }
    datasets = load_all_validation_datasets(data_config)
    rows = []
    trajectory_rows = []
    selection_rows = []
    rng = np.random.RandomState(int(config["seed"]) + 811)
    nu = float(config.get("nu", 2.0))
    base_beta = float(config.get("base_beta", 0.05))
    for dataset_index, (name, (hypergraph, metadata)) in enumerate(datasets.items()):
        multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
        simplegraph = project_temporal_hypergraph_to_simplegraph(hypergraph)
        rates, required = build_conflicting_lambda_rates(
            hypergraph,
            nu,
            base_beta,
            float(config["conflict_strength"]),
            float(config["time_drift"]),
        )
        parameters_h = HypergraphSIParameters(rates, nu)
        channel_data = _structural_channel_data(hypergraph, rates, nu)
        beta_global = _fit_structural_betas(channel_data, mode="global")
        beta_time = _fit_structural_betas(channel_data, mode="time", smoothness=0.0)
        beta_smooth = _fit_structural_betas(channel_data, mode="time", smoothness=45.0)
        beta_mean = _time_mean_required(required, hypergraph.time_points)
        beta_dynamics = _fit_dataset_dynamics_betas(
            hypergraph,
            multigraph,
            parameters_h,
            3 * int(config["state_samples_per_time"]),
            rng,
        )
        validation_states = []
        validation_probabilities = []
        for position, time_point in enumerate(hypergraph.time_points):
            fraction = min(0.42, 0.045 + 0.025 * position)
            time_states = []
            time_probabilities = []
            for _ in range(2 * int(config["state_samples_per_time"])):
                state = (rng.random_sample(len(hypergraph.nodes)) < fraction).astype(np.int8)
                if not state.any():
                    state[rng.randint(len(state))] = 1
                time_states.append(state)
                time_probabilities.append(
                    infection_probabilities(
                        state, hypergraph, time_point, parameters_h
                    )
                )
            validation_states.append(tuple(time_states))
            validation_probabilities.append(tuple(time_probabilities))
        beta_candidates = OrderedDict(
            [
                ("Direct", {t: base_beta for t in hypergraph.time_points}),
                ("Mean", beta_mean),
                ("Global", beta_global),
                ("Temporal", beta_time),
                ("Smooth", beta_smooth),
                ("Dynamics", beta_dynamics),
            ]
        )
        validation_scores = {}
        for candidate_name, candidate_beta in beta_candidates.items():
            score, _, _ = _evaluate_probability_method(
                validation_probabilities,
                validation_states,
                multigraph,
                PairwiseSIParameters(candidate_beta, INDEPENDENT),
            )
            validation_scores[candidate_name] = score
        selected_name = min(validation_scores, key=validation_scores.get)
        selected_beta = beta_candidates[selected_name]
        selection_rows.append(
            {
                "dataset": name,
                "selected_parameterization": selected_name,
                "validation_mae": float(validation_scores[selected_name]),
                "direct_validation_mae": float(validation_scores["Direct"]),
            }
        )
        methods = OrderedDict(
            [
                ("Direct-MG", (multigraph, PairwiseSIParameters(base_beta, INDEPENDENT))),
                ("Mean-MG", (multigraph, PairwiseSIParameters(beta_mean, INDEPENDENT))),
                ("Global-Qdagger-MG", (multigraph, PairwiseSIParameters(beta_global, INDEPENDENT))),
                ("Temporal-Qdagger-MG", (multigraph, PairwiseSIParameters(beta_time, INDEPENDENT))),
                ("Smooth-Qdagger-MG", (multigraph, PairwiseSIParameters(beta_smooth, INDEPENDENT))),
                ("Dynamics-Qdagger-MG", (multigraph, PairwiseSIParameters(beta_dynamics, INDEPENDENT))),
                ("Selected-Qdagger-MG", (multigraph, PairwiseSIParameters(selected_beta, INDEPENDENT))),
                ("Temporal-Qdagger-G", (simplegraph, PairwiseSIParameters(beta_time, INDEPENDENT))),
            ]
        )
        states_by_time = []
        source_probabilities = []
        source_start = time.perf_counter()
        for position, time_point in enumerate(hypergraph.time_points):
            fraction = min(0.42, 0.05 + 0.025 * position)
            time_states = []
            time_probabilities = []
            for _ in range(int(config["state_samples_per_time"])):
                state = (rng.random_sample(len(hypergraph.nodes)) < fraction).astype(np.int8)
                if not state.any():
                    state[rng.randint(len(state))] = 1
                time_states.append(state)
                time_probabilities.append(
                    infection_probabilities(
                        state, hypergraph, time_point, parameters_h
                    )
                )
            states_by_time.append(tuple(time_states))
            source_probabilities.append(tuple(time_probabilities))
        source_elapsed = time.perf_counter() - source_start
        for method, (network, parameters) in methods.items():
            mae, worst, elapsed = _evaluate_probability_method(
                source_probabilities, states_by_time, network, parameters
            )
            rows.append(
                {
                    "dataset": name,
                    "kind": metadata["kind"],
                    "nodes": int(metadata["nodes"]),
                    "events": int(metadata["events"]),
                    "method": method,
                    "probability_mae": mae,
                    "probability_worst": worst,
                    "evaluation_seconds": elapsed,
                    "source_evaluation_seconds": float(source_elapsed),
                    "interaction_count": _interaction_count(network),
                }
            )
        run_count = int(config["trajectory_runs"])
        density_source = []
        density_direct = []
        density_approx = []
        for run_index in range(run_count):
            uniforms = rng.random_sample(
                (len(hypergraph.time_points), len(hypergraph.nodes))
            )
            initial = np.zeros(len(hypergraph.nodes), dtype=np.int8)
            seed_count = max(1, int(round(0.01 * len(initial))))
            initial[
                rng.choice(len(initial), size=seed_count, replace=False)
            ] = 1
            source = simulate_temporal_si(
                hypergraph,
                parameters_h,
                initial,
                uniforms=uniforms,
            )
            direct = simulate_temporal_si(
                multigraph,
                PairwiseSIParameters(base_beta, INDEPENDENT),
                initial,
                uniforms=uniforms,
            )
            approximate = simulate_temporal_si(
                multigraph,
                PairwiseSIParameters(selected_beta, INDEPENDENT),
                initial,
                uniforms=uniforms,
            )
            density_source.append(source.infection_density)
            density_direct.append(direct.infection_density)
            density_approx.append(approximate.infection_density)
        density_source = np.mean(density_source, axis=0)
        density_direct = np.mean(density_direct, axis=0)
        density_approx = np.mean(density_approx, axis=0)
        trajectory_rows.append(
            {
                "dataset": name,
                "kind": metadata["kind"],
                "direct_trajectory_mae": float(
                    np.mean(np.abs(density_direct - density_source))
                ),
                "approximate_trajectory_mae": float(
                    np.mean(np.abs(density_approx - density_source))
                ),
                "direct_final_error": float(
                    abs(density_direct[-1] - density_source[-1])
                ),
                "approximate_final_error": float(
                    abs(density_approx[-1] - density_source[-1])
                ),
                "source_density": density_source,
                "direct_density": density_direct,
                "approximate_density": density_approx,
            }
        )
    return {
        "experiment": "A8",
        "dataset_count": len(datasets),
        "state_rows": rows,
        "trajectory_rows": trajectory_rows,
        "selection_rows": selection_rows,
    }

def _group_dw_step(opinions, group, mu, variance_threshold):
    result = opinions.copy()
    values = opinions[list(group)]
    if float(np.var(values)) <= variance_threshold:
        mean = float(values.mean())
        result[list(group)] = values + mu * (mean - values)
    return result

def _pairwise_dw_step(opinions, group, mu, confidence):
    result = opinions.copy()
    increments = np.zeros(len(opinions), dtype=float)
    counts = np.zeros(len(opinions), dtype=float)
    for left_index in range(len(group)):
        for right_index in range(left_index + 1, len(group)):
            left = group[left_index]
            right = group[right_index]
            if abs(opinions[left] - opinions[right]) <= confidence:
                increments[left] += mu * (opinions[right] - opinions[left])
                increments[right] += mu * (opinions[left] - opinions[right])
                counts[left] += 1.0
                counts[right] += 1.0
    active = counts > 0
    result[active] += increments[active] / counts[active]
    return np.clip(result, 0.0, 1.0)

def _opinion_clusters(opinions, threshold=0.05):
    ordered = np.sort(opinions)
    if not len(ordered):
        return 0
    return 1 + int(np.sum(np.diff(ordered) > threshold))

def _simulate_dw(initial, groups, mode, mu, threshold):
    opinions = initial.copy()
    variance = [float(np.var(opinions))]
    for group in groups:
        if mode == "group":
            opinions = _group_dw_step(opinions, group, mu, threshold)
        else:
            opinions = _pairwise_dw_step(opinions, group, mu, threshold)
        variance.append(float(np.var(opinions)))
    return opinions, np.asarray(variance)

def run_a9_gdw(config, seed=0):
    rng = np.random.RandomState(seed)
    nodes = int(config["nodes"])
    steps = int(config["time_steps"])
    groups = [
        tuple(
            int(value)
            for value in rng.choice(
                nodes, size=int(rng.randint(3, 7)), replace=False
            )
        )
        for _ in range(steps)
    ]
    group_mu = float(config["group_mu"])
    variance_threshold = float(config["group_variance_threshold"])
    training_initials = [
        rng.beta(2.0, 2.0, size=nodes)
        for _ in range(int(config["training_runs"]))
    ]
    test_initials = [
        rng.beta(1.7, 1.7, size=nodes)
        for _ in range(int(config["test_runs"]))
    ]

    def training_objective(values):
        mu, confidence = values
        errors = []
        for initial in training_initials:
            _, source_var = _simulate_dw(
                initial, groups, "group", group_mu, variance_threshold
            )
            _, target_var = _simulate_dw(
                initial, groups, "pair", mu, confidence
            )
            errors.append(np.mean((source_var - target_var) ** 2))
        return float(np.mean(errors))

    grid = []
    for mu in np.linspace(0.05, 0.5, 10):
        for confidence in np.linspace(0.1, 0.9, 17):
            grid.append((training_objective((mu, confidence)), mu, confidence))
    _, initial_mu, initial_confidence = min(grid)
    optimized = minimize(
        training_objective,
        [initial_mu, initial_confidence],
        method="L-BFGS-B",
        bounds=[(0.01, 0.5), (0.05, 1.0)],
        options={"maxiter": 120},
    )
    fitted = (float(optimized.x[0]), float(optimized.x[1]))
    methods = OrderedDict(
        [
            ("Naive pairwise", (group_mu, math.sqrt(12.0 * variance_threshold))),
            ("Approximate pairwise", fitted),
        ]
    )
    source_variances = []
    method_variances = {name: [] for name in methods}
    metrics = {name: [] for name in methods}
    for initial in test_initials:
        source_final, source_var = _simulate_dw(
            initial, groups, "group", group_mu, variance_threshold
        )
        source_variances.append(source_var)
        for name, (mu, confidence) in methods.items():
            target_final, target_var = _simulate_dw(
                initial, groups, "pair", mu, confidence
            )
            method_variances[name].append(target_var)
            metrics[name].append(
                {
                    "variance_mae": float(np.mean(np.abs(target_var - source_var))),
                    "final_variance_error": float(
                        abs(np.var(target_final) - np.var(source_final))
                    ),
                    "cluster_error": float(
                        abs(
                            _opinion_clusters(target_final)
                            - _opinion_clusters(source_final)
                        )
                    ),
                }
            )
    constructed_states = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.25, 0.5, 0.75]),
    )
    constructed_group = (0, 1, 2)
    group_next = [
        _group_dw_step(state, constructed_group, group_mu, variance_threshold)
        for state in constructed_states
    ]
    pair_thresholds = np.linspace(0.05, 1.0, 96)
    minimum_mismatch = float("inf")
    for confidence in pair_thresholds:
        mismatch = 0.0
        for state, target in zip(constructed_states, group_next):
            predicted = _pairwise_dw_step(
                state, constructed_group, fitted[0], confidence
            )
            mismatch = max(mismatch, float(np.max(np.abs(predicted - target))))
        minimum_mismatch = min(minimum_mismatch, mismatch)
    summary = {}
    for name in methods:
        summary[name] = {
            metric: float(np.mean([item[metric] for item in metrics[name]]))
            for metric in ("variance_mae", "final_variance_error", "cluster_error")
        }
    return {
        "experiment": "A9",
        "fitted_mu": fitted[0],
        "fitted_confidence": fitted[1],
        "training_objective": float(optimized.fun),
        "irreducible_constructed_state_mismatch": minimum_mismatch,
        "summary": summary,
        "source_variance": np.mean(source_variances, axis=0),
        "method_variance": {
            name: np.mean(values, axis=0)
            for name, values in method_variances.items()
        },
    }

def run_a10_efficiency(a2_result, a8_result):
    hierarchy = []
    reference = next(
        item for item in a2_result["candidates"] if item["name"] == "H-original"
    )
    for item in a2_result["candidates"]:
        hierarchy.append(
            {
                "name": item["name"],
                "endpoint_tv": item["endpoint_tv"],
                "runtime_seconds": item["runtime_seconds"],
                "speedup_vs_h": float(
                    reference["runtime_seconds"]
                    / max(item["runtime_seconds"], 1e-12)
                ),
                "interaction_ratio": float(
                    item["interaction_count"]
                    / max(reference["interaction_count"], 1)
                ),
                "complexity": item["complexity"],
            }
        )
    dataset_rows = []
    for row in a8_result["state_rows"]:
        if row["method"] not in (
            "Direct-MG",
            "Selected-Qdagger-MG",
            "Temporal-Qdagger-G",
        ):
            continue
        dataset_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "probability_mae": row["probability_mae"],
                "relative_evaluation_time": float(
                    row["evaluation_seconds"]
                    / max(row["source_evaluation_seconds"], 1e-12)
                ),
                "interaction_count": row["interaction_count"],
            }
        )
    return {
        "experiment": "A10",
        "hierarchy": hierarchy,
        "dataset_rows": dataset_rows,
    }
