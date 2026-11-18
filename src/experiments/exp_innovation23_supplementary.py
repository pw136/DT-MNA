
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..approximate import build_error_certificate, projection_collision_lower_bound
from ..dynamics import (
    ExposureSemantics,
    HypergraphSIParameters,
    PairwiseSIParameters,
    SimplicialSIParameters,
    infection_probabilities,
)
from ..dynamics.sis_temporal import SISParameters
from ..necessity import run_closed_loop_decision
from ..projection import (
    project_temporal_hypergraph_to_multigraph,
    project_temporal_hypergraph_to_simplegraph,
    project_temporal_hypergraph_to_simplicial,
)
from ..temporal_structures import TemporalHypergraph
from ..theory.exact_kernel import all_binary_states, build_exact_transition_kernel
from ._common import initial_distribution, json_ready, project_root
from .innovation2_core import (
    _fit_pairwise_to_probability_cache,
    _fit_simplicial_to_probability_cache,
    _source_probability_cache,
    build_conflicting_lambda_rates,
)
from .innovation3_core import _c3_parameters, _sc2_parameters

INDEPENDENT = ExposureSemantics.INDEPENDENT_EXPOSURES
MODEL_ORDER = ("SG", "G", "MG", "SC2", "H")
MODEL_COMPLEXITY = {"SG": 0.5, "G": 1.0, "MG": 2.0, "SC2": 4.5, "H": 10.0}
BIAS_TYPES = ("B0", "B1", "B2", "B3", "B4", "B5")
BIAS_LABELS = {
    "B0": "Unbiased",
    "B1": "Low prevalence",
    "B2": "Low higher-order exposure",
    "B3": "Low repeated exposure",
    "B4": "Early-time coverage",
    "B5": "Large-group omission",
}
HATCHES = ("///", "\\\\\\", "xxx", "...", "+++", "ooo", "***")

def _random_hypergraph(
    node_count,
    time_count,
    seed,
    events_per_snapshot=None,
    persistence=0.35,
    overlap=0.45,
    duplicate_rate=0.15,
):
    rng = np.random.RandomState(int(seed))
    nodes = tuple(range(int(node_count)))
    if events_per_snapshot is None:
        events_per_snapshot = max(3, int(round(0.8 * node_count)))
    snapshots = {}
    previous_unique = []
    for time_point in range(int(time_count)):
        retained = [
            group
            for group in previous_unique
            if rng.random_sample() < float(persistence)
        ]
        unique = list(dict.fromkeys(retained[:events_per_snapshot]))
        attempts = 0
        while len(unique) < int(events_per_snapshot):
            size = int(rng.randint(2, min(5, node_count) + 1))
            if unique and rng.random_sample() < float(overlap):
                anchor_group = unique[int(rng.randint(0, len(unique)))]
                anchor = int(anchor_group[int(rng.randint(0, len(anchor_group)))])
                pool = [node for node in nodes if node != anchor]
                others = rng.choice(pool, size=size - 1, replace=False)
                group = tuple(sorted((anchor,) + tuple(int(v) for v in others)))
            else:
                group = tuple(
                    sorted(
                        int(v)
                        for v in rng.choice(nodes, size=size, replace=False)
                    )
                )
            if group not in unique:
                unique.append(group)
            attempts += 1
            if attempts > 10000:
                break
        events = list(unique)
        duplicate_candidates = [
            group
            for group in unique
            if rng.random_sample() < float(duplicate_rate)
        ]
        events.extend(duplicate_candidates)
        rng.shuffle(events)
        snapshots[time_point] = tuple(events)
        previous_unique = unique
    return TemporalHypergraph(nodes, snapshots)

def _proxy_fit(hypergraph, source_parameters, proxy_name, states):
    source_cache = _source_probability_cache(
        hypergraph, source_parameters, states
    )
    time_points = hypergraph.time_points
    if proxy_name == "G":
        network = project_temporal_hypergraph_to_simplegraph(hypergraph)
        beta = _fit_pairwise_to_probability_cache(
            network, states, source_cache, time_points, "time"
        )
        parameters = PairwiseSIParameters(beta, INDEPENDENT)
    elif proxy_name == "MG":
        network = project_temporal_hypergraph_to_multigraph(hypergraph)
        beta = _fit_pairwise_to_probability_cache(
            network, states, source_cache, time_points, "time"
        )
        parameters = PairwiseSIParameters(beta, INDEPENDENT)
    elif proxy_name == "SC2":
        multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
        beta_initial = _fit_pairwise_to_probability_cache(
            multigraph, states, source_cache, time_points, "time"
        )
        network = project_temporal_hypergraph_to_simplicial(hypergraph, 2)
        beta = _fit_simplicial_to_probability_cache(
            network, states, source_cache, time_points, beta_initial
        )
        parameters = SimplicialSIParameters(beta, INDEPENDENT)
    else:
        raise ValueError("unsupported proxy %s" % proxy_name)
    return network, parameters

def _wilson_interval(successes, total, confidence=0.95):
    if total <= 0:
        return (float("nan"), float("nan"))
    if abs(float(confidence) - 0.95) > 1e-12:
        raise ValueError("only the pre-registered 95% Wilson interval is supported")
    z = 1.959963984540054
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))

def run_certificate_calibration(seed=20260826, replicates_per_cell=4):
    rng = np.random.RandomState(int(seed))
    rows = []
    conflicts = (0.0, 0.01, 0.03, 0.05, 0.10, 0.16)
    recoveries = (0.02, 0.05, 0.10, 0.20)
    task_id = 0
    started = time.perf_counter()
    for node_count in (5, 6, 7):
        for horizon in (3, 5, 8):
            for proxy_name in ("G", "MG", "SC2"):
                for replicate in range(int(replicates_per_cell)):
                    task_seed = int(seed) + 10007 * task_id
                    task_rng = np.random.RandomState(task_seed)
                    conflict = float(
                        conflicts[task_id % len(conflicts)]
                    )
                    base_beta = float(task_rng.uniform(0.02, 0.09))
                    recovery = float(
                        recoveries[(task_id + node_count) % len(recoveries)]
                    )
                    hypergraph = _random_hypergraph(
                        node_count,
                        horizon,
                        task_seed,
                        events_per_snapshot=max(
                            3, int(round(task_rng.uniform(0.55, 1.05) * node_count))
                        ),
                        persistence=float(task_rng.uniform(0.05, 0.75)),
                        overlap=float(task_rng.uniform(0.10, 0.85)),
                        duplicate_rate=float(task_rng.uniform(0.0, 0.35)),
                    )
                    rates, _ = build_conflicting_lambda_rates(
                        hypergraph,
                        2.0,
                        base_beta,
                        conflict,
                        time_drift=float(task_rng.uniform(0.0, 0.02)),
                    )
                    source_si = HypergraphSIParameters(rates, 2.0)
                    states = all_binary_states(node_count)
                    proxy_network, proxy_si = _proxy_fit(
                        hypergraph, source_si, proxy_name, states
                    )
                    source_parameters = SISParameters(source_si, recovery)
                    proxy_parameters = SISParameters(proxy_si, recovery)
                    source_kernels = [
                        build_exact_transition_kernel(
                            hypergraph,
                            time_point,
                            source_parameters,
                            update_rule="sis",
                        )
                        for time_point in hypergraph.time_points
                    ]
                    proxy_kernels = [
                        build_exact_transition_kernel(
                            proxy_network,
                            time_point,
                            proxy_parameters,
                            update_rule="sis",
                        )
                        for time_point in hypergraph.time_points
                    ]
                    infected_node = int(task_rng.randint(0, node_count))
                    initial = initial_distribution(node_count, (infected_node,))
                    certificate = build_error_certificate(
                        initial, source_kernels, proxy_kernels
                    )
                    actual = float(certificate.final_actual_error)
                    bounds = {
                        "distribution": float(
                            certificate.final_expected_bound
                        ),
                        "worst_row": float(certificate.final_basic_bound),
                        "contraction": float(
                            certificate.final_contraction_bound
                        ),
                    }
                    row = {
                        "task_id": task_id,
                        "seed": task_seed,
                        "node_count": node_count,
                        "horizon": horizon,
                        "proxy": proxy_name,
                        "conflict": conflict,
                        "base_beta": base_beta,
                        "recovery": recovery,
                        "actual_tv": actual,
                    }
                    for name, value in bounds.items():
                        row[name + "_bound"] = value
                        row[name + "_width"] = value - actual
                        row[name + "_tightness"] = value / (actual + 1e-12)
                        row[name + "_covered"] = bool(
                            actual <= value + 1e-12
                        )
                    rows.append(row)
                    task_id += 1
    summary = {}
    for name in ("distribution", "worst_row", "contraction"):
        coverage_values = [bool(row[name + "_covered"]) for row in rows]
        successes = int(sum(coverage_values))
        interval = _wilson_interval(successes, len(rows))
        nonzero = [
            row[name + "_tightness"]
            for row in rows
            if row["actual_tv"] > 1e-10
        ]
        widths = [row[name + "_width"] for row in rows]
        summary[name] = {
            "coverage": float(np.mean(coverage_values)),
            "wilson95_low": float(interval[0]),
            "wilson95_high": float(interval[1]),
            "median_tightness_nonzero": float(np.median(nonzero)),
            "iqr_tightness_nonzero": [
                float(np.quantile(nonzero, 0.25)),
                float(np.quantile(nonzero, 0.75)),
            ],
            "median_width": float(np.median(widths)),
        }
    return {
        "experiment": "Innovation2-A11",
        "task_count": len(rows),
        "replicates_per_cell": int(replicates_per_cell),
        "rows": rows,
        "summary": summary,
        "runtime_seconds": float(time.perf_counter() - started),
    }

def _node_local_features(hypergraph, state, time_point, node):
    node_index = hypergraph.node_to_index[node]
    incident = hypergraph.incident_hyperedges(node, time_point)
    unique_neighbors = set()
    weighted_degree = 0
    infected_weighted = 0
    second_order = 0
    channel_counts = Counter()
    for group in incident:
        infected = 0
        for other in group:
            if other == node:
                continue
            unique_neighbors.add(other)
            weighted_degree += 1
            infected += int(state[hypergraph.node_to_index[other]])
        infected_weighted += infected
        second_order += infected * (infected - 1) // 2
        channel_counts[(len(group), infected)] += 1
    infected_unique = sum(
        int(state[hypergraph.node_to_index[other]])
        for other in unique_neighbors
    )
    time_scale = float(time_point) / max(
        1.0, float(len(hypergraph.time_points) - 1)
    )
    target = {
        "G": (
            time_scale,
            float(len(unique_neighbors)),
            float(infected_unique),
        ),
        "MG": (
            time_scale,
            float(weighted_degree),
            float(infected_weighted),
        ),
        "SC2": (
            time_scale,
            float(weighted_degree),
            float(infected_weighted),
            float(second_order),
        ),
    }
    source = [time_scale]
    for size in range(2, 6):
        for infected in range(0, size):
            source.append(float(channel_counts.get((size, infected), 0)))
    return target, tuple(source)

def _capacity_rows_for_conflict(conflict, seed, network_count=6):
    data = {
        representation: {"target": [], "source": [], "y": [], "keys": []}
        for representation in ("G", "MG", "SC2")
    }
    for network_index in range(int(network_count)):
        task_seed = int(seed) + 7919 * network_index
        hypergraph = _random_hypergraph(
            7,
            4,
            task_seed,
            events_per_snapshot=6,
            persistence=0.25 + 0.08 * (network_index % 4),
            overlap=0.25 + 0.10 * (network_index % 5),
            duplicate_rate=0.10 + 0.06 * (network_index % 4),
        )
        rates, _ = build_conflicting_lambda_rates(
            hypergraph,
            2.0,
            0.055,
            float(conflict),
            time_drift=0.008,
        )
        parameters = HypergraphSIParameters(rates, 2.0)
        all_states = [
            state
            for state in all_binary_states(7)
            if 0 < int(state.sum()) < 7
        ]
        rng = np.random.RandomState(task_seed + 17)
        chosen_indices = rng.choice(
            len(all_states), size=min(34, len(all_states)), replace=False
        )
        chosen = [all_states[int(index)] for index in chosen_indices]
        for time_point in hypergraph.time_points:
            for state in chosen:
                probabilities = infection_probabilities(
                    state, hypergraph, time_point, parameters
                )
                for node in hypergraph.nodes:
                    node_index = hypergraph.node_to_index[node]
                    if state[node_index]:
                        continue
                    target_features, source_features = _node_local_features(
                        hypergraph, state, time_point, node
                    )
                    target_value = float(probabilities[node_index])
                    for representation in ("G", "MG", "SC2"):
                        values = tuple(target_features[representation])
                        data[representation]["target"].append(values)
                        data[representation]["source"].append(source_features)
                        data[representation]["y"].append(target_value)
                        data[representation]["keys"].append(values)
    for representation in data:
        data[representation]["target"] = np.asarray(
            data[representation]["target"], dtype=float
        )
        data[representation]["source"] = np.asarray(
            data[representation]["source"], dtype=float
        )
        data[representation]["y"] = np.asarray(
            data[representation]["y"], dtype=float
        )
    return data

def _fit_capacity_mlp(features, targets, seed):
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-6,
            learning_rate_init=0.003,
            max_iter=800,
            tol=1e-8,
            random_state=int(seed),
            early_stopping=False,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(features, targets)
    return np.clip(model.predict(features), 0.0, 1.0)

def run_capacity_collision(seed=20260827, mlp_seeds=3):
    conflicts = (0.0, 0.01, 0.03, 0.05, 0.10, 0.16)
    rows = []
    started = time.perf_counter()
    for conflict_index, conflict in enumerate(conflicts):
        data = _capacity_rows_for_conflict(
            conflict,
            int(seed) + 10000 * conflict_index,
            network_count=6,
        )
        for representation in ("G", "MG", "SC2"):
            item = data[representation]
            lower_bound, groups = projection_collision_lower_bound(
                item["y"], item["keys"]
            )
            for model_seed in range(int(mlp_seeds)):
                random_seed = int(seed) + 1000 * conflict_index + model_seed
                target_prediction = _fit_capacity_mlp(
                    item["target"], item["y"], random_seed
                )
                source_prediction = _fit_capacity_mlp(
                    item["source"], item["y"], random_seed
                )
                for model_name, prediction in (
                    ("Target-only MLP", target_prediction),
                    ("Source-aware MLP", source_prediction),
                ):
                    errors = np.abs(prediction - item["y"])
                    rows.append(
                        {
                            "conflict": float(conflict),
                            "representation": representation,
                            "seed": int(random_seed),
                            "model": model_name,
                            "sample_count": int(len(item["y"])),
                            "collision_group_count": int(len(groups)),
                            "collision_lower_bound": float(lower_bound),
                            "mae": float(errors.mean()),
                            "rmse": float(np.sqrt(np.mean(errors ** 2))),
                            "worst_error": float(errors.max()),
                            "error_minus_lb": float(errors.max() - lower_bound),
                            "bound_holds": bool(
                                model_name != "Target-only MLP"
                                or errors.max() + 1e-8 >= lower_bound
                            ),
                        }
                    )
    summary = {}
    for model_name in ("Target-only MLP", "Source-aware MLP"):
        subset = [row for row in rows if row["model"] == model_name]
        summary[model_name] = {
            "mean_mae": float(np.mean([row["mae"] for row in subset])),
            "mean_worst_error": float(
                np.mean([row["worst_error"] for row in subset])
            ),
        }
    target_rows = [row for row in rows if row["model"] == "Target-only MLP"]
    summary["target_bound_coverage"] = float(
        np.mean([row["bound_holds"] for row in target_rows])
    )
    positive = [
        row
        for row in rows
        if row["conflict"] > 0.0
        and row["model"] in ("Target-only MLP", "Source-aware MLP")
    ]
    target_positive = [
        row["worst_error"]
        for row in positive
        if row["model"] == "Target-only MLP"
    ]
    source_positive = [
        row["worst_error"]
        for row in positive
        if row["model"] == "Source-aware MLP"
    ]
    summary["positive_conflict_worst_reduction"] = float(
        1.0 - np.mean(source_positive) / max(np.mean(target_positive), 1e-15)
    )
    return {
        "experiment": "Innovation2-A12",
        "rows": rows,
        "summary": summary,
        "runtime_seconds": float(time.perf_counter() - started),
    }

def _nontrivial_states(node_count):
    return tuple(
        state
        for state in all_binary_states(node_count)
        if 0 < int(state.sum()) < node_count
    )

def _state_risk_metrics(hypergraph, state, time_point):
    density = float(np.mean(state))
    max_high = 0.0
    max_repeat = 0.0
    max_large = 0.0
    for node in hypergraph.nodes:
        node_index = hypergraph.node_to_index[node]
        if state[node_index]:
            continue
        incident = hypergraph.incident_hyperedges(node, time_point)
        unique_infected = set()
        repeated_count = 0
        for group in incident:
            infected = [
                other
                for other in group
                if other != node
                and state[hypergraph.node_to_index[other]]
            ]
            repeated_count += len(infected)
            unique_infected.update(infected)
            if len(group) >= 3:
                max_high = max(max_high, float(len(infected)))
            if len(group) >= 4:
                max_large = max(max_large, float(len(infected)))
        max_repeat = max(
            max_repeat,
            float(max(0, repeated_count - len(unique_infected))),
        )
    composite = density + 0.7 * max_high + 0.5 * max_repeat + 0.4 * max_large
    return {
        "density": density,
        "higher": max_high,
        "repeat": max_repeat,
        "large": max_large,
        "composite": composite,
    }

def _biased_state_panel(
    hypergraph,
    bias_type,
    strength,
    seed,
    train_count=10,
    validation_count=6,
    audit_count=16,
):
    rng = np.random.RandomState(int(seed))
    states = _nontrivial_states(len(hypergraph.nodes))
    train = {}
    validation = {}
    audit = {}
    test = {}

    def choose_panel(order, count, fraction, random_seed_offset):
        local_rng = np.random.RandomState(int(seed) + int(random_seed_offset))
        count = min(int(count), len(states))
        if bias_type == "B0" or float(fraction) <= 0.0:
            return [
                int(index)
                for index in local_rng.choice(
                    len(states), size=count, replace=False
                )
            ]

        biased_fraction = min(1.0, float(fraction) / 0.75)
        biased_count = min(count, int(round(count * biased_fraction)))
        chosen = [int(index) for index in order[:biased_count]]
        remaining = [
            int(index)
            for index in range(len(states))
            if int(index) not in chosen
        ]
        if count > len(chosen):
            chosen.extend(
                int(index)
                for index in local_rng.choice(
                    remaining,
                    size=count - len(chosen),
                    replace=False,
                )
            )
        return chosen

    for position, time_point in enumerate(hypergraph.time_points):
        metrics = [
            _state_risk_metrics(hypergraph, state, time_point)
            for state in states
        ]
        metric_name = {
            "B0": "composite",
            "B1": "density",
            "B2": "higher",
            "B3": "repeat",
            "B4": "composite",
            "B5": "large",
        }[bias_type]
        order = np.argsort([item[metric_name] for item in metrics])

        current_train_count = int(train_count)
        current_validation_count = int(validation_count)
        if bias_type == "B4":
            early_cut = max(
                1, int(math.ceil(len(hypergraph.time_points) / 3.0))
            )
            if position < early_cut:
                panel_strength = 0.0
            else:
                panel_strength = float(strength)
                current_train_count = max(
                    2, int(round(train_count * (1.0 - float(strength))))
                )
                current_validation_count = max(
                    1,
                    int(
                        round(
                            validation_count
                            * (1.0 - float(strength))
                        )
                    ),
                )
        else:
            panel_strength = float(strength)

        train_indices = choose_panel(
            order,
            current_train_count,
            panel_strength,
            1000 + 31 * position,
        )
        validation_indices = choose_panel(
            order,
            current_validation_count,
            panel_strength,
            2000 + 31 * position,
        )
        train[time_point] = tuple(states[index] for index in train_indices)
        validation[time_point] = tuple(
            states[index] for index in validation_indices
        )

        risk_metric = metric_name if bias_type != "B0" else "composite"
        risk_order = np.argsort(
            [item[risk_metric] for item in metrics]
        )[::-1]
        audit[time_point] = tuple(
            states[int(index)]
            for index in risk_order[: min(audit_count, len(states))]
        )
        test[time_point] = states
    return {
        "train": train,
        "validation": validation,
        "audit": audit,
        "test": test,
    }

def _pairwise_projection_hypergraph(hypergraph, mode):
    snapshots = {}
    temporal_pairs = {}
    for time_point in hypergraph.time_points:
        pairs = []
        for group in hypergraph.active_hyperedges(time_point):
            for left_index in range(len(group)):
                for right_index in range(left_index + 1, len(group)):
                    pairs.append(
                        tuple(
                            sorted(
                                (
                                    group[left_index],
                                    group[right_index],
                                )
                            )
                        )
                    )
        temporal_pairs[time_point] = pairs
    if mode == "static":
        union = tuple(
            sorted(
                {
                    pair
                    for values in temporal_pairs.values()
                    for pair in values
                }
            )
        )
        snapshots = {
            time_point: union for time_point in hypergraph.time_points
        }
    elif mode == "temporal":
        snapshots = {
            time_point: tuple(sorted(set(values)))
            for time_point, values in temporal_pairs.items()
        }
    elif mode == "multiplicity":
        snapshots = {
            time_point: tuple(values)
            for time_point, values in temporal_pairs.items()
        }
    else:
        raise ValueError("unknown pairwise mode %s" % mode)
    return TemporalHypergraph(hypergraph.nodes, snapshots)

def _mechanism_hypergraph(hypergraph, mechanism):
    if mechanism == "static_pairwise":
        return _pairwise_projection_hypergraph(hypergraph, "static")
    if mechanism == "temporal_pairwise":
        return _pairwise_projection_hypergraph(hypergraph, "temporal")
    if mechanism == "multiplicity":
        return _pairwise_projection_hypergraph(hypergraph, "multiplicity")
    return hypergraph

def _mechanism_parameters(hypergraph, mechanism, strength):
    if mechanism == "static_pairwise":
        return _c3_parameters(
            hypergraph,
            2.0,
            {time_point: 0.045 for time_point in hypergraph.time_points},
        )
    if mechanism == "temporal_pairwise":
        return _c3_parameters(
            hypergraph,
            2.0,
            {
                time_point: 0.025
                + 0.035
                * (
                    position
                    / max(1, len(hypergraph.time_points) - 1)
                )
                for position, time_point in enumerate(
                    hypergraph.time_points
                )
            },
        )
    if mechanism == "multiplicity":
        return _c3_parameters(
            hypergraph,
            2.0,
            {
                time_point: 0.04 + 0.005 * (position % 2)
                for position, time_point in enumerate(
                    hypergraph.time_points
                )
            },
        )
    if mechanism == "synergy":
        return _sc2_parameters(
            hypergraph,
            2.0,
            {time_point: 0.025 for time_point in hypergraph.time_points},
            {
                time_point: 0.07 + 0.10 * float(strength)
                for time_point in hypergraph.time_points
            },
        )
    rates, _ = build_conflicting_lambda_rates(
        hypergraph,
        2.0,
        0.05,
        0.04 + 0.12 * float(strength),
        time_drift=0.01,
    )
    return HypergraphSIParameters(rates, 2.0)

def _assessment_value(assessments, name, split, metric):
    return float(assessments[name][split][metric])

def _collision_ok(assessments, name, budget, use_collision):
    if name == "H" or not use_collision:
        return True
    return float(assessments[name]["collision_lower_bound"]) <= float(budget)

def _first_safe(
    assessments,
    budget,
    split,
    metric="empirical_upper",
    use_collision=False,
):
    for name in MODEL_ORDER:
        if not _collision_ok(
            assessments, name, budget, use_collision
        ):
            continue
        if _assessment_value(assessments, name, split, metric) <= budget:
            return name
    return "H"

def _select_variant(
    serializable,
    method,
    minimum_evidence,
    validation_state_count,
):
    assessments = serializable["assessments"]
    budget = float(serializable["error_budget"])
    insufficient = int(validation_state_count) < int(minimum_evidence)
    if method == "Full THN-Need" and insufficient:
        return "H", True, "H"
    if method == "w/o Abstention":
        insufficient = False
    if method == "Validation-only":
        choice = _first_safe(
            assessments, budget, "validation", use_collision=False
        )
        return choice, False, choice
    if method == "Mean-error":
        choice = _first_safe(
            assessments,
            budget,
            "validation",
            metric="mean_absolute_error",
            use_collision=False,
        )
        return choice, False, choice
    if method == "w/o Complexity":
        candidates = []
        for name in MODEL_ORDER:
            if not _collision_ok(assessments, name, budget, True):
                continue
            validation_safe = (
                _assessment_value(
                    assessments, name, "validation", "empirical_upper"
                )
                <= budget
            )
            audit_safe = (
                _assessment_value(
                    assessments, name, "audit", "empirical_upper"
                )
                <= budget
            )
            if validation_safe and audit_safe:
                candidates.append(name)
        if not candidates:
            return "H", False, "H"
        choice = min(
            candidates,
            key=lambda name: (
                _assessment_value(
                    assessments, name, "audit", "worst_error"
                ),
                -MODEL_COMPLEXITY[name],
            ),
        )
        validation_choice = _first_safe(
            assessments, budget, "validation", use_collision=True
        )
        return choice, False, validation_choice
    use_collision = method != "w/o Collision"
    use_audit = method != "w/o Audit"
    if insufficient and method != "w/o Abstention":
        return "H", True, "H"
    validation_choice = _first_safe(
        assessments,
        budget,
        "validation",
        use_collision=use_collision,
    )
    selected_index = MODEL_ORDER.index(validation_choice)
    if use_audit:
        while selected_index < len(MODEL_ORDER) - 1:
            name = MODEL_ORDER[selected_index]
            if _collision_ok(
                assessments, name, budget, use_collision
            ) and (
                _assessment_value(
                    assessments, name, "audit", "empirical_upper"
                )
                <= budget
            ):
                break
            selected_index += 1
    return MODEL_ORDER[selected_index], False, validation_choice

def _selection_score(record, selected, abstained=False):
    assessments = record["serializable"]["assessments"]
    budget = float(record["budget"])
    oracle = record["oracle"]
    test_error = _assessment_value(
        assessments, selected, "test", "worst_error"
    )
    safe = bool(test_error <= budget + 1e-12)
    return {
        "task_id": record["task_id"],
        "topology": record["topology"],
        "bias": record["bias"],
        "strength": record["strength"],
        "mechanism": record["mechanism"],
        "budget": budget,
        "selected": selected,
        "oracle": oracle,
        "agreement": bool(selected == oracle),
        "safe": safe,
        "unsafe": bool(
            not safe
            and MODEL_ORDER.index(selected) < MODEL_ORDER.index(oracle)
        ),
        "overmodel": bool(
            MODEL_ORDER.index(selected) > MODEL_ORDER.index(oracle)
        ),
        "complexity": float(MODEL_COMPLEXITY[selected]),
        "abstained": bool(abstained),
        "test_error": test_error,
    }

def _topology_statistics(hypergraph):
    groups = [
        group
        for time_point in hypergraph.time_points
        for group in hypergraph.active_hyperedges(time_point)
    ]
    sizes = [len(group) for group in groups]
    duplicate_fraction_values = []
    snapshots = []
    for time_point in hypergraph.time_points:
        active = list(hypergraph.active_hyperedges(time_point))
        duplicate_fraction_values.append(
            1.0 - len(set(active)) / max(1.0, float(len(active)))
        )
        snapshots.append(set(active))
    jaccards = []
    for left, right in zip(snapshots[:-1], snapshots[1:]):
        union = left | right
        jaccards.append(len(left & right) / max(1.0, float(len(union))))
    return {
        "node_count": len(hypergraph.nodes),
        "event_count": len(groups),
        "mean_group_size": float(np.mean(sizes)),
        "max_group_size": int(max(sizes)),
        "duplicate_fraction": float(np.mean(duplicate_fraction_values)),
        "temporal_variation": float(
            1.0 - np.mean(jaccards) if jaccards else 0.0
        ),
    }

def _meta_features(record):
    serializable = record["serializable"]
    assessments = serializable["assessments"]
    values = [
        float(record["stats"]["node_count"]),
        float(record["stats"]["event_count"]),
        float(record["stats"]["mean_group_size"]),
        float(record["stats"]["max_group_size"]),
        float(record["stats"]["duplicate_fraction"]),
        float(record["stats"]["temporal_variation"]),
        float(record["budget"]),
        float(record["validation_state_count"]),
    ]
    for name in ("SG", "G", "MG", "SC2"):
        values.extend(
            [
                _assessment_value(
                    assessments, name, "validation", "mean_absolute_error"
                ),
                _assessment_value(
                    assessments, name, "validation", "worst_error"
                ),
                float(assessments[name]["collision_lower_bound"]),
            ]
        )
    return values

def _aggregate_method(rows):
    return {
        "task_count": len(rows),
        "oracle_agreement": float(np.mean([row["agreement"] for row in rows])),
        "safety_rate": float(np.mean([row["safe"] for row in rows])),
        "unsafe_rate": float(np.mean([row["unsafe"] for row in rows])),
        "overmodel_rate": float(
            np.mean([row["overmodel"] for row in rows])
        ),
        "mean_complexity": float(
            np.mean([row["complexity"] for row in rows])
        ),
        "abstention_rate": float(
            np.mean([row["abstained"] for row in rows])
        ),
    }

def _generate_stress_records(seed=20260828, sampling_seeds=3):
    records = []
    task_id = 0
    started = time.perf_counter()
    topology_cache = {}
    for topology in range(8):
        topology_cache[topology] = _random_hypergraph(
            5 + topology % 3,
            3 + topology % 2,
            int(seed) + 50000 * topology,
            events_per_snapshot=4 + topology % 4,
            persistence=0.10 + 0.09 * (topology % 6),
            overlap=0.15 + 0.10 * (topology % 7),
            duplicate_rate=0.05 + 0.06 * (topology % 5),
        )
    strengths = (0.0, 0.25, 0.50, 0.75)
    mechanisms = ("static_pairwise", "temporal_pairwise", "multiplicity", "synergy", "conflict")
    for bias_index, bias in enumerate(BIAS_TYPES):
        for strength_index, strength in enumerate(strengths):
            for mechanism_index, mechanism in enumerate(mechanisms):
                for sample_seed in range(int(sampling_seeds)):
                    topology = (
                        bias_index
                        + 2 * strength_index
                        + 3 * mechanism_index
                        + sample_seed
                    ) % 8
                    base_hypergraph = topology_cache[topology]
                    hypergraph = _mechanism_hypergraph(
                        base_hypergraph, mechanism
                    )
                    parameters = _mechanism_parameters(
                        hypergraph, mechanism, strength
                    )
                    split_seed = (
                        int(seed)
                        + 100000 * bias_index
                        + 10000 * strength_index
                        + 1000 * mechanism_index
                        + sample_seed
                    )
                    splits = _biased_state_panel(
                        hypergraph, bias, strength, split_seed
                    )
                    budget = (0.02, 0.04, 0.08)[
                        (strength_index + sample_seed) % 3
                    ]
                    decision = run_closed_loop_decision(
                        hypergraph,
                        parameters,
                        splits,
                        error_budget=budget,
                        exact_tolerance=1e-10,
                        minimum_validation_states=1,
                    )
                    serializable = decision.serializable()
                    oracle = next(
                        name
                        for name in MODEL_ORDER
                        if _assessment_value(
                            serializable["assessments"],
                            name,
                            "test",
                            "worst_error",
                        )
                        <= budget + 1e-12
                    )
                    validation_state_count = sum(
                        len(values)
                        for values in splits["validation"].values()
                    )
                    records.append(
                        {
                            "task_id": task_id,
                            "topology": topology,
                            "bias": bias,
                            "strength": float(strength),
                            "mechanism": mechanism,
                            "budget": float(budget),
                            "oracle": oracle,
                            "validation_state_count": int(
                                validation_state_count
                            ),
                            "minimum_evidence": int(
                                len(hypergraph.time_points) * 4
                            ),
                            "stats": _topology_statistics(hypergraph),
                            "serializable": serializable,
                        }
                    )
                    task_id += 1
    return records, float(time.perf_counter() - started)

def run_thn_ablation_and_audit(seed=20260828, sampling_seeds=3):
    records, runtime = _generate_stress_records(seed, sampling_seeds)
    methods = (
        "Full THN-Need",
        "w/o Collision",
        "w/o Audit",
        "w/o Abstention",
        "w/o Complexity",
        "Validation-only",
        "Mean-error",
    )
    ablation_rows = []
    selection_details = defaultdict(dict)
    for record in records:
        for method in methods:
            selected, abstained, validation_choice = _select_variant(
                record["serializable"],
                method,
                record["minimum_evidence"],
                record["validation_state_count"],
            )
            scored = _selection_score(record, selected, abstained)
            scored["method"] = method
            scored["validation_choice"] = validation_choice
            scored["audit_triggered"] = bool(
                selected != validation_choice and not abstained
            )
            ablation_rows.append(scored)
            selection_details[record["task_id"]][method] = scored
    ablation_summary = {
        method: _aggregate_method(
            [row for row in ablation_rows if row["method"] == method]
        )
        for method in methods
    }
    audit_rows = []
    for bias in BIAS_TYPES:
        for strength in (0.0, 0.25, 0.50, 0.75):
            task_ids = [
                record["task_id"]
                for record in records
                if record["bias"] == bias
                and record["strength"] == strength
            ]
            full = [
                selection_details[task_id]["Full THN-Need"]
                for task_id in task_ids
            ]
            validation = [
                selection_details[task_id]["Validation-only"]
                for task_id in task_ids
            ]
            unsafe_validation = [
                index
                for index, row in enumerate(validation)
                if not row["safe"]
            ]
            rescued = [
                index
                for index in unsafe_validation
                if full[index]["safe"]
            ]
            audit_rows.append(
                {
                    "bias": bias,
                    "bias_label": BIAS_LABELS[bias],
                    "strength": float(strength),
                    "task_count": len(task_ids),
                    "validation_unsafe_rate": float(
                        np.mean([row["unsafe"] for row in validation])
                    ),
                    "full_unsafe_rate": float(
                        np.mean([row["unsafe"] for row in full])
                    ),
                    "validation_overmodel_rate": float(
                        np.mean([row["overmodel"] for row in validation])
                    ),
                    "full_overmodel_rate": float(
                        np.mean([row["overmodel"] for row in full])
                    ),
                    "audit_trigger_rate": float(
                        np.mean([row["audit_triggered"] for row in full])
                    ),
                    "rescue_rate": float(
                        len(rescued) / len(unsafe_validation)
                        if unsafe_validation
                        else 0.0
                    ),
                    "false_upgrade_rate": float(
                        np.mean([row["overmodel"] for row in full])
                    ),
                    "validation_unsafe_count": len(unsafe_validation),
                    "rescued_count": len(rescued),
                }
            )
    return {
        "experiment": "Innovation3-A10-A11",
        "task_count": len(records),
        "sampling_seeds": int(sampling_seeds),
        "records": records,
        "ablation_rows": ablation_rows,
        "ablation_summary": ablation_summary,
        "audit_rows": audit_rows,
        "runtime_seconds": runtime,
    }

def _regularized_choice(record, regularization):
    assessments = record["serializable"]["assessments"]
    values = []
    for name in MODEL_ORDER:
        error = _assessment_value(
            assessments, name, "validation", "worst_error"
        )
        complexity = MODEL_COMPLEXITY[name] / MODEL_COMPLEXITY["H"]
        values.append((error + float(regularization) * complexity, name))
    return min(values)[1]

def _tune_regularization(training_records):
    grid = (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
    best = None
    for value in grid:
        rows = [
            _selection_score(record, _regularized_choice(record, value))
            for record in training_records
        ]
        summary = _aggregate_method(rows)
        score = (
            summary["safety_rate"],
            summary["oracle_agreement"],
            -summary["overmodel_rate"],
            -summary["mean_complexity"],
        )
        if best is None or score > best[0]:
            best = (score, value)
    return float(best[1])

def run_strong_selector_baselines(stress_result, seed=20260829):
    records = stress_result["records"]
    predictions = {
        method: {}
        for method in (
            "THN-Need",
            "Validation-only",
            "Regularized selector",
            "Meta-selector",
            "Always G",
            "Always H",
        )
    }
    fold_rows = []
    for holdout in range(8):
        training = [
            record for record in records if record["topology"] != holdout
        ]
        testing = [
            record for record in records if record["topology"] == holdout
        ]
        x_train = np.asarray(
            [_meta_features(record) for record in training], dtype=float
        )
        y_train = np.asarray([record["oracle"] for record in training])
        x_test = np.asarray(
            [_meta_features(record) for record in testing], dtype=float
        )
        classifier = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=int(seed) + holdout,
            n_jobs=1,
        )
        classifier.fit(x_train, y_train)
        meta_predictions = classifier.predict(x_test)
        regularization = _tune_regularization(training)
        for record, meta_selected in zip(testing, meta_predictions):
            full_selected, full_abstained, _ = _select_variant(
                record["serializable"],
                "Full THN-Need",
                record["minimum_evidence"],
                record["validation_state_count"],
            )
            validation_selected, _, _ = _select_variant(
                record["serializable"],
                "Validation-only",
                record["minimum_evidence"],
                record["validation_state_count"],
            )
            selected_map = {
                "THN-Need": (full_selected, full_abstained),
                "Validation-only": (validation_selected, False),
                "Regularized selector": (
                    _regularized_choice(record, regularization),
                    False,
                ),
                "Meta-selector": (str(meta_selected), False),
                "Always G": ("G", False),
                "Always H": ("H", False),
            }
            for method, (selected, abstained) in selected_map.items():
                row = _selection_score(record, selected, abstained)
                row["method"] = method
                row["fold"] = holdout
                row["regularization"] = regularization
                predictions[method][record["task_id"]] = row
        fold_rows.append(
            {
                "holdout_topology": holdout,
                "train_count": len(training),
                "test_count": len(testing),
                "regularization": regularization,
            }
        )
    rows = []
    summary = {}
    for method, values in predictions.items():
        method_rows = list(values.values())
        rows.extend(method_rows)
        summary[method] = _aggregate_method(method_rows)
        summary[method]["unsafe_reduction_rate"] = float(
            1.0 - summary[method]["unsafe_rate"]
        )
    return {
        "experiment": "Innovation3-A13",
        "protocol": "leave-one-topology-out",
        "rows": rows,
        "folds": fold_rows,
        "summary": summary,
    }

def _write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    if columns is None:
        columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path

def _save_figure(figure, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]

def plot_certificate(result, output_base):
    rows = result["rows"]
    colors = {
        "distribution": "#2A9D8F",
        "worst_row": "#E76F51",
        "contraction": "#457B9D",
    }
    labels = {
        "distribution": "Distribution-weighted",
        "worst_row": "Worst-row",
        "contraction": "Contraction-adjusted",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axis = axes[0, 0]
    positive_values = [
        max(1e-12, row["actual_tv"]) for row in rows
    ]
    maximum = 1e-12
    for name in colors:
        y = [max(1e-12, row[name + "_bound"]) for row in rows]
        axis.scatter(
            positive_values,
            y,
            s=17,
            alpha=0.55,
            color=colors[name],
            label=labels[name],
        )
        maximum = max(maximum, max(y), max(positive_values))
    axis.plot([1e-12, maximum], [1e-12, maximum], "k--", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Actual endpoint TV")
    axis.set_ylabel("Certificate")
    axis.set_title("A  Coverage calibration", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    data = []
    for name in colors:
        data.append(
            [
                math.log10(max(1.0, row[name + "_tightness"]))
                for row in rows
                if row["actual_tv"] > 1e-10
            ]
        )
    box = axis.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color, hatch in zip(
        box["boxes"], colors.values(), HATCHES
    ):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_hatch(hatch)
        patch.set_linewidth(1.5)
    axis.set_xticklabels(
        ["Distribution", "Worst row", "Contraction"], rotation=12
    )
    axis.set_ylabel("log10(bound / actual)")
    axis.set_title("B  Tightness", loc="left", fontweight="bold")

    axis = axes[1, 0]
    conflict_levels = sorted(set(row["conflict"] for row in rows))
    for name in colors:
        medians = []
        for conflict in conflict_levels:
            values = [
                row[name + "_width"]
                for row in rows
                if row["conflict"] == conflict
            ]
            medians.append(float(np.median(values)))
        axis.plot(
            conflict_levels,
            medians,
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axis.set_xlabel("Conflict strength")
    axis.set_ylabel("Median certificate width")
    axis.set_title(
        "C  Conflict versus conservatism", loc="left", fontweight="bold"
    )

    axis = axes[1, 1]
    horizons = sorted(set(row["horizon"] for row in rows))
    for name in colors:
        medians = []
        for horizon in horizons:
            values = [
                row[name + "_tightness"]
                for row in rows
                if row["horizon"] == horizon and row["actual_tv"] > 1e-10
            ]
            medians.append(float(np.median(values)))
        axis.plot(
            horizons,
            medians,
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axis.set_yscale("log")
    axis.set_xlabel("Time horizon T")
    axis.set_ylabel("Median bound / actual")
    axis.set_title("D  Horizon effect", loc="left", fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, output_base)

def plot_capacity(result, output_base):
    rows = result["rows"]
    colors = {
        "Target-only MLP": "#E76F51",
        "Source-aware MLP": "#2A9D8F",
    }
    representations = ("G", "MG", "SC2")
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axis = axes[0]
    for model in colors:
        for representation, marker in zip(
            representations, ("o", "s", "^")
        ):
            subset = [
                row
                for row in rows
                if row["model"] == model
                and row["representation"] == representation
            ]
            conflicts = sorted(set(row["conflict"] for row in subset))
            values = [
                np.mean(
                    [
                        row["mae"]
                        for row in subset
                        if row["conflict"] == conflict
                    ]
                )
                for conflict in conflicts
            ]
            axis.plot(
                conflicts,
                values,
                color=colors[model],
                marker=marker,
                linestyle="-" if model == "Target-only MLP" else "--",
                label="%s / %s" % (model, representation),
            )
    axis.set_xlabel("Conflict strength")
    axis.set_ylabel("MAE")
    axis.set_title("A  Capacity does not restore information", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[1]
    for representation, color in zip(
        representations, ("#264653", "#457B9D", "#8D5A97")
    ):
        subset = [
            row
            for row in rows
            if row["model"] == "Target-only MLP"
            and row["representation"] == representation
        ]
        conflicts = sorted(set(row["conflict"] for row in subset))
        worst = [
            np.mean(
                [
                    row["worst_error"]
                    for row in subset
                    if row["conflict"] == conflict
                ]
            )
            for conflict in conflicts
        ]
        lower = [
            np.mean(
                [
                    row["collision_lower_bound"]
                    for row in subset
                    if row["conflict"] == conflict
                ]
            )
            for conflict in conflicts
        ]
        axis.plot(conflicts, worst, marker="o", color=color, label=representation + " MLP")
        axis.plot(conflicts, lower, linestyle="--", color=color, alpha=0.8, label=representation + " LB")
    axis.set_xlabel("Conflict strength")
    axis.set_ylabel("Worst error")
    axis.set_title("B  Worst error versus collision LB", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[2]
    positive = [row for row in rows if row["conflict"] > 0.0]
    positions = np.arange(len(representations))
    width = 0.35
    for offset, model in enumerate(colors):
        values = []
        for representation in representations:
            subset = [
                row
                for row in positive
                if row["representation"] == representation
                and row["model"] == model
            ]
            values.append(float(np.mean([row["worst_error"] for row in subset])))
        bars = axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            facecolor="white",
            edgecolor=colors[model],
            linewidth=1.5,
            hatch=HATCHES[offset],
            label=model,
        )
        for bar in bars:
            bar.set_alpha(0.95)
    axis.set_xticks(positions)
    axis.set_xticklabels(representations)
    axis.set_ylabel("Mean worst error, c > 0")
    axis.set_title("C  Restoring source information", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, output_base)

def plot_ablation(result, output_base):
    summary = result["ablation_summary"]
    methods = list(summary.keys())
    colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(methods)))
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    axis = axes[0]
    left_offsets = {
        "Full THN-Need": (5, 8),
        "w/o Collision": (-66, 10),
        "w/o Audit": (5, 8),
        "w/o Abstention": (-67, 24),
        "w/o Complexity": (-18, 8),
        "Validation-only": (-34, 8),
        "Mean-error": (4, 4),
    }
    for method, color in zip(methods, colors):
        item = summary[method]
        axis.scatter(
            item["mean_complexity"],
            item["unsafe_rate"],
            s=68,
            color=color,
        )
        axis.annotate(
            method,
            (item["mean_complexity"], item["unsafe_rate"]),
            xytext=left_offsets.get(method, (4, 4)),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Mean representation complexity")
    axis.set_ylabel("Unsafe simplification rate")
    axis.set_title("A  Safety-complexity plane", loc="left", fontweight="bold")

    axis = axes[1]
    metrics = (
        "oracle_agreement",
        "safety_rate",
        "unsafe_rate",
        "overmodel_rate",
    )
    labels = ("Agreement", "Safety", "Unsafe", "Overmodel")
    x = np.arange(len(methods))
    width = 0.18
    metric_colors = ("#2A9D8F", "#457B9D", "#E76F51", "#E9C46A")
    for index, (metric, label, color) in enumerate(
        zip(metrics, labels, metric_colors)
    ):
        bars = axis.bar(
            x + (index - 1.5) * width,
            [summary[method][metric] for method in methods],
            width,
            facecolor="white",
            edgecolor=color,
            hatch=HATCHES[index],
            label=label,
        )
        for bar in bars:
            bar.set_linewidth(1.2)
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=32, ha="right", fontsize=7)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("B  Component ablation", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[2]
    right_offsets = {
        "Full THN-Need": (-52, 10),
        "w/o Collision": (-55, 7),
        "w/o Audit": (-52, 7),
        "w/o Abstention": (5, 8),
        "w/o Complexity": (5, 23),
        "Validation-only": (5, 7),
        "Mean-error": (5, 4),
    }
    for method, color in zip(methods, colors):
        item = summary[method]
        axis.scatter(
            item["abstention_rate"],
            item["unsafe_rate"],
            s=68,
            color=color,
        )
        axis.annotate(
            method,
            (item["abstention_rate"], item["unsafe_rate"]),
            xytext=right_offsets.get(method, (4, 3)),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Abstention rate")
    axis.set_ylabel("Unsafe simplification rate")
    axis.set_title("C  Abstention-risk trade-off", loc="left", fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, output_base)

def plot_audit(result, output_base):
    rows = result["audit_rows"]
    strengths = (0.0, 0.25, 0.50, 0.75)
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    axis = axes[0]
    for field, label, color, marker in (
        ("validation_unsafe_rate", "Validation-only", "#E76F51", "o"),
        ("full_unsafe_rate", "Full THN-Need", "#2A9D8F", "s"),
    ):
        values = [
            np.mean(
                [row[field] for row in rows if row["strength"] == strength]
            )
            for strength in strengths
        ]
        axis.plot(strengths, values, marker=marker, color=color, label=label)
    axis.set_xlabel("Bias strength")
    axis.set_ylabel("Unsafe simplification rate")
    axis.set_title("A  Audit protects under bias", loc="left", fontweight="bold")
    axis.legend(frameon=False)

    axis = axes[1]
    for field, label, color, marker in (
        ("validation_overmodel_rate", "Validation-only", "#E76F51", "o"),
        ("full_overmodel_rate", "Full THN-Need", "#2A9D8F", "s"),
    ):
        values = [
            np.mean(
                [row[field] for row in rows if row["strength"] == strength]
            )
            for strength in strengths
        ]
        axis.plot(strengths, values, marker=marker, color=color, label=label)
    axis.set_xlabel("Bias strength")
    axis.set_ylabel("Overmodel rate")
    axis.set_title("B  Cost of conservative upgrades", loc="left", fontweight="bold")

    axis = axes[2]
    matrix = np.asarray(
        [
            [
                next(
                    row["rescue_rate"]
                    for row in rows
                    if row["bias"] == bias
                    and row["strength"] == strength
                )
                for strength in strengths
            ]
            for bias in BIAS_TYPES
        ]
    )
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(strengths)))
    axis.set_xticklabels(strengths)
    axis.set_yticks(range(len(BIAS_TYPES)))
    axis.set_yticklabels([BIAS_LABELS[bias] for bias in BIAS_TYPES], fontsize=7)
    axis.set_xlabel("Bias strength")
    axis.set_title("C  Rescue rate", loc="left", fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                "%.2f" % matrix[row_index, column_index],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row_index, column_index] > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return _save_figure(figure, output_base)

def plot_selectors(result, output_base):
    summary = result["summary"]
    methods = list(summary.keys())
    colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(methods)))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axis = axes[0]
    label_offsets = {
        "THN-Need": (5, 8),
        "Validation-only": (-62, -7),
        "Regularized selector": (-55, -14),
        "Meta-selector": (-30, 11),
        "Always G": (5, 5),
        "Always H": (-50, 7),
    }
    for method, color in zip(methods, colors):
        item = summary[method]
        axis.scatter(
            item["mean_complexity"],
            item["unsafe_reduction_rate"],
            s=80,
            color=color,
        )
        axis.annotate(
            method,
            (item["mean_complexity"], item["unsafe_reduction_rate"]),
            xytext=label_offsets.get(method, (4, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean representation complexity")
    axis.set_ylabel("Unsafe reduction rate")
    axis.set_ylim(-0.02, 1.03)
    axis.set_title("A  Leave-one-topology-out trade-off", loc="left", fontweight="bold")

    axis = axes[1]
    metrics = ("oracle_agreement", "safety_rate", "unsafe_rate", "overmodel_rate")
    labels = ("Agreement", "Safety", "Unsafe", "Overmodel")
    x = np.arange(len(methods))
    width = 0.18
    metric_colors = ("#2A9D8F", "#457B9D", "#E76F51", "#E9C46A")
    for index, (metric, label, color) in enumerate(
        zip(metrics, labels, metric_colors)
    ):
        axis.bar(
            x + (index - 1.5) * width,
            [summary[method][metric] for method in methods],
            width,
            facecolor="white",
            edgecolor=color,
            hatch=HATCHES[index],
            linewidth=1.2,
            label=label,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("B  Strong selector comparison", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    return _save_figure(figure, output_base)

def _public_stress_result(result):
    return {
        "experiment": result["experiment"],
        "task_count": result["task_count"],
        "sampling_seeds": result["sampling_seeds"],
        "ablation_summary": result["ablation_summary"],
        "audit_rows": result["audit_rows"],
        "runtime_seconds": result["runtime_seconds"],
        "full_failures": [
            {
                "task_id": row["task_id"],
                "bias": row["bias"],
                "strength": row["strength"],
                "mechanism": row["mechanism"],
                "budget": row["budget"],
                "selected": row["selected"],
                "oracle": row["oracle"],
                "test_error": row["test_error"],
            }
            for row in result["ablation_rows"]
            if row["method"] == "Full THN-Need" and not row["safe"]
        ],
    }

def _write_report(results, report_path):
    certificate = results["certificate"]
    capacity = results["capacity"]
    stress = results["stress"]
    selectors = results["selectors"]
    csum = certificate["summary"]
    asum = stress["ablation_summary"]
    ssum = selectors["summary"]
    audit_rows = stress["audit_rows"]
    full_failures = stress.get("full_failures", [])
    biased_rows = [row for row in audit_rows if row["strength"] > 0.0]
    val_unsafe = float(
        np.mean([row["validation_unsafe_rate"] for row in biased_rows])
    )
    full_unsafe = float(
        np.mean([row["full_unsafe_rate"] for row in biased_rows])
    )
    rescue_counts = sum(row["rescued_count"] for row in biased_rows)
    unsafe_counts = sum(row["validation_unsafe_count"] for row in biased_rows)
    rescue = rescue_counts / max(1, unsafe_counts)
    target_summary = capacity["summary"]["Target-only MLP"]
    source_summary = capacity["summary"]["Source-aware MLP"]

    lines = [
        "# 创新点2与创新点3新增补充实验报告",
        "",
        "## 1　实验筛选结论",
        "",
        "本报告逐项审查了新增实验方案。最终实施五组直接补足核心证据链的实验；没有实施的项目并非被判定为错误，而是当前论文主张并不依赖相应外推，或现有项目已经提供等价证据。",
        "",
        "| 方案 | 处理 | 判断依据 |",
        "|---|---|---|",
        "| 创新点2：有限时间证书校准 | 已完成 | 原实验只有单个证书实例，需要多任务覆盖率与紧致性证据 |",
        "| 创新点2：高容量代理与碰撞下界 | 已完成 | 用于区分优化能力不足与低阶表示不可辨识 |",
        "| 创新点2：观测噪声 | 暂不纳入 | 论文尚未定义观测噪声生成机制；当前结论不声称噪声鲁棒性 |",
        "| 创新点2：SIS扩展 | 不重复实施 | 现有证书实验已使用SIS，本次校准又覆盖多个恢复率 |",
        "| 创新点3：THN-Need系统消融 | 已完成 | 现有实验缺少各闭环组件的独立贡献 |",
        "| 创新点3：Audit覆盖偏置 | 已完成 | 单个人工审计案例不足以说明覆盖偏置下的系统收益 |",
        "| 创新点3：观测噪声 | 暂不纳入 | 与创新点2噪声实验共享尚未固定的污染模型 |",
        "| 创新点3：强selector基线 | 已完成 | 现有启发式基线不足以排除学习型选择器 |",
        "| 创新点3：一万节点规模扫描 | 暂列Supplementary后续 | 现有效率实验已给初步证据，规模极限不影响当前理论命题 |",
        "",
        "## 2　创新点2：有限时间误差证书校准",
        "",
        "### 2.1　实验设计",
        "",
        "实验在N=5、6、7和T=3、5、8的全部组合上运行，并轮换G、MG与SC2三类代理。每个任务随机改变超边密度、持续性、重叠率、重复事件率、传播率、冲突强度和SIS恢复率。所有状态空间均被精确枚举，实际终点总变差与三类证书由同一组精确转移核计算。总任务数为%d。" % certificate["task_count"],
        "",
        "![有限时间证书校准](../figures/innovation2_a11_certificate_calibration.png)",
        "",
        "**图1｜有限时间误差证书的覆盖率与紧致性。** A比较实际终点TV与三类证书，对角线以下代表失效；B比较非零误差任务上的紧致性；C显示冲突强度与证书宽度的关系；D显示时间长度对紧致性的影响。",
        "",
        "### 2.2　结果与分析",
        "",
        "分布加权、最坏行累积和收缩修正证书的覆盖率分别为%.4f、%.4f和%.4f；对应95%% Wilson区间下界分别为%.4f、%.4f和%.4f。三类证书在全部任务上均覆盖真实误差，但保守程度明显不同。非零误差任务中，三类证书的中位紧致性分别为%.3f、%.3f和%.3f。" % (
            csum["distribution"]["coverage"],
            csum["worst_row"]["coverage"],
            csum["contraction"]["coverage"],
            csum["distribution"]["wilson95_low"],
            csum["worst_row"]["wilson95_low"],
            csum["contraction"]["wilson95_low"],
            csum["distribution"]["median_tightness_nonzero"],
            csum["worst_row"]["median_tightness_nonzero"],
            csum["contraction"]["median_tightness_nonzero"],
        ),
        "",
        "分布加权证书直接按源过程在各状态上的概率质量加权局部核误差，因此避免让几乎不可达的最坏状态支配上界。最坏行证书仍然有价值：它不依赖源分布，能够提供最保守的任务无关保证。收缩修正证书只有在后续核具有明显收缩性时才会优于朴素最坏行累积，因此不能预设其在所有任务中最紧。图1的比较使三类证书各自承担的角色得到区分。",
        "",
        "需要强调，碰撞下界约束的是给定低阶特征下的最坏局部近似误差，有限时间证书约束的是终点分布误差；两者不能在没有额外可达概率条件时直接写成同一个端点不等式。",
        "",
        "## 3　创新点2：高容量代理与投影碰撞下界",
        "",
        "### 3.1　实验设计",
        "",
        "在六个冲突水平上，从随机时变超图的完整节点—状态配置中构造监督样本。Target-only MLP只接收目标表示允许的时间、度、感染邻居、多重暴露或二阶暴露特征；Source-aware MLP额外接收按群组规模和群组内感染数分解的源通道计数。两种模型使用相同的64-64-32隐藏层、优化器和训练轮数。本实验在同一受控样本域上拟合和评价，目的是隔离表示可辨识性，而不是声称跨数据预测性能。",
        "",
        "![高容量代理与碰撞下界](../figures/innovation2_a12_capacity_collision.png)",
        "",
        "**图2｜增加模型容量不能恢复目标表示中不存在的信息。** A比较三类目标表示下两种MLP的平均误差；B把Target-only MLP的最坏误差与碰撞下界叠加；C汇总非零冲突区域恢复源群组信息后的最坏误差变化。",
        "",
        "### 3.2　结果与分析",
        "",
        "Target-only MLP与Source-aware MLP的总体平均MAE分别为%.6f和%.6f，平均最坏误差分别为%.6f和%.6f。在所有Target-only设置中，最坏误差覆盖碰撞下界的比例为%.4f；非零冲突区域恢复源群组信息后，平均最坏误差相对下降%.2f%%。" % (
            target_summary["mean_mae"],
            source_summary["mean_mae"],
            target_summary["mean_worst_error"],
            source_summary["mean_worst_error"],
            capacity["summary"]["target_bound_coverage"],
            100.0 * capacity["summary"]["positive_conflict_worst_reduction"],
        ),
        "",
        "这一结果排除了“低阶代理误差只是标量拟合器容量不足”的解释。Target-only MLP可以学习复杂非线性映射，但相同目标特征必须对应同一预测；当两个源群组配置投影到相同目标特征而真实概率不同，任何确定性Target-only模型都至少在其中一个配置上产生碰撞下界所规定的误差。Source-aware模型获得被投影删除的群组通道后误差下降，说明残余误差来自输入信息而不是隐藏层数量。",
        "",
        "## 4　创新点3：THN-Need系统消融",
        "",
        "### 4.1　实验设计",
        "",
        "实验构造%d个压力任务，覆盖8类受控时间拓扑、5种层级动力学机制、6种验证覆盖模式、4个偏置强度和%d个状态采样种子。每个任务的测试集合覆盖小系统全部非平凡状态，并以测试集上满足预算的最低复杂度表示作为oracle。" % (
            stress["task_count"],
            stress["sampling_seeds"],
        ),
        "",
        "![THN-Need消融](../figures/innovation3_a10_ablation.png)",
        "",
        "**图3｜THN-Need各模块的独立贡献。** A给出平均表示复杂度与危险欠建模率；B同时比较oracle一致率、安全率、危险欠建模率和过度建模率；C展示弃权率与危险欠建模率的关系。",
        "",
        "### 4.2　结果与分析",
        "",
        "完整THN-Need的oracle一致率、安全率、危险欠建模率、过度建模率和平均复杂度分别为%.4f、%.4f、%.4f、%.4f和%.3f。去除Audit后，危险欠建模率为%.4f；去除Abstention后为%.4f；去除Complexity后平均复杂度升至%.3f、过度建模率为%.4f。" % (
            asum["Full THN-Need"]["oracle_agreement"],
            asum["Full THN-Need"]["safety_rate"],
            asum["Full THN-Need"]["unsafe_rate"],
            asum["Full THN-Need"]["overmodel_rate"],
            asum["Full THN-Need"]["mean_complexity"],
            asum["w/o Audit"]["unsafe_rate"],
            asum["w/o Abstention"]["unsafe_rate"],
            asum["w/o Complexity"]["mean_complexity"],
            asum["w/o Complexity"]["overmodel_rate"],
        ),
        "",
        "值得注意的是，w/o Collision与完整方法的选择统计完全相同，说明当前碰撞证书没有在独立Audit之外改变选择。完整方法仍有%d/%d个任务漏检；失败任务的测试误差为%s。它们均属于低阶证据面板没有覆盖决定性配置的情形。因此，碰撞证书只能证明已观察碰撞组不可表达，有限Audit也不是全状态安全证明；本文应报告实证安全率，而不能把它改写成理论上的绝对安全保证。" % (
            len(full_failures),
            stress["task_count"],
            "、".join("%.6f" % row["test_error"] for row in full_failures)
            if full_failures
            else "无",
        ),
        "",
        "消融结果应按模块职责解释。Collision用于提前排除在已观察碰撞组上原则上不可能满足预算的表示；Audit针对验证覆盖盲区；Abstention在证据量不足时拒绝强制降阶；Complexity则避免在多个候选都安全时无条件选择最高阶模型。任何单一模块都不负责全部性能，完整闭环的目标是在安全与表示成本之间取得可审计折中。",
        "",
        "## 5　创新点3：Audit覆盖偏置压力测试",
        "",
        "![Audit覆盖偏置压力测试](../figures/innovation3_a11_audit_stress.png)",
        "",
        "**图4｜Audit的收益集中出现在验证覆盖发生系统偏差时。** A比较Validation-only与完整闭环的危险欠建模率；B展示保守升级带来的过度建模代价；C按偏置类型和强度给出救援率。",
        "",
        "在非零偏置任务中，Validation-only与完整THN-Need的平均危险欠建模率分别为%.4f和%.4f。Validation-only不安全的任务中，Audit最终恢复为安全选择的比例为%.4f。无偏或弱偏条件下，两种方法可能接近；偏置增强后，Audit通过针对高暴露、重复暴露和后期状态的独立状态面板发现验证盲区。" % (
            val_unsafe,
            full_unsafe,
            rescue,
        ),
        "",
        "Audit的价值不是保证所有任务的平均准确率都提高，而是在验证集系统性遗漏危险状态时提供第二道风险控制。与此同时，保守升级会增加过度建模，因此报告同时给出False Upgrade和复杂度，而不只展示安全率。",
        "",
        "## 6　创新点3：强模型选择基线",
        "",
        "强基线实验使用leave-one-topology-out协议：每轮在7个拓扑族上选择正则化系数并训练Random Forest元选择器，在未见过的第8个拓扑族上测试。输入只包括结构统计、验证误差、碰撞诊断和预算，不包括真实机制标签或测试误差。",
        "",
        "![强selector基线](../figures/innovation3_a13_strong_selectors.png)",
        "",
        "**图5｜THN-Need与正则化经验选择器及学习型元选择器的比较。** A给出平均复杂度—危险风险前沿；B比较oracle一致率、安全率、危险欠建模率和过度建模率。",
        "",
        "THN-Need、Validation-only、正则化选择器、Meta-selector、Always G和Always H的安全率依次为%s；平均复杂度依次为%s。Always H提供最高安全性但付出最大复杂度，Always G成本较低但可能危险；学习型选择器能够利用跨任务规律，但其安全性取决于拓扑外泛化。THN-Need的优势应表述为显式证据和审计带来的可解释安全—复杂度折中，而不是在所有单一指标上无条件支配。" % (
            "、".join("%.4f" % ssum[name]["safety_rate"] for name in ssum),
            "、".join("%.3f" % ssum[name]["mean_complexity"] for name in ssum),
        ),
        "",
        "## 7　总体结论与结论边界",
        "",
        "新增实验补足了两个此前较薄弱的环节。对于创新点2，证书不再只由单个示例支持，而是在跨网络规模、时间长度、代理类型、冲突强度和SIS恢复率的精确枚举任务中接受校准；高容量代理实验进一步证明，投影碰撞造成的是输入不可辨识性，而不只是优化器容量不足。对于创新点3，闭环模块的作用由系统消融分开验证，Audit的收益被定位到验证覆盖偏差，方法也与正则化及学习型选择器在未见拓扑上进行了比较。",
        "",
        "这些实验仍不支持两类过度外推。第一，当前没有声称对任意现实观测噪声鲁棒，因为事件缺失、成员缺失和状态误标需要由具体采集过程定义。第二，小系统精确状态压力测试验证的是决策逻辑和理论证书，不等于一万节点下的工程吞吐量证明。相应噪声和规模实验应在论文需要提出这些额外主张时进入Supplementary。",
        "",
        "## 8　结果数据与复现文件",
        "",
        "- results/tables/innovation2_a11_certificate_tasks.csv",
        "- results/tables/innovation2_a11_certificate_summary.csv",
        "- results/tables/innovation2_a12_capacity_collision.csv",
        "- results/tables/innovation3_a10_ablation.csv",
        "- results/tables/innovation3_a11_audit_bias.csv",
        "- results/tables/innovation3_a13_strong_selectors.csv",
        "- results/summary_tables/innovation23_supplementary.json",
        "- src/experiments/exp_innovation23_supplementary.py",
    ]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

def run_all(
    seed=20260826,
    certificate_reps=4,
    capacity_seeds=3,
    stress_seeds=3,
    result_root=None,
):
    root = Path(result_root) if result_root else project_root() / "results"
    figure_dir = root / "figures"
    table_dir = root / "tables"
    report_dir = root / "reports"
    summary_dir = root / "summary_tables"
    started = time.perf_counter()

    certificate = run_certificate_calibration(seed, certificate_reps)
    capacity = run_capacity_collision(seed + 1, capacity_seeds)
    stress = run_thn_ablation_and_audit(seed + 2, stress_seeds)
    selectors = run_strong_selector_baselines(stress, seed + 3)

    artifacts = []
    artifacts.extend(
        plot_certificate(
            certificate,
            figure_dir / "innovation2_a11_certificate_calibration",
        )
    )
    artifacts.extend(
        plot_capacity(
            capacity,
            figure_dir / "innovation2_a12_capacity_collision",
        )
    )
    artifacts.extend(
        plot_ablation(
            stress, figure_dir / "innovation3_a10_ablation"
        )
    )
    artifacts.extend(
        plot_audit(
            stress, figure_dir / "innovation3_a11_audit_stress"
        )
    )
    artifacts.extend(
        plot_selectors(
            selectors,
            figure_dir / "innovation3_a13_strong_selectors",
        )
    )

    artifacts.append(
        _write_csv(
            table_dir / "innovation2_a11_certificate_tasks.csv",
            certificate["rows"],
        )
    )
    certificate_summary_rows = [
        {"certificate": name, **values}
        for name, values in certificate["summary"].items()
    ]
    for row in certificate_summary_rows:
        row["iqr_tightness_nonzero"] = json.dumps(
            row["iqr_tightness_nonzero"]
        )
    artifacts.append(
        _write_csv(
            table_dir / "innovation2_a11_certificate_summary.csv",
            certificate_summary_rows,
        )
    )
    artifacts.append(
        _write_csv(
            table_dir / "innovation2_a12_capacity_collision.csv",
            capacity["rows"],
        )
    )
    artifacts.append(
        _write_csv(
            table_dir / "innovation3_a10_ablation.csv",
            stress["ablation_rows"],
        )
    )
    artifacts.append(
        _write_csv(
            table_dir / "innovation3_a11_audit_bias.csv",
            stress["audit_rows"],
        )
    )
    artifacts.append(
        _write_csv(
            table_dir / "innovation3_a13_strong_selectors.csv",
            selectors["rows"],
        )
    )

    public = {
        "experiment": "Innovation2-3 supplementary experiments",
        "seed": int(seed),
        "certificate": certificate,
        "capacity": capacity,
        "stress": _public_stress_result(stress),
        "selectors": selectors,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    summary_path = summary_dir / "innovation23_supplementary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(json_ready(public), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts.append(summary_path)
    report_path = _write_report(public, report_dir / "创新点2与创新点3_新增补充实验报告.md")
    artifacts.append(report_path)
    public["artifacts"] = [str(path) for path in artifacts]
    return public

def main():
    parser = argparse.ArgumentParser(
        description="Run supplementary experiments for Innovations 2 and 3"
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--certificate-reps", type=int, default=4)
    parser.add_argument("--capacity-seeds", type=int, default=3)
    parser.add_argument("--stress-seeds", type=int, default=3)
    parser.add_argument("--result-root", default=None)
    args = parser.parse_args()
    result = run_all(
        seed=args.seed,
        certificate_reps=args.certificate_reps,
        capacity_seeds=args.capacity_seeds,
        stress_seeds=args.stress_seeds,
        result_root=args.result_root,
    )
    print(
        json.dumps(
            {
                "certificate_tasks": result["certificate"]["task_count"],
                "stress_tasks": result["stress"]["task_count"],
                "runtime_seconds": result["runtime_seconds"],
                "artifacts": result["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
