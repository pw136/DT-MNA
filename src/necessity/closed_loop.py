
from __future__ import division

from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar

from ..approximate import cvar, projection_collision_lower_bound
from ..dynamics import (
    ExposureSemantics,
    PairwiseSIParameters,
    SimplicialSIParameters,
    infection_probabilities,
    simulate_temporal_si,
)
from ..projection import (
    project_temporal_hypergraph_to_multigraph,
    project_temporal_hypergraph_to_simplegraph,
    project_temporal_hypergraph_to_simplicial,
)
from ..temporal_structures import TemporalSimpleGraph

INDEPENDENT = ExposureSemantics.INDEPENDENT_EXPOSURES
MODEL_ORDER = ("SG", "G", "MG", "SC2", "H")
MODEL_COMPLEXITY = {
    "SG": 0.5,
    "G": 1.0,
    "MG": 2.0,
    "SC2": 4.5,
    "H": 10.0,
}

@dataclass
class ErrorSummary:
    sample_count: int
    mean_absolute_error: float
    root_mean_square_error: float
    worst_error: float
    cvar95: float
    empirical_upper: float

@dataclass
class CandidateAssessment:
    name: str
    label: str
    complexity: float
    parameter_count: int
    network: object
    parameters: object
    qstar_feasible: bool
    collision_lower_bound: float
    train: ErrorSummary
    validation: ErrorSummary
    audit: ErrorSummary
    test: ErrorSummary
    validation_safe: bool
    audit_safe: bool
    test_safe: bool

    def serializable(self):
        return {
            "name": self.name,
            "label": self.label,
            "complexity": float(self.complexity),
            "parameter_count": int(self.parameter_count),
            "qstar_feasible": bool(self.qstar_feasible),
            "collision_lower_bound": float(self.collision_lower_bound),
            "train": _summary_dict(self.train),
            "validation": _summary_dict(self.validation),
            "audit": _summary_dict(self.audit),
            "test": _summary_dict(self.test),
            "validation_safe": bool(self.validation_safe),
            "audit_safe": bool(self.audit_safe),
            "test_safe": bool(self.test_safe),
        }

@dataclass
class ClosedLoopDecision:
    decision: str
    selected_name: str
    selected_network: object
    selected_parameters: object
    error_budget: float
    exact_tolerance: float
    validation_choice: str
    upgrade_path: tuple
    reason_codes: tuple
    assessments: tuple
    oracle_test_choice: str
    correct_vs_test_oracle: bool
    safe_on_test: bool
    minimal_on_test: bool

    def serializable(self):
        return {
            "decision": self.decision,
            "selected_name": self.selected_name,
            "error_budget": float(self.error_budget),
            "exact_tolerance": float(self.exact_tolerance),
            "validation_choice": self.validation_choice,
            "upgrade_path": list(self.upgrade_path),
            "reason_codes": list(self.reason_codes),
            "oracle_test_choice": self.oracle_test_choice,
            "correct_vs_test_oracle": bool(self.correct_vs_test_oracle),
            "safe_on_test": bool(self.safe_on_test),
            "minimal_on_test": bool(self.minimal_on_test),
            "assessments": {
                assessment.name: assessment.serializable()
                for assessment in self.assessments
            },
        }

def _summary_dict(summary):
    return {
        "sample_count": int(summary.sample_count),
        "mean_absolute_error": float(summary.mean_absolute_error),
        "root_mean_square_error": float(summary.root_mean_square_error),
        "worst_error": float(summary.worst_error),
        "cvar95": float(summary.cvar95),
        "empirical_upper": float(summary.empirical_upper),
    }

def _static_simple_graph(temporal_hypergraph):
    temporal = project_temporal_hypergraph_to_simplegraph(temporal_hypergraph)
    union = tuple(
        sorted(
            {
                edge
                for time_point in temporal.time_points
                for edge in temporal.active_edges(time_point)
            },
            key=lambda edge: (
                temporal.node_to_index[edge[0]],
                temporal.node_to_index[edge[1]],
            ),
        )
    )
    return TemporalSimpleGraph(
        temporal.nodes,
        {time_point: union for time_point in temporal.time_points},
    )

def build_candidate_networks(temporal_hypergraph):

    return {
        "SG": _static_simple_graph(temporal_hypergraph),
        "G": project_temporal_hypergraph_to_simplegraph(temporal_hypergraph),
        "MG": project_temporal_hypergraph_to_multigraph(temporal_hypergraph),
        "SC2": project_temporal_hypergraph_to_simplicial(
            temporal_hypergraph, 2
        ),
        "H": temporal_hypergraph,
    }

def generate_state_splits(
    temporal_hypergraph,
    seed=0,
    train_count=4,
    validation_count=4,
    audit_count=4,
    test_count=8,
):

    rng = np.random.RandomState(int(seed))
    fractions = np.asarray([0.04, 0.08, 0.14, 0.22, 0.34, 0.48, 0.62])
    split_counts = {
        "train": int(train_count),
        "validation": int(validation_count),
        "audit": int(audit_count),
        "test": int(test_count),
    }
    result = {}
    node_count = len(temporal_hypergraph.nodes)
    for split_offset, (split, count) in enumerate(split_counts.items()):
        by_time = {}
        for time_point in temporal_hypergraph.time_points:
            states = []
            groups = sorted(
                temporal_hypergraph.active_hyperedges(time_point),
                key=lambda group: (-len(group), tuple(group)),
            )
            pair_counts = {}
            for group in groups:
                for left_index in range(len(group)):
                    for right_index in range(left_index + 1, len(group)):
                        pair = tuple(
                            sorted(
                                (group[left_index], group[right_index]),
                                key=temporal_hypergraph.node_to_index.get,
                            )
                        )
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
            repeated_pairs = sorted(
                pair_counts,
                key=lambda pair: (
                    -pair_counts[pair],
                    temporal_hypergraph.node_to_index[pair[0]],
                    temporal_hypergraph.node_to_index[pair[1]],
                ),
            )
            for index in range(count):
                if repeated_pairs and index == 0:
                    pair = repeated_pairs[split_offset % len(repeated_pairs)]
                    state = np.zeros(node_count, dtype=np.int8)
                    infected_node = pair[split_offset % 2]
                    state[
                        temporal_hypergraph.node_to_index[infected_node]
                    ] = 1
                    states.append(state)
                    continue
                if groups and index < min(count, 5):
                    probe_index = index - 1
                    group = groups[
                        (probe_index // 2 + split_offset) % len(groups)
                    ]
                    infected = min(
                        len(group) - 1,
                        1 + (probe_index % 2),
                    )
                    state = np.zeros(node_count, dtype=np.int8)
                    for node in group[:infected]:
                        state[
                            temporal_hypergraph.node_to_index[node]
                        ] = 1
                    states.append(state)
                    continue
                fraction = float(
                    fractions[
                        (index + len(states) + int(time_point)) % len(fractions)
                    ]
                )
                infected = max(
                    1,
                    min(node_count - 1, int(round(fraction * node_count))),
                )
                chosen = rng.choice(node_count, size=infected, replace=False)
                state = np.zeros(node_count, dtype=np.int8)
                state[chosen] = 1
                states.append(state)
            by_time[time_point] = tuple(states)
        result[split] = by_time
    return result

def _pairwise_exposures(network, state, time_point, weighted):
    values = np.zeros(len(network.nodes), dtype=float)
    for node_index, node in enumerate(network.nodes):
        if state[node_index]:
            continue
        if weighted:
            values[node_index] = sum(
                weight
                for other, weight in network.weighted_neighbors(
                    node, time_point
                )
                if state[network.node_to_index[other]]
            )
        else:
            values[node_index] = sum(
                state[network.node_to_index[other]]
                for other in network.neighbors(node, time_point)
            )
    return values[:, None]

def _simplicial_exposures(network, state, time_point):
    values = np.zeros((len(network.nodes), 2), dtype=float)
    for node_index, node in enumerate(network.nodes):
        if state[node_index]:
            continue
        first = 0
        second = 0
        for cell in network.incident_maximal_simplices(node, time_point):
            infected = sum(
                state[network.node_to_index[other]]
                for other in cell
                if other != node
            )
            first += infected
            second += infected * (infected - 1) // 2
        values[node_index] = (first, second)
    return values

def _features(name, network, state, time_point):
    if name in ("SG", "G"):
        return _pairwise_exposures(
            network, state, time_point, weighted=False
        )
    if name == "MG":
        return _pairwise_exposures(
            network, state, time_point, weighted=True
        )
    if name == "SC2":
        return _simplicial_exposures(network, state, time_point)
    raise ValueError("features are not required for %s" % name)

def _collect_samples(
    source_network,
    source_parameters,
    candidate_name,
    candidate_network,
    states_by_time,
):
    times = []
    targets = []
    feature_rows = []
    collision_keys = []
    for time_point in source_network.time_points:
        for state in states_by_time[time_point]:
            state = np.asarray(state, dtype=np.int8)
            probabilities = infection_probabilities(
                state, source_network, time_point, source_parameters
            )
            features = _features(
                candidate_name, candidate_network, state, time_point
            )
            for node_index in range(len(source_network.nodes)):
                if state[node_index]:
                    continue
                row = tuple(float(value) for value in features[node_index])
                times.append(time_point)
                targets.append(float(probabilities[node_index]))
                feature_rows.append(row)
                if candidate_name == "SG":
                    collision_keys.append(row)
                else:
                    collision_keys.append((time_point,) + row)
    return {
        "times": np.asarray(times),
        "targets": np.asarray(targets, dtype=float),
        "features": np.asarray(feature_rows, dtype=float),
        "collision_keys": tuple(collision_keys),
    }

def _pair_prediction(beta, exposures):
    return 1.0 - np.power(1.0 - float(beta), exposures)

def _sc2_prediction(beta1, beta2, features):
    return 1.0 - (
        np.power(1.0 - float(beta1), features[:, 0])
        * np.power(1.0 - float(beta2), features[:, 1])
    )

def _fit_pairwise(samples, temporal):
    times = samples["times"]
    targets = samples["targets"]
    exposures = samples["features"][:, 0]
    parameter_times = tuple(sorted(set(times))) if temporal else (None,)
    values = {}
    for time_point in parameter_times:
        mask = np.ones(len(targets), dtype=bool)
        if temporal:
            mask = times == time_point

        def objective(beta):
            residual = _pair_prediction(beta, exposures[mask]) - targets[mask]
            return float(np.mean(residual ** 2))

        fitted = minimize_scalar(
            objective,
            bounds=(0.0, 0.95),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 500},
        )
        values[time_point] = float(fitted.x)
    if temporal:
        return values
    return {time_point: values[None] for time_point in sorted(set(times))}

def _fit_sc2(samples):
    times = samples["times"]
    targets = samples["targets"]
    features = samples["features"]
    rates = {}
    for time_point in sorted(set(times)):
        mask = times == time_point
        hazards = -np.log1p(
            -np.clip(targets[mask], 0.0, 1.0 - 1e-15)
        )
        fitted = lsq_linear(
            features[mask],
            hazards,
            bounds=(0.0, -np.log(0.05)),
            lsmr_tol="auto",
        )
        rates[(time_point, 1)] = float(1.0 - np.exp(-fitted.x[0]))
        rates[(time_point, 2)] = float(1.0 - np.exp(-fitted.x[1]))
    return rates

def _predict(name, parameters, samples):
    if name in ("SG", "G", "MG"):
        beta = np.asarray(
            [parameters[int(time_point)] for time_point in samples["times"]],
            dtype=float,
        )
        exposures = samples["features"][:, 0]
        return 1.0 - np.power(1.0 - beta, exposures)
    if name == "SC2":
        beta1 = np.asarray(
            [parameters[(int(time_point), 1)] for time_point in samples["times"]],
            dtype=float,
        )
        beta2 = np.asarray(
            [parameters[(int(time_point), 2)] for time_point in samples["times"]],
            dtype=float,
        )
        features = samples["features"]
        return 1.0 - (
            np.power(1.0 - beta1, features[:, 0])
            * np.power(1.0 - beta2, features[:, 1])
        )
    raise ValueError("unsupported candidate %s" % name)

def _summarize_errors(targets, predictions, confidence_margin):
    errors = np.abs(
        np.asarray(predictions, dtype=float)
        - np.asarray(targets, dtype=float)
    )
    if not len(errors):
        return ErrorSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    worst = float(errors.max())
    return ErrorSummary(
        int(len(errors)),
        float(errors.mean()),
        float(np.sqrt(np.mean(errors ** 2))),
        worst,
        float(cvar(errors, 0.95)),
        min(1.0, worst + float(confidence_margin)),
    )

def _assessment_for_candidate(
    name,
    networks,
    source_parameters,
    state_splits,
    error_budget,
    exact_tolerance,
    confidence_margin,
):
    source = networks["H"]
    if name == "H":
        empty = ErrorSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return CandidateAssessment(
            name,
            "Temporal hypergraph",
            MODEL_COMPLEXITY[name],
            len(getattr(source_parameters, "lambda_rates", {})),
            source,
            source_parameters,
            True,
            0.0,
            empty,
            empty,
            empty,
            empty,
            True,
            True,
            True,
        )

    samples = {
        split: _collect_samples(
            source,
            source_parameters,
            name,
            networks[name],
            state_splits[split],
        )
        for split in ("train", "validation", "audit", "test")
    }
    if name == "SG":
        raw_parameters = _fit_pairwise(samples["train"], temporal=False)
        parameters = PairwiseSIParameters(raw_parameters, INDEPENDENT)
        label = "Static simple graph"
    elif name in ("G", "MG"):
        raw_parameters = _fit_pairwise(samples["train"], temporal=True)
        parameters = PairwiseSIParameters(raw_parameters, INDEPENDENT)
        label = (
            "Temporal simple graph"
            if name == "G"
            else "Temporal multigraph"
        )
    else:
        raw_parameters = _fit_sc2(samples["train"])
        parameters = SimplicialSIParameters(raw_parameters, INDEPENDENT)
        label = "Temporal order-two simplicial model"

    summaries = {}
    for split in ("train", "validation", "audit", "test"):
        prediction = _predict(name, raw_parameters, samples[split])
        summaries[split] = _summarize_errors(
            samples[split]["targets"], prediction, confidence_margin
        )
    collision_targets = np.concatenate(
        (samples["train"]["targets"], samples["validation"]["targets"])
    )
    collision_features = (
        samples["train"]["collision_keys"]
        + samples["validation"]["collision_keys"]
    )
    lower, _ = projection_collision_lower_bound(
        collision_targets, collision_features
    )
    parameter_count = (
        len(raw_parameters)
        if name != "SC2"
        else len(raw_parameters)
    )
    return CandidateAssessment(
        name,
        label,
        MODEL_COMPLEXITY[name],
        int(parameter_count),
        networks[name],
        parameters,
        bool(summaries["train"].worst_error <= exact_tolerance),
        float(lower),
        summaries["train"],
        summaries["validation"],
        summaries["audit"],
        summaries["test"],
        bool(summaries["validation"].empirical_upper <= error_budget),
        bool(summaries["audit"].empirical_upper <= error_budget),
        bool(summaries["test"].worst_error <= error_budget),
    )

def _decision_label(selected_name, exact, high_order_proven=True):
    if selected_name == "SG":
        return "STATIC_GRAPH_SUFFICIENT"
    if selected_name == "G":
        return (
            "TEMPORAL_GRAPH_EXACT"
            if exact
            else "TEMPORAL_GRAPH_APPROXIMATE"
        )
    if selected_name == "MG":
        return "TEMPORAL_MULTIGRAPH_REQUIRED"
    if selected_name == "SC2":
        return "TEMPORAL_LOW_ORDER_REQUIRED"
    return (
        "TEMPORAL_HYPERGRAPH_REQUIRED"
        if high_order_proven
        else "INSUFFICIENT_EVIDENCE_SAFE_H"
    )

def run_closed_loop_decision(
    temporal_hypergraph,
    source_parameters,
    state_splits,
    error_budget=0.03,
    exact_tolerance=1e-10,
    confidence_margin=0.0,
    minimum_validation_states=1,
):

    networks = build_candidate_networks(temporal_hypergraph)
    assessments = tuple(
        _assessment_for_candidate(
            name,
            networks,
            source_parameters,
            state_splits,
            float(error_budget),
            float(exact_tolerance),
            float(confidence_margin),
        )
        for name in MODEL_ORDER
    )
    by_name = {assessment.name: assessment for assessment in assessments}
    validation_state_count = sum(
        len(states) for states in state_splits["validation"].values()
    )
    insufficient_states = (
        validation_state_count < int(minimum_validation_states)
    )
    validation_choice = (
        "H"
        if insufficient_states
        else next(
            assessment.name
            for assessment in assessments
            if assessment.validation_safe
        )
    )
    selected_index = MODEL_ORDER.index(validation_choice)
    upgrade_path = [validation_choice]
    while not by_name[MODEL_ORDER[selected_index]].audit_safe:
        selected_index += 1
        upgrade_path.append(MODEL_ORDER[selected_index])
    selected_name = MODEL_ORDER[selected_index]
    selected = by_name[selected_name]
    oracle_test_choice = next(
        assessment.name for assessment in assessments if assessment.test_safe
    )
    reason_codes = []
    if selected_name != "SG":
        reason_codes.append("static_graph_exceeds_budget")
    if selected_name not in ("SG", "G"):
        reason_codes.append("simple_graph_exceeds_budget")
    if selected_name in ("SC2", "H"):
        reason_codes.append("pairwise_models_exceed_budget")
    if selected_name == "H":
        if by_name["SC2"].collision_lower_bound > error_budget:
            reason_codes.append("projection_collision_proves_high_order_need")
        else:
            reason_codes.append("safe_fallback_after_low_order_failure")
    if insufficient_states:
        reason_codes.append("insufficient_validation_state_coverage")
    if len(upgrade_path) > 1:
        reason_codes.append("audit_triggered_upgrade")
    if selected.qstar_feasible:
        reason_codes.append("exact_mapping_detected")
    else:
        reason_codes.append("approximate_mapping_within_budget")
    high_order_proven = bool(
        by_name["SC2"].collision_lower_bound > error_budget
        and not insufficient_states
    )
    return ClosedLoopDecision(
        _decision_label(
            selected_name, selected.qstar_feasible, high_order_proven
        ),
        selected_name,
        selected.network,
        selected.parameters,
        float(error_budget),
        float(exact_tolerance),
        validation_choice,
        tuple(upgrade_path),
        tuple(reason_codes),
        assessments,
        oracle_test_choice,
        bool(selected_name == oracle_test_choice),
        bool(selected.test_safe),
        bool(selected_name == oracle_test_choice),
    )

def simulate_selected_model(decision, initial_state, seed=None, uniforms=None):

    return simulate_temporal_si(
        decision.selected_network,
        decision.selected_parameters,
        initial_state,
        seed=seed,
        uniforms=uniforms,
    )
