
from __future__ import division

import itertools
import math
import time
from collections import OrderedDict

import numpy as np

from ..dynamics import (
    ExposureSemantics,
    HypergraphSIParameters,
    simulate_temporal_si,
)
from ..necessity import (
    generate_state_splits,
    run_closed_loop_decision,
    simulate_selected_model,
)
from ..temporal_structures import TemporalHypergraph
from ..theory import (
    C3ProjectionRoute,
    build_temporal_c1_lambda_rates,
    build_temporal_c3_lambda_rates,
)
from .conversion_validation_data import load_all_validation_datasets
from .innovation2_core import build_conflicting_lambda_rates

INDEPENDENT = ExposureSemantics.INDEPENDENT_EXPOSURES
ORDER = ("SG", "G", "MG", "SC2", "H")
COMPLEXITY = {"SG": 0.5, "G": 1.0, "MG": 2.0, "SC2": 4.5, "H": 10.0}

def _all_states(hypergraph):
    return tuple(
        np.asarray(bits, dtype=np.int8)
        for bits in itertools.product((0, 1), repeat=len(hypergraph.nodes))
        if 0 < sum(bits) < len(bits)
    )

def _exhaustive_splits(hypergraph):
    states = _all_states(hypergraph)
    return {
        split: {
            time_point: states for time_point in hypergraph.time_points
        }
        for split in ("train", "validation", "audit", "test")
    }

def _split_from_states(hypergraph, train, validation, audit, test):
    values = {
        "train": tuple(train),
        "validation": tuple(validation),
        "audit": tuple(audit),
        "test": tuple(test),
    }
    return {
        split: {
            time_point: states for time_point in hypergraph.time_points
        }
        for split, states in values.items()
    }

def _c3_parameters(hypergraph, nu, beta_by_time):
    rates = build_temporal_c3_lambda_rates(
        hypergraph,
        2,
        nu,
        beta_by_time,
        exposure_semantics=INDEPENDENT,
        route=C3ProjectionRoute.DIRECT,
    )
    return HypergraphSIParameters(rates, nu)

def _sc2_parameters(hypergraph, nu, beta1_by_time, beta2_by_time):
    beta = {}
    for time_point in hypergraph.time_points:
        beta[(time_point, 1)] = float(beta1_by_time[time_point])
        beta[(time_point, 2)] = float(beta2_by_time[time_point])
    rates = build_temporal_c1_lambda_rates(
        hypergraph,
        2,
        nu,
        beta,
        exposure_semantics=INDEPENDENT,
    )
    return HypergraphSIParameters(rates, nu)

def _mechanism_cases(nu):
    static_g = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1), (1, 2), (2, 3)],
            1: [(0, 1), (1, 2), (2, 3)],
            2: [(0, 1), (1, 2), (2, 3)],
        },
    )
    temporal_g = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1), (2, 3)],
            1: [(0, 2), (1, 3)],
            2: [(0, 3), (1, 2)],
        },
    )
    multiplicity = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1), (0, 1), (2, 3)],
            1: [(0, 1), (0, 1), (2, 3)],
            2: [(0, 2), (0, 2), (1, 3)],
        },
    )
    sc2 = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1, 2), (1, 2, 3)],
            1: [(0, 1, 3), (0, 2, 3)],
            2: [(0, 1, 2), (0, 2, 3)],
        },
    )
    high = TemporalHypergraph(
        range(5),
        {
            0: [(0, 1, 2), (0, 1, 2, 3), (1, 3, 4)],
            1: [(0, 2, 4), (0, 1, 3, 4), (1, 2, 3)],
            2: [(0, 1, 4), (0, 2, 3, 4), (1, 2, 4)],
        },
    )
    high_rates, _ = build_conflicting_lambda_rates(
        high, nu, 0.05, 0.12
    )
    return OrderedDict(
        [
            (
                "static_pairwise",
                (
                    static_g,
                    _c3_parameters(
                        static_g,
                        nu,
                        {time_point: 0.07 for time_point in static_g.time_points},
                    ),
                    "SG",
                ),
            ),
            (
                "temporal_pairwise",
                (
                    temporal_g,
                    _c3_parameters(
                        temporal_g,
                        nu,
                        {
                            time_point: 0.055 + 0.018 * time_point
                            for time_point in temporal_g.time_points
                        },
                    ),
                    "G",
                ),
            ),
            (
                "multiplicity_only",
                (
                    multiplicity,
                    _c3_parameters(
                        multiplicity,
                        nu,
                        {
                            time_point: 0.08
                            for time_point in multiplicity.time_points
                        },
                    ),
                    "MG",
                ),
            ),
            (
                "second_order_synergy",
                (
                    sc2,
                    _sc2_parameters(
                        sc2,
                        nu,
                        {time_point: 0.03 for time_point in sc2.time_points},
                        {time_point: 0.14 for time_point in sc2.time_points},
                    ),
                    "SC2",
                ),
            ),
            (
                "group_specific_conflict",
                (high, HypergraphSIParameters(high_rates, nu), "H"),
            ),
        ]
    )

def _assessment_map(decision):
    return {item.name: item for item in decision.assessments}

def _decision_record(case_name, expected, decision, runtime):
    selected = _assessment_map(decision)[decision.selected_name]
    return {
        "case": case_name,
        "expected": expected,
        "selected": decision.selected_name,
        "oracle": decision.oracle_test_choice,
        "decision": decision.decision,
        "correct": bool(decision.selected_name == expected),
        "correct_vs_oracle": bool(decision.correct_vs_test_oracle),
        "safe_on_test": bool(decision.safe_on_test),
        "test_worst_error": float(selected.test.worst_error),
        "validation_choice": decision.validation_choice,
        "upgrade_path": list(decision.upgrade_path),
        "runtime_seconds": float(runtime),
        "serializable": decision.serializable(),
    }

def run_a0_mechanism_oracle(config):
    rows = []
    for case_name, (hypergraph, parameters, expected) in _mechanism_cases(
        float(config["nu"])
    ).items():
        started = time.perf_counter()
        decision = run_closed_loop_decision(
            hypergraph,
            parameters,
            _exhaustive_splits(hypergraph),
            error_budget=float(config["error_budget"]),
            exact_tolerance=float(config["exact_tolerance"]),
        )
        rows.append(
            _decision_record(
                case_name,
                expected,
                decision,
                time.perf_counter() - started,
            )
        )
    return {
        "rows": rows,
        "selection_accuracy": float(
            np.mean([row["correct"] for row in rows])
        ),
        "oracle_accuracy": float(
            np.mean([row["correct_vs_oracle"] for row in rows])
        ),
        "safety_rate": float(
            np.mean([row["safe_on_test"] for row in rows])
        ),
    }

def _synergy_hypergraph():
    return TemporalHypergraph(
        range(5),
        {
            0: [(0, 1, 2), (0, 1, 2), (2, 3, 4)],
            1: [(0, 2, 3), (1, 3, 4), (1, 3, 4)],
            2: [(0, 1, 4), (1, 2, 3), (0, 2, 4)],
        },
    )

def run_a1_synergy_budget_phase(config):
    hypergraph = _synergy_hypergraph()
    splits = _exhaustive_splits(hypergraph)
    rows = []
    for synergy in config["synergy_levels"]:
        parameters = _sc2_parameters(
            hypergraph,
            float(config["nu"]),
            {time_point: 0.035 for time_point in hypergraph.time_points},
            {
                time_point: float(synergy)
                for time_point in hypergraph.time_points
            },
        )
        for budget in config["budget_levels"]:
            decision = run_closed_loop_decision(
                hypergraph,
                parameters,
                splits,
                error_budget=float(budget),
                exact_tolerance=float(config["exact_tolerance"]),
            )
            rows.append(
                {
                    "synergy": float(synergy),
                    "budget": float(budget),
                    "selected": decision.selected_name,
                    "oracle": decision.oracle_test_choice,
                    "correct": bool(decision.correct_vs_test_oracle),
                    "safe": bool(decision.safe_on_test),
                }
            )
    return {
        "rows": rows,
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "safety_rate": float(np.mean([row["safe"] for row in rows])),
    }

def _temporal_synergy_hypergraph():
    groups = [(0, 1, 2), (1, 2, 3), (0, 3, 4)]
    return TemporalHypergraph(
        range(5), {time_point: groups for time_point in range(4)}
    )

def run_a2_temporal_synergy_phase(config):
    hypergraph = _temporal_synergy_hypergraph()
    splits = _exhaustive_splits(hypergraph)
    rows = []
    for temporal_strength in config["temporal_levels"]:
        beta1 = {
            time_point: min(
                0.3,
                max(
                    0.002,
                    0.04
                    + float(temporal_strength)
                    * math.sin(2.0 * math.pi * time_point / 4.0),
                ),
            )
            for time_point in hypergraph.time_points
        }
        for synergy in config["synergy_levels"]:
            parameters = _sc2_parameters(
                hypergraph,
                float(config["nu"]),
                beta1,
                {
                    time_point: float(synergy)
                    for time_point in hypergraph.time_points
                },
            )
            decision = run_closed_loop_decision(
                hypergraph,
                parameters,
                splits,
                error_budget=float(config["error_budget"]),
                exact_tolerance=float(config["exact_tolerance"]),
            )
            rows.append(
                {
                    "temporal_strength": float(temporal_strength),
                    "synergy": float(synergy),
                    "selected": decision.selected_name,
                    "oracle": decision.oracle_test_choice,
                    "correct": bool(decision.correct_vs_test_oracle),
                }
            )
    return {
        "rows": rows,
        "accuracy": float(np.mean([row["correct"] for row in rows])),
    }

def _conflict_hypergraph():
    return TemporalHypergraph(
        range(6),
        {
            0: [(0, 1, 2), (0, 1, 2, 3), (2, 4, 5)],
            1: [(0, 2, 4), (1, 2, 3, 5), (0, 4, 5)],
            2: [(0, 1, 4, 5), (1, 2, 3), (2, 3, 4)],
        },
    )

def run_a3_collision_certificate(config):
    hypergraph = _conflict_hypergraph()
    splits = _exhaustive_splits(hypergraph)
    rows = []
    for conflict in config["conflict_levels"]:
        rates, _ = build_conflicting_lambda_rates(
            hypergraph,
            float(config["nu"]),
            0.05,
            float(conflict),
        )
        decision = run_closed_loop_decision(
            hypergraph,
            HypergraphSIParameters(rates, float(config["nu"])),
            splits,
            error_budget=float(config["error_budget"]),
            exact_tolerance=float(config["exact_tolerance"]),
        )
        assessments = _assessment_map(decision)
        for name in ("G", "MG", "SC2"):
            item = assessments[name]
            observed = max(
                item.train.worst_error, item.validation.worst_error
            )
            rows.append(
                {
                    "conflict": float(conflict),
                    "model": name,
                    "collision_lower_bound": float(
                        item.collision_lower_bound
                    ),
                    "observed_worst_error": float(observed),
                    "bound_holds": bool(
                        item.collision_lower_bound
                        <= observed + 1e-12
                    ),
                    "selected": decision.selected_name,
                }
            )
    return {
        "rows": rows,
        "coverage_rate": float(
            np.mean([row["bound_holds"] for row in rows])
        ),
    }

def run_a4_audit_fallback(config):
    hypergraph = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1, 2), (1, 2, 3)],
            1: [(0, 1, 3), (0, 2, 3)],
        },
    )
    parameters = _sc2_parameters(
        hypergraph,
        float(config["nu"]),
        {time_point: 0.03 for time_point in hypergraph.time_points},
        {time_point: 0.12 for time_point in hypergraph.time_points},
    )
    all_states = _all_states(hypergraph)
    low = [state for state in all_states if int(state.sum()) == 1]
    high = [state for state in all_states if int(state.sum()) >= 2]

    splits = _split_from_states(
        hypergraph, all_states, low, high, all_states
    )
    assessments_by_budget = []
    chosen = None
    for budget in np.linspace(0.005, 0.12, 24):
        decision = run_closed_loop_decision(
            hypergraph,
            parameters,
            splits,
            error_budget=float(budget),
            exact_tolerance=float(config["exact_tolerance"]),
        )
        if len(decision.upgrade_path) > 1:
            chosen = decision
            break
        assessments_by_budget.append(
            {
                "budget": float(budget),
                "validation_choice": decision.validation_choice,
                "selected": decision.selected_name,
            }
        )
    if chosen is None:
        chosen = run_closed_loop_decision(
            hypergraph,
            parameters,
            splits,
            error_budget=float(config["error_budget"]),
            exact_tolerance=float(config["exact_tolerance"]),
        )
    return {
        "decision": chosen.serializable(),
        "upgrade_triggered": bool(len(chosen.upgrade_path) > 1),
        "upgrade_path": list(chosen.upgrade_path),
        "selected": chosen.selected_name,
        "oracle": chosen.oracle_test_choice,
        "safe_on_test": bool(chosen.safe_on_test),
        "search_trace": assessments_by_budget,
    }

def run_a5_sample_coverage(config):
    cases = list(_mechanism_cases(float(config["nu"])).items())
    rows = []
    time_count = 3
    minimum_states = time_count * 3
    for state_count in config["coverage_state_counts"]:
        decisions = []
        for replicate in range(int(config["coverage_repetitions"])):
            case_name, (hypergraph, parameters, _) = cases[
                replicate % len(cases)
            ]
            splits = generate_state_splits(
                hypergraph,
                seed=int(config["seed"]) + 1000 * state_count + replicate,
                train_count=max(2, int(state_count)),
                validation_count=int(state_count),
                audit_count=max(2, int(state_count)),
                test_count=8,
            )
            decision = run_closed_loop_decision(
                hypergraph,
                parameters,
                splits,
                error_budget=float(config["error_budget"]),
                exact_tolerance=float(config["exact_tolerance"]),
                minimum_validation_states=minimum_states,
            )
            abstained = decision.decision.startswith(
                "INSUFFICIENT_EVIDENCE"
            )
            decisions.append(
                {
                    "case": case_name,
                    "correct": bool(decision.correct_vs_test_oracle),
                    "safe": bool(decision.safe_on_test),
                    "abstained": bool(abstained),
                }
            )
        rows.append(
            {
                "states_per_time": int(state_count),
                "correct_rate": float(
                    np.mean([item["correct"] for item in decisions])
                ),
                "safety_rate": float(
                    np.mean([item["safe"] for item in decisions])
                ),
                "abstention_rate": float(
                    np.mean([item["abstained"] for item in decisions])
                ),
            }
        )
    return {"rows": rows}

def _dataset_parameters(hypergraph, mechanism, nu):
    if mechanism == "pairwise":
        return _c3_parameters(
            hypergraph,
            nu,
            {
                time_point: 0.035
                + 0.012
                * (position / max(1, len(hypergraph.time_points) - 1))
                for position, time_point in enumerate(
                    hypergraph.time_points
                )
            },
        )
    if mechanism == "sc2":
        return _sc2_parameters(
            hypergraph,
            nu,
            {time_point: 0.03 for time_point in hypergraph.time_points},
            {time_point: 0.09 for time_point in hypergraph.time_points},
        )
    rates, _ = build_conflicting_lambda_rates(
        hypergraph, nu, 0.045, 0.10
    )
    return HypergraphSIParameters(rates, nu)

def _median_runtime(function, repetitions=3):
    values = []
    result = None
    for _ in range(int(repetitions)):
        started = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - started)
    return float(np.median(values)), result

def run_a6_eight_datasets(config):
    loaded = load_all_validation_datasets(config)
    rows = []
    efficiency = []
    for dataset_index, (name, (hypergraph, metadata)) in enumerate(
        loaded.items()
    ):
        for mechanism_index, mechanism in enumerate(
            ("pairwise", "sc2", "conflict")
        ):
            parameters = _dataset_parameters(
                hypergraph, mechanism, float(config["nu"])
            )
            splits = generate_state_splits(
                hypergraph,
                seed=int(config["seed"])
                + 100 * dataset_index
                + 10 * mechanism_index,
                train_count=int(config["dataset_train_states"]),
                validation_count=int(
                    config["dataset_validation_states"]
                ),
                audit_count=int(config["dataset_audit_states"]),
                test_count=int(config["dataset_test_states"]),
            )
            started = time.perf_counter()
            decision = run_closed_loop_decision(
                hypergraph,
                parameters,
                splits,
                error_budget=float(config["error_budget"]),
                exact_tolerance=float(config["exact_tolerance"]),
            )
            runtime = time.perf_counter() - started
            selected = _assessment_map(decision)[decision.selected_name]
            rows.append(
                {
                    "dataset": name,
                    "kind": metadata["kind"],
                    "mechanism": mechanism,
                    "selected": decision.selected_name,
                    "oracle": decision.oracle_test_choice,
                    "correct": bool(decision.correct_vs_test_oracle),
                    "safe": bool(decision.safe_on_test),
                    "test_worst_error": float(selected.test.worst_error),
                    "runtime_seconds": float(runtime),
                    "validation_choice": decision.validation_choice,
                    "upgrades": int(len(decision.upgrade_path) - 1),
                    "max_group_size": int(
                        max(
                            len(group)
                            for time_point in hypergraph.time_points
                            for group in hypergraph.active_hyperedges(
                                time_point
                            )
                        )
                    ),
                    "serializable": decision.serializable(),
                }
            )
            if mechanism == "pairwise":
                initial = next(
                    iter(splits["test"][hypergraph.time_points[0]])
                )
                rng = np.random.RandomState(
                    int(config["seed"]) + dataset_index
                )
                uniforms = rng.random_sample(
                    (
                        len(hypergraph.time_points),
                        len(hypergraph.nodes),
                    )
                )
                source_time, _ = _median_runtime(
                    lambda: simulate_temporal_si(
                        hypergraph,
                        parameters,
                        initial,
                        uniforms=uniforms,
                    )
                )
                selected_time, _ = _median_runtime(
                    lambda: simulate_selected_model(
                        decision, initial, uniforms=uniforms
                    )
                )
                efficiency.append(
                    {
                        "dataset": name,
                        "selected": decision.selected_name,
                        "source_seconds": source_time,
                        "selected_seconds": selected_time,
                        "speedup": float(
                            source_time / max(selected_time, 1e-15)
                        ),
                        "complexity_ratio": float(
                            10.0 / COMPLEXITY[decision.selected_name]
                        ),
                    }
                )
    return {
        "rows": rows,
        "efficiency": efficiency,
        "selection_accuracy": float(
            np.mean([row["correct"] for row in rows])
        ),
        "safety_rate": float(np.mean([row["safe"] for row in rows])),
    }

def _first_safe_by_metric(serializable, metric, budget):
    for name in ORDER:
        value = serializable["assessments"][name]["validation"][metric]
        if value <= budget:
            return name
    return "H"

def _score_baseline(cases, chooser):
    rows = []
    for case in cases:
        oracle = case["oracle"]
        selected = chooser(case)
        assessment = case["serializable"]["assessments"][selected]
        safe = assessment["test"]["worst_error"] <= case["budget"]
        rows.append(
            {
                "selected": selected,
                "oracle": oracle,
                "correct": selected == oracle,
                "safe": safe,
                "overmodel": COMPLEXITY[selected] > COMPLEXITY[oracle],
                "unsafe_simplification": (
                    COMPLEXITY[selected] < COMPLEXITY[oracle] and not safe
                ),
            }
        )
    return {
        "selection_accuracy": float(
            np.mean([row["correct"] for row in rows])
        ),
        "safety_rate": float(np.mean([row["safe"] for row in rows])),
        "overmodel_rate": float(
            np.mean([row["overmodel"] for row in rows])
        ),
        "unsafe_simplification_rate": float(
            np.mean([row["unsafe_simplification"] for row in rows])
        ),
    }

def run_a7_baseline_comparison(config, a0, a1, a2, a6):
    cases = []
    for row in a0["rows"]:
        cases.append(
            {
                "selected": row["selected"],
                "oracle": row["oracle"],
                "serializable": row["serializable"],
                "budget": float(config["error_budget"]),
                "max_group_size": 4,
            }
        )
    for source in (a6["rows"],):
        for row in source:
            cases.append(
                {
                    "selected": row["selected"],
                    "oracle": row["oracle"],
                    "serializable": row["serializable"],
                    "budget": float(config["error_budget"]),
                    "max_group_size": row["max_group_size"],
                }
            )
    methods = OrderedDict(
        [
            ("THN-Need closed loop", lambda case: case["selected"]),
            ("Validation-only", lambda case: case["serializable"]["validation_choice"]),
            (
                "Mean-error selector",
                lambda case: _first_safe_by_metric(
                    case["serializable"],
                    "mean_absolute_error",
                    case["budget"],
                ),
            ),
            (
                "Hyperedge-size rule",
                lambda case: "H" if case["max_group_size"] > 2 else "G",
            ),
            ("Always G", lambda case: "G"),
            ("Always H", lambda case: "H"),
        ]
    )
    return {
        "case_count": len(cases),
        "methods": {
            name: _score_baseline(cases, chooser)
            for name, chooser in methods.items()
        },
    }

def run_a8_reliability(a0, a3, a6):
    certificate_rows = a3["rows"]
    validation_rows = []
    for row in a0["rows"] + a6["rows"]:
        selected = row["selected"]
        assessment = row["serializable"]["assessments"][selected]
        validation_rows.append(
            {
                "validation_safe": assessment["validation_safe"],
                "test_safe": assessment["test_safe"],
            }
        )
    validation_safe = [
        row for row in validation_rows if row["validation_safe"]
    ]
    generalization_rate = float(
        np.mean([row["test_safe"] for row in validation_safe])
    )
    return {
        "collision_bound_coverage": float(
            np.mean([row["bound_holds"] for row in certificate_rows])
        ),
        "validation_to_test_safety_rate": generalization_rate,
        "evaluated_decisions": len(validation_rows),
    }

def run_all_innovation3(config):
    started = time.perf_counter()
    a0 = run_a0_mechanism_oracle(config)
    a1 = run_a1_synergy_budget_phase(config)
    a2 = run_a2_temporal_synergy_phase(config)
    a3 = run_a3_collision_certificate(config)
    a4 = run_a4_audit_fallback(config)
    a5 = run_a5_sample_coverage(config)
    a6 = run_a6_eight_datasets(config)
    a7 = run_a7_baseline_comparison(config, a0, a1, a2, a6)
    a8 = run_a8_reliability(a0, a3, a6)
    acceptance = {
        "mechanism_oracle_accuracy": bool(a0["oracle_accuracy"] == 1.0),
        "mechanism_safety": bool(a0["safety_rate"] == 1.0),
        "synergy_phase_safety": bool(a1["safety_rate"] == 1.0),
        "collision_bound_coverage": bool(
            a3["coverage_rate"] >= 0.999
        ),
        "audit_fallback_safe": bool(a4["safe_on_test"]),
        "dataset_safety": bool(a6["safety_rate"] >= 0.95),
        "baseline_tradeoff_advantage": bool(
            a7["methods"]["THN-Need closed loop"]["safety_rate"]
            >= a7["methods"]["Validation-only"]["safety_rate"]
            and a7["methods"]["THN-Need closed loop"][
                "selection_accuracy"
            ]
            > a7["methods"]["Hyperedge-size rule"][
                "selection_accuracy"
            ]
            and a7["methods"]["THN-Need closed loop"][
                "overmodel_rate"
            ]
            < a7["methods"]["Hyperedge-size rule"]["overmodel_rate"]
        ),
        "validation_test_reliability": bool(
            a8["validation_to_test_safety_rate"] >= 0.95
        ),
    }
    acceptance["overall_passed"] = bool(all(acceptance.values()))
    return {
        "experiment": "innovation3_higher_order_necessity",
        "config": dict(config),
        "A0": a0,
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": a5,
        "A6": a6,
        "A7": a7,
        "A8": a8,
        "acceptance": acceptance,
        "runtime_seconds": float(time.perf_counter() - started),
    }
