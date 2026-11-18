
from __future__ import unicode_literals

import argparse
import json
from pathlib import Path

import numpy as np

from ._common import json_ready, project_root, write_result
from .innovation2_core import (
    run_a0_exact_recovery,
    run_a1_conflict_scan,
    run_a2_surrogate_hierarchy,
    run_a3_error_certificate,
    run_a4_temporal_drift,
    run_a5_generalization,
    run_a6_ablation,
    run_a7_adaptive_reduction,
    run_a8_dataset_suite,
    run_a9_gdw,
    run_a10_efficiency,
)
from .innovation2_outputs import write_all_figures, write_all_tables

DEFAULT_CONFIG = project_root() / "configs" / "innovation2.json"

def _load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _method_value(rows, conflict, method, key):
    return next(
        row[key]
        for row in rows
        if row["conflict"] == float(conflict) and row["method"] == method
    )

def _acceptance(result):
    highest_conflict = max(
        row["conflict"] for row in result["A1"]["summary"]
    )
    direct_high = _method_value(
        result["A1"]["summary"], highest_conflict, "Direct copy", "mean_mae"
    )
    proposed_high = _method_value(
        result["A1"]["summary"], highest_conflict, "Dynamics aware", "mean_mae"
    )
    temporal_improvement = []
    for scenario, values in result["A4"]["scenarios"].items():
        if scenario == "constant":
            continue
        temporal_improvement.append(
            min(
                values["rmse"]["Per-time"],
                values["rmse"]["Smooth"],
                values["rmse"]["Piecewise"],
            )
            < values["rmse"]["Global"]
        )
    dataset_pairs = []
    for trajectory in result["A8"]["trajectory_rows"]:
        dataset_pairs.append(
            trajectory["approximate_trajectory_mae"]
            <= trajectory["direct_trajectory_mae"] + 1e-12
        )
    dataset_names = list(
        dict.fromkeys(row["dataset"] for row in result["A8"]["state_rows"])
    )
    state_pairs = []
    for dataset in dataset_names:
        direct = next(
            row["probability_mae"]
            for row in result["A8"]["state_rows"]
            if row["dataset"] == dataset and row["method"] == "Direct-MG"
        )
        selected = next(
            row["probability_mae"]
            for row in result["A8"]["state_rows"]
            if row["dataset"] == dataset
            and row["method"] == "Selected-Qdagger-MG"
        )
        state_pairs.append(selected <= direct + 1e-12)
    gdw = result["A9"]["summary"]
    checks = {
        "exact_recovery": bool(result["A0"]["passed"]),
        "no_qstar_detected": bool(
            all(
                row["qstar_feasible_rate"] == 0.0
                for row in result["A1"]["summary"]
                if row["conflict"] > 0.0
            )
        ),
        "approximation_improves_high_conflict": bool(proposed_high < direct_high),
        "finite_time_certificate_covers": bool(result["A3"]["passed"]),
        "temporal_models_improve_drift": bool(all(temporal_improvement)),
        "adaptive_uses_multiple_levels": bool(
            len(set(result["A7"]["adaptive"]["selected_names"])) >= 3
            and max(result["A7"]["adaptive"]["selected_errors"])
            <= result["A7"]["error_budget"] + 1e-12
        ),
        "dataset_trajectory_noninferiority_rate": float(np.mean(dataset_pairs)),
        "dataset_state_improvement_rate": float(np.mean(state_pairs)),
        "gdw_approximation_improves": bool(
            gdw["Approximate pairwise"]["variance_mae"]
            < gdw["Naive pairwise"]["variance_mae"]
            and result["A9"]["irreducible_constructed_state_mismatch"] > 0.0
        ),
    }
    checks["overall_passed"] = bool(
        checks["exact_recovery"]
        and checks["no_qstar_detected"]
        and checks["approximation_improves_high_conflict"]
        and checks["finite_time_certificate_covers"]
        and checks["temporal_models_improve_drift"]
        and checks["adaptive_uses_multiple_levels"]
        and checks["dataset_trajectory_noninferiority_rate"] >= 0.75
        and checks["dataset_state_improvement_rate"] >= 0.75
        and checks["gdw_approximation_improves"]
    )
    return checks

def run_experiment(config_path=DEFAULT_CONFIG, output_dir=None, figure_dir=None):
    config = _load_config(config_path)
    seed = int(config["seed"])
    nu = float(config["nu"])
    result = {
        "experiment": "innovation2_approximate_transformability",
        "config": config,
    }
    result["A0"] = run_a0_exact_recovery(nu)
    result["A1"] = run_a1_conflict_scan(
        config["conflict_levels"],
        config["controlled_replicates"],
        seed + 1,
    )
    result["A2"], context = run_a2_surrogate_hierarchy(
        nu,
        float(config["error_budget"]),
    )
    result["A3"] = run_a3_error_certificate(context)
    result["A4"] = run_a4_temporal_drift(seed + 4)
    result["A5"] = run_a5_generalization(nu)
    result["A6"] = run_a6_ablation(seed + 6)
    result["A7"] = run_a7_adaptive_reduction(
        seed + 7,
        error_budget=0.025,
    )
    dataset_config = dict(config["dataset_suite"])
    dataset_config.update(
        {
            "seed": seed + 8,
            "nu": nu,
            "base_beta": float(config["base_beta"]),
        }
    )
    result["A8"] = run_a8_dataset_suite(dataset_config)
    result["A9"] = run_a9_gdw(config["gdw"], seed + 9)
    result["A10"] = run_a10_efficiency(result["A2"], result["A8"])
    result["acceptance"] = _acceptance(result)
    summary_dir = (
        Path(output_dir)
        if output_dir is not None
        else project_root() / "results" / "summary_tables"
    )
    figures = (
        Path(figure_dir)
        if figure_dir is not None
        else project_root() / "results" / "figures"
    )
    result_path = write_result(
        "innovation2_approximate_transformability",
        result,
        summary_dir,
    )
    table_paths = write_all_tables(result, summary_dir)
    figure_paths = write_all_figures(result, figures)
    return json_ready(result), result_path, table_paths, figure_paths

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    args = parser.parse_args()
    result, result_path, table_paths, figure_paths = run_experiment(
        args.config,
        args.output_dir,
        args.figure_dir,
    )
    print(result_path)
    print("tables=%d figures=%d overall_passed=%s" % (
        len(table_paths),
        len(figure_paths),
        result["acceptance"]["overall_passed"],
    ))

if __name__ == "__main__":
    main()
