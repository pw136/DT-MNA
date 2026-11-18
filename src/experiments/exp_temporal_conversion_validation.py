
import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

from ..metrics.temporal_structure import temporal_variation_metrics
from ._common import project_root, write_result
from .conversion_validation_core import (
    run_exact_exhaustive_validation,
    validate_dataset_dynamics,
    validate_negative_controls,
    validate_structural_conversion,
    validate_temporal_reordering,
    validate_window_sensitivity,
)
from .conversion_validation_data import load_all_validation_datasets
from .conversion_validation_outputs import create_all_figures, create_all_reports

DEFAULT_CONFIG = project_root() / "configs" / "temporal_conversion_validation.json"

def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config

def _dataset_acceptance(structure, dynamics, negative, reordering, windows):
    return OrderedDict(
        [
            ("structure", bool(structure["passed"])),
            ("dynamics", bool(dynamics["passed"])),
            ("negative", bool(negative["passed"])),
            ("reordering", bool(reordering["passed"])),
            ("windows", bool(windows["passed"])),
        ]
    )

def run_experiment(config=None, verbose=False):

    config = dict(load_config() if config is None else config)
    started = time.perf_counter()
    loaded = load_all_validation_datasets(config)
    exact = run_exact_exhaustive_validation(config)

    metadata = OrderedDict()
    structural = OrderedDict()
    dynamics = OrderedDict()
    negative_controls = OrderedDict()
    reordering = OrderedDict()
    window_sensitivity = OrderedDict()
    by_dataset = OrderedDict()

    for index, (name, (temporal, dataset_metadata)) in enumerate(loaded.items()):
        dataset_started = time.perf_counter()
        dataset_metadata = dict(dataset_metadata)
        dataset_metadata["temporal_variation"] = temporal_variation_metrics(temporal)
        metadata[name] = dataset_metadata

        structure_result, representations = validate_structural_conversion(
            temporal, int(config["kappa"])
        )
        dynamic_result = validate_dataset_dynamics(
            representations, config, int(config["seed"]) + 100 * index
        )
        negative_result = validate_negative_controls(
            representations, config, int(config["seed"]) + 100 * index + 1
        )
        reordering_result = validate_temporal_reordering(
            temporal,
            representations,
            config,
            int(config["seed"]) + 100 * index + 2,
        )
        window_result = validate_window_sensitivity(
            temporal, config, int(config["seed"]) + 100 * index + 3
        )

        structural[name] = structure_result
        dynamics[name] = dynamic_result
        negative_controls[name] = negative_result
        reordering[name] = reordering_result
        window_sensitivity[name] = window_result
        by_dataset[name] = _dataset_acceptance(
            structure_result,
            dynamic_result,
            negative_result,
            reordering_result,
            window_result,
        )
        metadata[name]["runtime_seconds"] = float(
            time.perf_counter() - dataset_started
        )
        if verbose:
            verdict = "PASS" if all(by_dataset[name].values()) else "FAIL"
            print(
                "%s: %s (%.2fs)"
                % (name, verdict, metadata[name]["runtime_seconds"]),
                flush=True,
            )

    all_dataset_gates = all(
        all(gates.values()) for gates in by_dataset.values()
    )
    acceptance = {
        "exact_state_space": bool(exact["passed"]),
        "by_dataset": by_dataset,
        "all_dataset_gates_passed": bool(all_dataset_gates),
        "overall_passed": bool(exact["passed"] and all_dataset_gates),
    }
    return {
        "experiment": "temporal_conversion_validation",
        "design": {
            "validation_experiments": [
                "V1_snapshot_structure",
                "V2_exact_temporal_dynamics",
                "V3_negative_controls",
                "V4_temporal_reordering",
                "V5_window_sensitivity",
            ],
            "synthetic_dataset_count": 4,
            "real_dataset_count": 4,
            "total_dataset_count": 8,
            "synthetic_realization_note": (
                "The paper publishes SSC summary statistics but not the exact "
                "four random edge lists. SSC1-SSC4 are deterministic regenerated "
                "realizations calibrated to Supplementary Table S2."
            ),
        },
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "config_path": config.get("_config_path"),
        "datasets": metadata,
        "exact": exact,
        "structural": structural,
        "dynamics": dynamics,
        "negative_controls": negative_controls,
        "reordering": reordering,
        "window_sensitivity": window_sensitivity,
        "acceptance": acceptance,
        "runtime_seconds": float(time.perf_counter() - started),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_experiment(config, verbose=not args.quiet)
    root = project_root()
    figure_dir = Path(args.figure_dir) if args.figure_dir else root / "results" / "figures"
    report_dir = Path(args.report_dir) if args.report_dir else root / "results" / "reports"
    figure_paths = create_all_figures(result, figure_dir)
    report_paths = create_all_reports(result, report_dir)
    result["artifacts"] = {
        "figures": figure_paths,
        "reports": report_paths,
    }
    result_path = write_result(
        "temporal_conversion_validation", result, args.output_dir
    )
    print("Result: %s" % result_path.resolve())
    print(
        "Overall: %s"
        % ("PASS" if result["acceptance"]["overall_passed"] else "FAIL")
    )

if __name__ == "__main__":
    main()
