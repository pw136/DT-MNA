
import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..dynamics import simulate_temporal_si
from ._common import json_ready, project_root
from .conversion_validation_core import (
    _activity_seed,
    build_positive_parameters,
    build_representations,
)
from .conversion_validation_data import load_all_validation_datasets
from .exp_temporal_conversion_validation import load_config

PATHS = OrderedDict(
    [
        (
            "H_to_C_C1",
            {
                "panel": "A",
                "title": r"$H(t)\rightarrow C_{\kappa}(t)$  (C1t)",
                "source": "H",
                "target": "C",
                "source_parameters": "H_C1",
                "target_parameters": "C_C1",
            },
        ),
        (
            "C_to_MG_C2",
            {
                "panel": "B",
                "title": r"$C_{\kappa}(t)\rightarrow MG(t)$  (C2t)",
                "source": "C",
                "target": "MG",
                "source_parameters": "C_C2",
                "target_parameters": "MG",
            },
        ),
        (
            "H_to_C_to_MG_C3",
            {
                "panel": "C",
                "title": r"$H(t)\rightarrow C_{\kappa}(t)\rightarrow MG(t)$  (C3t)",
                "source": "H",
                "target": "MG",
                "source_parameters": "H_C3_via",
                "target_parameters": "MG",
            },
        ),
        (
            "H_to_MG_direct_C3",
            {
                "panel": "D",
                "title": r"$H(t)\rightarrow MG_{\mathrm{direct}}(t)$  (C3t)",
                "source": "H",
                "target": "MG_direct",
                "source_parameters": "H_C3_direct",
                "target_parameters": "MG",
            },
        ),
    ]
)

def _parameter_config(config, beta):
    current = dict(config)
    current["beta_pairwise"] = float(beta)
    current["beta_higher_order"] = float(beta) / 3.0
    return current

def _simulation_designs(network, beta_values, simulation_count, seed):
    designs = OrderedDict()
    node_count = len(network.nodes)
    time_count = len(network.time_points)
    active_seed_index = network.node_to_index[_activity_seed(network)]
    for beta_index, beta in enumerate(beta_values):
        rng = np.random.RandomState(seed + 10000 * beta_index)
        beta_designs = []
        for simulation in range(simulation_count):
            initial = np.zeros(node_count, dtype=np.int8)
            if simulation == 0:
                initial_index = active_seed_index
            else:
                initial_index = int(rng.randint(node_count))
            initial[initial_index] = 1
            uniforms = rng.random_sample((time_count, node_count))
            beta_designs.append((initial, uniforms))
        designs[float(beta)] = beta_designs
    return designs

def _mean_and_sem(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if len(values) <= 1:
        sem = np.zeros(values.shape[1], dtype=float)
    else:
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return mean, sem

def run_experiment(config=None):
    config = dict(load_config() if config is None else config)
    dataset_name = str(config.get("curve_overlay_dataset", "High-School"))
    beta_values = tuple(
        float(value)
        for value in config.get("curve_overlay_beta_values", [0.01, 0.05, 0.10])
    )
    simulation_count = int(config.get("curve_overlay_simulations", 200))
    seed = int(config["seed"]) + 22000

    loaded = load_all_validation_datasets(config)
    if dataset_name not in loaded:
        raise KeyError("unknown curve-overlay dataset: %s" % dataset_name)
    hypergraph, metadata = loaded[dataset_name]
    representations = build_representations(hypergraph, int(config["kappa"]))
    designs = _simulation_designs(
        hypergraph, beta_values, simulation_count, seed
    )

    paths = OrderedDict((label, OrderedDict()) for label in PATHS)
    global_max_mean_gap = 0.0
    global_max_realization_gap = 0.0
    all_paths_identical = True

    for beta in beta_values:
        parameters = build_positive_parameters(
            representations, _parameter_config(config, beta)
        )
        for label, specification in PATHS.items():
            source_curves = []
            target_curves = []
            identical_count = 0
            for initial, uniforms in designs[beta]:
                source = simulate_temporal_si(
                    representations[specification["source"]],
                    parameters[specification["source_parameters"]],
                    initial,
                    uniforms=uniforms,
                )
                target = simulate_temporal_si(
                    representations[specification["target"]],
                    parameters[specification["target_parameters"]],
                    initial,
                    uniforms=uniforms,
                )
                source_curves.append(source.infection_density)
                target_curves.append(target.infection_density)
                identical_count += int(np.array_equal(source.states, target.states))

            source_array = np.asarray(source_curves, dtype=float)
            target_array = np.asarray(target_curves, dtype=float)
            source_mean, source_sem = _mean_and_sem(source_array)
            target_mean, target_sem = _mean_and_sem(target_array)
            max_mean_gap = float(np.max(np.abs(source_mean - target_mean)))
            max_realization_gap = float(
                np.max(np.abs(source_array - target_array))
            )
            path_match_rate = float(identical_count / simulation_count)
            paths[label][str(beta)] = {
                "time": list(range(len(source_mean))),
                "source_mean": source_mean.tolist(),
                "source_sem": source_sem.tolist(),
                "target_mean": target_mean.tolist(),
                "target_sem": target_sem.tolist(),
                "max_mean_curve_gap": max_mean_gap,
                "max_realization_density_gap": max_realization_gap,
                "common_random_path_match_rate": path_match_rate,
            }
            global_max_mean_gap = max(global_max_mean_gap, max_mean_gap)
            global_max_realization_gap = max(
                global_max_realization_gap, max_realization_gap
            )
            all_paths_identical = all_paths_identical and identical_count == simulation_count

    tolerance = float(config["exact_tolerance"])
    return {
        "experiment": "v2_trajectory_overlay",
        "dataset": dataset_name,
        "dataset_metadata": metadata,
        "beta_values": beta_values,
        "higher_order_beta_rule": "beta_higher_order = beta_pairwise / 3",
        "simulation_count_per_beta": simulation_count,
        "initial_condition": (
            "one infected node per realization; the first uses the most active "
            "node and the remaining realizations use seeded uniform node draws"
        ),
        "coupling": "common random numbers for each source-target pair",
        "paths": paths,
        "global_max_mean_curve_gap": global_max_mean_gap,
        "global_max_realization_density_gap": global_max_realization_gap,
        "all_sample_paths_identical": bool(all_paths_identical),
        "passed": bool(
            all_paths_identical
            and global_max_mean_gap < tolerance
            and global_max_realization_gap < tolerance
        ),
    }

def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    return plt

def plot_result(result, output_stem):
    plt = _setup_matplotlib()
    from matplotlib.lines import Line2D

    beta_values = [float(value) for value in result["beta_values"]]
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(beta_values)))
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for axis, (label, specification) in zip(axes, PATHS.items()):
        for color, beta in zip(colors, beta_values):
            row = result["paths"][label][str(beta)]
            time = np.asarray(row["time"], dtype=float)
            source = np.asarray(row["source_mean"], dtype=float)
            target = np.asarray(row["target_mean"], dtype=float)
            sem = np.asarray(row["target_sem"], dtype=float)
            axis.fill_between(
                time,
                np.maximum(0.0, target - sem),
                np.minimum(1.0, target + sem),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axis.plot(time, target, color=color, linewidth=2.0)
            axis.plot(
                time,
                source,
                linestyle="none",
                marker="o",
                markersize=4.1,
                markerfacecolor="white",
                markeredgewidth=1.2,
                color=color,
            )
        axis.set_title(
            "%s  %s" % (specification["panel"], specification["title"]),
            loc="left",
            fontweight="bold",
        )
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.text(
            0.98,
            0.05,
            r"max $|\Delta I(t)|$ = %.1e" % max(
                result["paths"][label][str(beta)]["max_mean_curve_gap"]
                for beta in beta_values
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    for axis in axes[2:]:
        axis.set_xlabel("Time step")
    for axis in axes[::2]:
        axis.set_ylabel(r"Mean infected density $I(t)$")
    axes[0].set_ylim(bottom=0.0)

    beta_handles = [
        Line2D([0], [0], color=color, linewidth=2.0, label=r"$\beta$ = %.3f" % beta)
        for color, beta in zip(colors, beta_values)
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            label="Source representation",
        ),
        Line2D([0], [0], color="black", linewidth=2.0, label="Target representation"),
    ]
    figure.legend(
        handles=beta_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=len(beta_handles) + len(style_handles),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in (".png", ".pdf", ".svg"):
        path = output_stem.with_suffix(suffix)
        figure.savefig(str(path), bbox_inches="tight")
        outputs[suffix[1:]] = str(path)
    plt.close(figure)
    return outputs

def write_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conversion",
                "beta",
                "time_step",
                "source_mean_density",
                "source_sem",
                "target_mean_density",
                "target_sem",
                "absolute_mean_gap",
                "path_match_rate",
            ]
        )
        for label in PATHS:
            for beta in result["beta_values"]:
                row = result["paths"][label][str(float(beta))]
                for index, time_step in enumerate(row["time"]):
                    writer.writerow(
                        [
                            label,
                            float(beta),
                            time_step,
                            row["source_mean"][index],
                            row["source_sem"][index],
                            row["target_mean"][index],
                            row["target_sem"][index],
                            abs(
                                row["source_mean"][index]
                                - row["target_mean"][index]
                            ),
                            row["common_random_path_match_rate"],
                        ]
                    )
    return output_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "temporal_conversion_validation.json"),
    )
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = run_experiment(config)
    root = project_root()
    figure_dir = Path(args.figure_dir) if args.figure_dir else root / "results" / "figures"
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / "summary_tables"
    figures = plot_result(
        result, figure_dir / "conversion_v2_trajectory_overlay"
    )
    csv_path = write_csv(
        result, output_dir / "conversion_v2_trajectory_overlay.csv"
    )
    json_path = output_dir / "conversion_v2_trajectory_overlay.json"
    result["artifacts"] = {
        "figures": figures,
        "csv": str(csv_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result), handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("Result: %s" % json_path.resolve())
    print("Figure: %s" % Path(figures["png"]).resolve())
    print("Overall: %s" % ("PASS" if result["passed"] else "FAIL"))

if __name__ == "__main__":
    main()
