
import argparse
import json
from pathlib import Path

import numpy as np

from ..metrics.trajectory_distance import binary_population_entropy
from ..theory.exact_kernel import all_binary_states

def project_root():
    return Path(__file__).resolve().parents[2]

def initial_distribution(node_count, infected_nodes):
    states = all_binary_states(node_count)
    target = np.zeros(node_count, dtype=np.int8)
    target[list(infected_nodes)] = 1
    matches = np.where(np.all(states == target, axis=1))[0]
    distribution = np.zeros(len(states), dtype=float)
    distribution[int(matches[0])] = 1.0
    return distribution

def expected_density(distribution, states):
    return float(np.dot(distribution, states.mean(axis=1)))

def expected_density_trajectory(distributions, states):
    return [expected_density(distribution, states) for distribution in distributions]

def expected_entropy(distribution, states):
    entropies = np.asarray(
        [binary_population_entropy(state) for state in states], dtype=float
    )
    return float(np.dot(distribution, entropies))

def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_ready(item) for item in value]
    return value

def write_result(name, result, output_dir=None):
    output = Path(output_dir) if output_dir else project_root() / "results" / "summary_tables"
    output.mkdir(parents=True, exist_ok=True)
    path = output / (name + ".json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result), handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path

def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser

def maybe_plot_curves(curves, labels, output_path, ylabel="infected density"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 3.8), dpi=160)
    for curve, label in zip(curves, labels):
        axis.plot(range(len(curve)), curve, label=label, linewidth=1.8)
    axis.set_xlabel("time step")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(str(output_path))
    plt.close(figure)
    return output_path
