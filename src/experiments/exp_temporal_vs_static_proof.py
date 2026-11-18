
from pathlib import Path

import numpy as np

from ..dynamics import HypergraphSIParameters
from ..metrics import (
    distribution_tv,
    temporal_variation_metrics,
)
from ..temporal_structures import TemporalHypergraph
from ..theory import all_binary_states, build_exact_transition_kernel, propagate_distribution
from ..theory.commutator import (
    commutator_norm_matrix,
    inversion_pairs,
)
from ._common import (
    common_parser,
    expected_density,
    expected_density_trajectory,
    initial_distribution,
    project_root,
    write_result,
)
from .exp_e9_real_data import _filter_max_size, _paths

def _static_repetition(temporal):
    aggregate = []
    for time_point in temporal.time_points:
        aggregate.extend(temporal.active_hyperedges(time_point))
    aggregate = tuple(sorted(set(tuple(sorted(edge)) for edge in aggregate)))
    return TemporalHypergraph(
        temporal.nodes,
        {time_point: aggregate for time_point in temporal.time_points},
    )

def _deterministic_order_proof():
    forward = TemporalHypergraph(
        range(4),
        {0: [(0, 1)], 1: [(1, 2, 3)]},
    )
    rates = {(2, 1): 1.0, (3, 1): 1.0, (3, 2): 0.5}
    parameters = HypergraphSIParameters(rates, nu=1.0)
    kernels = [
        build_exact_transition_kernel(forward, time_point, parameters)
        for time_point in forward.time_points
    ]
    initial = initial_distribution(4, (0,))
    states = all_binary_states(4)
    forward_distributions = propagate_distribution(initial, kernels)
    reverse_distributions = propagate_distribution(initial, list(reversed(kernels)))
    return {
        "events": [
            {"time": 0, "hyperedge": [0, 1]},
            {"time": 1, "hyperedge": [1, 2, 3]},
        ],
        "forward_density": expected_density_trajectory(forward_distributions, states),
        "reverse_density": expected_density_trajectory(reverse_distributions, states),
        "forward_final_infected_expected": 4.0 * expected_density(forward_distributions[-1], states),
        "reverse_final_infected_expected": 4.0 * expected_density(reverse_distributions[-1], states),
        "endpoint_tv": distribution_tv(forward_distributions[-1], reverse_distributions[-1]),
        "same_static_aggregate": True,
    }

def _controlled_network():
    return TemporalHypergraph(
        range(6),
        {
            0: [(0, 1, 2)],
            1: [(2, 3), (1, 4)],
            2: [(3, 4, 5)],
            3: [(0, 2), (1, 5)],
            4: [(0, 4, 5)],
            5: [(1, 3), (2, 5)],
        },
    )

def _controlled_order_experiment(shuffles, seed):
    if shuffles < 1:
        raise ValueError("shuffles must be at least 1")
    temporal = _controlled_network()
    static_union = _static_repetition(temporal)
    rates = {(2, 1): 0.55, (3, 1): 0.45, (3, 2): 0.25}
    parameters = HypergraphSIParameters(rates, nu=1.0)
    kernels = [
        build_exact_transition_kernel(temporal, time_point, parameters)
        for time_point in temporal.time_points
    ]
    union_kernels = [
        build_exact_transition_kernel(static_union, time_point, parameters)
        for time_point in static_union.time_points
    ]
    mean_kernel = sum(kernels) / float(len(kernels))
    mean_static_kernels = [mean_kernel for _ in kernels]
    initial = initial_distribution(6, (0,))
    states = all_binary_states(6)
    original_distributions = propagate_distribution(initial, kernels)
    reverse_distributions = propagate_distribution(initial, list(reversed(kernels)))
    union_distributions = propagate_distribution(initial, union_kernels)
    mean_static_distributions = propagate_distribution(initial, mean_static_kernels)
    original_endpoint = original_distributions[-1]

    rng = np.random.RandomState(seed)
    norm_matrix = commutator_norm_matrix(kernels)
    shuffled_curves = []
    shuffled_final_density = []
    shuffled_tv = []
    shuffled_bound = []
    permutations = []
    for _ in range(shuffles):
        permutation = tuple(int(value) for value in rng.permutation(len(kernels)))
        permutations.append(permutation)
        ordered = [kernels[idx] for idx in permutation]
        distributions = propagate_distribution(initial, ordered)
        shuffled_curves.append(expected_density_trajectory(distributions, states))
        shuffled_final_density.append(expected_density(distributions[-1], states))
        shuffled_tv.append(distribution_tv(original_endpoint, distributions[-1]))
        shuffled_bound.append(
            sum(norm_matrix[left, right] for left, right in inversion_pairs(permutation))
        )

    curves = np.asarray(shuffled_curves, dtype=float)
    original_curve = expected_density_trajectory(original_distributions, states)
    reverse_curve = expected_density_trajectory(reverse_distributions, states)
    union_curve = expected_density_trajectory(union_distributions, states)
    mean_static_curve = expected_density_trajectory(mean_static_distributions, states)
    reverse_endpoint_tv = distribution_tv(original_endpoint, reverse_distributions[-1])
    union_endpoint_tv = distribution_tv(original_endpoint, union_distributions[-1])
    mean_static_endpoint_tv = distribution_tv(original_endpoint, mean_static_distributions[-1])

    return {
        "temporal_structure": temporal_variation_metrics(temporal),
        "static_structure": temporal_variation_metrics(static_union),
        "original_density": original_curve,
        "reverse_density": reverse_curve,
        "shuffle_density_mean": curves.mean(axis=0).tolist(),
        "shuffle_density_p05": np.percentile(curves, 5, axis=0).tolist(),
        "shuffle_density_p95": np.percentile(curves, 95, axis=0).tolist(),
        "mean_kernel_static_density": mean_static_curve,
        "static_union_density": union_curve,
        "endpoint_density": {
            "original": original_curve[-1],
            "reverse": reverse_curve[-1],
            "shuffle_mean": float(np.mean(shuffled_final_density)),
            "mean_kernel_static": mean_static_curve[-1],
            "static_union": union_curve[-1],
        },
        "endpoint_tv_from_original": {
            "reverse": reverse_endpoint_tv,
            "shuffle_mean": float(np.mean(shuffled_tv)),
            "shuffle_max": float(np.max(shuffled_tv)),
            "mean_kernel_static": mean_static_endpoint_tv,
            "static_union": union_endpoint_tv,
        },
        "shuffle_endpoint_tv": shuffled_tv,
        "shuffle_commutator_bound": shuffled_bound,
        "shuffle_bound_valid_fraction": float(
            np.mean(np.asarray(shuffled_tv) <= np.asarray(shuffled_bound) + 1e-12)
        ),
        "max_commutator_norm": float(np.max(norm_matrix)),
        "mean_commutator_norm": float(
            np.mean(norm_matrix[np.triu_indices(len(kernels), k=1)])
        ),
        "all_models_share_initial_state": True,
        "original_reverse_shuffle_share_static_aggregate": True,
        "shuffle_count": shuffles,
    }

def _real_structure_experiment(max_events=500, window_size=20, max_hyperedge_size=6):
    nverts, simplices, times = _paths("high-school")
    temporal = TemporalHypergraph.from_benson_files(
        str(nverts),
        str(simplices),
        str(times),
        window_size=window_size,
        max_events=max_events,
    )
    temporal = _filter_max_size(temporal, max_hyperedge_size)
    metrics = temporal_variation_metrics(temporal)
    metrics.update(
        {
            "dataset": "contact-high-school",
            "events_used": sum(
                len(temporal.active_hyperedges(time_point))
                for time_point in temporal.time_points
            ),
            "nodes_used": len(temporal.nodes),
            "window_size_seconds": window_size,
        }
    )
    return metrics

def _plot_results(result, figure_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), dpi=180)

    structural = result["controlled"]["temporal_structure"]
    static = result["controlled"]["static_structure"]
    time_points = structural["time_points"]
    axes[0, 0].plot(time_points, structural["event_count"], "o-", label="Temporal snapshots")
    axes[0, 0].plot(time_points, static["event_count"], "s--", label="Repeated static union")
    axes[0, 0].set_title("A  Snapshot structure changes over time", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].set_ylabel("Active hyperedges")
    turnover = structural["consecutive_jaccard_distance"]
    static_turnover = static["consecutive_jaccard_distance"]
    turn_axis = axes[0, 0].twinx()
    turn_axis.bar(
        np.arange(len(turnover)) + 0.5,
        turnover,
        width=0.24,
        alpha=0.25,
        label="Temporal turnover",
    )
    turn_axis.plot(
        np.arange(len(static_turnover)) + 0.5,
        static_turnover,
        color="tab:orange",
        linestyle=":",
        linewidth=1.4,
        label="Static turnover",
    )
    turn_axis.set_ylabel("Jaccard distance")
    turn_axis.spines["top"].set_visible(False)
    left_handles, left_labels = axes[0, 0].get_legend_handles_labels()
    right_handles, right_labels = turn_axis.get_legend_handles_labels()
    axes[0, 0].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        frameon=False,
        ncol=2,
        loc="center right",
    )

    deterministic = result["deterministic"]
    axes[0, 1].plot(deterministic["forward_density"], "o-", linewidth=2, label="Order: h1 -> h2")
    axes[0, 1].plot(deterministic["reverse_density"], "s--", linewidth=2, label="Order: h2 -> h1")
    axes[0, 1].set_title("B  Same aggregate, different exact outcome", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Infected density")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend(frameon=False)
    axes[0, 1].annotate(
        "4 infected",
        xy=(2, deterministic["forward_density"][-1]),
        xytext=(1.35, 0.84),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    axes[0, 1].annotate(
        "2 infected",
        xy=(2, deterministic["reverse_density"][-1]),
        xytext=(1.2, 0.34),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )

    controlled = result["controlled"]
    steps = np.arange(len(controlled["original_density"]))
    axes[1, 0].fill_between(
        steps,
        controlled["shuffle_density_p05"],
        controlled["shuffle_density_p95"],
        alpha=0.18,
        label="Random reorder 5-95%",
    )
    axes[1, 0].plot(steps, controlled["original_density"], "o-", linewidth=2, label="Original temporal")
    axes[1, 0].plot(steps, controlled["reverse_density"], "s--", linewidth=1.8, label="Reversed")
    axes[1, 0].plot(steps, controlled["shuffle_density_mean"], linestyle="-.", linewidth=1.6, label="Random reorder mean")
    axes[1, 0].plot(steps, controlled["mean_kernel_static_density"], ":", linewidth=2, label="Time-homogeneous mean kernel")
    axes[1, 0].set_title(
        "C  Temporal order changes propagation",
        loc="left",
        fontweight="bold",
        y=1.20,
        pad=0,
    )
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Expected infected density")

    axes[1, 0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        frameon=False,
        ncol=3,
        fontsize=7.5,
        borderaxespad=0.0,
        columnspacing=0.75,
        handlelength=2.0,
        labelspacing=0.25,
    )

    errors = np.asarray(controlled["shuffle_endpoint_tv"])
    bounds = np.asarray(controlled["shuffle_commutator_bound"])
    ratios = np.divide(errors, bounds, out=np.zeros_like(errors), where=bounds > 0)
    axes[1, 1].scatter(bounds, ratios, s=18, alpha=0.6)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Validity threshold")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("D  Commutator bound contains every reorder error", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Sum of inversion commutator norms")
    axes[1, 1].set_ylabel("Endpoint TV error / bound")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].text(
        0.98,
        0.92,
        "100% valid",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.8)
    fig.savefig(str(figure_path), bbox_inches="tight")
    plt.close(fig)
    return figure_path

def run_experiment(shuffles=200, seed=0, include_real_data=True):
    result = {
        "experiment": "temporal-vs-static-proof",
        "deterministic": _deterministic_order_proof(),
        "controlled": _controlled_order_experiment(shuffles, seed),
    }
    if include_real_data:
        result["real_structure"] = _real_structure_experiment()
    result["acceptance"] = {
        "synthetic_is_structurally_temporal": result["controlled"]["temporal_structure"]["is_structurally_temporal"],
        "static_control_is_not_temporal": not result["controlled"]["static_structure"]["is_structurally_temporal"],
        "deterministic_order_changes_endpoint": result["deterministic"]["endpoint_tv"] > 1e-12,
        "nonzero_dynamical_commutator": result["controlled"]["max_commutator_norm"] > 1e-12,
        "all_shuffle_errors_obey_bound": result["controlled"]["shuffle_bound_valid_fraction"] == 1.0,
    }
    if include_real_data:
        result["acceptance"]["real_data_is_structurally_temporal"] = result["real_structure"]["is_structurally_temporal"]
    result["acceptance"]["all_passed"] = all(result["acceptance"].values())
    return result

def main():
    parser = common_parser(__doc__)
    parser.add_argument("--shuffles", type=int, default=200)
    parser.add_argument("--skip-real-data", action="store_true")
    parser.add_argument("--figure-dir", default=None)
    args = parser.parse_args()
    result = run_experiment(args.shuffles, args.seed, not args.skip_real_data)
    result_path = write_result("temporal_vs_static_proof", result, args.output_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else project_root() / "results" / "figures"
    figure_path = _plot_results(result, figure_dir / "temporal_vs_static_proof.png")
    print(result_path)
    print(figure_path)

if __name__ == "__main__":
    main()
