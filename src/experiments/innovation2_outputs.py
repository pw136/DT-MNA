
from __future__ import unicode_literals

import csv
from pathlib import Path

import numpy as np

COLORS = {
    "Direct copy": "#7f7f7f",
    "Arithmetic mean": "#bcbd22",
    "Frequency weighted": "#9467bd",
    "Hazard projection": "#1f77b4",
    "Dynamics aware": "#d62728",
    "Direct-MG": "#7f7f7f",
    "Temporal-Qdagger-MG": "#1f77b4",
    "Dynamics-Qdagger-MG": "#d62728",
    "Selected-Qdagger-MG": "#d62728",
}

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )
    return plt

def _save(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(str(path), dpi=220, bbox_inches="tight")
    return path

def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return path
    if columns is None:
        columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path

def plot_conflict_scan(a1, path):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    methods = a1["methods"]
    for method in methods:
        rows = [row for row in a1["summary"] if row["method"] == method]
        x = [row["conflict"] for row in rows]
        color = COLORS.get(method, None)
        axes[0].plot(
            x,
            [row["mean_mae"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=method,
            color=color,
        )
        axes[1].plot(
            x,
            [row["mean_worst"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=method,
            color=color,
        )
    axes[0].set_title("A  Typical-state error", loc="left", fontweight="bold")
    axes[1].set_title("B  Worst-channel error", loc="left", fontweight="bold")
    axes[0].set_ylabel("Mean absolute probability error")
    axes[1].set_ylabel("Worst probability error")
    for axis in axes:
        axis.set_xlabel("Constraint conflict strength")
        axis.set_ylim(bottom=0.0)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    return _save(figure, path)

def plot_surrogate_pareto(a2, path):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    front = set(a2["pareto_names"])
    offset_left = {
        "G-global": (4, 4),
        "G-time": (4, 8),
        "MG-global": (4, 6),
        "MG-smooth": (4, 7),
        "MG-time": (4, 6),
        "SC2-time": (4, 5),
        "H-original": (4, 5),
    }
    offset_right = {
        "G-global": (4, 4),
        "G-time": (4, 10),
        "MG-global": (-46, 5),
        "MG-smooth": (4, 4),
        "MG-time": (-43, -10),
        "SC2-time": (4, 5),
        "H-original": (4, 5),
    }
    for item in a2["candidates"]:
        marker = "o" if item["name"] in front else "x"
        color = "#d62728" if item["name"] in front else "#7f7f7f"
        axes[0].scatter(
            item["complexity"],
            item["endpoint_tv"],
            marker=marker,
            s=46,
            color=color,
            zorder=3,
        )
        axes[0].annotate(
            item["name"],
            (item["complexity"], item["endpoint_tv"]),
            xytext=offset_left[item["name"]],
            textcoords="offset points",
            fontsize=8,
        )
        axes[1].scatter(
            item["max_kernel_tv"],
            item["runtime_seconds"],
            marker=marker,
            s=46,
            color=color,
            zorder=3,
        )
        axes[1].annotate(
            item["name"],
            (item["max_kernel_tv"], item["runtime_seconds"]),
            xytext=offset_right[item["name"]],
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].axhline(
        a2["error_budget"],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        label="Error budget",
    )
    axes[0].set_xlabel("Representation complexity score")
    axes[0].set_ylabel("Endpoint TV error")
    axes[0].set_title("A  Accuracy-complexity frontier", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Maximum one-step kernel TV")
    axes[1].set_ylabel("Exact-kernel evaluation time (s)")
    axes[1].set_title("B  Accuracy-computation tradeoff", loc="left", fontweight="bold")
    return _save(figure, path)

def plot_error_certificate(a3, path):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    steps = np.arange(len(a3["actual_endpoint_tv"]))
    axes[0].plot(
        steps,
        a3["actual_endpoint_tv"],
        color="#111111",
        marker="o",
        linewidth=2.0,
        label="Actual endpoint TV",
    )
    axes[0].plot(
        steps,
        a3["expected_bounds"],
        color="#2ca02c",
        marker="s",
        linewidth=1.6,
        label="Distribution-weighted bound",
    )
    axes[0].plot(
        steps,
        a3["contraction_bounds"],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="Contraction bound",
    )
    axes[0].plot(
        steps,
        a3["basic_bounds"],
        color="#d62728",
        linestyle=":",
        linewidth=1.8,
        label="Worst-row bound",
    )
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Total variation distance")
    axes[0].set_title("A  Finite-time certificate", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)
    time_points = np.arange(len(a3["per_time_max_kernel_tv"]))
    width = 0.36
    axes[1].bar(
        time_points - width / 2.0,
        a3["per_time_expected_kernel_tv"],
        width,
        color="#2ca02c",
        label="Expected row TV",
    )
    axes[1].bar(
        time_points + width / 2.0,
        a3["per_time_max_kernel_tv"],
        width,
        color="#d62728",
        label="Maximum row TV",
    )
    axes[1].axhline(
        a3["irreducible_collision_lower_bound"],
        color="#9467bd",
        linestyle="--",
        linewidth=1.4,
        label="Collision lower bound",
    )
    axes[1].set_xlabel("Snapshot")
    axes[1].set_ylabel("One-step error")
    axes[1].set_title("B  Local error profile", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7)
    return _save(figure, path)

def plot_temporal_drift(a4, path):
    plt = _pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    styles = {
        "Global": ("#7f7f7f", ":"),
        "Per-time": ("#ff7f0e", "-"),
        "Smooth": ("#1f77b4", "-"),
        "Piecewise": ("#d62728", "--"),
    }
    for axis, (scenario, values) in zip(axes.flat, a4["scenarios"].items()):
        truth = np.asarray(values["truth"])
        axis.plot(
            np.arange(len(truth)),
            truth,
            color="#111111",
            linewidth=2.2,
            label="True central parameter",
        )
        for method, estimates in values["estimates"].items():
            color, linestyle = styles[method]
            axis.plot(
                np.arange(len(truth)),
                estimates,
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                label=method,
            )
        axis.set_title(
            scenario.capitalize(),
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Time")
        axis.set_ylabel("Pairwise beta")
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    return _save(figure, path)

def plot_adaptive_timeline(a7, path):
    plt = _pyplot()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 4.9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    matrix = np.asarray(a7["error_matrix"]).T
    image = axes[0].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(a7["error_budget"] * 2.5, float(matrix.max())),
    )
    axes[0].set_yticks(np.arange(len(a7["model_names"])))
    axes[0].set_yticklabels(a7["model_names"])
    axes[0].set_ylabel("Candidate representation")
    axes[0].set_title(
        "A  Per-snapshot approximation error",
        loc="left",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02)
    colorbar.set_label("Probability MAE")
    selected = np.asarray(a7["adaptive"]["selected_indices"], dtype=float)
    axes[1].step(
        np.arange(len(selected)),
        selected,
        where="mid",
        color="#1f77b4",
        linewidth=2.0,
    )
    axes[1].scatter(
        np.arange(len(selected)),
        selected,
        color="#1f77b4",
        s=18,
        zorder=3,
    )
    axes[1].set_yticks(np.arange(len(a7["model_names"])))
    axes[1].set_yticklabels(a7["model_names"])
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Selected")
    axes[1].set_title(
        "B  Lowest admissible representation with switch cost",
        loc="left",
        fontweight="bold",
    )
    return _save(figure, path)

def plot_dataset_results(a8, path):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    datasets = []
    for row in a8["state_rows"]:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
    methods = ("Direct-MG", "Temporal-Qdagger-MG", "Selected-Qdagger-MG")
    offsets = (-0.22, 0.0, 0.22)
    markers = ("o", "s", "^")
    for method, offset, marker in zip(methods, offsets, markers):
        values = [
            next(
                row["probability_mae"]
                for row in a8["state_rows"]
                if row["dataset"] == dataset and row["method"] == method
            )
            for dataset in datasets
        ]
        axes[0].scatter(
            np.arange(len(datasets)) + offset,
            values,
            marker=marker,
            s=42,
            color=COLORS.get(method, None),
            label=method,
        )
    axes[0].set_xticks(np.arange(len(datasets)))
    axes[0].set_xticklabels(datasets, rotation=35, ha="right")
    axes[0].set_ylabel("Held-out conditional-probability MAE")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("A  Eight-dataset generalization", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)
    direct = np.asarray(
        [row["direct_trajectory_mae"] for row in a8["trajectory_rows"]]
    )
    approximate = np.asarray(
        [row["approximate_trajectory_mae"] for row in a8["trajectory_rows"]]
    )
    upper = max(float(direct.max()), float(approximate.max()), 1e-4)
    axes[1].plot([0, upper], [0, upper], color="#7f7f7f", linestyle="--", linewidth=1.0)
    label_offsets = {
        "SSC1": (10, 10),
        "SSC2": (5, 5),
        "SSC3": (8, -17),
        "SSC4": (5, 5),
        "Email-Enron": (5, 5),
        "NDC-classes-full": (10, -10),
        "Primary-School": (8, 12),
        "High-School": (10, 6),
    }
    for index, row in enumerate(a8["trajectory_rows"]):
        axes[1].scatter(
            row["direct_trajectory_mae"],
            row["approximate_trajectory_mae"],
            marker="o" if row["kind"] == "synthetic" else "s",
            s=46,
            color="#d62728",
        )
        axes[1].annotate(
            row["dataset"],
            (row["direct_trajectory_mae"], row["approximate_trajectory_mae"]),
            xytext=label_offsets[row["dataset"]],
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xlabel("Direct mapping trajectory MAE")
    axes[1].set_ylabel("Approximate transform trajectory MAE")
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title("B  Paired trajectory comparison", loc="left", fontweight="bold")
    return _save(figure, path)

def plot_gdw(a9, path):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    steps = np.arange(len(a9["source_variance"]))
    axes[0].plot(
        steps,
        a9["source_variance"],
        color="#111111",
        linewidth=2.1,
        label="Group DW",
    )
    axes[0].plot(
        steps,
        a9["method_variance"]["Naive pairwise"],
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.6,
        label="Naive pairwise",
    )
    axes[0].plot(
        steps,
        a9["method_variance"]["Approximate pairwise"],
        color="#d62728",
        linewidth=1.8,
        label="Approximate pairwise",
    )
    axes[0].set_xlabel("Group event")
    axes[0].set_ylabel("Opinion variance")
    axes[0].set_title("A  Ensemble-average GDW dynamics", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    metrics = ("variance_mae", "final_variance_error", "cluster_error")
    labels = ("Trajectory variance", "Final variance", "Cluster count")
    x = np.arange(len(metrics))
    width = 0.34
    axes[1].bar(
        x - width / 2.0,
        [a9["summary"]["Naive pairwise"][metric] for metric in metrics],
        width,
        color="#7f7f7f",
        label="Naive pairwise",
    )
    axes[1].bar(
        x + width / 2.0,
        [a9["summary"]["Approximate pairwise"][metric] for metric in metrics],
        width,
        color="#d62728",
        label="Approximate pairwise",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Mean test error")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean test error (log scale)")
    axes[1].set_title("B  Held-out task errors", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    return _save(figure, path)

def plot_efficiency(a10, path):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    label_offsets = {
        "G-global": (4, 4),
        "G-time": (4, 5),
        "MG-global": (4, 5),
        "MG-smooth": (4, 5),
        "MG-time": (4, 5),
        "SC2-time": (4, 5),
        "H-original": (4, 5),
    }
    for item in a10["hierarchy"]:
        axes[0].scatter(
            item["endpoint_tv"],
            item["speedup_vs_h"],
            s=50,
            color="#1f77b4" if item["name"] != "H-original" else "#111111",
        )
        axes[0].annotate(
            item["name"],
            (item["endpoint_tv"], item["speedup_vs_h"]),
            xytext=label_offsets[item["name"]],
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].axvline(0.08, color="#d62728", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Endpoint TV error")
    axes[0].set_ylabel("Exact-kernel speedup vs H")
    axes[0].set_xlim(left=0.0)
    axes[0].set_title("A  Small-network speed-accuracy", loc="left", fontweight="bold")
    methods = ("Direct-MG", "Selected-Qdagger-MG", "Temporal-Qdagger-G")
    for method, marker in zip(methods, ("o", "s", "^")):
        rows = [row for row in a10["dataset_rows"] if row["method"] == method]
        axes[1].scatter(
            [row["probability_mae"] for row in rows],
            [row["relative_evaluation_time"] for row in rows],
            marker=marker,
            s=46,
            label=method,
        )
    axes[1].set_xlabel("Conditional-probability MAE")
    axes[1].set_ylabel("Evaluation time / hypergraph time")
    axes[1].set_xlim(left=0.0)
    axes[1].set_title("B  Dataset-level cost-accuracy", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7)
    return _save(figure, path)

def write_all_tables(result, table_dir):
    table_dir = Path(table_dir)
    paths = []
    paths.append(write_csv(table_dir / "innovation2_a1_conflict_scan.csv", result["A1"]["summary"]))
    paths.append(write_csv(table_dir / "innovation2_a2_surrogate_hierarchy.csv", result["A2"]["candidates"]))
    certificate_rows = []
    for index in range(len(result["A3"]["per_time_max_kernel_tv"])):
        certificate_rows.append(
            {
                "time": index,
                "expected_kernel_tv": result["A3"]["per_time_expected_kernel_tv"][index],
                "max_kernel_tv": result["A3"]["per_time_max_kernel_tv"][index],
                "actual_endpoint_tv": result["A3"]["actual_endpoint_tv"][index + 1],
                "expected_bound": result["A3"]["expected_bounds"][index + 1],
                "basic_bound": result["A3"]["basic_bounds"][index + 1],
                "contraction_bound": result["A3"]["contraction_bounds"][index + 1],
            }
        )
    paths.append(write_csv(table_dir / "innovation2_a3_error_certificate.csv", certificate_rows))
    temporal_rows = []
    for scenario, values in result["A4"]["scenarios"].items():
        for method, rmse in values["rmse"].items():
            temporal_rows.append({"scenario": scenario, "method": method, "parameter_rmse": rmse})
    paths.append(write_csv(table_dir / "innovation2_a4_temporal_drift.csv", temporal_rows))
    paths.append(write_csv(table_dir / "innovation2_a5_generalization.csv", result["A5"]["rows"]))
    paths.append(write_csv(table_dir / "innovation2_a6_ablation.csv", result["A6"]["rows"]))
    adaptive_rows = []
    for index, selected in enumerate(result["A7"]["adaptive"]["selected_names"]):
        adaptive_rows.append(
            {
                "time": index,
                "conflict_strength": result["A7"]["conflict_schedule"][index],
                "selected_model": selected,
                "selected_error": result["A7"]["adaptive"]["selected_errors"][index],
            }
        )
    paths.append(write_csv(table_dir / "innovation2_a7_adaptive_selection.csv", adaptive_rows))
    paths.append(write_csv(table_dir / "innovation2_a8_state_errors.csv", result["A8"]["state_rows"]))
    paths.append(write_csv(table_dir / "innovation2_a8_model_selection.csv", result["A8"]["selection_rows"]))
    trajectory_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in ("source_density", "direct_density", "approximate_density")
        }
        for row in result["A8"]["trajectory_rows"]
    ]
    paths.append(write_csv(table_dir / "innovation2_a8_trajectory_errors.csv", trajectory_rows))
    gdw_rows = [
        dict({"method": method}, **metrics)
        for method, metrics in result["A9"]["summary"].items()
    ]
    paths.append(write_csv(table_dir / "innovation2_a9_gdw.csv", gdw_rows))
    paths.append(write_csv(table_dir / "innovation2_a10_efficiency.csv", result["A10"]["hierarchy"]))
    return paths

def write_all_figures(result, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(plot_conflict_scan(result["A1"], figure_dir / "innovation2_conflict_error.png"))
    paths.append(plot_surrogate_pareto(result["A2"], figure_dir / "innovation2_surrogate_pareto.png"))
    paths.append(plot_error_certificate(result["A3"], figure_dir / "innovation2_error_certificate.png"))
    paths.append(plot_temporal_drift(result["A4"], figure_dir / "innovation2_temporal_drift.png"))
    paths.append(plot_adaptive_timeline(result["A7"], figure_dir / "innovation2_adaptive_timeline.png"))
    paths.append(plot_dataset_results(result["A8"], figure_dir / "innovation2_eight_datasets.png"))
    paths.append(plot_gdw(result["A9"], figure_dir / "innovation2_gdw.png"))
    paths.append(plot_efficiency(result["A10"], figure_dir / "innovation2_efficiency.png"))
    return paths
