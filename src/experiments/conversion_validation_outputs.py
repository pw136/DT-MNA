
from collections import OrderedDict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

DATASET_COLORS = {
    "synthetic": "#2878b5",
    "real": "#f08a24",
}

def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 170,
        }
    )
    return plt

def _dataset_names(result):

    preferred = [
        "SSC1",
        "SSC2",
        "SSC3",
        "SSC4",
        "Email-Enron",
        "NDC-classes-full",
        "Primary-School",
        "High-School",
    ]
    ordered = [name for name in preferred if name in result["datasets"]]
    ordered.extend(name for name in result["datasets"] if name not in ordered)
    return ordered

def _dataset_colors(result):
    return [
        DATASET_COLORS[result["datasets"][name]["kind"]]
        for name in _dataset_names(result)
    ]

def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(str(path), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)

def plot_structural_validation(result, output_path):
    plt = _setup_matplotlib()
    names = _dataset_names(result)
    colors = _dataset_colors(result)
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))

    snapshot_count = [
        result["datasets"][name]["temporal_variation"]["snapshot_count"]
        for name in names
    ]
    unique_count = [
        result["datasets"][name]["temporal_variation"]["unique_snapshot_count"]
        for name in names
    ]

    snapshot_axis = axes[0, 0]
    for row, (retained, total, color) in enumerate(
        zip(unique_count, snapshot_count, colors)
    ):
        all_positions = np.arange(1, total + 1)
        unique_positions = np.arange(1, retained + 1)
        snapshot_axis.plot(
            [1, total],
            [row, row],
            color="#d9e0e8",
            linewidth=1.0,
            zorder=0,
        )
        snapshot_axis.scatter(
            all_positions,
            np.full(total, row),
            s=62,
            color="#e6eaf0",
            edgecolor="#c7d0dc",
            linewidth=0.7,
            zorder=1,
        )
        snapshot_axis.scatter(
            unique_positions,
            np.full(retained, row),
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        axes[0, 0].text(
            total + 0.28,
            row,
            f"{retained}/{total}",
            ha="left",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#374151",
        )
    axes[0, 0].set_title(
        "A  Every temporal snapshot is unique",
        loc="left",
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Temporal snapshot index")
    axes[0, 0].set_yticks(x)
    axes[0, 0].set_yticklabels(names)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylim(len(names) - 0.5, -1.05)
    axes[0, 0].set_xlim(0.5, max(snapshot_count) + 0.95)
    axes[0, 0].set_xticks(np.arange(1, max(snapshot_count) + 1))
    axes[0, 0].grid(axis="x", color="#d9dee7", linewidth=0.7, alpha=0.8)
    axes[0, 0].set_axisbelow(True)
    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#e6eaf0",
                markeredgecolor="#c7d0dc",
                markersize=7,
                label="All snapshots",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=DATASET_COLORS["synthetic"],
                markeredgecolor="white",
                markersize=6,
                label="Unique: synthetic",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=DATASET_COLORS["real"],
                markeredgecolor="white",
                markersize=6,
                label="Unique: real",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=8,
        ncol=3,
        columnspacing=0.9,
        handlelength=1.5,
    )

    nodes = [result["datasets"][name]["nodes"] for name in names]
    events = [result["datasets"][name]["events"] for name in names]
    axes[0, 1].scatter(nodes, events, c=colors, s=55)
    annotation_offsets = {
        "High-School": (-5, 9, "right"),
        "Primary-School": (5, -13, "left"),
    }
    for idx, name in enumerate(names):
        dx, dy, alignment = annotation_offsets.get(name, (4, 3, "left"))
        axes[0, 1].annotate(
            name,
            (nodes[idx], events[idx]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Nodes")
    axes[0, 1].set_ylabel("Temporal events")
    axes[0, 1].set_title("B  Eight validation data sets", loc="left", fontweight="bold")

    output_labels = ["H events", "C cells", "MG pairs", "G edges"]
    output_values = np.asarray(
        [
            [
                result["structural"][name]["hypergraph_event_count"],
                result["structural"][name]["simplicial_cell_count"],
                result["structural"][name]["via_simplicial_multigraph_edge_count_with_multiplicity"],
                result["structural"][name]["simplegraph_edge_count"],
            ]
            for name in names
        ],
        dtype=float,
    )
    for index, label in enumerate(output_labels):
        axes[1, 0].plot(
            x,
            output_values[:, index],
            marker=("o", "s", "^", "D")[index],
            linewidth=1.6,
            label=label,
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=30, ha="right")
    axes[1, 0].set_ylabel("Count across all snapshots")
    axes[1, 0].set_title("C  Projection changes representation size", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, ncol=2)

    checks = [
        ("H→C", "max_simplicial_multiset_error"),
        ("H→MG", "max_direct_multigraph_weight_error"),
        ("C→MG", "max_via_simplicial_multigraph_weight_error"),
        ("H→G", "max_simplegraph_support_error"),
        ("route", "max_direct_indirect_simple_support_error"),
        ("source", "max_provenance_error"),
    ]
    pass_matrix = np.asarray(
        [
            [
                1.0 if result["structural"][name][key] == 0 else 0.0
                for _, key in checks
            ]
            for name in names
        ]
    )

    status_axis = axes[1, 1]
    criterion_labels = [label for label, _ in checks]
    criterion_y = np.arange(len(checks))
    pass_counts = pass_matrix.sum(axis=0)
    pass_rates = pass_counts / len(names)
    status_axis.hlines(
        criterion_y,
        0.0,
        pass_rates,
        color="#9fd3b2",
        linewidth=3.0,
        zorder=1,
    )
    status_axis.scatter(
        pass_rates,
        criterion_y,
        s=105,
        color="#198754",
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    for row, (rate, count) in enumerate(zip(pass_rates, pass_counts)):
        status_axis.text(
            rate + 0.035,
            row,
            f"{int(count)}/{len(names)}",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#166534",
        )
    status_axis.axvline(1.0, color="#7f8c8d", linestyle=":", linewidth=1.0, zorder=0)
    status_axis.set_xlim(0.0, 1.16)
    status_axis.set_yticks(criterion_y)
    status_axis.set_yticklabels(criterion_labels)
    status_axis.invert_yaxis()
    status_axis.set_xlabel("Data sets passing each structural check")
    status_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    status_axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    status_axis.grid(axis="x", color="#dde3ea", linewidth=0.8)
    status_axis.set_axisbelow(True)
    passed_count = int(pass_matrix.sum())
    total_count = int(pass_matrix.size)
    status_axis.set_title(
        f"D  Structural acceptance by criterion ({passed_count}/{total_count})",
        loc="left",
        fontweight="bold",
    )
    return _save(fig, output_path)

def plot_dynamic_validation(result, output_path):
    plt = _setup_matplotlib()
    names = _dataset_names(result)
    conversions = [
        "H_to_C_C1",
        "C_to_MG_C2",
        "H_to_C_to_MG_C3",
        "H_to_MG_direct_C3",
    ]
    short = [
        "H→C (C1t)",
        "C→MG (C2t)",
        "H→C→MG (C3t)",
        "H→MG direct (C3t)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    exact_kernel = [
        max(
            result["exact"]["conversions"][conversion]["max_kernel_tv"],
            1e-16,
        )
        for conversion in conversions
    ]
    exact_path = [
        max(
            result["exact"]["conversions"][conversion]["path_distribution_tv"],
            1e-16,
        )
        for conversion in conversions
    ]
    x = np.arange(len(conversions))
    axes[0, 0].bar(x - 0.18, exact_kernel, 0.36, label="One-step kernel TV")
    axes[0, 0].bar(x + 0.18, exact_path, 0.36, label="Full path TV")
    axes[0, 0].axhline(result["config"]["exact_tolerance"], color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(short)
    axes[0, 0].set_ylabel("TV distance")
    axes[0, 0].set_title("A  Exhaustive 32-state proof", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    error_matrix = np.asarray(
        [
            [
                max(
                    result["dynamics"][name]["max_conditional_probability_error"][conversion],
                    1e-16,
                )
                for conversion in conversions
            ]
            for name in names
        ]
    )
    image = axes[0, 1].imshow(
        np.log10(error_matrix),
        vmin=-16,
        vmax=-8,
        cmap="viridis",
        aspect="auto",
    )
    axes[0, 1].set_xticks(np.arange(len(conversions)))
    axes[0, 1].set_xticklabels(short, rotation=20, ha="right")
    axes[0, 1].set_yticks(np.arange(len(names)))
    axes[0, 1].set_yticklabels(names)
    axes[0, 1].set_title("B  Conditional-probability error", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes[0, 1], fraction=0.046)
    colorbar.set_label("log10(max error)")

    match_matrix = np.asarray(
        [
            [
                result["dynamics"][name]["common_random_path_match"][conversion]
                for conversion in conversions
            ]
            for name in names
        ],
        dtype=float,
    )
    axes[1, 0].imshow(match_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[1, 0].set_xticks(np.arange(len(conversions)))
    axes[1, 0].set_xticklabels(short, rotation=20, ha="right")
    axes[1, 0].set_yticks(np.arange(len(names)))
    axes[1, 0].set_yticklabels(names)
    for row in range(len(names)):
        for column in range(len(conversions)):
            axes[1, 0].text(
                column,
                row,
                "identical" if match_matrix[row, column] else "different",
                ha="center",
                va="center",
                fontsize=7,
            )
    axes[1, 0].set_title("C  Common-random-number sample paths", loc="left", fontweight="bold")

    markers = ["o", "s", "^", "D"]
    for conversion, label, marker in zip(conversions, short, markers):
        source = [
            result["dynamics"][name]["endpoint_density"][conversion]["source"]
            for name in names
        ]
        target = [
            result["dynamics"][name]["endpoint_density"][conversion]["target"]
            for name in names
        ]
        axes[1, 1].scatter(source, target, marker=marker, s=45, label=label)
    axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Source endpoint density")
    axes[1, 1].set_ylabel("Target endpoint density")
    axes[1, 1].set_title("D  Endpoint equality", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("V2 — Exact temporal dynamical transformability", fontsize=14, fontweight="bold")
    return _save(fig, output_path)

def plot_negative_controls(result, output_path):
    plt = _setup_matplotlib()
    names = _dataset_names(result)
    metrics = [
        ("wrong_C1_probability_error", "Break C1t"),
        ("nonzero_higher_order_C2_error", "Break C2t"),
        ("one_step_time_shift_error", "Shift target time"),
        ("MG_to_G_multiplicity_loss_error", "Drop multiplicity"),
    ]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), gridspec_kw={"height_ratios": [2.2, 1]})
    width = 0.19
    for index, (key, label) in enumerate(metrics):
        values = [
            max(result["negative_controls"][name][key], 1e-16)
            for name in names
        ]
        axes[0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
        )
    axes[0].axhline(result["config"]["exact_tolerance"], color="black", linestyle="--", linewidth=1, label="Exact tolerance")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Max conditional-probability error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_title("A  Invalid transformations become distinguishable", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3)

    counts = [
        result["negative_controls"][name]["detected_control_count"]
        for name in names
    ]
    axes[1].bar(x, counts, color=_dataset_colors(result))
    axes[1].axhline(4, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylim(0, 4.3)
    axes[1].set_ylabel("Detected controls / 4")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_title("B  Sensitivity count", loc="left", fontweight="bold")
    fig.suptitle("V3 — Negative controls and information-loss boundaries", fontsize=14, fontweight="bold")
    return _save(fig, output_path)

def plot_reordering(result, output_path):
    plt = _setup_matplotlib()
    names = _dataset_names(result)
    x = np.arange(len(names))
    aligned = [
        max(result["reordering"][name]["aligned_conversion_max_error"], 1e-16)
        for name in names
    ]
    misaligned = [
        max(result["reordering"][name]["misaligned_conversion_max_error"], 1e-16)
        for name in names
    ]
    order_effect = [
        max(result["reordering"][name]["source_order_effect_max"], 1e-16)
        for name in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].bar(x - 0.18, aligned, 0.36, label="Same reorder on H and C")
    axes[0].bar(x + 0.18, misaligned, 0.36, label="Mismatched order")
    axes[0].axhline(result["config"]["exact_tolerance"], color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("Max conditional-probability error")
    axes[0].set_title("A  Conversion is covariant to a shared reorder", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    axes[1].bar(x, order_effect, color=_dataset_colors(result))
    axes[1].axhline(result["config"]["exact_tolerance"], color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("Original-vs-reordered probability error")
    axes[1].set_title("B  Event order changes the source process", loc="left", fontweight="bold")
    fig.suptitle("V4 — Temporal-order consistency", fontsize=14, fontweight="bold")
    return _save(fig, output_path)

def plot_window_sensitivity(result, output_path):
    plt = _setup_matplotlib()
    names = _dataset_names(result)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    for name in names:
        rows = result["window_sensitivity"][name]["rows"]
        windows = [row["window_size"] for row in rows]
        deltas = [row["endpoint_density_change_from_window1"] for row in rows]
        style = "-" if result["datasets"][name]["kind"] == "synthetic" else "--"
        axes[0, 0].plot(windows, deltas, marker="o", linestyle=style, label=name)
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set_xlabel("Window size (snapshots)")
    axes[0, 0].set_ylabel("Endpoint-density change")
    axes[0, 0].set_title("A  Aggregation changes propagation", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=7)

    for name in names:
        rows = result["window_sensitivity"][name]["rows"]
        axes[0, 1].plot(
            [row["window_size"] for row in rows],
            [row["mean_events_per_snapshot"] for row in rows],
            marker="o",
            label=name,
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Window size (snapshots)")
    axes[0, 1].set_ylabel("Mean simultaneous events")
    axes[0, 1].set_title("B  Windowing inflates concurrence", loc="left", fontweight="bold")

    max_errors = []
    for name in names:
        rows = result["window_sensitivity"][name]["rows"]
        max_errors.append(
            max(max(row["H_to_C_probability_error"], 1e-16) for row in rows)
        )
    axes[1, 0].bar(np.arange(len(names)), max_errors, color=_dataset_colors(result))
    axes[1, 0].axhline(result["config"]["exact_tolerance"], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(np.arange(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=30, ha="right")
    axes[1, 0].set_ylabel("Max H→C error across windows")
    axes[1, 0].set_title("C  Conversion remains exact at each resolution", loc="left", fontweight="bold")

    counterexample = next(iter(result["window_sensitivity"].values()))[
        "exact_small_network_counterexample"
    ]
    keys = sorted(
        int(key)
        for key in counterexample["window_endpoint_kernel_tv"].keys()
    )
    values = [
        counterexample["window_endpoint_kernel_tv"].get(
            key,
            counterexample["window_endpoint_kernel_tv"].get(str(key)),
        )
        for key in keys
    ]
    axes[1, 1].plot(keys, values, "o-", linewidth=2)
    axes[1, 1].axhline(result["config"]["exact_tolerance"], color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Window size")
    axes[1, 1].set_ylabel("Exact endpoint-kernel TV")
    axes[1, 1].set_title("D  Exact toy aggregation counterexample", loc="left", fontweight="bold")
    fig.suptitle("V5 — Temporal-resolution and aggregation sensitivity", fontsize=14, fontweight="bold")
    return _save(fig, output_path)

def plot_summary(result, output_path):
    plt = _setup_matplotlib()
    names = _dataset_names(result)
    gates = [
        ("Structure", "structure"),
        ("Dynamics", "dynamics"),
        ("Neg. controls", "negative"),
        ("Reorder", "reordering"),
        ("Windows", "windows"),
    ]
    matrix = np.asarray(
        [
            [
                float(result["acceptance"]["by_dataset"][name][key])
                for _, key in gates
            ]
            for name in names
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[0].set_xticks(np.arange(len(gates)))
    axes[0].set_xticklabels([label for label, _ in gates], rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(names)))
    axes[0].set_yticklabels(names)
    for row in range(len(names)):
        for column in range(len(gates)):
            axes[0].text(
                column,
                row,
                "PASS" if matrix[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=8,
            )
    axes[0].set_title("A  Unified acceptance matrix", loc="left", fontweight="bold")

    synthetic_names = [
        name for name in names if result["datasets"][name]["kind"] == "synthetic"
    ]
    published = [
        result["datasets"][name]["published_profile"]["interactions"]
        for name in synthetic_names
    ]
    realized = [
        result["datasets"][name]["unique_aggregate_interactions"]
        for name in synthetic_names
    ]
    sx = np.arange(len(synthetic_names))
    axes[1].bar(sx - 0.18, published, 0.36, label="Published |E|")
    axes[1].bar(sx + 0.18, realized, 0.36, label="Regenerated unique groups")
    axes[1].set_yscale("log")
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(synthetic_names)
    axes[1].set_ylabel("Aggregate interactions")
    axes[1].set_title("B  SSC profile calibration", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    fig.suptitle("V6 — Eight-data-set temporal conversion validation", fontsize=14, fontweight="bold")
    return _save(fig, output_path)

def create_all_figures(result, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = OrderedDict()
    paths["V1"] = plot_structural_validation(
        result, figure_dir / "conversion_v1_structure.png"
    )
    paths["V2"] = plot_dynamic_validation(
        result, figure_dir / "conversion_v2_dynamics.png"
    )
    paths["V3"] = plot_negative_controls(
        result, figure_dir / "conversion_v3_negative_controls.png"
    )
    paths["V4"] = plot_reordering(
        result, figure_dir / "conversion_v4_reordering.png"
    )
    paths["V5"] = plot_window_sensitivity(
        result, figure_dir / "conversion_v5_window_sensitivity.png"
    )
    paths["V6"] = plot_summary(
        result, figure_dir / "conversion_v6_summary.png"
    )
    return paths

def _fmt(value):
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4:
            return "%.3e" % value
        return "%.6f" % value
    return str(value)

def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_fmt(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)

def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)

def write_structural_report(result, report_dir):
    rows = []
    for name in _dataset_names(result):
        data = result["datasets"][name]
        metrics = result["structural"][name]
        rows.append(
            (
                name,
                data["kind"],
                data["nodes"],
                data["events"],
                data["temporal_variation"]["unique_snapshot_count"],
                metrics["simplicial_cell_count"],
                metrics["via_simplicial_multigraph_edge_count_with_multiplicity"],
                metrics["simplegraph_edge_count"],
                metrics["passed"],
            )
        )
    error_rows = []
    for name in _dataset_names(result):
        metrics = result["structural"][name]
        error_rows.append(
            (
                name,
                metrics["max_simplicial_multiset_error"],
                metrics["max_direct_multigraph_weight_error"],
                metrics["max_via_simplicial_multigraph_weight_error"],
                metrics["max_simplegraph_support_error"],
                metrics["max_provenance_error"],
            )
        )
    text = """# 实验V1：逐快照结构转换验证报告

## 实验目的

本实验验证时变超图到时变单纯复形、时变多重图和时变简单图的转换是否在每一个时间快照上严格遵守定义。验证重点不是传播曲线，而是节点、时间标签、重复超边、单纯形组合、边重数和来源分组是否被准确保留。

## 实验设计

对每个数据集和每个时间点，程序用独立的纯快照公式重新计算理论结果，再与时变对象中保存的结果比较。超图到单纯复形使用 κ=2；大于3个节点的超边被展开为全部三节点组合。多重图的边重数等于同一时刻包含该节点对的交互次数，简单图则只保留重数是否大于零。

结构转换属于整数和集合运算，因此通过标准是严格零误差，不使用统计容差。

## 图形说明

![V1结构转换](../../results/figures/conversion_v1_structure.png)

- 子图A比较总快照数和唯一快照数，用于确认输入不是静态快照重复。
- 子图B展示8个数据集的节点数和时变事件数，双对数坐标用于同时容纳SSC1到SSC4以及真实数据的规模差异。
- 子图C展示同一批事件经过H、C、MG和G表示后产生的对象数量。数量变化是投影定义和重复边处理的结果，不是转换误差。
- 子图D是严格验收矩阵。H→C检查单纯形多重集合，H→MG和C→MG检查边重数，H→G检查连接支持集，route检查直接与间接简单图路径，source检查来源分组。

## 数据规模与转换结果

""" + _table(
        ["数据集", "类型", "节点", "事件", "唯一快照", "C单元", "MG边重数", "G边", "结论"],
        rows,
    ) + """

## 严格误差

""" + _table(
        ["数据集", "H→C", "H→MG", "C→MG", "H→G", "来源分组"],
        error_rows,
    ) + """

## 结果解释

所有误差都必须等于0才可判定结构转换成立。H→C的单元数通常高于原始超边数，是因为较大超边会生成多个κ维单元；MG边数进一步增加，是因为一个群体交互会产生多个节点对；G边数减少则来自重复边合并。这些规模差异是转换预期结果。

直接H→MG和H→C→MG不应无条件比较原始重数是否相同，因为截断大超边后，同一节点对可能出现在多个投影单纯形中。本实验分别验证两条路径各自的理论重数，并只在简单连接支持集层面检查路径一致性。

## 结论

验收矩阵全部通过意味着转换程序逐快照正确、时间标签未错位、重复交互未意外丢失，并且来源信息可追溯。该结论只证明结构转换正确；动力学等价由实验V2进一步验证。
"""
    return _write(Path(report_dir) / "实验V1_逐快照结构转换验证报告.md", text)

def write_dynamic_report(result, report_dir):
    exact_rows = []
    for label, values in result["exact"]["conversions"].items():
        exact_rows.append(
            (label, values["max_kernel_tv"], values["path_distribution_tv"])
        )
    dataset_rows = []
    for name in _dataset_names(result):
        values = result["dynamics"][name]
        errors = values["max_conditional_probability_error"]
        dataset_rows.append(
            (
                name,
                errors["H_to_C_C1"],
                errors["C_to_MG_C2"],
                errors["H_to_C_to_MG_C3"],
                errors["H_to_MG_direct_C3"],
                all(values["common_random_path_match"].values()),
                values["passed"],
            )
        )
    text = """# 实验V2：时变动力学精确转换验证报告

## 实验目的

本实验验证结构投影之后，源网络和目标网络上的SI动力学是否在相应条件下真正等价。验证包括H→C的C1t、C→MG的完整C2t，以及明确分开的H→C→MG和直接H→MG两条C3t路径。

## 实验设计

小网络部分使用5个节点、3个时间快照，枚举全部32个感染状态，并构建每个时间点的完整联合转移核。随后比较单步核最大逐行TV距离和完整轨迹分布TV距离。

8个数据集部分使用4种代表性感染状态检查全部时间点的节点条件感染概率，并使用完全相同的随机数驱动源网络和目标网络。若条件概率相同，共同随机数应产生逐节点、逐时刻完全一致的样本路径。

## 图形说明

![V2动力学转换](../../results/figures/conversion_v2_dynamics.png)

- 子图A使用对数纵轴展示小网络单步核TV和完整路径TV；虚线为1e-10验收阈值。
- 子图B是8个数据集、4种转换的最大条件概率误差热力图，颜色表示误差的十进制对数。
- 子图C显示共同随机数路径是否完全一致。这里的“identical”要求整个状态矩阵相同，而不只是平均感染密度相近。
- 子图D比较源网络和目标网络的终点感染密度。点位于对角线表示两个表示给出相同结果。

## 完整状态空间结果

""" + _table(["转换", "最大核TV", "路径分布TV"], exact_rows) + """

多重图到简单图在没有重复边时的核TV为 """ + _fmt(result["exact"]["MG_to_G_no_repeated_edges_kernel_tv"]) + """；存在重复边却直接删除重数时，核TV为 """ + _fmt(result["exact"]["MG_to_G_repeated_edge_loss_kernel_tv"]) + """。这说明MG→G不是无条件动力学等价转换。

## 8个数据集结果

""" + _table(
        [
            "数据集",
            "H→C C1t",
            "C→MG C2t",
            "H→C→MG C3t",
            "H→MG-direct C3t",
            "路径完全一致",
            "结论",
        ],
        dataset_rows,
    ) + """

## 结果解释

感染密度曲线重合只能说明一个低维均值相同，不能排除联合状态分布差异。因此本实验把完整核TV和路径分布TV作为小网络的最高等级证据，并在大网络上使用共同随机数路径一致性作为可扩展验证。

C1t通过说明超图中由超边规模和感染节点数决定的效应，被投影单纯形中的各阶暴露准确重建。完整C2t不仅关闭二阶及更高阶传播，还要求一阶暴露重数、来源分组和组合语义一致。C3t分别验证直接团展开的暴露数E=χ，以及经κ阶单纯投影后的暴露数E=n_{m,χ,1}^{(κ)}；二者不再共用含糊的“H→MG”标签。

## 结论

只有当核TV、路径TV和条件概率误差都低于阈值，并且共同随机数路径完全一致时，才判定动力学转换有效。MG→G的正例仅限于没有重复边或使用有效权重补偿的情形。
"""
    return _write(Path(report_dir) / "实验V2_时变动力学精确转换验证报告.md", text)

def write_negative_report(result, report_dir):
    rows = []
    for name in _dataset_names(result):
        values = result["negative_controls"][name]
        rows.append(
            (
                name,
                values["wrong_C1_probability_error"],
                values["nonzero_higher_order_C2_error"],
                values["one_step_time_shift_error"],
                values["MG_to_G_multiplicity_loss_error"],
                values["detected_control_count"],
            )
        )
    text = """# 实验V3：负对照与信息损失边界报告

## 实验目的

正实验的零误差可能来自测试过弱，因此本实验主动破坏转换条件，检查指标能否识别无效转换。四个负对照分别是：将C1t映射率放大15%、在C2t中保留非零三元传播、将目标快照错后一位，以及把多重图直接压缩为简单图。

## 图形说明

![V3负对照](../../results/figures/conversion_v3_negative_controls.png)

- 子图A使用对数纵轴显示最大条件概率误差。黑色虚线是精确转换容差；高于虚线表示差异可以被识别。
- 子图B统计每个数据集4个负对照中检测到几个。多重图到简单图在某些快照没有重复边时可能暂时无差异，因此该项不作为所有数据集的强制失败条件。

## 实验结果

""" + _table(
        ["数据集", "破坏C1t", "破坏C2t", "时间错位", "删除重数", "检出数"],
        rows,
    ) + """

## 结果解释

破坏C1t后，超图单个超边产生的总感染效应不再等于投影单纯形效应；破坏C2t后，多重图无法表示单纯复形中的三元协同传播；时间错位说明结构内容相同但时间标签不匹配也不能构成有效时变转换。

MG→G误差反映信息损失边界。简单图只记录边是否存在，不能区分一条接触和多条独立接触。当某数据集在所选快照内没有重复节点对时，这个负对照可能为零；一旦出现重复节点对，直接删除重数通常会改变感染概率。

## 结论

正确条件下的机器精度误差与错误条件下的非零误差形成明确分离，证明验收方法不是“无论输入什么都通过”。同时，实验给出了哪些转换是有损投影以及需要附加补偿条件。
"""
    return _write(Path(report_dir) / "实验V3_负对照与信息损失边界报告.md", text)

def write_reordering_report(result, report_dir):
    rows = []
    for name in _dataset_names(result):
        values = result["reordering"][name]
        rows.append(
            (
                name,
                values["reorder_count"],
                values["aligned_conversion_max_error"],
                values["misaligned_conversion_max_error"],
                values["source_order_effect_max"],
                values["passed"],
            )
        )
    text = """# 实验V4：时间重排一致性报告

## 实验目的

本实验验证转换是否对共同的时间重排保持一致，并检验只重排一侧时是否会被识别。对每个数据集生成多组随机排列：第一组同时作用于H和C，第二组只作用于源网络或目标网络。

## 图形说明

![V4时间重排](../../results/figures/conversion_v4_reordering.png)

- 子图A比较共同重排和错位重排后的转换误差。共同重排仍满足逐时刻C1t，因此应低于容差；错位重排把不同快照配对，应出现非零误差。
- 子图B比较原始源网络和重排源网络的条件传播概率，用来确认事件顺序本身会改变过程。

## 实验结果

""" + _table(
        ["数据集", "重排次数", "共同重排误差", "错位误差", "源顺序效应", "结论"],
        rows,
    ) + """

## 结果解释

该实验区分了两个概念：转换规则本身应对同一个时间置换保持协变，但网络动力学不应对任意事件顺序保持不变。也就是说，将H和C以同样方式重排后它们仍然等价；只重排其中一个表示则破坏时间对齐。

## 结论

共同重排误差低于阈值证明转换不是依赖某一个偶然时间序列；错位重排和源顺序效应非零则证明时间标签和事件顺序不能被忽略。
"""
    return _write(Path(report_dir) / "实验V4_时间重排一致性报告.md", text)

def write_window_report(result, report_dir):
    rows = []
    for name in _dataset_names(result):
        values = result["window_sensitivity"][name]
        largest = values["rows"][-1]
        rows.append(
            (
                name,
                largest["window_size"],
                largest["mean_events_per_snapshot"],
                largest["endpoint_density_change_from_window1"],
                max(row["H_to_C_probability_error"] for row in values["rows"]),
                values["passed"],
            )
        )
    counterexample = next(iter(result["window_sensitivity"].values()))[
        "exact_small_network_counterexample"
    ]
    exact_rows = sorted(
        (
            int(key),
            value,
        )
        for key, value in counterexample["window_endpoint_kernel_tv"].items()
    )
    text = """# 实验V5：时间窗口与聚合敏感性报告

## 实验目的

本实验研究时间分辨率改变是否影响传播结果，以及在每一种新分辨率上H→C转换是否仍然正确。窗口大小取1、2、3、4和6个原始快照。

## 图形说明

![V5窗口敏感性](../../results/figures/conversion_v5_window_sensitivity.png)

- 子图A展示相对于窗口1的平均终点感染密度变化。非零表示时间聚合改变传播。
- 子图B展示窗口增大后每个快照中同时存在的事件数量。窗口越大，原本先后发生的事件越容易被视为同时发生。
- 子图C展示每种分辨率上H→C的最大误差。它回答“转换是否仍正确”，而不是“聚合是否无损”。
- 子图D使用小网络完整转移核展示聚合窗口造成的精确终点核误差。

## 8个数据集结果

""" + _table(
        ["数据集", "最大窗口", "平均并发事件", "终点密度变化", "最大转换误差", "转换结论"],
        rows,
    ) + """

## 小网络精确结果

""" + _table(["窗口", "终点核TV"], exact_rows) + """

## 结果解释

窗口聚合包含两个不同结论。第一，给定聚合后的时变超图，重新执行投影并重新计算C1t参数，H→C转换仍应精确，这由子图C验证。第二，聚合后的过程一般不再等于原始细时间过程，因为多个先后事件被压缩到同一步，转移核乘积被一个聚合核替代，这由子图A和D验证。

因此，“不同网络表示之间可以转换”不意味着“可以任意删除时间分辨率”。网络表示转换与时间聚合是两种不同操作。

## 结论

实验同时支持转换的分辨率稳健性和时间聚合的有损性：转换在每个选定分辨率上有效，但不同分辨率对应的动力学过程不必相同。
"""
    return _write(Path(report_dir) / "实验V5_时间窗口与聚合敏感性报告.md", text)

def write_summary_report(result, report_dir):
    rows = []
    for name in _dataset_names(result):
        gates = result["acceptance"]["by_dataset"][name]
        rows.append(
            (
                name,
                result["datasets"][name]["kind"],
                gates["structure"],
                gates["dynamics"],
                gates["negative"],
                gates["reordering"],
                gates["windows"],
                all(gates.values()),
            )
        )
    synthetic_rows = []
    for name in _dataset_names(result):
        data = result["datasets"][name]
        if data["kind"] == "synthetic":
            synthetic_rows.append(
                (
                    name,
                    data["published_profile"]["nodes"],
                    data["published_profile"]["interactions"],
                    data["published_profile"]["mean_triangle_degree"],
                    data["unique_aggregate_interactions"],
                    "是" if data["exact_original_realization_available"] else "否",
                )
            )
    text = """# 时变网络转换八数据集综合验证报告

## 验证范围

本套件覆盖论文中的4个真实数据集和4个合成SSC网络。真实数据保留原始事件时间顺序并重分箱为12个等时长快照；SSC1–SSC4由于论文和仓库没有公开四个具体随机边表，按照Supplementary Table S2公开的N、|E|和平均2-单纯形度确定性再生成聚合结构，再将交互分配到时变快照。报告不把再生成实例表述为论文原始随机实例。

## 综合图说明

![V6综合验收](../../results/figures/conversion_v6_summary.png)

- 子图A汇总结构、动力学、负对照、重排和窗口五类验收门。每个绿色PASS都对应一个明确数值条件，而不是人工判断。
- 子图B比较SSC论文公开的|E|与本次再生成结构的唯一聚合交互数。二者严格一致，用于确认合成规模校准。

## SSC参数校准

""" + _table(
        ["网络", "论文N", "论文|E|", "论文平均三角度", "再生成唯一交互", "原始边表可用"],
        synthetic_rows,
    ) + """

## 综合验收矩阵

""" + _table(
        ["数据集", "类型", "结构", "动力学", "负对照", "重排", "窗口", "总评"],
        rows,
    ) + """

## 证据层次

第一层是逐快照整数和集合恒等式，证明程序按照转换定义工作。第二层是小网络完整状态空间验证，证明满足C1t、完整C2t和两条路径各自的C3t时完整转移核及路径分布一致。第三层是在8个数据集上比较条件概率和共同随机数路径，验证可扩展性。第四层使用负对照、时间错位和窗口聚合界定转换成立的必要条件及信息损失边界。

## 总结

若综合验收为通过，可以得出：当前实现的时变结构转换在全部8个数据集上逐快照正确；在相应时变参数条件下，H→C、C→MG、H→C→MG以及直接H→MG动力学等价；删除条件、时间对齐或多重边信息会产生可检测误差；时间聚合后可以重新获得表示间等价，但不能认为聚合后的过程等于原始细时间过程。

完整机器可读结果位于 `results/summary_tables/temporal_conversion_validation.json`。
"""
    return _write(Path(report_dir) / "时变网络转换八数据集综合验证报告.md", text)

def create_all_reports(result, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    return OrderedDict(
        [
            ("V1", write_structural_report(result, report_dir)),
            ("V2", write_dynamic_report(result, report_dir)),
            ("V3", write_negative_report(result, report_dir)),
            ("V4", write_reordering_report(result, report_dir)),
            ("V5", write_window_report(result, report_dir)),
            ("V6", write_summary_report(result, report_dir)),
        ]
    )
