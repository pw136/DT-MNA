
from __future__ import division

import csv
from pathlib import Path

import numpy as np

MODEL_ORDER = ("SG", "G", "MG", "SC2", "H")
MODEL_INDEX = {name: index for index, name in enumerate(MODEL_ORDER)}
MODEL_COLORS = {
    "SG": "#9aa4b2",
    "G": "#2878b5",
    "MG": "#2ca02c",
    "SC2": "#f08a24",
    "H": "#d62728",
}

def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt

def _save(fig, path, apply_tight_layout=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if apply_tight_layout:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(str(path), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path.resolve())

def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )
    return str(path.resolve())

def write_innovation3_tables(result, output_dir):
    output_dir = Path(output_dir)
    paths = {}
    a0_rows = [
        {key: value for key, value in row.items() if key != "serializable"}
        for row in result["A0"]["rows"]
    ]
    paths["A0"] = _write_csv(
        output_dir / "innovation3_a0_mechanism_oracle.csv", a0_rows
    )
    paths["A1"] = _write_csv(
        output_dir / "innovation3_a1_synergy_budget_phase.csv",
        result["A1"]["rows"],
    )
    paths["A2"] = _write_csv(
        output_dir / "innovation3_a2_temporal_synergy_phase.csv",
        result["A2"]["rows"],
    )
    paths["A3"] = _write_csv(
        output_dir / "innovation3_a3_collision_certificate.csv",
        result["A3"]["rows"],
    )
    upgrade_rows = []
    decision = result["A4"]["decision"]
    for name in MODEL_ORDER:
        assessment = decision["assessments"][name]
        upgrade_rows.append(
            {
                "model": name,
                "validation_worst_error": assessment["validation"][
                    "worst_error"
                ],
                "audit_worst_error": assessment["audit"]["worst_error"],
                "test_worst_error": assessment["test"]["worst_error"],
                "validation_safe": assessment["validation_safe"],
                "audit_safe": assessment["audit_safe"],
                "test_safe": assessment["test_safe"],
                "selected": name == decision["selected_name"],
            }
        )
    paths["A4"] = _write_csv(
        output_dir / "innovation3_a4_audit_upgrade.csv", upgrade_rows
    )
    paths["A5"] = _write_csv(
        output_dir / "innovation3_a5_sample_coverage.csv",
        result["A5"]["rows"],
    )
    a6_rows = [
        {key: value for key, value in row.items() if key != "serializable"}
        for row in result["A6"]["rows"]
    ]
    paths["A6"] = _write_csv(
        output_dir / "innovation3_a6_eight_datasets.csv", a6_rows
    )
    paths["efficiency"] = _write_csv(
        output_dir / "innovation3_a9_efficiency.csv",
        result["A6"]["efficiency"],
    )
    baseline_rows = [
        {"method": method, **metrics}
        for method, metrics in result["A7"]["methods"].items()
    ]
    paths["A7"] = _write_csv(
        output_dir / "innovation3_a7_baseline_comparison.csv",
        baseline_rows,
    )
    paths["A8"] = _write_csv(
        output_dir / "innovation3_a8_reliability.csv",
        [result["A8"]],
    )
    acceptance_rows = [
        {"gate": gate, "passed": passed}
        for gate, passed in result["acceptance"].items()
    ]
    paths["acceptance"] = _write_csv(
        output_dir / "innovation3_acceptance.csv", acceptance_rows
    )
    return paths

def plot_oracle_closed_loop(result, output_path):
    plt = _setup_matplotlib()
    rows = result["A0"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    matrix = np.zeros((len(MODEL_ORDER), len(MODEL_ORDER)), dtype=int)
    for row in rows:
        matrix[MODEL_INDEX[row["expected"]], MODEL_INDEX[row["selected"]]] += 1
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    axes[0].set_xticks(range(len(MODEL_ORDER)))
    axes[0].set_xticklabels(MODEL_ORDER)
    axes[0].set_yticks(range(len(MODEL_ORDER)))
    axes[0].set_yticklabels(MODEL_ORDER)
    axes[0].set_xlabel("Closed-loop selection")
    axes[0].set_ylabel("Known minimal representation")
    axes[0].set_title("A  Mechanism-level oracle agreement", loc="left", fontweight="bold")
    for row_index in range(len(MODEL_ORDER)):
        for column in range(len(MODEL_ORDER)):
            axes[0].text(
                column,
                row_index,
                str(matrix[row_index, column]),
                ha="center",
                va="center",
            )

    x = np.arange(len(rows))
    budget = float(result["config"]["error_budget"])
    for model in MODEL_ORDER:
        values = [
            max(
                row["serializable"]["assessments"][model]["test"][
                    "worst_error"
                ],
                1e-16,
            )
            for row in rows
        ]
        axes[1].plot(
            x,
            values,
            marker="o",
            linewidth=1.3,
            color=MODEL_COLORS[model],
            label=model,
        )
    axes[1].axhline(budget, color="#111111", linestyle="--", linewidth=1, label="Budget")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [row["case"].replace("_", "\n") for row in rows],
        rotation=0,
    )
    axes[1].set_ylabel("Independent-test worst probability error")
    axes[1].set_title("B  Why each rung is the lowest safe choice", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=3)
    fig.suptitle("Innovation 3 — Known mechanisms and closed-loop choices", fontweight="bold")
    return _save(fig, output_path)

def _decision_cmap(plt):
    from matplotlib.colors import ListedColormap

    return ListedColormap([MODEL_COLORS[name] for name in MODEL_ORDER])

def plot_synergy_phase(result, output_path):
    plt = _setup_matplotlib()
    rows = result["A1"]["rows"]
    synergies = sorted({row["synergy"] for row in rows})
    budgets = sorted({row["budget"] for row in rows})
    selected = np.zeros((len(budgets), len(synergies)))
    oracle = np.zeros_like(selected)
    for row in rows:
        y = budgets.index(row["budget"])
        x = synergies.index(row["synergy"])
        selected[y, x] = MODEL_INDEX[row["selected"]]
        oracle[y, x] = MODEL_INDEX[row["oracle"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    fig.subplots_adjust(left=0.075, right=0.895, bottom=0.15, top=0.86, wspace=0.25)
    for axis, matrix, title in (
        (axes[0], selected, "A  Closed-loop decision"),
        (axes[1], oracle, "B  Independent-test oracle"),
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=-0.5,
            vmax=4.5,
            cmap=_decision_cmap(plt),
        )
        axis.set_xticks(range(len(synergies)))
        axis.set_xticklabels(["%.3g" % value for value in synergies])
        axis.set_yticks(range(len(budgets)))
        axis.set_yticklabels(["%.3g" % value for value in budgets])
        axis.set_xlabel("Second-order synergy β₂")
        axis.set_ylabel("Allowed worst error ε")
        axis.set_title(title, loc="left", fontweight="bold")
        for row_index in range(len(budgets)):
            for column in range(len(synergies)):
                axis.text(
                    column,
                    row_index,
                    MODEL_ORDER[int(matrix[row_index, column])],
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    colorbar_axis = fig.add_axes([0.915, 0.18, 0.018, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=range(5))
    colorbar.ax.set_yticklabels(MODEL_ORDER)
    fig.suptitle("Accuracy budget determines whether higher order is necessary", fontweight="bold")
    return _save(fig, output_path, apply_tight_layout=False)

def plot_temporal_synergy_phase(result, output_path):
    plt = _setup_matplotlib()
    rows = result["A2"]["rows"]
    temporal = sorted({row["temporal_strength"] for row in rows})
    synergy = sorted({row["synergy"] for row in rows})
    matrix = np.zeros((len(synergy), len(temporal)))
    correct = np.zeros_like(matrix)
    for row in rows:
        x = temporal.index(row["temporal_strength"])
        y = synergy.index(row["synergy"])
        matrix[y, x] = MODEL_INDEX[row["selected"]]
        correct[y, x] = float(row["correct"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    image = axes[0].imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=-0.5,
        vmax=4.5,
        cmap=_decision_cmap(plt),
    )
    axes[0].set_xticks(range(len(temporal)))
    axes[0].set_xticklabels(["%.3g" % value for value in temporal])
    axes[0].set_yticks(range(len(synergy)))
    axes[0].set_yticklabels(["%.3g" % value for value in synergy])
    axes[0].set_xlabel("Temporal rate variation strength")
    axes[0].set_ylabel("Second-order synergy β₂")
    axes[0].set_title("A  Time and order are separated", loc="left", fontweight="bold")
    for row_index in range(len(synergy)):
        for column in range(len(temporal)):
            axes[0].text(
                column,
                row_index,
                MODEL_ORDER[int(matrix[row_index, column])],
                ha="center",
                va="center",
                fontsize=7,
            )
    colorbar = fig.colorbar(
        image, ax=axes[0], fraction=0.046, pad=0.04, ticks=range(5)
    )
    colorbar.ax.set_yticklabels(MODEL_ORDER)
    axes[1].imshow(
        correct,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )
    axes[1].set_xticks(range(len(temporal)))
    axes[1].set_xticklabels(["%.3g" % value for value in temporal])
    axes[1].set_yticks(range(len(synergy)))
    axes[1].set_yticklabels(["%.3g" % value for value in synergy])
    axes[1].set_xlabel("Temporal rate variation strength")
    axes[1].set_ylabel("Second-order synergy β₂")
    axes[1].set_title("B  Agreement with oracle", loc="left", fontweight="bold")
    for row_index in range(len(synergy)):
        for column in range(len(temporal)):
            axes[1].text(
                column,
                row_index,
                "PASS" if correct[row_index, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.suptitle("Two-axis necessity map: temporal information versus higher order", fontweight="bold")
    return _save(fig, output_path)

def plot_collision_certificate(result, output_path):
    plt = _setup_matplotlib()
    rows = result["A3"]["rows"]
    conflicts = sorted({row["conflict"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for model in ("G", "MG", "SC2"):
        subset = [row for row in rows if row["model"] == model]
        axes[0].plot(
            conflicts,
            [row["collision_lower_bound"] for row in subset],
            marker="o",
            color=MODEL_COLORS[model],
            label="%s lower bound" % model,
        )
        axes[0].plot(
            conflicts,
            [row["observed_worst_error"] for row in subset],
            marker="x",
            linestyle="--",
            color=MODEL_COLORS[model],
            label="%s observed" % model,
        )
    axes[0].axhline(
        float(result["config"]["error_budget"]),
        color="#111111",
        linestyle=":",
        label="Budget",
    )
    axes[0].set_xlabel("Constraint-conflict strength")
    axes[0].set_ylabel("Worst probability error")
    axes[0].set_title("A  Collision lower bound is respected", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    selected = []
    for conflict in conflicts:
        selected.append(
            next(
                row["selected"]
                for row in rows
                if row["conflict"] == conflict
            )
        )
    axes[1].step(
        conflicts,
        [MODEL_INDEX[name] for name in selected],
        where="mid",
        marker="o",
        color="#2878b5",
    )
    axes[1].set_yticks(range(len(MODEL_ORDER)))
    axes[1].set_yticklabels(MODEL_ORDER)
    axes[1].set_xlabel("Constraint-conflict strength")
    axes[1].set_ylabel("Selected representation")
    axes[1].set_title("B  Irreducible conflict forces safe upgrades", loc="left", fontweight="bold")
    fig.suptitle("Projection collisions certify when graph-only models cannot work", fontweight="bold")
    return _save(fig, output_path)

def plot_audit_and_coverage(result, output_path):
    plt = _setup_matplotlib()
    decision = result["A4"]["decision"]
    coverage = result["A5"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for offset, split, label in (
        (-width, "validation", "Validation"),
        (0.0, "audit", "Safety audit"),
        (width, "test", "Independent test"),
    ):
        values = [
            max(
                decision["assessments"][name][split]["worst_error"],
                1e-16,
            )
            for name in MODEL_ORDER
        ]
        axes[0].bar(x + offset, values, width, label=label)
    axes[0].axhline(
        float(result["config"]["error_budget"]),
        color="#111111",
        linestyle="--",
        label="Budget",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODEL_ORDER)
    axes[0].set_ylabel("Worst probability error")
    axes[0].set_title(
        "A  Audit catches a validation-only underfit",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    states = [row["states_per_time"] for row in coverage]
    axes[1].plot(
        states,
        [row["correct_rate"] for row in coverage],
        marker="o",
        label="Oracle agreement",
    )
    axes[1].plot(
        states,
        [row["safety_rate"] for row in coverage],
        marker="s",
        label="Test safety",
    )
    axes[1].plot(
        states,
        [row["abstention_rate"] for row in coverage],
        marker="^",
        label="Safe abstention",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Validation states per snapshot")
    axes[1].set_ylabel("Fraction over controlled cases")
    axes[1].set_title(
        "B  More state coverage replaces abstention with decisions",
        loc="left",
        fontweight="bold",
    )
    axes[1].legend(frameon=False)
    fig.suptitle("Closed-loop safety: audit upgrade and evidence-aware abstention", fontweight="bold")
    return _save(fig, output_path)

def plot_eight_datasets(result, output_path):
    plt = _setup_matplotlib()
    rows = result["A6"]["rows"]
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    mechanisms = ("pairwise", "sc2", "conflict")
    selected = np.zeros((len(datasets), len(mechanisms)))
    oracle = np.zeros_like(selected)
    errors = np.zeros_like(selected)
    for row in rows:
        y = datasets.index(row["dataset"])
        x = mechanisms.index(row["mechanism"])
        selected[y, x] = MODEL_INDEX[row["selected"]]
        oracle[y, x] = MODEL_INDEX[row["oracle"]]
        errors[y, x] = row["test_worst_error"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    for axis, matrix, title in (
        (axes[0], selected, "A  Closed-loop choice"),
        (axes[1], oracle, "B  Independent-test oracle"),
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            vmin=-0.5,
            vmax=4.5,
            cmap=_decision_cmap(plt),
        )
        axis.set_xticks(range(len(mechanisms)))
        axis.set_xticklabels(mechanisms, rotation=20, ha="right")
        axis.set_yticks(range(len(datasets)))
        axis.set_yticklabels(datasets)
        axis.set_title(title, loc="left", fontweight="bold")
        for y in range(len(datasets)):
            for x in range(len(mechanisms)):
                axis.text(
                    x,
                    y,
                    MODEL_ORDER[int(matrix[y, x])],
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    colorbar = fig.colorbar(
        image, ax=axes[:2], fraction=0.025, pad=0.02, ticks=range(5)
    )
    colorbar.ax.set_yticklabels(MODEL_ORDER)
    log_errors = np.log10(np.maximum(errors, 1e-16))
    error_image = axes[2].imshow(
        log_errors,
        aspect="auto",
        cmap="magma",
        vmin=-16.0,
        vmax=np.log10(float(result["config"]["error_budget"])),
    )
    axes[2].set_xticks(range(len(mechanisms)))
    axes[2].set_xticklabels(mechanisms, rotation=20, ha="right")
    axes[2].set_yticks(range(len(datasets)))
    axes[2].set_yticklabels(datasets)
    axes[2].set_title("C  Selected-model test error", loc="left", fontweight="bold")
    for y in range(len(datasets)):
        for x in range(len(mechanisms)):
            axes[2].text(
                x,
                y,
                "%.3g" % errors[y, x],
                ha="center",
                va="center",
                color="white" if log_errors[y, x] < -8.0 else "black",
                fontsize=6.5,
            )
    error_colorbar = fig.colorbar(
        error_image, ax=axes[2], fraction=0.046, pad=0.04
    )
    error_colorbar.set_label("log10(worst error)")
    fig.suptitle("Eight temporal structures under three known dynamical mechanisms", fontweight="bold")
    return _save(fig, output_path)

def plot_baseline_comparison(result, output_path):
    plt = _setup_matplotlib()
    methods = list(result["A7"]["methods"])
    metrics = (
        ("selection_accuracy", "Oracle agreement"),
        ("safety_rate", "Test safety"),
        ("overmodel_rate", "Over-modelling"),
        ("unsafe_simplification_rate", "Unsafe simplification"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    x = np.arange(len(methods))
    width = 0.19
    for index, (metric, label) in enumerate(metrics):
        axes[0].bar(
            x + (index - 1.5) * width,
            [result["A7"]["methods"][name][metric] for name in methods],
            width,
            label=label,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction of evaluation cases")
    axes[0].set_title("A  Accuracy–safety–complexity comparison", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    thn = result["A7"]["methods"]["THN-Need closed loop"]
    markers = {
        "THN-Need closed loop": "D",
        "Validation-only": "o",
        "Mean-error selector": "s",
        "Hyperedge-size rule": "^",
        "Always G": "v",
        "Always H": "P",
    }
    label_offsets = {
        "THN-Need closed loop": (6, 8),
        "Validation-only": (6, -14),
        "Mean-error selector": (6, -4),
        "Hyperedge-size rule": (-118, 8),
        "Always G": (6, 8),
        "Always H": (-72, -14),
    }
    for name in methods:
        point = result["A7"]["methods"][name]
        axes[1].scatter(
            point["overmodel_rate"],
            point["unsafe_simplification_rate"],
            s=80 if name == "THN-Need closed loop" else 48,
            c="#2878b5" if name == "THN-Need closed loop" else "#9aa4b2",
            marker=markers[name],
            zorder=3 if name == "THN-Need closed loop" else 2,
        )
        axes[1].annotate(
            name,
            (
                point["overmodel_rate"],
                point["unsafe_simplification_rate"],
            ),
            xytext=label_offsets[name],
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Over-modelling rate")
    axes[1].set_ylabel("Unsafe simplification rate")
    axes[1].set_title("B  Conservative and unsafe extremes are both visible", loc="left", fontweight="bold")
    axes[1].axvline(thn["overmodel_rate"], color="#2878b5", alpha=0.2)
    fig.suptitle("Necessity decisions versus common heuristic selectors", fontweight="bold")
    return _save(fig, output_path)

def plot_efficiency(result, output_path):
    plt = _setup_matplotlib()
    rows = result["A6"]["efficiency"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    x = [row["complexity_ratio"] for row in rows]
    y = [row["speedup"] for row in rows]
    colors = [MODEL_COLORS[row["selected"]] for row in rows]
    axes[0].scatter(x, y, c=colors, s=50)
    label_offsets = {
        "SSC1": (5, 9),
        "SSC2": (5, -12),
        "SSC3": (5, 1),
        "SSC4": (5, -7),
        "Email-Enron": (5, 5),
        "NDC-classes-full": (5, -7),
        "Primary-School": (-100, 9),
        "High-School": (-88, -10),
    }
    short_labels = {
        "Primary-School": "Primary-School",
        "High-School": "High-School",
    }
    for row in rows:
        axes[0].annotate(
            short_labels.get(row["dataset"], row["dataset"]),
            (row["complexity_ratio"], row["speedup"]),
            xytext=label_offsets[row["dataset"]],
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].axhline(1.0, color="#111111", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Representation-complexity reduction (H / selected)")
    axes[0].set_ylabel("Measured trajectory speedup (H / selected)")
    axes[0].set_xlim(2.05, 5.75)
    axes[0].set_title("A  Structural reduction and observed runtime", loc="left", fontweight="bold")
    dataset_names = [row["dataset"] for row in rows]
    axes[1].bar(
        np.arange(len(rows)),
        [row["source_seconds"] for row in rows],
        0.38,
        label="H source",
    )
    axes[1].bar(
        np.arange(len(rows)) + 0.38,
        [row["selected_seconds"] for row in rows],
        0.38,
        label="Selected",
    )
    axes[1].set_xticks(np.arange(len(rows)) + 0.19)
    axes[1].set_xticklabels(dataset_names, rotation=30, ha="right")
    axes[1].set_ylabel("Seconds per trajectory (median of 3)")
    axes[1].set_title("B  Closed-loop model execution cost", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    fig.suptitle("Computational consequence of choosing the lowest sufficient representation", fontweight="bold")
    return _save(fig, output_path)

def create_innovation3_figures(result, output_dir):
    output_dir = Path(output_dir)
    return {
        "oracle": plot_oracle_closed_loop(
            result, output_dir / "innovation3_oracle_closed_loop.png"
        ),
        "synergy_phase": plot_synergy_phase(
            result, output_dir / "innovation3_synergy_budget_phase.png"
        ),
        "temporal_phase": plot_temporal_synergy_phase(
            result, output_dir / "innovation3_temporal_synergy_phase.png"
        ),
        "collision": plot_collision_certificate(
            result, output_dir / "innovation3_collision_certificate.png"
        ),
        "audit_coverage": plot_audit_and_coverage(
            result, output_dir / "innovation3_audit_and_coverage.png"
        ),
        "datasets": plot_eight_datasets(
            result, output_dir / "innovation3_eight_datasets.png"
        ),
        "baselines": plot_baseline_comparison(
            result, output_dir / "innovation3_baseline_comparison.png"
        ),
        "efficiency": plot_efficiency(
            result, output_dir / "innovation3_efficiency.png"
        ),
    }
