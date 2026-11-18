
from dataclasses import dataclass

import numpy as np

@dataclass
class QStarFeasibilityResult:
    feasible: bool
    exact_parameter: object
    required_parameters: tuple
    conflict_std: float
    conflict_range: float
    max_constraint_gap: float
    lower_bound: float
    reason: str

def _as_vector(values, label):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not len(vector):
        raise ValueError("%s must not be empty" % label)
    if not np.all(np.isfinite(vector)):
        raise ValueError("%s must contain finite values" % label)
    return vector

def check_scalar_qstar_feasibility(
    targets,
    designs,
    bounds=(0.0, 1.0),
    tolerance=1e-12,
):

    targets = _as_vector(targets, "targets")
    designs = _as_vector(designs, "designs")
    if targets.shape != designs.shape:
        raise ValueError("targets and designs must have the same shape")
    lower, upper = (float(bounds[0]), float(bounds[1]))
    active = np.abs(designs) > tolerance
    impossible_zero = (~active) & (np.abs(targets) > tolerance)
    if np.any(impossible_zero):
        return QStarFeasibilityResult(
            False,
            None,
            tuple(),
            float("inf"),
            float("inf"),
            float(np.max(np.abs(targets[impossible_zero]))),
            float(np.max(np.abs(targets[impossible_zero]))),
            "a zero design coefficient is paired with a nonzero target",
        )
    required = targets[active] / designs[active]
    if not len(required):
        exact = min(max(0.0, lower), upper)
        return QStarFeasibilityResult(
            True, exact, tuple(), 0.0, 0.0, 0.0, 0.0, "all constraints are null"
        )
    conflict_range = float(required.max() - required.min())
    conflict_std = float(required.std())
    center = float((required.max() + required.min()) / 2.0)
    max_gap = float(np.max(np.abs(required - center)))
    in_bounds = bool(
        required.min() >= lower - tolerance and required.max() <= upper + tolerance
    )
    feasible = bool(conflict_range <= tolerance and in_bounds)
    exact = float(required.mean()) if feasible else None
    if not in_bounds:
        reason = "the common required parameter lies outside the admissible bounds"
    elif not feasible:
        reason = "different higher-order channels require incompatible parameters"
    else:
        reason = "all active constraints share one admissible parameter"
    return QStarFeasibilityResult(
        feasible,
        exact,
        tuple(float(value) for value in required),
        conflict_std,
        conflict_range,
        max_gap,
        0.5 * conflict_range,
        reason,
    )

def projection_collision_lower_bound(source_values, target_features):

    values = _as_vector(source_values, "source_values")
    if len(values) != len(target_features):
        raise ValueError("source_values and target_features must have equal length")
    grouped = {}
    for value, feature in zip(values, target_features):
        if isinstance(feature, np.ndarray):
            feature = tuple(float(item) for item in feature.reshape(-1))
        elif isinstance(feature, list):
            feature = tuple(feature)
        grouped.setdefault(feature, []).append(float(value))
    bounds = {
        feature: 0.5 * (max(items) - min(items))
        for feature, items in grouped.items()
        if len(items) > 1
    }
    if not bounds:
        return 0.0, {}
    return float(max(bounds.values())), bounds
