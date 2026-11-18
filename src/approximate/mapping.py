
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

@dataclass
class ApproximateFit:
    parameters: object
    predictions: np.ndarray
    residuals: np.ndarray
    weighted_rmse: float
    mean_absolute_error: float
    worst_error: float
    objective: float
    converged: bool
    method: str

def _vector(values, label):
    result = np.asarray(values, dtype=float).reshape(-1)
    if not len(result):
        raise ValueError("%s must not be empty" % label)
    if not np.all(np.isfinite(result)):
        raise ValueError("%s must contain finite values" % label)
    return result

def _weights(values, length):
    if values is None:
        return np.ones(length, dtype=float)
    result = _vector(values, "weights")
    if len(result) != length:
        raise ValueError("weights have the wrong length")
    if np.any(result < 0.0) or result.sum() <= 0.0:
        raise ValueError("weights must be nonnegative with positive sum")
    return result

def _summarize(parameters, predictions, targets, weights, objective, converged, method):
    residuals = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    normalized = weights / weights.sum()
    return ApproximateFit(
        parameters=parameters,
        predictions=np.asarray(predictions, dtype=float),
        residuals=residuals,
        weighted_rmse=float(np.sqrt(np.dot(normalized, residuals ** 2))),
        mean_absolute_error=float(np.dot(normalized, np.abs(residuals))),
        worst_error=float(np.max(np.abs(residuals))),
        objective=float(objective),
        converged=bool(converged),
        method=method,
    )

def pairwise_effect(beta, exposure_counts):

    beta = np.asarray(beta, dtype=float)
    counts = np.asarray(exposure_counts, dtype=float)
    if np.any(beta < 0.0) or np.any(beta >= 1.0):
        raise ValueError("beta must be in [0, 1)")
    if np.any(counts < 0.0):
        raise ValueError("exposure counts must be nonnegative")
    return 1.0 - np.power(1.0 - beta, counts)

def required_pairwise_beta(source_effects, exposure_counts, tolerance=1e-15):
    effects = _vector(source_effects, "source_effects")
    counts = _vector(exposure_counts, "exposure_counts")
    if effects.shape != counts.shape:
        raise ValueError("source_effects and exposure_counts must have equal shape")
    if np.any(effects < -tolerance) or np.any(effects >= 1.0):
        raise ValueError("source effects must lie in [0, 1)")
    if np.any(counts <= 0.0):
        raise ValueError("exposure counts must be positive")
    effects = np.clip(effects, 0.0, 1.0 - tolerance)
    return 1.0 - np.power(1.0 - effects, 1.0 / counts)

def cvar(values, quantile=0.95):
    values = _vector(values, "values")
    if quantile <= 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in (0, 1]")
    count = max(1, int(np.ceil((1.0 - quantile) * len(values))))
    return float(np.sort(values)[-count:].mean())

def fit_pairwise_hazard(
    source_effects,
    exposure_counts,
    weights=None,
    beta_bounds=(0.0, 0.95),
):

    effects = _vector(source_effects, "source_effects")
    counts = _vector(exposure_counts, "exposure_counts")
    if effects.shape != counts.shape:
        raise ValueError("source_effects and exposure_counts must have equal shape")
    weights = _weights(weights, len(effects))
    if np.any(counts <= 0.0):
        raise ValueError("exposure counts must be positive")
    effects = np.clip(effects, 0.0, 1.0 - 1e-15)
    hazards = -np.log1p(-effects)
    denominator = float(np.dot(weights, counts ** 2))
    if denominator <= 0.0:
        raise ValueError("the weighted design has zero norm")
    fitted_hazard = float(np.dot(weights, counts * hazards) / denominator)
    hazard_bounds = (
        -np.log1p(-float(beta_bounds[0])),
        -np.log1p(-float(beta_bounds[1])),
    )
    fitted_hazard = min(max(fitted_hazard, hazard_bounds[0]), hazard_bounds[1])
    beta = 1.0 - np.exp(-fitted_hazard)
    predictions = pairwise_effect(beta, counts)
    objective = float(np.dot(weights, (counts * fitted_hazard - hazards) ** 2))
    return _summarize(
        float(beta),
        predictions,
        effects,
        weights,
        objective,
        True,
        "weighted_hazard_projection",
    )

def fit_dynamics_aware_pairwise(
    source_effects,
    exposure_counts,
    weights=None,
    beta_bounds=(0.0, 0.95),
    tail_weight=0.25,
    tail_quantile=0.95,
):

    effects = _vector(source_effects, "source_effects")
    counts = _vector(exposure_counts, "exposure_counts")
    if effects.shape != counts.shape:
        raise ValueError("source_effects and exposure_counts must have equal shape")
    weights = _weights(weights, len(effects))
    normalized = weights / weights.sum()

    def objective(beta):
        residual = np.abs(pairwise_effect(beta, counts) - effects)
        mean_square = float(np.dot(normalized, residual ** 2))
        return mean_square + float(tail_weight) * cvar(residual, tail_quantile) ** 2

    result = minimize_scalar(
        objective,
        bounds=(float(beta_bounds[0]), float(beta_bounds[1])),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    predictions = pairwise_effect(float(result.x), counts)
    return _summarize(
        float(result.x),
        predictions,
        effects,
        weights,
        result.fun,
        result.success,
        "dynamics_aware_probability_tail",
    )

def fit_nonnegative_mapping(
    design_matrix,
    targets,
    weights=None,
    l1_penalty=0.0,
    upper_bound=0.95,
    initial=None,
):

    matrix = np.asarray(design_matrix, dtype=float)
    targets = _vector(targets, "targets")
    if matrix.ndim != 2 or matrix.shape[0] != len(targets):
        raise ValueError("design_matrix must have one row per target")
    weights = _weights(weights, len(targets))
    if initial is None:
        initial = np.zeros(matrix.shape[1], dtype=float)
    initial = np.asarray(initial, dtype=float)

    def objective(parameters):
        residuals = np.dot(matrix, parameters) - targets
        return float(np.dot(weights, residuals ** 2) + l1_penalty * parameters.sum())

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(0.0, float(upper_bound)) for _ in range(matrix.shape[1])],
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    predictions = np.dot(matrix, result.x)
    return _summarize(
        tuple(float(value) for value in result.x),
        predictions,
        targets,
        weights,
        result.fun,
        result.success,
        "weighted_nonnegative_mapping",
    )

def fit_temporal_scalar(
    targets_by_time,
    designs_by_time,
    weights_by_time=None,
    smoothness=0.0,
    bounds=(0.0, 3.0),
):

    targets = [_vector(values, "targets") for values in targets_by_time]
    designs = [_vector(values, "designs") for values in designs_by_time]
    if len(targets) != len(designs) or not targets:
        raise ValueError("temporal targets and designs must have equal nonzero length")
    if weights_by_time is None:
        weights_by_time = [None] * len(targets)
    weights = [_weights(value, len(targets[idx])) for idx, value in enumerate(weights_by_time)]
    diagonal = np.zeros(len(targets), dtype=float)
    rhs = np.zeros(len(targets), dtype=float)
    for time_index in range(len(targets)):
        if targets[time_index].shape != designs[time_index].shape:
            raise ValueError("each temporal target and design must have equal shape")
        diagonal[time_index] = np.dot(
            weights[time_index], designs[time_index] ** 2
        )
        rhs[time_index] = np.dot(
            weights[time_index], designs[time_index] * targets[time_index]
        )
    system = np.diag(diagonal)
    smoothness = float(smoothness)
    for time_index in range(1, len(targets)):
        system[time_index - 1, time_index - 1] += smoothness
        system[time_index, time_index] += smoothness
        system[time_index - 1, time_index] -= smoothness
        system[time_index, time_index - 1] -= smoothness
    parameters = np.linalg.solve(system + 1e-15 * np.eye(len(targets)), rhs)
    parameters = np.clip(parameters, float(bounds[0]), float(bounds[1]))
    return parameters

def fit_piecewise_scalar(
    targets_by_time,
    designs_by_time,
    weights_by_time=None,
    switch_penalty=0.0,
    bounds=(0.0, 3.0),
):

    targets = [_vector(values, "targets") for values in targets_by_time]
    designs = [_vector(values, "designs") for values in designs_by_time]
    if len(targets) != len(designs) or not targets:
        raise ValueError("temporal targets and designs must have equal nonzero length")
    if weights_by_time is None:
        weights_by_time = [None] * len(targets)
    weights = [_weights(value, len(targets[idx])) for idx, value in enumerate(weights_by_time)]
    length = len(targets)
    segment_cost = np.zeros((length, length), dtype=float)
    segment_value = np.zeros((length, length), dtype=float)
    for start in range(length):
        cumulative_denominator = 0.0
        cumulative_rhs = 0.0
        cumulative_target_square = 0.0
        for end in range(start, length):
            cumulative_denominator += float(
                np.dot(weights[end], designs[end] ** 2)
            )
            cumulative_rhs += float(
                np.dot(weights[end], designs[end] * targets[end])
            )
            cumulative_target_square += float(
                np.dot(weights[end], targets[end] ** 2)
            )
            value = cumulative_rhs / max(cumulative_denominator, 1e-15)
            value = min(max(value, float(bounds[0])), float(bounds[1]))
            segment_value[start, end] = value
            segment_cost[start, end] = (
                cumulative_target_square
                - 2.0 * value * cumulative_rhs
                + value * value * cumulative_denominator
            )
    best = np.full(length + 1, np.inf, dtype=float)
    previous = np.full(length + 1, -1, dtype=int)
    best[0] = -float(switch_penalty)
    for end in range(1, length + 1):
        for start in range(end):
            candidate = (
                best[start]
                + segment_cost[start, end - 1]
                + float(switch_penalty)
            )
            if candidate < best[end]:
                best[end] = candidate
                previous[end] = start
    parameters = np.zeros(length, dtype=float)
    change_points = []
    end = length
    while end > 0:
        start = int(previous[end])
        parameters[start:end] = segment_value[start, end - 1]
        if start > 0:
            change_points.append(start)
        end = start
    return parameters, tuple(sorted(change_points)), float(best[length])
