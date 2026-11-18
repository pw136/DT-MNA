
from dataclasses import dataclass

import numpy as np

@dataclass
class AdaptiveSelectionResult:
    selected_indices: tuple
    selected_names: tuple
    selected_errors: tuple
    switch_count: int
    objective: float
    feasible: bool

def pareto_front(records, error_key="error", cost_key="cost"):

    ordered = sorted(records, key=lambda item: (item[cost_key], item[error_key]))
    result = []
    best_error = float("inf")
    for record in ordered:
        error = float(record[error_key])
        if error < best_error - 1e-15:
            result.append(record)
            best_error = error
    return tuple(result)

def select_minimal_surrogate(
    candidates,
    error_budget,
    error_key="error",
    complexity_key="complexity",
):
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate[error_key]) <= float(error_budget)
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda item: (float(item[complexity_key]), float(item[error_key])),
    )

def adaptive_model_sequence(
    errors,
    complexities,
    model_names=None,
    error_budget=None,
    complexity_weight=0.0,
    switch_cost=0.0,
):

    errors = np.asarray(errors, dtype=float)
    complexities = np.asarray(complexities, dtype=float).reshape(-1)
    if errors.ndim != 2 or errors.shape[1] != len(complexities):
        raise ValueError("errors must have one column per complexity")
    if model_names is None:
        model_names = tuple("M%d" % idx for idx in range(errors.shape[1]))
    if len(model_names) != errors.shape[1]:
        raise ValueError("model_names has the wrong length")
    allowed = np.isfinite(errors)
    if error_budget is not None:
        budget = np.asarray(error_budget, dtype=float)
        if budget.ndim == 0:
            budget = np.full(errors.shape[0], float(budget))
        if len(budget) != errors.shape[0]:
            raise ValueError("error_budget has the wrong length")
        allowed &= errors <= budget[:, None]
    if np.any(~allowed.any(axis=1)):
        return AdaptiveSelectionResult(tuple(), tuple(), tuple(), 0, float("inf"), False)
    cost = errors + float(complexity_weight) * complexities[None, :]
    cost[~allowed] = np.inf
    dynamic = np.full_like(cost, np.inf)
    previous = np.full(errors.shape, -1, dtype=int)
    dynamic[0] = cost[0]
    for time_index in range(1, errors.shape[0]):
        for model_index in range(errors.shape[1]):
            transition = dynamic[time_index - 1] + float(switch_cost)
            transition[model_index] = dynamic[time_index - 1, model_index]
            best_previous = int(np.argmin(transition))
            dynamic[time_index, model_index] = (
                transition[best_previous] + cost[time_index, model_index]
            )
            previous[time_index, model_index] = best_previous
    selected = [int(np.argmin(dynamic[-1]))]
    for time_index in range(errors.shape[0] - 1, 0, -1):
        selected.append(int(previous[time_index, selected[-1]]))
    selected.reverse()
    switch_count = sum(
        int(selected[index] != selected[index - 1])
        for index in range(1, len(selected))
    )
    return AdaptiveSelectionResult(
        tuple(selected),
        tuple(model_names[index] for index in selected),
        tuple(float(errors[time_index, index]) for time_index, index in enumerate(selected)),
        int(switch_count),
        float(np.min(dynamic[-1])),
        True,
    )
