
from dataclasses import dataclass

import numpy as np

from ..metrics.kernel_distance import distribution_tv, kernel_row_tv
from ..theory.exact_kernel import propagate_distribution

@dataclass
class ErrorCertificate:
    per_time_max_kernel_tv: tuple
    per_time_expected_kernel_tv: tuple
    contractions: tuple
    basic_bounds: tuple
    expected_bounds: tuple
    contraction_bounds: tuple
    actual_endpoint_tv: tuple
    final_basic_bound: float
    final_expected_bound: float
    final_contraction_bound: float
    final_actual_error: float
    covered_by_basic: bool
    covered_by_expected: bool
    covered_by_contraction: bool

def node_probability_kernel_bound(probability_errors):

    errors = np.clip(np.asarray(probability_errors, dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - errors))

def dobrushin_coefficient(kernel):

    kernel = np.asarray(kernel, dtype=float)
    maximum = 0.0
    for row_index in range(len(kernel)):
        distances = 0.5 * np.abs(kernel[row_index + 1 :] - kernel[row_index]).sum(axis=1)
        if len(distances):
            maximum = max(maximum, float(distances.max()))
    return min(1.0, maximum)

def cumulative_tv_bound(step_errors, contractions=None):
    errors = tuple(float(value) for value in step_errors)
    if contractions is None:
        return min(1.0, float(sum(errors)))
    contractions = tuple(float(value) for value in contractions)
    if len(contractions) != len(errors):
        raise ValueError("contractions and step_errors must have equal length")
    total = 0.0
    for time_index, error in enumerate(errors):
        attenuation = 1.0
        for later in contractions[time_index + 1 :]:
            attenuation *= later
        total += error * attenuation
    return min(1.0, float(total))

def build_error_certificate(
    initial_distribution,
    source_kernels,
    surrogate_kernels,
    reachable_rows=None,
    tolerance=1e-12,
):
    source_kernels = tuple(source_kernels)
    surrogate_kernels = tuple(surrogate_kernels)
    if len(source_kernels) != len(surrogate_kernels) or not source_kernels:
        raise ValueError("kernel sequences must have equal nonzero length")
    source_trajectory = propagate_distribution(initial_distribution, source_kernels)
    surrogate_trajectory = propagate_distribution(initial_distribution, surrogate_kernels)
    maxima = []
    expected = []
    contractions = []
    actual = [0.0]
    basic = [0.0]
    expected_bounds = [0.0]
    contracted = [0.0]
    for time_index, (left, right) in enumerate(zip(source_kernels, surrogate_kernels)):
        row_distances = kernel_row_tv(left, right)
        rows = None if reachable_rows is None else tuple(reachable_rows[time_index])
        if rows is None:
            selected = row_distances
        elif rows:
            selected = row_distances[list(rows)]
        else:
            selected = np.asarray([0.0])
        maxima.append(float(selected.max()))
        expected.append(float(np.dot(source_trajectory[time_index], row_distances)))
        contractions.append(dobrushin_coefficient(right))
        actual.append(
            distribution_tv(
                source_trajectory[time_index + 1],
                surrogate_trajectory[time_index + 1],
            )
        )
        basic.append(cumulative_tv_bound(maxima))
        expected_bounds.append(cumulative_tv_bound(expected))
        contracted.append(cumulative_tv_bound(maxima, contractions))
    final_actual = float(actual[-1])
    return ErrorCertificate(
        tuple(maxima),
        tuple(expected),
        tuple(contractions),
        tuple(basic),
        tuple(expected_bounds),
        tuple(contracted),
        tuple(actual),
        float(basic[-1]),
        float(expected_bounds[-1]),
        float(contracted[-1]),
        final_actual,
        bool(final_actual <= basic[-1] + tolerance),
        bool(final_actual <= expected_bounds[-1] + tolerance),
        bool(final_actual <= contracted[-1] + tolerance),
    )
