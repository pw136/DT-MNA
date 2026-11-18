
import numpy as np

from .kernel_distance import distribution_tv

def infection_density(states):
    states = np.asarray(states, dtype=float)
    return states.mean(axis=-1)

def binary_population_entropy(value):

    array = np.asarray(value, dtype=float)
    if array.ndim > 0 and array.size > 1 and np.all((array == 0) | (array == 1)):
        probability = float(array.mean())
    else:
        probability = float(array)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * np.log2(probability)
        - (1.0 - probability) * np.log2(1.0 - probability)
    )

def entropy_trajectory(states):
    return np.asarray(
        [binary_population_entropy(state) for state in np.asarray(states)],
        dtype=float,
    )

def infection_times(states):
    states = np.asarray(states, dtype=np.int8)
    result = np.full(states.shape[1], np.inf)
    for node_idx in range(states.shape[1]):
        infected = np.where(states[:, node_idx] == 1)[0]
        if len(infected):
            result[node_idx] = infected[0]
    return result

def normalized_rmse(theory, simulation, epsilon=1e-15):
    theory = np.asarray(theory, dtype=float)
    simulation = np.asarray(simulation, dtype=float)
    if theory.shape != simulation.shape:
        raise ValueError("theory and simulation must have equal shape")
    scale = float(simulation.max() - simulation.min())
    rmse = float(np.sqrt(np.mean((theory - simulation) ** 2)))
    return rmse / max(scale, epsilon)

def path_distribution_tv(paths_left, paths_right):
    keys = set(paths_left) | set(paths_right)
    return 0.5 * sum(
        abs(paths_left.get(key, 0.0) - paths_right.get(key, 0.0)) for key in keys
    )

def endpoint_tv(trajectory_left, trajectory_right):
    return distribution_tv(trajectory_left[-1], trajectory_right[-1])

