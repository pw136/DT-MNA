
from dataclasses import dataclass

import numpy as np

from .si_temporal import SimulationResult, infection_probabilities, resolve_rate

@dataclass
class SISParameters:
    transmission: object
    recovery: object

    def recovery_rate(self, time_point):
        value = resolve_rate(self.recovery, time_point)
        if value < 0.0 or value > 1.0:
            raise ValueError("recovery probability must be in [0, 1]")
        return value

def simulate_temporal_sis(
    temporal_network,
    parameters,
    initial_state,
    time_points=None,
    seed=None,
    infection_uniforms=None,
    recovery_uniforms=None,
):
    points = tuple(temporal_network.time_points if time_points is None else time_points)
    state = np.asarray(initial_state, dtype=np.int8).copy()
    if state.shape != (len(temporal_network.nodes),):
        raise ValueError("initial_state has the wrong shape")
    rng = np.random.RandomState(seed)
    states = [state.copy()]
    probabilities = []
    for step, time_point in enumerate(points):
        p_infection = infection_probabilities(
            state, temporal_network, time_point, parameters.transmission
        )
        probabilities.append(p_infection.copy())
        infection_draws = (
            np.asarray(infection_uniforms)[step]
            if infection_uniforms is not None
            else rng.random_sample(len(state))
        )
        recovery_draws = (
            np.asarray(recovery_uniforms)[step]
            if recovery_uniforms is not None
            else rng.random_sample(len(state))
        )
        recovery = parameters.recovery_rate(time_point)
        previous = state.copy()
        state[(previous == 0) & (infection_draws < p_infection)] = 1
        state[(previous == 1) & (recovery_draws < recovery)] = 0
        states.append(state.copy())
    return SimulationResult(
        states=np.asarray(states, dtype=np.int8),
        time_points=points,
        conditional_probabilities=tuple(probabilities),
    )

