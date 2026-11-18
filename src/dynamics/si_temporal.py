
from math import factorial
from dataclasses import dataclass

import numpy as np

from .update_semantics import ExposureSemantics, coerce_exposure_semantics
from ..temporal_structures import (
    TemporalHypergraph,
    TemporalMultiGraph,
    TemporalSimpleGraph,
    TemporalSimplicialComplex,
)

def resolve_rate(source, time_point, *keys):

    if callable(source):
        return float(source(time_point, *keys))
    if hasattr(source, "get"):
        candidates = [(time_point,) + keys, keys]
        if len(keys) == 1:
            candidates.extend([(time_point, keys[0]), keys[0]])
        candidates.extend([time_point, "default"])
        for candidate in candidates:
            try:
                if candidate in source:
                    return float(source[candidate])
            except TypeError:
                pass
        raise KeyError("no rate for time=%r keys=%r" % (time_point, keys))
    return float(source)

def _check_probability(value, label):
    tolerance = 1e-12
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError("%s must be in [0, 1], got %.16g" % (label, value))
    return min(1.0, max(0.0, float(value)))

def _choose(n, k):
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

@dataclass
class HypergraphSIParameters:
    lambda_rates: object
    nu: float = 1.0

    def lambda_rate(self, time_point, hyperedge_size, infected_count):
        return resolve_rate(
            self.lambda_rates, time_point, hyperedge_size, infected_count
        )

@dataclass
class SimplicialSIParameters:
    beta_rates: object
    exposure_semantics: ExposureSemantics = ExposureSemantics.GROUPED_ADDITIVE

    def beta_rate(self, time_point, order):
        return resolve_rate(self.beta_rates, time_point, order)

@dataclass
class PairwiseSIParameters:
    beta: object
    exposure_semantics: ExposureSemantics = ExposureSemantics.INDEPENDENT_EXPOSURES

    def beta_rate(self, time_point):
        return resolve_rate(self.beta, time_point)

@dataclass
class SimulationResult:
    states: np.ndarray
    time_points: tuple
    conditional_probabilities: tuple

    @property
    def infection_density(self):
        return self.states.mean(axis=1)

    @property
    def infection_times(self):
        result = np.full(self.states.shape[1], np.inf)
        for node_idx in range(self.states.shape[1]):
            infected = np.where(self.states[:, node_idx] == 1)[0]
            if len(infected):
                result[node_idx] = infected[0]
        return result

    @property
    def infection_order(self):
        return tuple(np.argsort(self.infection_times, kind="mergesort"))

def _state_by_node(state, network):
    state = np.asarray(state, dtype=np.int8)
    if state.shape != (len(network.nodes),):
        raise ValueError("state must have shape (%d,)" % len(network.nodes))
    if not np.all((state == 0) | (state == 1)):
        raise ValueError("state must be binary")
    return state, {node: int(state[idx]) for idx, node in enumerate(network.nodes)}

def _simplicial_cell_effect(infected_count, time_point, parameters):
    semantics = coerce_exposure_semantics(parameters.exposure_semantics)
    if infected_count <= 0:
        return 0.0
    if semantics == ExposureSemantics.GROUPED_ADDITIVE:
        effect = sum(
            _choose(infected_count, order) * parameters.beta_rate(time_point, order)
            for order in range(1, infected_count + 1)
        )
        return _check_probability(effect, "grouped simplex infection effect")

    survival = 1.0
    for order in range(1, infected_count + 1):
        beta = _check_probability(
            parameters.beta_rate(time_point, order), "simplex beta"
        )
        survival *= (1.0 - beta) ** _choose(infected_count, order)
    return 1.0 - survival

def _hypergraph_probabilities(state_by_node, network, time_point, parameters):
    result = np.zeros(len(network.nodes), dtype=float)
    for node_idx, node in enumerate(network.nodes):
        if state_by_node[node] == 1:
            continue
        survival = 1.0
        for hyperedge in network.incident_hyperedges(node, time_point):
            infected_count = sum(
                state_by_node[other] for other in hyperedge if other != node
            )
            if infected_count <= 0:
                continue
            rate = parameters.lambda_rate(
                time_point, len(hyperedge), infected_count
            )
            effect = _check_probability(
                rate * (infected_count ** parameters.nu),
                "hyperedge infection effect",
            )
            survival *= 1.0 - effect
        result[node_idx] = 1.0 - survival
    return result

def _simplicial_probabilities(state_by_node, network, time_point, parameters):
    result = np.zeros(len(network.nodes), dtype=float)
    for node_idx, node in enumerate(network.nodes):
        if state_by_node[node] == 1:
            continue
        survival = 1.0
        semantics = coerce_exposure_semantics(parameters.exposure_semantics)
        if semantics == ExposureSemantics.GROUPED_ADDITIVE:
            for source_group in network.incident_source_groups(node, time_point):
                source_effect = 0.0
                for cell in source_group:
                    if node not in cell:
                        continue
                    infected_count = sum(
                        state_by_node[other] for other in cell if other != node
                    )
                    source_effect += _simplicial_cell_effect(
                        infected_count, time_point, parameters
                    )
                source_effect = _check_probability(
                    source_effect, "projected source-group infection effect"
                )
                survival *= 1.0 - source_effect
        else:
            for cell in network.incident_maximal_simplices(node, time_point):
                infected_count = sum(
                    state_by_node[other] for other in cell if other != node
                )
                effect = _simplicial_cell_effect(
                    infected_count, time_point, parameters
                )
                survival *= 1.0 - effect
        result[node_idx] = 1.0 - survival
    return result

def _multigraph_probabilities(state_by_node, network, time_point, parameters):
    beta = _check_probability(parameters.beta_rate(time_point), "pairwise beta")
    semantics = coerce_exposure_semantics(parameters.exposure_semantics)
    result = np.zeros(len(network.nodes), dtype=float)
    for node_idx, node in enumerate(network.nodes):
        if state_by_node[node] == 1:
            continue
        if semantics == ExposureSemantics.GROUPED_ADDITIVE:
            survival = 1.0
            for source_block in network.incident_source_blocks(node, time_point):
                infected_count = sum(
                    sum(
                        state_by_node[other]
                        for other in group
                        if other != node
                    )
                    for group in source_block
                    if node in group
                )
                effect = _check_probability(
                    beta * infected_count, "grouped pairwise infection effect"
                )
                survival *= 1.0 - effect
            result[node_idx] = 1.0 - survival
        else:
            exposure_count = sum(
                weight
                for other, weight in network.weighted_neighbors(node, time_point)
                if state_by_node[other] == 1
            )
            result[node_idx] = 1.0 - (1.0 - beta) ** exposure_count
    return result

def _simplegraph_probabilities(state_by_node, network, time_point, parameters):
    beta = _check_probability(parameters.beta_rate(time_point), "pairwise beta")
    result = np.zeros(len(network.nodes), dtype=float)
    for node_idx, node in enumerate(network.nodes):
        if state_by_node[node] == 1:
            continue
        infected_neighbors = sum(
            state_by_node[other] for other in network.neighbors(node, time_point)
        )
        result[node_idx] = 1.0 - (1.0 - beta) ** infected_neighbors
    return result

def infection_probabilities(state, temporal_network, time_point, parameters, update_rule="si"):

    if hasattr(parameters, "transmission"):
        parameters = parameters.transmission
    _, state_by_node = _state_by_node(state, temporal_network)
    if isinstance(temporal_network, TemporalHypergraph):
        if not isinstance(parameters, HypergraphSIParameters):
            raise TypeError("TemporalHypergraph requires HypergraphSIParameters")
        return _hypergraph_probabilities(
            state_by_node, temporal_network, time_point, parameters
        )
    if isinstance(temporal_network, TemporalSimplicialComplex):
        if not isinstance(parameters, SimplicialSIParameters):
            raise TypeError("simplicial dynamics requires SimplicialSIParameters")
        return _simplicial_probabilities(
            state_by_node, temporal_network, time_point, parameters
        )
    if isinstance(temporal_network, TemporalMultiGraph):
        if not isinstance(parameters, PairwiseSIParameters):
            raise TypeError("multigraph dynamics requires PairwiseSIParameters")
        return _multigraph_probabilities(
            state_by_node, temporal_network, time_point, parameters
        )
    if isinstance(temporal_network, TemporalSimpleGraph):
        if not isinstance(parameters, PairwiseSIParameters):
            raise TypeError("simple-graph dynamics requires PairwiseSIParameters")
        return _simplegraph_probabilities(
            state_by_node, temporal_network, time_point, parameters
        )
    raise TypeError("unsupported temporal network type: %s" % type(temporal_network).__name__)

def simulate_temporal_si(
    temporal_network,
    parameters,
    initial_state,
    time_points=None,
    seed=None,
    uniforms=None,
):

    points = tuple(temporal_network.time_points if time_points is None else time_points)
    state, _ = _state_by_node(initial_state, temporal_network)
    state = state.copy()
    rng = np.random.RandomState(seed)
    if uniforms is not None:
        uniforms = np.asarray(uniforms, dtype=float)
        expected = (len(points), len(temporal_network.nodes))
        if uniforms.shape != expected:
            raise ValueError("uniforms must have shape %r" % (expected,))

    states = [state.copy()]
    probabilities = []
    for step, time_point in enumerate(points):
        current_probabilities = infection_probabilities(
            state, temporal_network, time_point, parameters
        )
        probabilities.append(current_probabilities.copy())
        draws = uniforms[step] if uniforms is not None else rng.random_sample(len(state))
        susceptible = state == 0
        state = np.where(susceptible & (draws < current_probabilities), 1, state).astype(np.int8)
        states.append(state.copy())
    return SimulationResult(
        states=np.asarray(states, dtype=np.int8),
        time_points=points,
        conditional_probabilities=tuple(probabilities),
    )
