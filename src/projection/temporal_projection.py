
from .snapshot_projection import hypergraph_to_simplicial_snapshot
from ..temporal_structures import (
    TemporalMultiGraph,
    TemporalSimpleGraph,
    TemporalSimplicialComplex,
)

def project_temporal_hypergraph_to_simplicial(temporal_hypergraph, kappa):
    source_groups = {}
    source_events = {}
    snapshots = {}
    for time_point in temporal_hypergraph.time_points:
        events = temporal_hypergraph.active_hyperedges(time_point)
        blocks = tuple(
            hypergraph_to_simplicial_snapshot((hyperedge,), kappa)
            for hyperedge in events
        )
        source_groups[time_point] = blocks
        source_events[time_point] = events
        snapshots[time_point] = tuple(cell for block in blocks for cell in block)
    return TemporalSimplicialComplex(
        temporal_hypergraph.nodes,
        snapshots,
        max_order=kappa,
        source_groups=source_groups,
        source_events=source_events,
    )

def project_temporal_hypergraph_to_multigraph(temporal_hypergraph):
    source_blocks = {
        time_point: tuple(
            (hyperedge,)
            for hyperedge in temporal_hypergraph.active_hyperedges(time_point)
        )
        for time_point in temporal_hypergraph.time_points
    }
    return TemporalMultiGraph(
        temporal_hypergraph.nodes,
        {
            time_point: temporal_hypergraph.active_hyperedges(time_point)
            for time_point in temporal_hypergraph.time_points
        },
        source_blocks=source_blocks,
    )

def project_temporal_simplicial_to_multigraph(temporal_simplicial):
    return TemporalMultiGraph(
        temporal_simplicial.nodes,
        {
            time_point: temporal_simplicial.active_maximal_simplices(time_point)
            for time_point in temporal_simplicial.time_points
        },
        source_blocks={
            time_point: temporal_simplicial.active_source_groups(time_point)
            for time_point in temporal_simplicial.time_points
        },
    )

def project_temporal_hypergraph_to_simplegraph(temporal_hypergraph):
    multi = project_temporal_hypergraph_to_multigraph(temporal_hypergraph)
    snapshots = {
        time_point: multi.active_edges(time_point, repeated=False)
        for time_point in multi.time_points
    }
    return TemporalSimpleGraph(temporal_hypergraph.nodes, snapshots)
