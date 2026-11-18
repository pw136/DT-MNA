
from ..dynamics import HypergraphSIParameters
from ..metrics import max_kernel_tv
from ..temporal_structures import TemporalHypergraph
from ..theory import build_exact_transition_kernel
from ..theory.exact_kernel import kernel_product
from ._common import common_parser, write_result

def _window_aggregate(temporal, window_size):
    snapshots = {}
    for start in range(0, len(temporal.time_points), window_size):
        events = []
        for time_point in temporal.time_points[start : start + window_size]:
            events.extend(temporal.active_hyperedges(time_point))
        snapshots[start // window_size] = tuple(events)
    return TemporalHypergraph(temporal.nodes, snapshots)

def run_experiment():
    temporal = TemporalHypergraph(
        range(4),
        {
            0: [(0, 1)],
            1: [(1, 2, 3)],
            2: [(0, 3)],
            3: [(0, 1, 2)],
            4: [(2, 3)],
            5: [(0, 2, 3)],
        },
    )
    rates = {(2, 1): 0.25, (3, 1): 0.18, (3, 2): 0.09}
    params = HypergraphSIParameters(rates, 1.0)
    original_kernels = [build_exact_transition_kernel(temporal, t, params) for t in temporal.time_points]
    original_product = kernel_product(original_kernels)
    errors = {}
    for window_size in (1, 2, 3, 6):
        aggregated = _window_aggregate(temporal, window_size)
        kernels = [build_exact_transition_kernel(aggregated, t, params) for t in aggregated.time_points]
        errors[window_size] = max_kernel_tv(original_product, kernel_product(kernels))
    safe = [size for size, error in errors.items() if error < 1e-10]
    return {
        "experiment": "E5",
        "window_endpoint_kernel_tv": errors,
        "safe_window_sizes": safe,
        "minimum_safe_resolution": min(safe) if safe else None,
    }

def main():
    args = common_parser(__doc__).parse_args()
    path = write_result("e5_window_size", run_experiment(), args.output_dir)
    print(path)

if __name__ == "__main__":
    main()

