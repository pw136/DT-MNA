
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.dynamics import HypergraphSIParameters  
from src.theory import all_binary_states  
from src.experiments.exp_innovation23_supplementary import (  
    _proxy_fit,
    _random_hypergraph,
)
from src.experiments.innovation2_core import (  
    INDEPENDENT,
    build_conflicting_lambda_rates,
    _fit_pairwise_to_probability_cache,
    _fit_simplicial_to_probability_cache,
    _source_probability_cache,
)
from src.projection import (  
    project_temporal_hypergraph_to_multigraph,
    project_temporal_hypergraph_to_simplegraph,
    project_temporal_hypergraph_to_simplicial,
)

OUT_DIR = ROOT / "results" / "rev3" / "X1"
BOUND = 0.35
SEED = 20260826
REPLICATES = 4

def _parameter_values(proxy_name, parameters):
    for attr in ("beta", "rates", "beta_rates"):
        value = getattr(parameters, attr, None)
        if isinstance(value, dict):
            return list(value.values())
    raise AttributeError("cannot extract fitted values from %r" % parameters)

def _proxy_fit_with_inits(hypergraph, source_parameters, proxy_name, states):

    source_cache = _source_probability_cache(
        hypergraph, source_parameters, states
    )
    time_points = hypergraph.time_points
    intermediate = []
    if proxy_name == "G":
        network = project_temporal_hypergraph_to_simplegraph(hypergraph)
        beta = _fit_pairwise_to_probability_cache(
            network, states, source_cache, time_points, "time"
        )
        from src.dynamics import PairwiseSIParameters

        parameters = PairwiseSIParameters(beta, INDEPENDENT)
    elif proxy_name == "MG":
        network = project_temporal_hypergraph_to_multigraph(hypergraph)
        beta = _fit_pairwise_to_probability_cache(
            network, states, source_cache, time_points, "time"
        )
        from src.dynamics import PairwiseSIParameters

        parameters = PairwiseSIParameters(beta, INDEPENDENT)
    elif proxy_name == "SC2":
        multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
        beta_initial = _fit_pairwise_to_probability_cache(
            multigraph, states, source_cache, time_points, "time"
        )
        intermediate = list(beta_initial.values())
        network = project_temporal_hypergraph_to_simplicial(hypergraph, 2)
        beta = _fit_simplicial_to_probability_cache(
            network, states, source_cache, time_points, beta_initial
        )
        from src.dynamics import SimplicialSIParameters

        parameters = SimplicialSIParameters(beta, INDEPENDENT)
    else:
        raise ValueError("unsupported proxy %s" % proxy_name)
    return network, parameters, intermediate

def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(int(SEED))  
    conflicts = (0.0, 0.01, 0.03, 0.05, 0.10, 0.16)
    recoveries = (0.02, 0.05, 0.10, 0.20)
    task_id = 0
    records = []
    for node_count in (5, 6, 7):
        for horizon in (3, 5, 8):
            for proxy_name in ("G", "MG", "SC2"):
                for replicate in range(int(REPLICATES)):
                    task_seed = int(SEED) + 10007 * task_id
                    task_rng = np.random.RandomState(task_seed)
                    conflict = float(conflicts[task_id % len(conflicts)])
                    base_beta = float(task_rng.uniform(0.02, 0.09))
                    float(recoveries[(task_id + node_count) % len(recoveries)])
                    hypergraph = _random_hypergraph(
                        node_count,
                        horizon,
                        task_seed,
                        events_per_snapshot=max(
                            3,
                            int(round(task_rng.uniform(0.55, 1.05)
                                       * node_count)),
                        ),
                        persistence=float(task_rng.uniform(0.05, 0.75)),
                        overlap=float(task_rng.uniform(0.10, 0.85)),
                        duplicate_rate=float(task_rng.uniform(0.0, 0.35)),
                    )
                    rates, _ = build_conflicting_lambda_rates(
                        hypergraph,
                        2.0,
                        base_beta,
                        conflict,
                        time_drift=float(task_rng.uniform(0.0, 0.02)),
                    )
                    source_si = HypergraphSIParameters(rates, 2.0)
                    states = all_binary_states(node_count)
                    _, parameters, intermediate = _proxy_fit_with_inits(
                        hypergraph, source_si, proxy_name, states
                    )
                    values = _parameter_values(proxy_name, parameters)
                    records.append(
                        {
                            "task_id": task_id,
                            "proxy": proxy_name,
                            "node_count": node_count,
                            "horizon": horizon,
                            "n_parameters": len(values),
                            "max_value": float(max(values)),
                            "n_at_bound": int(
                                sum(v >= BOUND - 1e-9 for v in values)
                            ),
                            "intermediate_max": float(
                                max(intermediate) if intermediate else -1.0
                            ),
                            "intermediate_at_bound": int(
                                sum(v >= BOUND - 1e-9 for v in intermediate)
                            ),
                        }
                    )
                    task_id += 1

    all_max = max(r["max_value"] for r in records)
    all_at_bound = sum(r["n_at_bound"] for r in records)
    n_params = sum(r["n_parameters"] for r in records)
    inter_max = max(r["intermediate_max"] for r in records)
    inter_at_bound = sum(r["intermediate_at_bound"] for r in records)
    by_proxy = {}
    for proxy in ("G", "MG", "SC2"):
        sub = [r for r in records if r["proxy"] == proxy]
        by_proxy[proxy] = {
            "max_fitted": max(r["max_value"] for r in sub),
            "n_parameters": sum(r["n_parameters"] for r in sub),
            "n_at_bound": sum(r["n_at_bound"] for r in sub),
        }

    result = {
        "experiment": (
            "run_certificate_calibration "
            "(src/experiments/exp_innovation23_supplementary.py:188), "
            "108 enumerable-SIS tasks of Section 3.2.2"
        ),
        "bound": BOUND,
        "seed": SEED,
        "replicates_per_cell": REPLICATES,
        "n_tasks": task_id,
        "n_fitted_parameters": n_params,
        "max_fitted_value": all_max,
        "n_parameters_at_bound": all_at_bound,
        "intermediate_pairwise_init_for_SC2": {
            "max_fitted": inter_max,
            "n_at_bound": inter_at_bound,
        },
        "by_proxy": by_proxy,
        "bound_atol": 1e-9,
        "code_evidence": {
            "pairwise_bound": (
                "src/experiments/innovation2_core.py:449,465 "
                "bounds=(0.0, 0.35), method='bounded' (minimize_scalar, "
                "MSE in probability space)"
            ),
            "simplicial_bound": (
                "src/experiments/innovation2_core.py:508 "
                "bounds=[(0.0, 0.35), (0.0, 0.35)] (L-BFGS-B)"
            ),
            "task_grid": (
                "src/experiments/exp_innovation23_supplementary.py:195-198 "
                "nodes(5,6,7) x horizons(3,5,8) x proxies(G,MG,SC2) "
                "x 4 replicates"
            ),
            "fit_call": (
                "src/experiments/exp_innovation23_supplementary.py:137-166 "
                "(_proxy_fit)"
            ),
        },
        "task_records": records,
        "seconds": time.perf_counter() - t0,
    }
    (OUT_DIR / "bound_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "task_records"}, indent=2))

if __name__ == "__main__":
    main()
