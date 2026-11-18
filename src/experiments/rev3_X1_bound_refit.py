
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.dynamics import (  
    HypergraphSIParameters,
    SimplicialSIParameters,
    infection_probabilities,
)
from src.dynamics.sis_temporal import SISParameters  
from src.theory.exact_kernel import (  
    all_binary_states,
    build_exact_transition_kernel,
)
from src.approximate import build_error_certificate  
from src.projection import (  
    project_temporal_hypergraph_to_multigraph,
    project_temporal_hypergraph_to_simplicial,
)
from src.experiments.exp_innovation23_supplementary import (  
    _random_hypergraph,
)
from src.experiments.innovation2_core import (  
    INDEPENDENT,
    build_conflicting_lambda_rates,
    _fit_pairwise_to_probability_cache,
    _fit_simplicial_to_probability_cache,
    _source_probability_cache,
)
from src.experiments._common import initial_distribution  

OUT_DIR = ROOT / "results" / "rev3" / "X1"
SEED = 20260826
REPLICATES = 4
NEW_BOUND = 0.95

def _fit_simplicial_wide(simplicial, states, source_cache, time_points,
                         initial_beta_by_time, bound):

    rates = {}
    for time_point in time_points:
        def objective(values):
            params = SimplicialSIParameters(
                {(time_point, 1): float(values[0]),
                 (time_point, 2): float(values[1])},
                INDEPENDENT,
            )
            residuals = []
            for state_index, state in enumerate(states):
                predicted = infection_probabilities(
                    state, simplicial, time_point, params
                )
                susceptible = state == 0
                residuals.extend(
                    (predicted[susceptible]
                     - source_cache[time_point][
                         state_index, susceptible]).tolist()
                )
            return float(np.mean(np.asarray(residuals) ** 2))

        result = minimize(
            objective,
            [initial_beta_by_time[time_point], 0.005],
            method="L-BFGS-B",
            bounds=[(0.0, bound), (0.0, bound)],
            options={"maxiter": 250, "ftol": 1e-14},
        )
        rates[(time_point, 1)] = float(result.x[0])
        rates[(time_point, 2)] = float(result.x[1])
    return rates

def _regenerate(task_id):

    conflicts = (0.0, 0.01, 0.03, 0.05, 0.10, 0.16)
    recoveries = (0.02, 0.05, 0.10, 0.20)
    node_counts, horizons, proxies = (5, 6, 7), (3, 5, 8), ("G", "MG", "SC2")
    tid = 0
    for node_count in node_counts:
        for horizon in horizons:
            for proxy_name in proxies:
                for replicate in range(REPLICATES):
                    if tid != task_id:
                        tid += 1
                        continue
                    task_seed = int(SEED) + 10007 * tid
                    task_rng = np.random.RandomState(task_seed)
                    conflict = float(conflicts[tid % len(conflicts)])
                    base_beta = float(task_rng.uniform(0.02, 0.09))
                    recovery = float(
                        recoveries[(tid + node_count) % len(recoveries)]
                    )
                    hypergraph = _random_hypergraph(
                        node_count, horizon, task_seed,
                        events_per_snapshot=max(
                            3, int(round(task_rng.uniform(0.55, 1.05)
                                       * node_count))),
                        persistence=float(task_rng.uniform(0.05, 0.75)),
                        overlap=float(task_rng.uniform(0.10, 0.85)),
                        duplicate_rate=float(task_rng.uniform(0.0, 0.35)),
                    )
                    rates, _ = build_conflicting_lambda_rates(
                        hypergraph, 2.0, base_beta, conflict,
                        time_drift=float(task_rng.uniform(0.0, 0.02)),
                    )
                    infected_node = int(task_rng.randint(0, node_count))
                    return (hypergraph, rates, proxy_name, recovery,
                            node_count, infected_node, task_seed, conflict,
                            base_beta)
    raise ValueError(task_id)

def _certificate_metrics(hypergraph, rates, proxy_si, recovery,
                         node_count, infected_node):
    source_si = HypergraphSIParameters(rates, 2.0)
    source_parameters = SISParameters(source_si, recovery)
    proxy_parameters = SISParameters(proxy_si, recovery)
    source_kernels = [
        build_exact_transition_kernel(hypergraph, t, source_parameters,
                                      update_rule="sis")
        for t in hypergraph.time_points
    ]
    proxy_net = project_temporal_hypergraph_to_simplicial(hypergraph, 2)
    proxy_kernels = [
        build_exact_transition_kernel(proxy_net, t, proxy_parameters,
                                      update_rule="sis")
        for t in hypergraph.time_points
    ]
    initial = initial_distribution(node_count, (infected_node,))
    certificate = build_error_certificate(initial, source_kernels,
                                          proxy_kernels)
    return {
        "actual_tv": float(certificate.final_actual_error),
        "distribution_bound": float(certificate.final_expected_bound),
        "worst_row_bound": float(certificate.final_basic_bound),
        "contraction_bound": float(certificate.final_contraction_bound),
    }

def main():
    t0 = time.perf_counter()
    check = json.loads((OUT_DIR / "bound_check.json").read_text())
    affected = [r["task_id"] for r in check["task_records"]
                if r["n_at_bound"] > 0]
    stored = pd.read_csv(
        ROOT / "results" / "tables" / "innovation2_a11_certificate_tasks.csv")

    rows = []
    for task_id in affected:
        (hypergraph, rates, proxy_name, recovery, node_count,
         infected_node, task_seed, conflict, base_beta) = _regenerate(
            task_id)
        assert proxy_name == "SC2"
        states = all_binary_states(node_count)
        source_si = HypergraphSIParameters(rates, 2.0)
        source_cache = _source_probability_cache(hypergraph, source_si,
                                                 states)
        time_points = hypergraph.time_points
        multigraph = project_temporal_hypergraph_to_multigraph(hypergraph)
        beta_initial = _fit_pairwise_to_probability_cache(
            multigraph, states, source_cache, time_points, "time")
        simplicial = project_temporal_hypergraph_to_simplicial(hypergraph, 2)

        beta_old = _fit_simplicial_to_probability_cache(
            simplicial, states, source_cache, time_points, beta_initial)
        beta_new = _fit_simplicial_wide(
            simplicial, states, source_cache, time_points, beta_initial,
            NEW_BOUND)

        metrics_old = _certificate_metrics(
            hypergraph, rates, SimplicialSIParameters(beta_old, INDEPENDENT),
            recovery, node_count, infected_node)
        metrics_new = _certificate_metrics(
            hypergraph, rates, SimplicialSIParameters(beta_new, INDEPENDENT),
            recovery, node_count, infected_node)

        stored_row = stored[stored.task_id == task_id].iloc[0]
        rows.append({
            "task_id": int(task_id),
            "seed": task_seed, "conflict": conflict,
            "base_beta": base_beta, "recovery": recovery,
            "params_old": {str(k): v for k, v in beta_old.items()},
            "params_new": {str(k): v for k, v in beta_new.items()},
            "metrics_old_recomputed": metrics_old,
            "metrics_old_stored": {
                k: float(stored_row[k]) for k in metrics_old},
            "metrics_new_095": metrics_new,
        })

    merged = stored.copy()
    for row in rows:
        for k, v in row["metrics_new_095"].items():
            merged.loc[merged.task_id == row["task_id"], k] = v
    for name in ("distribution", "worst_row", "contraction"):
        merged[name + "_width"] = merged[name + "_bound"] - merged["actual_tv"]
        merged[name + "_tightness"] = (
            merged[name + "_bound"] / (merged["actual_tv"] + 1e-12))
        merged[name + "_covered"] = (
            merged["actual_tv"] <= merged[name + "_bound"] + 1e-12)

    def summary(df):
        out = {}
        for name in ("distribution", "worst_row", "contraction"):
            nonzero = df.loc[df.actual_tv > 1e-10,
                             name + "_tightness"]
            out[name] = {
                "coverage": float(df[name + "_covered"].mean()),
                "median_tightness_nonzero": float(nonzero.median()),
                "median_width": float(df[name + "_width"].median()),
                "n_saturated_bound_ge_1": int(
                    (df[name + "_bound"] >= 1 - 1e-12).sum()),
            }
        return out

    result = {
        "bound_old": 0.35, "bound_new": NEW_BOUND,
        "affected_tasks": affected,
        "task_rows": rows,
        "summary_stored": summary(stored),
        "summary_refit": summary(merged),
        "seconds": time.perf_counter() - t0,
    }
    (OUT_DIR / "bound_refit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "task_rows"}, indent=2))
    for row in rows:
        print("task", row["task_id"],
              "old", {k: round(v, 6)
                      for k, v in row["metrics_old_recomputed"].items()},
              "stored", {k: round(v, 6)
                         for k, v in row["metrics_old_stored"].items()},
              "new", {k: round(v, 6)
                      for k, v in row["metrics_new_095"].items()})

if __name__ == "__main__":
    main()
