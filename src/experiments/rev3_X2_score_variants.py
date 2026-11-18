
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dynamics.si_temporal import HypergraphSIParameters  
from src.temporal_structures.temporal_hypergraph import (  
    TemporalHypergraph,
)
from src.approximate.selection import adaptive_model_sequence  
from src.experiments.exp_results_revision import (  
    DYNAMIC_MODELS,
    ProbabilityEngine,
    simulate_batch,
)

OUT_DIR = ROOT / "results" / "rev3" / "X2"
SYN_DIR = ROOT / "results" / "results_revision_supplement" / "dynamic"
REAL_DIR = ROOT / "results" / "paper_core_extension" / "dynamic"

COST_S0 = np.array([1.0, 2.0, 4.5, 10.0])
COST_S1 = np.array([1.0, 4.0, 7.0, 10.0])
COST_S3 = np.array([1.17, 1.17, 2.15, 10.0])
COMPLEXITY_WEIGHT = 0.012
SWITCH_COST = 0.006
ERROR_BUDGET = 0.05
TIMING_REPEATS = 5

SCORES = {"S0": COST_S0, "S1": COST_S1, "S3": COST_S3}

def context_folders():
    folders = []
    for path in sorted(SYN_DIR.glob("synthetic_*")):
        folders.append(("synthetic", path))
    for path in sorted(REAL_DIR.iterdir()):
        if path.is_dir():
            folders.append(("real", path))
    return folders

def load_engine(folder):
    net = json.loads((folder / "network.json").read_text())
    network = TemporalHypergraph(
        net["nodes"],
        {int(t): [tuple(g) for g in groups]
         for t, groups in net["snapshots"].items()},
    )
    rates = {(int(t), int(m), int(c)): float(v)
             for t, m, c, v in net["lambda_rates"]}
    params = HypergraphSIParameters(rates, float(net["nu"]))
    return ProbabilityEngine(network, params)

def load_models(folder):
    fits = json.loads((folder / "fits.json").read_text())
    models = {model: [] for model in DYNAMIC_MODELS if model != "H"}
    for record in fits:
        models[record["model"]].append(record["parameters"][0])
    return {m: np.array(v) for m, v in models.items()}

def causal_sequence(errors, cost, budget):

    chosen = []
    for row in errors:
        allowed = row <= budget
        c = row + COMPLEXITY_WEIGHT * cost
        if chosen:
            c = c + SWITCH_COST * (np.arange(len(cost)) != chosen[-1])
        c[~allowed] = np.inf
        chosen.append(int(np.argmin(c)))
    return np.array(chosen, dtype=int)

def order_only_sequence(errors, budget):

    seq = []
    for row in errors:
        allowed = np.flatnonzero(row <= budget)
        seq.append(int(allowed[0]) if len(allowed) else len(DYNAMIC_MODELS) - 1)
    return np.array(seq, dtype=int)

def sequences_for(errors_val, errors_audit, variant):
    checked = np.maximum(errors_val, errors_audit)
    out = {}
    if variant == "S2":
        seq = order_only_sequence(checked, ERROR_BUDGET)
        out["Causal-checked"] = seq
        out["Causal-validation"] = order_only_sequence(errors_val, ERROR_BUDGET)
        out["Offline-DP"] = seq.copy()
    else:
        cost = SCORES[variant]
        offline = adaptive_model_sequence(
            checked, cost, DYNAMIC_MODELS, error_budget=ERROR_BUDGET,
            complexity_weight=COMPLEXITY_WEIGHT, switch_cost=SWITCH_COST)
        if not offline.feasible:
            raise AssertionError("infeasible")
        out["Causal-checked"] = causal_sequence(checked, cost, ERROR_BUDGET)
        out["Causal-validation"] = causal_sequence(
            errors_val, cost, ERROR_BUDGET)
        out["Offline-DP"] = np.array(offline.selected_indices)
    eligible = np.max(checked, axis=0) <= ERROR_BUDGET
    fixed = int(np.flatnonzero(eligible)[0])
    out["Fixed-validated"] = np.full(len(checked), fixed)
    for index, name in enumerate(DYNAMIC_MODELS):
        out[f"Fixed-{name}"] = np.full(len(checked), index)
    return out

def run_context(args):
    domain, folder, timing = args
    context = folder.name
    started = time.perf_counter()
    engine = load_engine(folder)
    models = load_models(folder)
    sel = json.loads((folder / "selection.json").read_text())
    errors = {k: np.array(v) for k, v in sel["errors"].items()}
    stored_seqs = {k: np.array(v) for k, v in sel["sequences"].items()}
    traj = np.load(folder / "trajectories.npz")
    initial, uniforms, reference = traj["initial"], traj["uniforms"], traj["source_states"]
    source_density = reference.mean(axis=2)

    s0 = sequences_for(errors["validation"], errors["audit"], "S0")
    repro = {m: bool(np.array_equal(s0[m], stored_seqs[m])) for m in s0}

    h_params = {}
    for pos, t in enumerate(engine.points):
        h_params[pos] = sum(1 for k in engine.parameters.lambda_rates
                            if k[0] == t)
    n_params = {"G-time": 1, "MG-time": 1, "C2-time": 2}
    rows, sim_rows = [], {}
    for variant in ("S0", "S1", "S2", "S3"):
        seqs = (s0 if variant == "S0" else
                sequences_for(errors["validation"], errors["audit"], variant))
        ref_seq = s0["Offline-DP"]
        for method, seq in seqs.items():
            indexes = (np.arange(len(seq)), seq)
            selected_test = errors["test"][indexes]
            fitted_scalars = int(sum(
                h_params[pos] if DYNAMIC_MODELS[idx] == "H"
                else n_params[DYNAMIC_MODELS[idx]]
                for pos, idx in enumerate(seq)))
            row = {"context": context, "domain": domain, "variant": variant,
                   "method": method,
                   "mean_score": float(np.mean(COST_S0[seq])),
                   "mean_score_own": float(np.mean(
                       SCORES[variant][seq])) if variant in SCORES else float(
                           np.mean(COST_S0[seq])),
                   "score_values": " ".join(str(int(x)) for x in seq),
                   "switches": int(np.sum(seq[1:] != seq[:-1])),
                   "pct_snapshots_equal_S0_offline": float(
                       np.mean(seq == ref_seq)),
                   "test_over_budget_rate": float(np.mean(
                       selected_test > ERROR_BUDGET + 1e-12)),
                   "test_worst": float(selected_test.max()),
                   "fitted_scalar_params": fitted_scalars}
            for pos, idx in enumerate(seq):
                stage = min(5, 6 * pos // len(seq))
                row[f"stage{stage}_{DYNAMIC_MODELS[idx]}"] = 1
            rows.append(row)

            if method in ("Causal-checked", "Causal-validation", "Offline-DP"):
                states = simulate_batch(engine, models, seq, initial, uniforms)
                density = states.mean(axis=2)
                sim_rows[(variant, method)] = {
                    "paired_density_mae": float(
                        np.abs(density - source_density).mean()),
                    "paired_path_match_rate": float(
                        np.mean(np.all(states == reference, axis=(1, 2)))),
                    "final_density_mae": float(
                        np.abs(density[:, -1] - source_density[:, -1]).mean()),
                }
            elif variant == "S0":
                stored = pd.read_csv(folder / "results.csv")
                srow = stored[stored["method"] == method].iloc[0]
                sim_rows[(variant, method)] = {
                    "paired_density_mae": float(srow["paired_density_mae"]),
                    "paired_path_match_rate": float(
                        srow["paired_path_match_rate"]),
                    "final_density_mae": float(srow["final_density_mae"]),
                }
            else:
                sim_rows[(variant, method)] = sim_rows[("S0", method)]
            for k, v in sim_rows.get((variant, method), {}).items():
                row[k] = v

    for row in rows:
        seq = np.array([int(x) for x in row["score_values"].split()])
        total = len(seq)
        for k, name in enumerate(DYNAMIC_MODELS):
            row[f"frac_{name}"] = float(np.mean(seq == k))

    timing_rows = []
    if timing:
        for method in ["Causal-checked", "Causal-validation", "Offline-DP",
                       "Fixed-validated", "Fixed-G-time", "Fixed-MG-time",
                       "Fixed-C2-time", "Fixed-H"]:
            seq = s0[method]
            times = []
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                simulate_batch(engine, models, seq, initial, uniforms)
                times.append(time.perf_counter() - t0)
            timing_rows.append({"context": context, "domain": domain,
                                "method": method,
                                "median_seconds": float(np.median(times)),
                                "min_seconds": float(min(times))})
    return {"rows": rows, "timing": timing_rows, "repro": repro,
            "context": context, "seconds": time.perf_counter() - started}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=0,
                        help="worker processes (0 = serial)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tasks = [(d, f, True) for d, f in context_folders()]
    if args.jobs:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            results = pool.map(run_context, tasks)
    else:
        results = [run_context(task) for task in tasks]

    rows = [r for res in results for r in res["rows"]]
    timing = [r for res in results for r in res["timing"]]
    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUT_DIR / "score_variant_rows.csv", index=False,
              encoding="utf-8-sig")
    td = pd.DataFrame(timing)
    td.to_csv(OUT_DIR / "timing_rows.csv", index=False, encoding="utf-8-sig")

    frac_cols = [f"frac_{m}" for m in DYNAMIC_MODELS]
    stage_cols = [c for c in df.columns if c.startswith("stage")]
    summary = (df.groupby(["variant", "method"])
               .agg(mean_score=("mean_score", "mean"),
                    paired_density_mae=("paired_density_mae", "mean"),
                    paired_path_match_rate=("paired_path_match_rate", "mean"),
                    final_density_mae=("final_density_mae", "mean"),
                    switches=("switches", "mean"),
                    pct_equal_S0=("pct_snapshots_equal_S0_offline", "mean"),
                    test_over_budget=("test_over_budget_rate", "mean"),
                    fitted_scalars=("fitted_scalar_params", "mean"),
                    **{c: (c, "mean") for c in frac_cols})
               .reset_index())
    summary.to_csv(OUT_DIR / "score_variant_summary.csv", index=False,
                   encoding="utf-8-sig")

    timing_summary = (td.groupby("method")
                      .agg(median_seconds=("median_seconds", "median"))
                      .reset_index())
    ref = float(timing_summary.loc[
        timing_summary.method == "Fixed-H", "median_seconds"].iloc[0])
    timing_summary["relative_to_H"] = timing_summary["median_seconds"] / ref
    timing_summary.to_csv(OUT_DIR / "timing_summary.csv", index=False,
                          encoding="utf-8-sig")

    repro = {res["context"]: res["repro"] for res in results}
    bad = {c: v for c, v in repro.items() if not all(v.values())}
    (OUT_DIR / "run_info.json").write_text(json.dumps({
        "contexts": [res["context"] for res in results],
        "score_vectors": {"S0": COST_S0.tolist(), "S1": COST_S1.tolist(),
                          "S2": "order-only", "S3": COST_S3.tolist()},
        "complexity_weight": COMPLEXITY_WEIGHT,
        "switch_cost": SWITCH_COST, "error_budget": ERROR_BUDGET,
        "timing_repeats": TIMING_REPEATS,
        "s0_reproduction": repro,
        "s0_mismatches": bad,
        "seconds_per_context": {res["context"]: res["seconds"]
                                for res in results},
        "total_seconds": time.perf_counter() - t0,
    }, indent=2, default=float))
    print("contexts:", len(results), "S0 mismatches:", bad)
    print(summary[summary.variant == "S0"].to_string(index=False))
    print("done in %.1fs" % (time.perf_counter() - t0), flush=True)

if __name__ == "__main__":
    main()
