
from __future__ import annotations

import json
import math
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
from src.experiments.exp_results_revision import (  
    DYNAMIC_MODELS,
    ProbabilityEngine,
    fit_dynamic,
    select_sequences,
    simulate_batch,
)

OUT_DIR = ROOT / "results" / "rev3" / "X3b"
SYN_DIR = ROOT / "results" / "results_revision_supplement" / "dynamic"
REAL_DIR = ROOT / "results" / "paper_core_extension" / "dynamic"

COST = np.array([1.0, 2.0, 4.5, 10.0])
ERROR_BUDGET = 0.05
COMPLEXITY_WEIGHT = 0.012
SWITCH_COST = 0.006
CAL_TOL = 0.01
CAL_SIMS = 400
CAL_ITERS = 24
CAL_LAM_MAX = 1e18
CAL_EXPANSIONS = 40
SEED = 20261117
STAGES = ("M1", "M4", "M3", "M2", "M1", "M3")

OPT_SYN = {"optimizer_starts": [0.003, 0.03, 0.15],
           "optimizer_maxiter": 250, "optimizer_maxfun": 500}
OPT_REAL = {"optimizer_starts": [0.003, 0.03, 0.15],
            "optimizer_maxiter": 1000, "optimizer_maxfun": 2000}

def base_config(opt):
    return {"alpha": 0.95, "tail_weight": 0.2,
            "tail_weight_grid": [0.0, 0.2], "validation_tail_weight": 0.2,
            "fit_probability_max": 0.95,
            "dynamic": {"error_budget": ERROR_BUDGET,
                        "tail_weight": 0.2,
                        "complexity_weight": COMPLEXITY_WEIGHT,
                        "switch_cost": SWITCH_COST},
            **opt}

def hazard(mech, lam, size, chi):

    if mech == "M1":
        lam_h = lam * chi
    elif mech == "M4":
        lam_h = lam * chi + 4.0 * lam * chi * (chi - 1)
    elif mech == "M3":
        lam_h = lam if 2 * chi >= size - 1 else 0.0
    elif mech == "M2":
        lam_h = lam * chi * chi
    else:
        raise ValueError(mech)
    return -math.expm1(-lam_h)

def stage_parameters(network, stage_lams, stage_for_time):

    rates = {}
    for t in network.time_points:
        mech = STAGES[stage_for_time(t)]
        lam = stage_lams[mech]
        for group in set(network.active_hyperedges(t)):
            m = len(group)
            for chi in range(1, m):
                rates[(t, m, chi)] = hazard(mech, lam, m, chi)
    return HypergraphSIParameters(rates, 0.0)

def mean_final_attack(network, mech, lam, initials, uniforms):

    rates = {}
    for t in network.time_points:
        for group in set(network.active_hyperedges(t)):
            m = len(group)
            for chi in range(1, m):
                rates[(t, m, chi)] = hazard(mech, lam, m, chi)
    engine = ProbabilityEngine(network, HypergraphSIParameters(rates, 0.0))
    state = initials.copy()
    for pos, t in enumerate(engine.points):
        probs = engine.source(state, t)
        state = np.maximum(state, (uniforms[:, pos] < probs).astype(np.int8))
    return float(state.mean())

def calibrate(network, mech, initials, uniforms):
    a0 = float(initials.mean())
    a_max = mean_final_attack(network, mech, CAL_LAM_MAX,
                              initials, uniforms)
    target = a0 + 0.5 * (a_max - a0)
    record = {"mechanism": mech, "a0": a0, "a_max": a_max,
              "target": target}
    if a_max - a0 <= 1e-12:

        achieved = mean_final_attack(network, mech, 1e-8,
                                     initials, uniforms)
        record.update({"lambda": 1e-8, "attack": achieved,
                       "success": abs(achieved - target) <= CAL_TOL,
                       "note": "a_max == a0; no reachable spread"})
        return record
    lo, hi = 1e-8, 1.0
    f_lo = mean_final_attack(network, mech, lo, initials, uniforms)
    f_hi = mean_final_attack(network, mech, hi, initials, uniforms)
    expansions = 0
    while f_hi < target and expansions < CAL_EXPANSIONS:
        hi *= 8.0
        f_hi = mean_final_attack(network, mech, hi, initials, uniforms)
        expansions += 1
    if f_hi < target:
        record.update({"lambda": None, "attack": f_hi,
                       "success": False,
                       "note": ("target %g unreachable: f(%g)=%g after %d "
                                "expansions" % (target, hi, f_hi,
                                                expansions))})
        return record
    if f_lo > target:
        record.update({"lambda": lo, "attack": f_lo,
                       "success": abs(f_lo - target) <= CAL_TOL,
                       "note": "f at smallest lambda already above target"})
        return record
    log_lo, log_hi = math.log(lo), math.log(hi)
    for _ in range(CAL_ITERS):
        mid = 0.5 * (log_lo + log_hi)
        f = mean_final_attack(network, mech, math.exp(mid),
                              initials, uniforms)
        if f < target:
            log_lo = mid
        else:
            log_hi = mid
    lam = math.exp(0.5 * (log_lo + log_hi))
    attack = mean_final_attack(network, mech, lam, initials, uniforms)
    record.update({"lambda": lam, "attack": attack,
                   "success": abs(attack - target) <= CAL_TOL,
                   "log_bracket": [log_lo, log_hi]})
    return record

def load_panels(folder):
    z = np.load(folder / "network.npz")
    panels = {}
    for key in z.files:
        split, t = key.rsplit("_", 1)
        panels.setdefault(split, {})[int(t)] = z[key]
    return panels

def run_context(args):
    domain, folder = args
    context = folder.name
    started = time.perf_counter()
    net = json.loads((folder / "network.json").read_text())
    network = TemporalHypergraph(
        net["nodes"],
        {int(t): [tuple(g) for g in groups]
         for t, groups in net["snapshots"].items()})
    points = network.time_points
    t_index = {t: i for i, t in enumerate(points)}
    stage_for_time = lambda t: min(5, 6 * t_index[t] // len(points))  
    traj = np.load(folder / "trajectories.npz")
    initial, uniforms = traj["initial"], traj["uniforms"]
    cal_initials = np.tile(initial, (CAL_SIMS // len(initial), 1))
    rng = np.random.RandomState(SEED + sum(map(ord, context)))
    cal_uniforms = rng.random_sample((len(cal_initials), len(points),
                                      network.nodes.__len__()))

    calibrations = {}
    for mech in dict.fromkeys(STAGES):
        calibrations[mech] = calibrate(network, mech, cal_initials,
                                       cal_uniforms)
    failed = [c for c in calibrations.values() if not c["success"]]
    if failed:
        out = {"context": context, "domain": domain,
               "calibrations": list(calibrations.values()),
               "calibration_failed": True,
               "seconds": time.perf_counter() - started}
        (OUT_DIR / f"context_{context}.json").write_text(
            json.dumps(out, indent=2, default=float))
        return [], out["calibrations"], context, out["seconds"]

    stage_lams = {m: c["lambda"] for m, c in calibrations.items()}

    params = stage_parameters(network, stage_lams, stage_for_time)
    engine = ProbabilityEngine(network, params)
    panels = load_panels(folder)
    config = base_config(OPT_SYN if domain == "synthetic" else OPT_REAL)

    fit_started = time.perf_counter()
    models, fits = fit_dynamic(engine, panels, config)
    fit_seconds = time.perf_counter() - fit_started

    errors = {s: np.zeros((len(points), len(DYNAMIC_MODELS)))
              for s in ("validation", "audit", "test")}
    for split in errors:
        for pos, t in enumerate(points):
            states = panels[split][t]
            target = engine.source(states, t)
            for idx, model in enumerate(DYNAMIC_MODELS):
                resid = np.abs(engine.predict(states, t, model,
                                              models.get(model)) - target)
                errors[split][pos, idx] = resid[states == 0].max()

    sequences = select_sequences(errors["validation"], errors["audit"],
                                 config)
    reference = simulate_batch(engine, models, sequences["Fixed-H"],
                               initial, uniforms)
    source_density = reference.mean(axis=2)

    rows, seq_out = [], {}
    for method, seq in sequences.items():
        seq_out[method] = [int(x) for x in seq]
        idx2 = (np.arange(len(seq)), seq)
        selected_test = errors["test"][idx2]
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            states = simulate_batch(engine, models, seq, initial, uniforms)
            times.append(time.perf_counter() - t0)
        density = states.mean(axis=2)
        row = {"context": context, "domain": domain, "method": method,
               "nodes": int(network.nodes.__len__()),
               "snapshots": len(points),
               "mean_complexity": float(COST[seq].mean()),
               "switches": int(np.sum(seq[1:] != seq[:-1])),
               "test_worst": float(selected_test.max()),
               "test_stage_exceed_rate": float(np.mean(
                   selected_test > ERROR_BUDGET + 1e-12)),
               "paired_density_mae": float(
                   np.abs(density - source_density).mean()),
               "mean_curve_mae": float(
                   np.abs(density.mean(axis=0) - source_density.mean(axis=0)
                          ).mean()),
               "final_density_mae": float(
                   np.abs(density[:, -1] - source_density[:, -1]).mean()),
               "state_mismatch_rate": float(np.mean(states != reference)),
               "paired_path_match_rate": float(
                   np.mean(np.all(states == reference, axis=(1, 2)))),
               "simulation_seconds": float(np.median(times))}
        for k, name in enumerate(DYNAMIC_MODELS):
            row[f"frac_{name}"] = float(np.mean(seq == k))
        for stage in range(6):
            mask = np.array([min(5, 6 * p // len(seq)) == stage
                             for p in range(len(seq))])
            for k, name in enumerate(DYNAMIC_MODELS):
                row[f"stage{stage + 1}_frac_{name}"] = float(
                    np.mean(seq[mask] == k))
        rows.append(row)

    out = {"context": context, "domain": domain,
           "calibrations": list(calibrations.values()),
           "stage_lams": stage_lams,
           "stage_map": {str(t): STAGES[stage_for_time(t)] for t in points},
           "sequences": seq_out,
           "errors": {k: v.tolist() for k, v in errors.items()},
           "fit_seconds": fit_seconds,
           "seconds": time.perf_counter() - started}
    (OUT_DIR / f"context_{context}.json").write_text(
        json.dumps(out, indent=2, default=float))
    return rows, out["calibrations"], context, out["seconds"]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tasks = ([("synthetic", p) for p in sorted(SYN_DIR.glob("synthetic_*"))] +
             [("real", p) for p in sorted(REAL_DIR.iterdir()) if p.is_dir()])
    if args.jobs:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            results = pool.map(run_context, tasks)
    else:
        results = [run_context(task) for task in tasks]

    rows = [r for res in results for r in res[0]]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dynamic_results.csv", index=False,
              encoding="utf-8-sig")
    if not df.empty:
        frac = [f"frac_{m}" for m in DYNAMIC_MODELS]
        stage = [c for c in df.columns if c.startswith("stage")]
        summary = (df.groupby("method")
                   .agg(mean_complexity=("mean_complexity", "mean"),
                        paired_density_mae=("paired_density_mae", "mean"),
                        paired_path_match_rate=(
                            "paired_path_match_rate", "mean"),
                        mean_curve_mae=("mean_curve_mae", "mean"),
                        final_density_mae=("final_density_mae", "mean"),
                        switches=("switches", "mean"),
                        test_stage_exceed=("test_stage_exceed_rate", "mean"),
                        simulation_seconds=("simulation_seconds", "median"),
                        **{c: (c, "mean") for c in frac + stage})
                   .reset_index())
        summary.to_csv(OUT_DIR / "summary.csv", index=False,
                       encoding="utf-8-sig")
    else:
        summary = None

    cals = [{"context": res[2], **c} for res in results for c in res[1]]
    pd.DataFrame(cals).to_csv(OUT_DIR / "calibrations.csv", index=False,
                              encoding="utf-8-sig")
    failed = [c for c in cals if not c["success"]]
    (OUT_DIR / "run_info.json").write_text(json.dumps({
        "seed": SEED, "cal_sims": CAL_SIMS,
        "cal_rule": "target = a0 + 0.5*(a_max - a0)",
        "cal_tol": CAL_TOL, "cal_lam_max": CAL_LAM_MAX,
        "cal_iters": CAL_ITERS, "cal_expansions": CAL_EXPANSIONS,
        "stages": list(STAGES),
        "scores": COST.tolist(), "eta_pi": COMPLEXITY_WEIGHT,
        "eta_sw": SWITCH_COST, "error_budget": ERROR_BUDGET,
        "failed_calibrations": failed,
        "seconds_per_context": {res[2]: res[3] for res in results},
        "total_seconds": time.perf_counter() - t0,
    }, indent=2, default=float))
    print("failed calibrations:", json.dumps(failed, default=float))
    if summary is not None:
        print(summary.to_string(index=False))
    print("done in %.1fs" % (time.perf_counter() - t0), flush=True)
    if failed:
        raise RuntimeError(
            "%d calibrations did not converge: %s"
            % (len(failed),
               [(c["context"], c["mechanism"]) for c in failed]))

if __name__ == "__main__":
    main()
