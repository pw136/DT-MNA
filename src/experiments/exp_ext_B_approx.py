
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    MECHANISMS,
    SparseEngine,
    fit_global_hazard,
    hazard_table,
    load_config,
    load_dataset,
    simulate_batch,
    snapshot_index_edges,
    write_json,
)
from src.experiments.exp_ext_C_selection import (  
    collision_lower_bound,
    sample_records,
)

OUT_DIR = ROOT / "results" / "external_baselines" / "B"
A_DIR = ROOT / "results" / "external_baselines" / "A"

CANDIDATES = ("G", "MG", "C2")
R_GRID = (10, 30, 100, 300, 1000, 2000)
RUNS_REF = 5000
RUNS_SOURCE = 300
N_RECORDS = 4000
P0 = 0.05
CHUNK = 500
BASE_SEED = 20261018

MODE_OFF = 500000
SEED_REF, SEED_MC, SEED_FIT, SEED_TEST, SEED_OURS = 0, 5000, 10000, 15000, 20000
TIMING_REPS = 3

CHUNK = 250

def mc_stats(prob_fn, runs, n, n_steps, seed, gamma, mode):

    tail = max(1, n_steps // 4)
    curve_sum = np.zeros(n_steps + 1)
    final_frac = np.empty(runs)
    marg_sum = np.zeros(n)
    pos = 0
    while pos < runs:
        take = min(CHUNK, runs - pos)
        s = seed + pos * 7919
        initial = (
            np.random.RandomState(s).random_sample((take, n)) < P0
        ).astype(np.int8)
        uniforms = np.random.RandomState(s + 1).random_sample(
            (take, n_steps, n)).astype(np.float32)
        rec_u = (np.random.RandomState(s + 2).random_sample(
            (take, n_steps, n)).astype(np.float32) if gamma > 0.0 else None)
        traj = simulate_batch(prob_fn, initial, uniforms, n_steps,
                              recovery_gamma=gamma, recovery_uniforms=rec_u)
        curve_sum += traj.mean(axis=2).sum(axis=0)
        final_frac[pos : pos + take] = traj[:, -1].mean(axis=1)
        if mode == "si":
            marg_sum += traj[:, -1].sum(axis=0)
        else:
            marg_sum += traj[:, -tail:].mean(axis=1).sum(axis=0)
        pos += take
    return {
        "curve": curve_sum / runs,
        "final_frac": final_frac,
        "marginals": marg_sum / runs,
    }

def make_prob_source(engine, mechanism, lam):
    def prob_fn(states, step):
        return engine.source_probabilities(states, step, mechanism, lam)
    return prob_fn

def make_prob_surrogate(engine, candidate, h):
    def prob_fn(states, step):
        feats = engine.features(states, step, candidate)
        return SparseEngine.predict(feats, h)
    return prob_fn

def timed_mc(prob_fn, runs, n, n_steps, seed, mode, gamma):

    t0 = time.perf_counter()
    stats = mc_stats(prob_fn, runs, n, n_steps, seed, gamma, mode)
    times = [time.perf_counter() - t0]
    for rep in range(1, TIMING_REPS):
        t0 = time.perf_counter()
        mc_stats(prob_fn, runs, n, n_steps, seed + rep * 37, gamma, mode)
        times.append(time.perf_counter() - t0)
    stats["seconds"] = float(np.median(times))
    return stats

def _full_pb(edge, ps):

    m = len(edge)
    full = np.zeros(m + 1)
    full[0] = 1.0
    for p in ps:
        full[1:] = full[1:] * (1.0 - p) + full[:-1] * p
        full[0] *= 1.0 - p
    return full

def _decon_pb(full, p):

    if p >= 1.0 - 1e-12:
        return None
    m = len(full) - 1
    doth = np.zeros(m)
    doth[0] = full[0] / (1.0 - p)
    for k in range(1, m):
        doth[k] = (full[k] - p * doth[k - 1]) / (1.0 - p)
    doth = np.clip(doth, 0.0, 1.0)
    s = doth.sum()
    return doth / s if s > 0 else None

def ibmf_q(snap, P, mechanism, lam, tables):

    n = len(P)
    edges = snapshot_index_edges(snap)
    cvec = snap.multiplicities.astype(np.float64)
    hazard = np.zeros(n)
    for r, edge in enumerate(edges):
        m = len(edge)
        ps = P[list(edge)]
        full = _full_pb(edge, ps)
        table = tables[m]
        for i in range(m):
            doth = _decon_pb(full, ps[i])
            if doth is None:
                continue
            hazard[edge[i]] += cvec[r] * float(doth @ table[:m])
    return -np.expm1(-hazard)

def mmca_q(snap, P, mechanism, lam, tables):

    n = len(P)
    edges = snapshot_index_edges(snap)
    cvec = snap.multiplicities.astype(np.float64)
    log_noninf = np.zeros(n)  
    for r, edge in enumerate(edges):
        m = len(edge)
        c = cvec[r]
        pw = np.exp(-c * tables[m])  
        ps = P[list(edge)]

        full = np.zeros(m + 1)
        full[0] = 1.0
        for p in ps:
            full[1:] = full[1:] * (1.0 - p) + full[:-1] * p
            full[0] *= 1.0 - p
        for i, p in enumerate(ps):

            if p >= 1.0 - 1e-12:
                g = 1.0  
            else:
                doth = np.zeros(m)
                doth[0] = full[0] / (1.0 - p)
                for k in range(1, m):
                    doth[k] = (full[k] - p * doth[k - 1]) / (1.0 - p)
                doth = np.clip(doth, 0.0, 1.0)
                s = doth.sum()
                if s > 0:
                    doth /= s
                else:
                    doth[0] = 1.0
                g = float(doth @ pw[:m])
            log_noninf[edge[i]] += math.log(max(g, 1e-300))
    return -np.expm1(log_noninf)

def mean_field_run(snapshots, n, mechanism, lam, gamma, method):

    tables = hazard_table(mechanism, lam, sizes=(2, 3, 4, 5, 6))
    P = np.full(n, P0)
    out = [P.copy()]
    for t, snap in enumerate(snapshots):
        if method == "ibmf":
            q = ibmf_q(snap, P, mechanism, lam, tables)
        else:
            q = mmca_q(snap, P, mechanism, lam, tables)
        if gamma > 0.0:
            P = (1.0 - gamma) * P + (1.0 - P) * q
        else:
            P = P + (1.0 - P) * q
        assert np.all(P >= -1e-9) and np.all(P <= 1.0 + 1e-9), (
            f"{method} marginals left [0,1] at step {t}: "
            f"min={P.min():.4g} max={P.max():.4g}"
        )
        np.clip(P, 0.0, 1.0, out=P)
        out.append(P.copy())
    return np.stack(out)

def timed_mean_field(snapshots, n, mechanism, lam, gamma, method):
    t0 = time.perf_counter()
    P = mean_field_run(snapshots, n, mechanism, lam, gamma, method)
    times = [time.perf_counter() - t0]
    for _ in range(1, TIMING_REPS):
        t0 = time.perf_counter()
        mean_field_run(snapshots, n, mechanism, lam, gamma, method)
        times.append(time.perf_counter() - t0)
    return P, float(np.median(times))

def record_panel(engine, mechanism, lam, traj, records):

    feats = {m: [] for m in CANDIDATES}
    src = []
    steps = sorted(set(int(r[1]) for r in records))
    for step in steps:
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        states = traj[runs, step].astype(np.float64)
        p = engine.source_probabilities(states, step, mechanism, lam)
        fmap = {m: engine.features(states, step, m) for m in CANDIDATES}
        for i in range(len(runs)):
            mask = states[i] == 0
            src.append(p[i][mask])
            for m in CANDIDATES:
                feats[m].append(fmap[m][i][mask])
    return {
        "source": np.concatenate(src),
        "features": {m: np.concatenate(feats[m]) for m in CANDIDATES},
    }

def source_traj_block(engine, mechanism, lam, runs, seed, mode, gamma):

    n, T = engine.n, engine.n_snapshots
    initial = (
        np.random.RandomState(seed).random_sample((runs, n)) < P0
    ).astype(np.int8)
    unif = np.random.RandomState(seed + 1).random_sample(
        (runs, T, n)).astype(np.float32)
    rec = (np.random.RandomState(seed + 2).random_sample(
        (runs, T, n)).astype(np.float32) if gamma > 0.0 else None)
    return simulate_batch(make_prob_source(engine, mechanism, lam),
                          initial, unif, T,
                          recovery_gamma=gamma, recovery_uniforms=rec)

def method_metrics(ref, est_curve, est_marg, est_ff, mode, n_steps):
    tail = max(1, n_steps // 4)
    out = {
        "curve_mae": float(np.abs(est_curve - ref["curve"]).mean()),
        "final_error": (
            float(abs(est_curve[-1] - ref["curve"][-1])) if mode == "si"
            else float(abs(est_curve[-tail:].mean()
                           - ref["curve"][-tail:].mean()))),
        "marginal_rmse": float(np.sqrt(
            np.mean((est_marg - ref["marginals"]) ** 2))),
        "spearman": float(spearmanr(est_marg, ref["marginals"]).statistic),
        "w1_final": (
            float(wasserstein_distance(ref["final_frac"], est_ff))
            if est_ff is not None else None),
    }
    return out

def run_setting(dataset, mechanism, mode, lam, gamma, config, seed0):
    seed0 += 0 if mode == "si" else MODE_OFF
    engine = SparseEngine(dataset.snapshots, dataset.n_nodes)
    n, T = dataset.n_nodes, dataset.n_snapshots
    sis = mode == "sis"

    ref = timed_mc(make_prob_source(engine, mechanism, lam),
                   RUNS_REF, n, T, seed0 + SEED_REF, mode, gamma)

    traj_fit = source_traj_block(engine, mechanism, lam, RUNS_SOURCE,
                                 seed0 + SEED_FIT, mode, gamma)
    fit_records = sample_records(traj_fit, N_RECORDS, seed0 + SEED_FIT + 1)
    panel_fit = record_panel(engine, mechanism, lam, traj_fit, fit_records)
    del traj_fit
    fitted, fit_seconds = {}, {}
    for cand in CANDIDATES:
        t0 = time.perf_counter()
        h, info = fit_global_hazard(panel_fit["features"][cand],
                                    panel_fit["source"], config)
        fit_seconds[cand] = time.perf_counter() - t0
        fitted[cand] = {"h": h.tolist(), "objective": info["objective"],
                        "success": info["success"]}

    traj_test = source_traj_block(engine, mechanism, lam, RUNS_SOURCE,
                                  seed0 + SEED_TEST, mode, gamma)
    test_records = sample_records(traj_test, N_RECORDS, seed0 + SEED_TEST + 1)
    panel_test = record_panel(engine, mechanism, lam, traj_test, test_records)
    del traj_test
    lb_share = {}
    for cand in CANDIDATES:
        h = np.asarray(fitted[cand]["h"])
        pred = SparseEngine.predict(panel_test["features"][cand], h)
        max_err = float(np.abs(pred - panel_test["source"]).max())
        lb = collision_lower_bound(panel_test["features"][cand],
                                   panel_test["source"])
        lb_share[cand] = {"lb_infty": lb, "max_err": max_err,
                          "share": lb / max_err if max_err > 0 else 0.0}

    methods = {}
    grids = {"Source-MC": make_prob_source(engine, mechanism, lam)}
    for cand in CANDIDATES:
        grids["MC-%s" % cand] = make_prob_surrogate(
            engine, cand, np.asarray(fitted[cand]["h"]))
    for name, pf in grids.items():
        rows = {}
        block = SEED_MC if name == "Source-MC" else SEED_OURS
        for ri, R in enumerate(R_GRID):
            est = timed_mc(pf, R, n, T, seed0 + block + ri * 101, mode, gamma)
            m = method_metrics(ref, est["curve"], est["marginals"],
                               est["final_frac"], mode, T)
            rows[str(R)] = {**m, "seconds": est["seconds"]}
        methods[name] = {"grid": rows}

    for name in ("IBMF", "MMCA"):
        P, sec = timed_mean_field(dataset.snapshots, n, mechanism, lam,
                                  gamma, name.lower())
        est_curve = P.mean(axis=1)
        tail = max(1, T // 4)
        est_marg = P[-1] if mode == "si" else P[-tail:].mean(axis=0)
        methods[name] = {
            **method_metrics(ref, est_curve, est_marg, None, mode, T),
            "seconds": sec,
        }

    return {
        "dataset": dataset.name, "mechanism": mechanism, "mode": mode,
        "lambda": lam, "gamma": gamma, "n_nodes": n, "n_snapshots": T,
        "reference": {"runs": RUNS_REF, "seconds": ref["seconds"],
                      "final_mean": float(ref["curve"][-1])},
        "fitted": fitted, "fit_seconds": fit_seconds,
        "lb_share": lb_share, "methods": methods,
    }

def main():
    import pandas as pd

    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    infra = ROOT / config["paths"]["infra_results"]
    calibration = json.loads((infra / "calibration.json").read_text())

    names = config["datasets"]["contact_names"] + [
        config["datasets"]["email_name"]]
    all_rows, violations = [], []
    started = time.perf_counter()
    for di, name in enumerate(names):
        dataset = load_dataset(name)
        for mi, mechanism in enumerate(MECHANISMS):
            si_lam = float(calibration[name][mechanism]["lambda"])
            aj = json.loads((A_DIR / ("setting_%s_%s.json" % (name, mechanism))
                             ).read_text())
            sis_lam = float(aj["sis_calibration"]["lambda"])
            gamma = float(aj["gamma"])
            for mode, lam, gam in (("si", si_lam, 0.0),
                                   ("sis", sis_lam, gamma)):
                seed0 = BASE_SEED + di * 1000 + mi * 97
                t0 = time.perf_counter()
                result = run_setting(dataset, mechanism, mode, lam, gam,
                                     config, seed0)
                result["seconds"] = time.perf_counter() - t0
                write_json(OUT_DIR / ("setting_%s_%s_%s.json"
                                      % (name, mechanism, mode)), result)
                for cand in CANDIDATES:
                    ls = result["lb_share"][cand]
                    all_rows.append({
                        "dataset": name, "mechanism": mechanism,
                        "mode": mode, "method": "LB-%s" % cand, "R": np.nan,
                        **ls, "seconds": np.nan,
                    })
                    if ls["lb_infty"] > ls["max_err"] + 1e-12:
                        violations.append((name, mechanism, mode, cand, ls))
                for mname, mdat in result["methods"].items():
                    if "grid" in mdat:
                        for R, row in mdat["grid"].items():
                            all_rows.append({
                                "dataset": name, "mechanism": mechanism,
                                "mode": mode, "method": mname,
                                "R": int(R), "seconds": row["seconds"],
                                **{k: v for k, v in row.items()
                                   if k != "seconds"},
                            })
                    else:
                        all_rows.append({
                            "dataset": name, "mechanism": mechanism,
                            "mode": mode, "method": mname, "R": np.nan,
                            "seconds": mdat["seconds"],
                            **{k: v for k, v in mdat.items()
                               if k != "seconds"},
                        })
                print("%s %s %s: %.0fs" % (name, mechanism, mode,
                                         result["seconds"]), flush=True)
        del dataset

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    write_json(OUT_DIR / "run_info.json",
               {"total_seconds": time.perf_counter() - started,
                "lb_violations": violations})
    if violations:
        print("LB VIOLATIONS:", violations, flush=True)
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
