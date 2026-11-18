
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    MECHANISMS,
    SparseEngine,
    _snapshot_core,
    aggregated_snapshots,
    load_config,
    load_dataset,
    permuted_snapshots,
    simulate_batch,
    snapshot_index_edges,
    write_json,
)

OUT_DIR = ROOT / "results" / "external_baselines" / "A"

VARIANTS = ("V0", "V1", "V2", "V3", "V4a", "V4b", "V5", "V6", "V7", "V8")
M1_ONLY = ("V7", "V8")
N_PERMUTATIONS = 20
RUNS = 500
CAL_RUNS = 400
BASE_SEED = 20261018
SIS_TARGET = 0.20
SIS_TOL = 0.01

def hour_window_ids(dataset):

    if dataset.stats["is_email"]:
        return [s.day_index // 7 for s in dataset.snapshots]
    t0 = dataset.first_timestamp
    delta = dataset.delta_seconds
    ids = []
    for s in dataset.snapshots:
        ts = t0 + s.window_index * delta
        hour = int((ts % 86400.0) // 3600.0)
        ids.append(s.day_index * 24 + hour)

    remap = {w: i for i, w in enumerate(sorted(set(ids)))}
    return [remap[w] for w in ids]

def day_window_ids(dataset):

    if dataset.stats["is_email"]:
        return [s.day_index // 28 for s in dataset.snapshots]
    return [s.day_index for s in dataset.snapshots]

def static_window_ids(dataset):
    return [0] * dataset.n_snapshots

def build_variant_snapshots(dataset, variant, perm_seed):

    snaps = dataset.snapshots
    t = len(snaps)
    n = dataset.n_nodes
    if variant == "V0":
        return [snaps]
    if variant == "V1":
        return [permuted_snapshots(snaps, np.arange(t - 1, -1, -1))]
    if variant == "V2":
        rng = np.random.RandomState(perm_seed)
        return [permuted_snapshots(snaps, rng.permutation(t))
                for _ in range(N_PERMUTATIONS)]
    if variant == "V3":
        rng = np.random.RandomState(perm_seed)
        perms = []
        days = [s.day_index for s in snaps]
        blocks = defaultdict(list)
        for i, d in enumerate(days):
            blocks[d].append(i)
        block_list = [blocks[d] for d in sorted(blocks)]
        for _ in range(N_PERMUTATIONS):
            perm = np.arange(t)
            for block in block_list:
                idx = np.array(block)
                perm[block] = idx[rng.permutation(len(idx))]
            perms.append(permuted_snapshots(snaps, perm))
        return perms
    if variant == "V4a":
        return [aggregated_snapshots(snaps, hour_window_ids(dataset), n)]
    if variant == "V4b":
        return [aggregated_snapshots(snaps, day_window_ids(dataset), n)]
    if variant == "V5":
        return [aggregated_snapshots(snaps, static_window_ids(dataset), n)]
    if variant == "V6":

        counter = defaultdict(float)
        for snap in snaps:
            for edge, c in zip(snapshot_index_edges(snap),
                               snap.multiplicities):
                counter[edge] += float(c)
        mean_c = float(np.mean(list(counter.values())))
        per_step = {h: mean_c / t for h in counter}
        window_snap = _snapshot_core(per_step, n, window_index=0,
                                     day_index=snaps[0].day_index)
        return [[window_snap] * t]
    return None  

def sim_engine_list(snap_lists, n, mechanism, lam, initial, uniforms,
                    gamma=0.0, recovery_uniforms=None, sis_last_frac=None,
                    m1_variant=None):

    t = len(snap_lists[0])
    states_list = [None] * len(snap_lists)
    for gi, snaps in enumerate(snap_lists):
        engine = SparseEngine(snaps, n)
        runs = uniforms.shape[0]
        if len(snap_lists) == 1:
            idx = np.arange(runs)  
        else:
            idx = np.arange(runs)[np.arange(runs) % N_PERMUTATIONS == gi]
        if len(idx) == 0:
            continue

        if m1_variant == "V7":

            w = engine.snapshots[0].pair_weights
            def prob_fn(states, step, w=w):
                z = w.dot(states.T).T
                return (-np.expm1(-lam * z)) * (1.0 - states)
        elif m1_variant == "V8":

            def prob_fn(states, step, engine=engine):
                z = engine.snapshots[step].pair_weights.dot(states.T).T
                return (-np.expm1(-lam * z)) * (1.0 - states)
        else:
            def prob_fn(states, step, engine=engine):
                return engine.source_probabilities(
                    states, step, mechanism, lam
                )

        traj = simulate_batch(
            prob_fn, initial[idx],
            uniforms[idx],
            t,
            recovery_gamma=gamma,
            recovery_uniforms=(None if recovery_uniforms is None
                               else recovery_uniforms[idx]),
        )
        states_list[gi] = (idx, traj)
    n_runs = uniforms.shape[0]
    out = np.empty((n_runs, t + 1, engine.n), dtype=np.int8)
    for idx, traj in states_list:
        out[idx] = traj
    return out

def infection_times_norm(traj):

    t = traj.shape[1] - 1
    infected = traj == 1
    any_inf = infected.any(axis=1)
    first = np.argmax(infected, axis=1)
    times = np.where(any_inf, first, t).astype(np.float64)
    return times / t

def variant_metrics(traj_src, traj_v, mode, delta_seconds):
    ar_src = traj_src[:, -1].mean(axis=1)
    ar_v = traj_v[:, -1].mean(axis=1)
    curve_src = traj_src.mean(axis=(0, 2))
    curve_v = traj_v.mean(axis=(0, 2))
    out = {
        "w1": float(wasserstein_distance(ar_src, ar_v)),
        "mean_curve_mae": float(np.abs(curve_v - curve_src).mean()),
        "exact_agreement": float(
            np.all(traj_v == traj_src, axis=(1, 2)).mean()
        ),
    }
    if mode == "si":
        final = curve_src[-1]
        half = 0.5 * final

        def t50(curve):
            idx = np.flatnonzero(curve >= half)
            return int(idx[0]) if len(idx) else len(curve) - 1

        dt50_steps = t50(curve_v) - t50(curve_src)
        unit = "days" if delta_seconds >= 86400 else "hours"
        scale = delta_seconds / 86400.0 if unit == "days" \
            else delta_seconds / 3600.0
        out["dt50"] = float(dt50_steps * scale)
        out["dt50_unit"] = unit
        out["w1_infection_times"] = float(
            wasserstein_distance(
                infection_times_norm(traj_src).ravel(),
                infection_times_norm(traj_v).ravel(),
            )
        )
    return out

def sis_prevalence(engine, mechanism, lam, initial, uniforms, rec_uniforms,
                   gamma):
    def prob_fn(states, step):
        return engine.source_probabilities(states, step, mechanism, lam)

    traj = simulate_batch(prob_fn, initial, uniforms, engine.n_snapshots,
                          recovery_gamma=gamma,
                          recovery_uniforms=rec_uniforms)
    tail = traj[:, int(math.ceil(0.75 * traj.shape[1])):, :]
    return float(tail.mean())

def calibrate_lambda_sis(engine, mechanism, gamma, config, initial, uniforms,
                         rec_uniforms):
    target, tol = SIS_TARGET, SIS_TOL
    cal = config["calibration"]
    lo_log, hi_log = (float(v) for v in cal["log_lambda_bracket"])
    max_iter = int(cal["max_bisection_iterations"])

    def rate(log_lam):
        return sis_prevalence(engine, mechanism, math.exp(log_lam), initial,
                              uniforms, rec_uniforms, gamma)

    r_lo, r_hi = rate(lo_log), rate(hi_log)
    if r_lo >= target:
        return {"lambda": float(math.exp(lo_log)), "prevalence": r_lo,
                "status": "unreachable_low"}
    if r_hi <= target:
        return {"lambda": float(math.exp(hi_log)), "prevalence": r_hi,
                "status": "unreachable_high"}
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo_log + hi_log)
        r = rate(mid)
        if best is None or abs(r - target) < abs(best["prevalence"] - target):
            best = {"lambda": float(math.exp(mid)), "prevalence": r}
        if abs(r - target) <= tol:
            break
        if r < target:
            lo_log = mid
        else:
            hi_log = mid
    best["status"] = "ok" if abs(best["prevalence"] - target) <= tol \
        else "unreachable"
    return best

def run_setting(dataset, mechanism, lam_si, config, seed0):
    rng = np.random.RandomState(seed0)
    n, t = dataset.n_nodes, dataset.n_snapshots
    is_email = dataset.stats["is_email"]

    gamma = (1.0 / 7.0) if is_email else dataset.n_days / dataset.n_snapshots

    uniforms = rng.random_sample((RUNS, t, n)).astype(np.float32)
    initial_si = np.zeros((RUNS, n), dtype=np.int8)
    seeds = np.random.RandomState(seed0 + 1).randint(n, size=RUNS)
    initial_si[np.arange(RUNS), seeds] = 1
    rec_uniforms = np.random.RandomState(seed0 + 2).random_sample(
        (RUNS, t, n)
    ).astype(np.float32)

    initial_sis = (np.random.RandomState(seed0 + 3).random_sample(
        (RUNS, n)
    ) < 0.05).astype(np.int8)
    empty = initial_sis.sum(axis=1) == 0
    initial_sis[np.flatnonzero(empty), seeds[empty]] = 1

    result = {
        "dataset": dataset.name, "mechanism": mechanism,
        "lambda_si": lam_si, "gamma": gamma, "T": t, "N": n,
        "si": {}, "sis": {},
    }

    v0_engine = SparseEngine(dataset.snapshots, n)
    cal_uniforms = np.random.RandomState(BASE_SEED).random_sample(
        (CAL_RUNS, t, n)
    ).astype(np.float32)
    cal_rec = np.random.RandomState(BASE_SEED + 7).random_sample(
        (CAL_RUNS, t, n)
    ).astype(np.float32)
    cal_init = (np.random.RandomState(BASE_SEED + 8).random_sample(
        (CAL_RUNS, n)
    ) < 0.05).astype(np.int8)
    sis_cal = calibrate_lambda_sis(
        v0_engine, mechanism, gamma, config, cal_init, cal_uniforms, cal_rec
    )
    del cal_uniforms, cal_rec, cal_init  
    result["sis_calibration"] = sis_cal
    lam_sis = sis_cal["lambda"]

    for mode, lam, init, rec in (
        ("si", lam_si, initial_si, None),
        ("sis", lam_sis, initial_sis, rec_uniforms),
    ):
        gam = gamma if mode == "sis" else 0.0
        traj_map = {}
        for variant in VARIANTS:
            if variant in M1_ONLY and mechanism != "M1":
                continue
            if variant == "V3" and is_email:
                result[mode][variant] = None
                continue
            t0 = time.perf_counter()
            if variant in M1_ONLY:
                if variant == "V7":
                    agg = aggregated_snapshots(
                        dataset.snapshots, static_window_ids(dataset), n
                    )
                    traj = sim_engine_list(
                        [agg], n, mechanism, lam, init, uniforms,
                        gamma=gam, recovery_uniforms=rec,
                        m1_variant="V7",
                    )
                else:
                    traj = sim_engine_list(
                        [dataset.snapshots], n, mechanism, lam, init,
                        uniforms, gamma=gam, recovery_uniforms=rec,
                        m1_variant="V8",
                    )
            else:
                snap_lists = build_variant_snapshots(
                    dataset, variant, perm_seed=seed0 + 5
                )
                traj = sim_engine_list(
                    snap_lists, n, mechanism, lam, init, uniforms,
                    gamma=gam, recovery_uniforms=rec,
                )
            traj_map[variant] = traj
            if variant == "V0":
                result[mode]["V0_seconds"] = time.perf_counter() - t0
                continue
            met = variant_metrics(traj_map["V0"], traj, mode,
                                  dataset.delta_seconds)
            met["seconds"] = time.perf_counter() - t0
            result[mode][variant] = met
        if mechanism == "M1":
            result[mode]["V7_eq_V5_agreement"] = float(
                np.all(traj_map["V7"] == traj_map["V5"], axis=(1, 2)).mean()
            )
            result[mode]["V8_eq_V0_agreement"] = float(
                np.all(traj_map["V8"] == traj_map["V0"], axis=(1, 2)).mean()
            )

    return result

def main():
    import pandas as pd

    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    infra = ROOT / config["paths"]["infra_results"]
    calibration = json.loads((infra / "calibration.json").read_text())
    names = config["datasets"]["contact_names"] + [config["datasets"]["email_name"]]

    rows = []
    started = time.perf_counter()
    for di, name in enumerate(names):
        dataset = load_dataset(name)
        for mi, mechanism in enumerate(MECHANISMS):
            lam = float(calibration[name][mechanism]["lambda"])
            seed0 = BASE_SEED + di * 1000 + mi * 97
            t0 = time.perf_counter()
            res = run_setting(dataset, mechanism, lam, config, seed0)
            res["seconds"] = time.perf_counter() - t0
            write_json(
                OUT_DIR / ("setting_%s_%s.json" % (name, mechanism)), res
            )
            for mode in ("si", "sis"):
                for variant, met in res[mode].items():
                    if not isinstance(met, dict):
                        continue
                    rows.append({"dataset": name, "mechanism": mechanism,
                                 "mode": mode, "variant": variant, **met})
            sis = res["sis_calibration"]
            print("%s %s: %.0fs | sis lambda=%.4g prev=%.3f %s" % (
                name, mechanism, res["seconds"], sis["lambda"],
                sis["prevalence"], sis["status"]), flush=True)
        del dataset

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    write_json(OUT_DIR / "run_info.json",
               {"total_seconds": time.perf_counter() - started})
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
