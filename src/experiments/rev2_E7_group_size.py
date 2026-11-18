
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    SparseEngine,
    _snapshot_core,
    attack_rate,
    calibrate_lambda,
    load_config,
    simulate_batch,
    write_json,
)

OUT_DIR = ROOT / "results" / "rev2" / "E7"

N_NODES = 1000
T_SNAPS = 100
GROUPS_PER_SNAP = 200
SIZES = (2, 3, 4, 6, 8, 10)
MECH = "M2"
RUNS = 100
TIMING_REPS = 3
SEED = 20261019
CAL_SEED = 20260725

def build_snapshots(m, seed=SEED):

    rng = np.random.RandomState(seed)
    snaps = []
    for t in range(T_SNAPS):
        counter = {}
        for members in rng.randint(0, N_NODES, size=(GROUPS_PER_SNAP, m)):
            edge = tuple(sorted(set(members.tolist())))
            if len(edge) < 2:  
                edge = tuple(sorted(
                    rng.randint(0, N_NODES, size=m).tolist()
                ))
            counter[edge] = counter.get(edge, 0) + 1
        snaps.append(_snapshot_core(counter, N_NODES, window_index=t,
                                    day_index=0))
    return snaps

def storage_units(snaps, model):

    if model == "H":
        return float(sum(s.multiplicities.sum() for s in snaps))
    if model == "G":
        return float(sum(s.pair_binary.nnz for s in snaps))
    if model == "MG":
        return float(sum(s.pair_weights.nnz for s in snaps))
    if model == "C2":
        return float(sum(s.cell_multiplicities.sum() for s in snaps))
    raise KeyError(model)

def make_prob(engine, model, lam):
    if model == "H":
        def prob_fn(states, step):
            return engine.source_probabilities(states, step, MECH, lam)
    else:
        def prob_fn(states, step):
            feats = engine.features(states, step, model)
            return SparseEngine.predict(
                feats, np.full(feats.shape[2], lam)
            )
    return prob_fn

def main():
    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    snaps3 = build_snapshots(3)
    eng3 = SparseEngine(snaps3, N_NODES)
    cal = calibrate_lambda(eng3, MECH, config=config)
    lam = float(cal["lambda"])
    print("lambda (m=3, AR=%.3f): %.6g" % (cal["attack_rate"], lam),
          flush=True)
    del snaps3, eng3

    rows = []
    details = {}
    for m in SIZES:
        t0 = time.perf_counter()
        snaps = build_snapshots(m)
        build_s = time.perf_counter() - t0
        engine = SparseEngine(snaps, N_NODES)

        initial = np.zeros((RUNS, N_NODES), dtype=np.int8)
        seeds = np.random.RandomState(SEED + 1).randint(
            N_NODES, size=RUNS
        )
        initial[np.arange(RUNS), seeds] = 1
        uniforms = np.random.RandomState(SEED + 2).random_sample(
            (RUNS, T_SNAPS, N_NODES)
        ).astype(np.float32)

        row = {"m": m, "build_seconds": build_s}
        for model in ("H", "MG", "G", "C2"):
            row["storage_%s" % model] = storage_units(snaps, model)
            prob_fn = make_prob(engine, model, lam)
            times = []
            for rep in range(TIMING_REPS):
                t0 = time.perf_counter()
                simulate_batch(prob_fn, initial, uniforms, T_SNAPS)
                times.append(time.perf_counter() - t0)
            row["sim_s_%s" % model] = float(np.median(times))
            print("m=%d %s: storage=%.0f sim=%.2fs" % (
                m, model, row["storage_%s" % model],
                row["sim_s_%s" % model]), flush=True)
        rows.append(row)
        details[m] = {"n_unique_mean": float(
            np.mean([s.n_unique for s in snaps]))}
        del snaps, engine, uniforms

    df = pd.DataFrame(rows)
    for model in ("MG", "G", "C2"):
        df["storage_ratio_%s_H" % model] = (
            df["storage_%s" % model] / df["storage_H"]
        )
        df["sim_ratio_%s_H" % model] = (
            df["sim_s_%s" % model] / df["sim_s_H"]
        )
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    write_json(OUT_DIR / "run_info.json", {
        "seed": SEED, "cal_seed": CAL_SEED,
        "N": N_NODES, "T": T_SNAPS,
        "groups_per_snapshot": GROUPS_PER_SNAP,
        "mechanism": MECH, "lambda_m3": lam,
        "calibration": cal, "runs": RUNS, "timing_reps": TIMING_REPS,
        "n_unique_mean_per_m": details,
        "total_seconds": time.perf_counter() - started,
    })
    print(df.to_string(index=False))
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
