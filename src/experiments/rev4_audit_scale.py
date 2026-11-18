
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    MECHANISMS, SparseEngine, fit_global_hazard, load_config, load_dataset, write_json)
from src.experiments.exp_ext_C_selection import (  
    BUDGETS, LADDER, LOWER, N_RECORDS, period_split, source_trajectories,
    sample_records)

OUT_DIR = ROOT / "results" / "rev4" / "audit_scale"
N_VAL = (10, 30, 100)
N_AUDIT = (0, 10, 30, 100)
REPS = 50
BASE_SEED = 20261111

def per_record_errors(engine, mechanism, lam, traj, records, union, fitted):

    out = {m: [] for m in LOWER}
    for step in sorted(set(int(r[1]) for r in records)):
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        states = traj[runs, step].astype(np.float64)
        sus = states == 0
        src = engine.source_probabilities(states, step, mechanism, lam)
        for m in LOWER:
            pred = SparseEngine.predict(engine.features(states, step, m, union=union),
                                        fitted[m])
            err = np.where(sus, np.abs(pred - src), 0.0).max(axis=1)
            out[m].append(err)
    return {m: np.concatenate(v) for m, v in out.items()}

def train_panel(engine, mechanism, lam, traj, records, union):
    feats = {m: [] for m in LOWER}
    src = []
    for step in sorted(set(int(r[1]) for r in records)):
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        states = traj[runs, step].astype(np.float64)
        sus = states == 0
        p = engine.source_probabilities(states, step, mechanism, lam)
        src.append(p[sus])
        for m in LOWER:
            feats[m].append(engine.features(states, step, m, union=union)[sus])
    return {m: np.concatenate(v) for m, v in feats.items()}, np.concatenate(src)

def select(val, aud, eps):

    m_val = next((m for m in LADDER if val[m] <= eps), "H")
    if aud is None:
        return m_val
    for m in LADDER[LADDER.index(m_val):]:
        if aud[m] <= eps:
            return m
    return "H"

def run_setting(dataset, mechanism, lam, config, seed0):
    train_snaps, test_snaps, _ = period_split(dataset, config)
    tr = SparseEngine(train_snaps, dataset.n_nodes)
    te = SparseEngine(test_snaps, dataset.n_nodes)
    union = tr.union_adjacency(0, tr.n_snapshots)
    traj_train, _, _ = source_trajectories(tr, mechanism, lam, 300, seed0)
    traj_test, _, _ = source_trajectories(te, mechanism, lam, 300, seed0 + 5000)
    fit_recs = sample_records(traj_train[:100], N_RECORDS, seed0 + 1)
    feats, src = train_panel(tr, mechanism, lam, traj_train[:100], fit_recs, union)
    fitted = {m: fit_global_hazard(feats[m], src, config)[0] for m in LOWER}

    pools = {}
    for k, (name, sub) in enumerate((("validation", traj_train[100:200]),
                                     ("audit", traj_train[200:300]))):
        recs = sample_records(sub, N_RECORDS, seed0 + 10 + k)
        pools[name] = per_record_errors(tr, mechanism, lam, sub, recs, union, fitted)
    test_recs = sample_records(traj_test, N_RECORDS, seed0 + 20)
    test_err = per_record_errors(te, mechanism, lam, traj_test, test_recs,
                                 te.union_adjacency(0, te.n_snapshots), fitted)
    test_max = {m: float(test_err[m].max()) for m in LOWER}
    test_max["H"] = 0.0

    rng = np.random.RandomState(seed0 + 30)
    rows = []
    for n_val in N_VAL:
        for n_aud in N_AUDIT:
            for rep in range(REPS):
                iv = rng.choice(len(pools["validation"]["G"]), n_val, replace=False)
                val = {m: float(pools["validation"][m][iv].max()) for m in LOWER}
                val["H"] = 0.0
                aud = None
                if n_aud:
                    ia = rng.choice(len(pools["audit"]["G"]), n_aud, replace=False)
                    aud = {m: float(pools["audit"][m][ia].max()) for m in LOWER}
                    aud["H"] = 0.0
                for eps in BUDGETS:
                    sel = select(val, aud, eps)
                    rows.append({"n_val": n_val, "n_audit": n_aud, "rep": rep,
                                 "epsilon": eps, "selected": sel,
                                 "test_max_err": test_max[sel],
                                 "violation": bool(test_max[sel] > eps + 1e-12)})
    return {"dataset": dataset.name, "mechanism": mechanism, "lambda": lam,
            "test_max_err": test_max, "fitted_h": {m: fitted[m].tolist() for m in LOWER},
            "rows": rows}

def main():
    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    infra = ROOT / config["paths"]["infra_results"]
    calibration = json.loads((infra / "calibration.json").read_text())
    names = config["datasets"]["contact_names"] + [config["datasets"]["email_name"]]
    only = sys.argv[1:]
    for di, name in enumerate(names):
        if only and name not in only:
            continue
        dataset = load_dataset(name)
        for mi, mechanism in enumerate(MECHANISMS):
            out = OUT_DIR / ("setting_%s_%s.json" % (name, mechanism))
            if out.exists():
                continue
            t0 = time.perf_counter()
            lam = float(calibration[name][mechanism]["lambda"])
            res = run_setting(dataset, mechanism, lam, config,
                              BASE_SEED + di * 1000 + mi * 97)
            res["seconds"] = time.perf_counter() - t0
            write_json(out, res)
            v = [r for r in res["rows"] if r["n_audit"] == 0]
            print("%s %s: %.0fs, violation rate without audit %.3f" % (
                name, mechanism, res["seconds"], np.mean([r["violation"] for r in v])),
                flush=True)
        del dataset

if __name__ == "__main__":
    main()
