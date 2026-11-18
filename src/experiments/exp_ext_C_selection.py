
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    MECHANISMS,
    SparseEngine,
    bernoulli_ml_fit,
    fit_global_hazard,
    load_config,
    load_dataset,
    simulate_batch,
    write_json,
)

OUT_DIR = ROOT / "results" / "external_baselines" / "C"

LADDER = ("SG", "G", "MG", "C2", "H")
COMPLEXITY = {"SG": 0.5, "G": 1.0, "MG": 2.0, "C2": 4.5, "H": 10.0}
LOWER = ("SG", "G", "MG", "C2")
BUDGETS = (0.01, 0.02, 0.05, 0.10)
RUNS_SOURCE = 300
RUNS_EVAL = 500
N_RECORDS = 4000
W1_TOLERANCE = 0.02
BASE_SEED = 20261018

METHOD_ORDER = (
    ["Ours(eps=%.2f)" % e for e in BUDGETS]
    + ["Lucas", "Lucas(no-refit)", "Kirkley", "Kirkley(no-refit)", "BIC"]
    + ["Always-%s" % m for m in LADDER]
)

def period_split(dataset, config):

    if dataset.stats["is_email"]:
        train_days = int(config["training_period"]["email_days"])
    else:
        train_days = int(math.ceil(dataset.n_days / 2.0))
    train = [s for s in dataset.snapshots if s.day_index < train_days]
    test = [s for s in dataset.snapshots if s.day_index >= train_days]
    return train, test, train_days

def random_initial(runs, n, seed):
    initial = np.zeros((runs, n), dtype=np.int8)
    seeds = np.random.RandomState(seed).randint(n, size=runs)
    initial[np.arange(runs), seeds] = 1
    return initial

def source_trajectories(engine, mechanism, lam, runs, seed):
    initial = random_initial(runs, engine.n, seed)
    uniforms = np.random.RandomState(seed + 1).random_sample(
        (runs, engine.n_snapshots, engine.n)
    ).astype(np.float32)

    def prob_fn(states, step):
        return engine.source_probabilities(states, step, mechanism, lam)

    traj = simulate_batch(prob_fn, initial, uniforms, engine.n_snapshots)
    return traj, initial, uniforms

def sample_records(traj, count, seed):

    n = traj.shape[2]
    infected = traj[:, :-1, :].sum(axis=2)  
    eligible = np.argwhere((infected >= 1) & (infected <= n - 1))
    rng = np.random.RandomState(seed)
    take = min(count, len(eligible))
    idx = rng.choice(len(eligible), size=take, replace=False)
    return eligible[idx]

def record_panel(engine, mechanism, lam, traj, records, union):

    feats = {m: [] for m in LOWER}
    src = []
    steps = sorted(set(int(r[1]) for r in records))
    for step in steps:
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        states = traj[runs, step].astype(np.float64)
        p = engine.source_probabilities(states, step, mechanism, lam)
        fmap = {m: engine.features(states, step, m, union=union)
                for m in LOWER}
        for i in range(len(runs)):
            mask = states[i] == 0
            src.append(p[i][mask])
            for m in LOWER:
                feats[m].append(fmap[m][i][mask])
    return {
        "source": np.concatenate(src),
        "features": {m: np.concatenate(feats[m]) for m in LOWER},
    }

def collision_lower_bound(features, source_p):

    keys, inverse = np.unique(features, axis=0, return_inverse=True)
    lb = 0.0
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    src_sorted = source_p[order]
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    ends = np.r_[starts[1:], len(inv_sorted)]
    for a, b in zip(starts, ends):
        if b - a >= 2:
            group = src_sorted[a:b]
            lb = max(lb, 0.5 * float(group.max() - group.min()))
    return float(lb)

def collect_susceptible_panel(engine, mechanism, lam, traj, runs_idx,
                              union):

    feats = {m: [] for m in LOWER}
    outcome, src = [], []
    n = traj.shape[2]
    for step in range(traj.shape[1] - 1):
        states = traj[runs_idx, step].astype(np.float64)
        mask = states == 0
        if not mask.any():
            continue
        p = engine.source_probabilities(states, step, mechanism, lam)
        src.append(p[mask])
        outcome.append(traj[runs_idx, step + 1][mask].astype(np.float64))
        for m in LOWER:
            f = engine.features(states, step, m, union=union)
            feats[m].append(f[mask])
    return (
        {m: np.concatenate(feats[m]) for m in LOWER},
        np.concatenate(outcome),
        np.concatenate(src),
    )

def log_likelihood(probs, outcomes):
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return float(np.sum(outcomes * np.log(p) + (1.0 - outcomes) * np.log1p(-p)))

def model_predict(engine, model, states, step, mechanism, lam, union):

    kind = model["kind"]
    if kind == "lower":
        feats = engine.features(states, step, model["candidate"], union=union)
        return SparseEngine.predict(feats, model["h"])
    if kind == "H":
        return engine.source_probabilities(states, step, mechanism, lam)
    if kind == "HS":
        return engine.source_probabilities(
            states, step, mechanism, lam,
            allowed_sizes=model["sizes"], lam_scale=model["multiplier"],
        )
    raise KeyError(kind)

def model_key(model):
    if model["kind"] == "lower":
        return ("lower", model["candidate"], tuple(np.round(model["h"], 12)))
    if model["kind"] == "H":
        return ("H",)
    return ("HS", tuple(sorted(model["sizes"])), round(model["multiplier"], 12))

def storage_units(model, test_engine, test_snaps, train_union_nnz):
    kind = model["kind"]
    if kind == "lower":
        cand = model["candidate"]
        if cand == "SG":
            return float(train_union_nnz)
        if cand == "G":
            return float(sum(s.pair_binary.nnz for s in test_snaps))
        if cand == "MG":
            return float(sum(s.pair_weights.nnz for s in test_snaps))
        if cand == "C2":
            return float(sum(s.cell_multiplicities.sum() for s in test_snaps))
        raise KeyError(cand)
    if kind == "H":
        return float(sum(s.multiplicities.sum() for s in test_snaps))
    if kind == "HS":
        total = 0.0
        for s in test_snaps:
            mask = np.isin(s.sizes, list(model["sizes"]))
            total += float(s.multiplicities[mask].sum())
        return total
    raise KeyError(kind)

def n_params(model):
    if model["kind"] == "lower":
        return int(len(model["h"]))
    return 1

def process_metrics(traj_src, traj_m):

    ar_src = traj_src[:, -1].mean(axis=1)
    ar_m = traj_m[:, -1].mean(axis=1)
    curve_src = traj_src.mean(axis=(0, 2))
    curve_m = traj_m.mean(axis=(0, 2))
    exact = np.all(traj_m == traj_src, axis=(1, 2)).mean()
    return {
        "w1_attack_rate": float(wasserstein_distance(ar_src, ar_m)),
        "mean_curve_mae": float(np.abs(curve_m - curve_src).mean()),
        "mean_paired_ar_diff": float(np.abs(ar_m - ar_src).mean()),
        "exact_agreement": float(exact),
    }

def node_errors(engine, model, mechanism, lam, union, traj, records):

    diffs = []
    steps = sorted(set(int(r[1]) for r in records))
    for step in steps:
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        states = traj[runs, step].astype(np.float64)
        mask = states == 0
        src = engine.source_probabilities(states, step, mechanism, lam)[mask]
        pred = model_predict(engine, model, states, step, mechanism, lam, union)[mask]
        diffs.append(np.abs(pred - src))
    d = np.concatenate(diffs)
    return float(d.max()), float(d.mean())

def run_setting(dataset, mechanism, lam, external, config, seed0):
    train_snaps, test_snaps, train_days = period_split(dataset, config)
    tr_eng = SparseEngine(train_snaps, dataset.n_nodes)
    te_eng = SparseEngine(test_snaps, dataset.n_nodes)
    union = tr_eng.union_adjacency(0, tr_eng.n_snapshots)

    traj_train, _, _ = source_trajectories(tr_eng, mechanism, lam, RUNS_SOURCE, seed0)
    traj_test, _, _ = source_trajectories(te_eng, mechanism, lam, RUNS_SOURCE,
                                          seed0 + 5000)
    splits = {
        "train": traj_train[:100],
        "validation": traj_train[100:200],
        "audit": traj_train[200:300],
    }
    panels = {}
    for k, (name, sub) in enumerate(splits.items()):
        records = sample_records(sub, N_RECORDS, seed0 + 100 + k)
        panels[name] = record_panel(tr_eng, mechanism, lam, sub, records, union)
    test_records = sample_records(traj_test, N_RECORDS, seed0 + 200)

    fitted, fit_info = {}, {}
    for m in LOWER:
        h, info = fit_global_hazard(
            panels["train"]["features"][m], panels["train"]["source"], config
        )
        fitted[m] = h
        fit_info[m] = info

    def val_error(m, split):
        pred = SparseEngine.predict(panels[split]["features"][m], fitted[m])
        return float(np.abs(pred - panels[split]["source"]).max())

    val_err = {m: val_error(m, "validation") for m in LOWER}
    audit_err = {m: val_error(m, "audit") for m in LOWER}
    lbs = {
        m: collision_lower_bound(
            panels["validation"]["features"][m], panels["validation"]["source"]
        )
        for m in LOWER
    }
    val_err["H"] = 0.0
    audit_err["H"] = 0.0
    lbs["H"] = 0.0

    ours = {}
    for eps in BUDGETS:
        eligible = [m for m in LADDER if lbs[m] <= eps]
        m_val = next(
            (m for m in eligible if val_err[m] <= eps), "H"
        )
        start = LADDER.index(m_val)
        selected = "H"
        for m in LADDER[start:]:
            if audit_err[m] <= eps:
                selected = m
                break
        ours[eps] = {
            "epsilon": eps, "lb": {m: lbs[m] for m in LADDER},
            "val_max_err": {m: val_err[m] for m in LADDER},
            "audit_max_err": {m: audit_err[m] for m in LADDER},
            "eligible": eligible, "m_val": m_val, "selected": selected,
            "delta_m": 0.0,
        }

    def sampled_hazard_features(size_set, records, traj, engine):
        rows = []
        steps = sorted(set(int(r[1]) for r in records))
        for step in steps:
            runs = [int(r[0]) for r in records if int(r[1]) == step]
            states = traj[runs, step].astype(np.float64)
            hfeat = engine.hazard_feature(
                states, step, mechanism, lam, allowed_sizes=size_set
            )
            mask = states == 0
            rows.append(hfeat[mask])
        return np.concatenate(rows)[:, None]

    train_records = sample_records(splits["train"], N_RECORDS, seed0 + 100)
    hs = {}
    for method, size_set in (
        ("Lucas", set(external["lucas_sizes"])),
        ("Kirkley", set(external["kirkley_sizes"])),
    ):
        size_set = set(int(s) for s in size_set)
        feats1d = sampled_hazard_features(
            size_set, train_records, splits["train"], tr_eng
        )
        mult, minfo = fit_global_hazard(
            feats1d, panels["train"]["source"], config, upper_bound=50.0
        )
        hs[method] = {"sizes": sorted(size_set),
                      "multiplier": float(mult[0]), "fit": minfo}

    feats_all, out_all, src_all = collect_susceptible_panel(
        tr_eng, mechanism, lam, traj_train, np.arange(100), union
    )
    n_obs = len(out_all)
    bic_table = {}
    for m in LOWER:
        h_ml, info = bernoulli_ml_fit(feats_all[m], out_all, config)
        k = len(h_ml)
        bic_table[m] = {
            "h": h_ml.tolist(), "k": k,
            "log_likelihood": info["log_likelihood"],
            "bic": k * math.log(n_obs) - 2.0 * info["log_likelihood"],
        }
    ll_h = log_likelihood(src_all, out_all)
    bic_table["H"] = {"h": [float(lam)], "k": 1, "log_likelihood": ll_h,
                      "bic": 1 * math.log(n_obs) - 2.0 * ll_h}
    bic_selected = min(LADDER, key=lambda m: bic_table[m]["bic"])

    models = {
        "Lucas": {"kind": "HS", "sizes": hs["Lucas"]["sizes"],
                  "multiplier": hs["Lucas"]["multiplier"]},
        "Lucas(no-refit)": {"kind": "HS", "sizes": hs["Lucas"]["sizes"],
                            "multiplier": 1.0},
        "Kirkley": {"kind": "HS", "sizes": hs["Kirkley"]["sizes"],
                    "multiplier": hs["Kirkley"]["multiplier"]},
        "Kirkley(no-refit)": {"kind": "HS", "sizes": hs["Kirkley"]["sizes"],
                              "multiplier": 1.0},
        "BIC": (
            {"kind": "lower", "candidate": bic_selected,
             "h": np.asarray(bic_table[bic_selected]["h"], dtype=float)}
            if bic_selected != "H" else {"kind": "H"}
        ),
    }
    for eps in BUDGETS:
        sel = ours[eps]["selected"]
        models["Ours(eps=%.2f)" % eps] = (
            {"kind": "lower", "candidate": sel, "h": fitted[sel]}
            if sel != "H" else {"kind": "H"}
        )
    for m in LOWER:
        models["Always-%s" % m] = {"kind": "lower", "candidate": m,
                                   "h": fitted[m]}
    models["Always-H"] = {"kind": "H"}

    eval_uniforms = np.random.RandomState(seed0 + 9000).random_sample(
        (RUNS_EVAL, te_eng.n_snapshots, te_eng.n)
    ).astype(np.float32)
    eval_initial = random_initial(RUNS_EVAL, te_eng.n, seed0 + 9001)

    def prob_source(states, step):
        return te_eng.source_probabilities(states, step, mechanism, lam)

    traj_src = simulate_batch(prob_source, eval_initial, eval_uniforms,
                              te_eng.n_snapshots)

    traj_cache = {}
    time_cache = {}
    rows = {}
    src_storage = float(sum(s.multiplicities.sum() for s in test_snaps))
    union_nnz = int(union.nnz)
    src_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        simulate_batch(prob_source, eval_initial, eval_uniforms,
                       te_eng.n_snapshots)
        src_times.append(time.perf_counter() - t0)
    src_seconds = float(np.median(src_times))

    for name in METHOD_ORDER:
        model = models[name]
        key = model_key(model)
        if key not in traj_cache:
            times = []
            traj = None
            for rep in range(5):
                t0 = time.perf_counter()

                def prob_fn(states, step, model=model):
                    return model_predict(te_eng, model, states, step,
                                         mechanism, lam, union)

                traj = simulate_batch(prob_fn, eval_initial, eval_uniforms,
                                      te_eng.n_snapshots)
                times.append(time.perf_counter() - t0)
            traj_cache[key] = traj
            time_cache[key] = float(np.median(times))
        traj_m = traj_cache[key]
        metrics = process_metrics(traj_src, traj_m)
        nmax, nmean = node_errors(te_eng, model, mechanism, lam, union,
                                  traj_test, test_records)
        storage = storage_units(model, te_eng, test_snaps, union_nnz)
        rows[name] = {
            "method": name,
            "model": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in model.items()},
            "selected": (model.get("candidate")
                         if model["kind"] == "lower"
                         else ("H" if model["kind"] == "H"
                               else "H_S(%s)" % ",".join(map(str, model["sizes"])))),
            "node_max_err": nmax,
            "node_mean_err": nmean,
            **metrics,
            "accurate": bool(metrics["w1_attack_rate"] <= W1_TOLERANCE),
            "n_params": n_params(model),
            "storage_units": storage,
            "storage_rel_source": storage / src_storage,
            "seconds_500runs": time_cache[key],
            "runtime_rel_source": time_cache[key] / src_seconds,
        }

    return {
        "dataset": dataset.name, "mechanism": mechanism, "lambda": lam,
        "train_days": train_days,
        "n_train_snapshots": len(train_snaps),
        "n_test_snapshots": len(test_snaps),
        "fitted_h": {m: fitted[m].tolist() for m in LOWER},
        "fit_info": fit_info,
        "ours": {"%.2f" % e: ours[e] for e in BUDGETS},
        "order_filtered": hs,
        "bic_table": bic_table, "bic_selected": bic_selected,
        "test_source_seconds_500runs": src_seconds,
        "test_source_storage": src_storage,
        "methods": rows,
    }

def main():
    import pandas as pd

    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    infra = ROOT / config["paths"]["infra_results"]
    calibration = json.loads((infra / "calibration.json").read_text())
    external = json.loads((infra / "external_methods.json").read_text())

    names = config["datasets"]["contact_names"] + [config["datasets"]["email_name"]]
    all_rows = []
    started = time.perf_counter()
    for di, name in enumerate(names):
        dataset = load_dataset(name)
        for mi, mechanism in enumerate(MECHANISMS):
            lam = float(calibration[name][mechanism]["lambda"])
            ext = {
                "lucas_sizes": range(
                    2, external[name]["lucas"]["max_kept_hyperedge_size"] + 1
                ),
                "kirkley_sizes": external[name]["kirkley"]["R_star_sizes"],
            }
            seed0 = BASE_SEED + di * 1000 + mi * 97
            t0 = time.perf_counter()
            result = run_setting(dataset, mechanism, lam, ext, config, seed0)
            result["seconds"] = time.perf_counter() - t0
            write_json(
                OUT_DIR / ("setting_%s_%s.json" % (name, mechanism)), result
            )
            for m in METHOD_ORDER:
                row = {"dataset": name, "mechanism": mechanism,
                       **{k: v for k, v in result["methods"][m].items()
                          if k != "model"}}
                all_rows.append(row)
            acc = sum(r["accurate"] for r in result["methods"].values())
            print("%s %s: %.0fs, %d/14 methods accurate" % (
                name, mechanism, result["seconds"], acc), flush=True)
        del dataset

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)

    grp = df.groupby("method", sort=False)
    table = pd.DataFrame({
        "fraction_accurate": grp["accurate"].mean(),
        "median_w1": grp["w1_attack_rate"].median(),
        "median_storage_rel": grp["storage_rel_source"].median(),
        "median_runtime_rel": grp["runtime_rel_source"].median(),
        "mean_n_params": grp["n_params"].mean(),
    }).reindex(METHOD_ORDER)
    table.to_csv(OUT_DIR / "method_summary.csv")
    write_json(OUT_DIR / "run_info.json",
               {"total_seconds": time.perf_counter() - started})
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
