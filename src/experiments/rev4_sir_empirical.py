
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
    write_json,
)
from src.experiments.exp_ext_C_selection import (  
    BUDGETS,
    LADDER,
    LOWER,
    N_RECORDS,
    W1_TOLERANCE,
    collision_lower_bound,
    model_key,
    model_predict,
    n_params,
    period_split,
    storage_units,
)

OUT_DIR = ROOT / "results" / "rev4" / "sir_empirical"
A_DIR = ROOT / "results" / "external_baselines" / "A"
RUNS_SOURCE = 300
RUNS_EVAL = 500
RUNS_CAL = 400
INITIAL_FRACTION = 0.05
TARGET_SIZE = 0.50
TOL = 0.01
BASE_SEED = 20261110

METHOD_ORDER = (
    ["Ours(eps=%.2f)" % e for e in BUDGETS]
    + ["Lucas", "Kirkley", "BIC"]
    + ["Always-%s" % m for m in LADDER]
)

def simulate_sir(prob_fn, initial, inf_u, rec_u, n_steps, gamma):

    state = initial.copy()
    out = [state.copy()]
    for step in range(n_steps):
        x = (state == 1).astype(np.float64)
        p = prob_fn(x, step)
        new_inf = (state == 0) & (inf_u[:, step] < p)
        new_rec = (state == 1) & (rec_u[:, step] < gamma)
        state = state.copy()
        state[new_rec] = 2
        state[new_inf] = 1
        out.append(state.copy())
    return np.stack(out, axis=1)

def random_initial(runs, n, seed):
    rng = np.random.RandomState(seed)
    init = (rng.random_sample((runs, n)) < INITIAL_FRACTION).astype(np.int8)
    empty = init.sum(axis=1) == 0
    init[np.flatnonzero(empty), rng.randint(n, size=int(empty.sum()))] = 1
    return init

def crn(runs, steps, n, seed):
    rng = np.random.RandomState(seed)
    inf_u = rng.random_sample((runs, steps, n)).astype(np.float32)
    rec_u = rng.random_sample((runs, steps, n)).astype(np.float32)
    return inf_u, rec_u

def final_size(traj):
    return (traj[:, -1] != 0).mean(axis=1)

def calibrate(engine, mechanism, gamma, seed):
    init = random_initial(RUNS_CAL, engine.n, seed)
    inf_u, rec_u = crn(RUNS_CAL, engine.n_snapshots, engine.n, seed + 1)
    evals = []

    def size(log_lam):
        lam = math.exp(log_lam)
        traj = simulate_sir(
            lambda x, t: engine.source_probabilities(x, t, mechanism, lam),
            init, inf_u, rec_u, engine.n_snapshots, gamma)
        value = float(final_size(traj).mean())
        evals.append((lam, value))
        return value

    lo, hi = -16.0, 7.0
    best = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v = size(mid)
        if best is None or abs(v - TARGET_SIZE) < abs(best[1] - TARGET_SIZE):
            best = (math.exp(mid), v)
        if abs(v - TARGET_SIZE) <= TOL:
            break
        if v < TARGET_SIZE:
            lo = mid
        else:
            hi = mid
    return {"lambda": best[0], "final_size": best[1],
            "status": "ok" if abs(best[1] - TARGET_SIZE) <= TOL else "unreachable",
            "evaluations": len(evals)}

def sample_records(traj, count, seed):

    inf = (traj[:, :-1, :] == 1).sum(axis=2)
    sus = (traj[:, :-1, :] == 0).sum(axis=2)
    eligible = np.argwhere((inf >= 1) & (sus >= 1))
    rng = np.random.RandomState(seed)
    take = min(count, len(eligible))
    return eligible[rng.choice(len(eligible), size=take, replace=False)]

def record_panel(engine, mechanism, lam, traj, records, union):
    feats = {m: [] for m in LOWER}
    src = []
    for step in sorted(set(int(r[1]) for r in records)):
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        st = traj[runs, step]
        x = (st == 1).astype(np.float64)
        sus = st == 0
        p = engine.source_probabilities(x, step, mechanism, lam)
        fmap = {m: engine.features(x, step, m, union=union) for m in LOWER}
        for i in range(len(runs)):
            src.append(p[i][sus[i]])
            for m in LOWER:
                feats[m].append(fmap[m][i][sus[i]])
    return {"source": np.concatenate(src),
            "features": {m: np.concatenate(feats[m]) for m in LOWER}}

def susceptible_transitions(engine, mechanism, lam, traj, union):

    feats = {m: [] for m in LOWER}
    outcome, src = [], []
    for step in range(traj.shape[1] - 1):
        st = traj[:, step]
        sus = st == 0
        if not sus.any():
            continue
        x = (st == 1).astype(np.float64)
        p = engine.source_probabilities(x, step, mechanism, lam)
        src.append(p[sus])
        outcome.append((traj[:, step + 1][sus] == 1).astype(np.float64))
        for m in LOWER:
            feats[m].append(engine.features(x, step, m, union=union)[sus])
    return ({m: np.concatenate(feats[m]) for m in LOWER},
            np.concatenate(outcome), np.concatenate(src))

def log_likelihood(probs, outcomes):
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return float(np.sum(outcomes * np.log(p) + (1.0 - outcomes) * np.log1p(-p)))

def node_errors(engine, model, mechanism, lam, union, traj, records):
    diffs = []
    for step in sorted(set(int(r[1]) for r in records)):
        runs = [int(r[0]) for r in records if int(r[1]) == step]
        st = traj[runs, step]
        x = (st == 1).astype(np.float64)
        sus = st == 0
        src = engine.source_probabilities(x, step, mechanism, lam)[sus]
        pred = model_predict(engine, model, x, step, mechanism, lam, union)[sus]
        diffs.append(np.abs(pred - src))
    d = np.concatenate(diffs)
    return float(d.max()), float(d.mean())

def run_setting(dataset, mechanism, gamma, external, config, seed0):
    full = SparseEngine(dataset.snapshots, dataset.n_nodes)
    cal = calibrate(full, mechanism, gamma, seed0)
    lam = cal["lambda"]
    result = {"dataset": dataset.name, "mechanism": mechanism, "gamma": gamma,
              "calibration": cal, "lambda": lam}

    if mechanism == "M1":
        init = random_initial(RUNS_EVAL, full.n, seed0 + 11)
        inf_u, rec_u = crn(RUNS_EVAL, full.n_snapshots, full.n, seed0 + 12)
        t_src = simulate_sir(
            lambda x, t: full.source_probabilities(x, t, mechanism, lam),
            init, inf_u, rec_u, full.n_snapshots, gamma)
        h_mg = np.array([lam])
        t_mg = simulate_sir(
            lambda x, t: SparseEngine.predict(full.features(x, t, "MG"), h_mg),
            init, inf_u, rec_u, full.n_snapshots, gamma)
        result["exact_mg"] = {
            "runs": RUNS_EVAL,
            "identical_fraction": float(np.all(t_src == t_mg, axis=(1, 2)).mean()),
            "w1_final_size": float(wasserstein_distance(final_size(t_src),
                                                        final_size(t_mg))),
        }

    train_snaps, test_snaps, train_days = period_split(dataset, config)
    tr = SparseEngine(train_snaps, dataset.n_nodes)
    te = SparseEngine(test_snaps, dataset.n_nodes)
    union = tr.union_adjacency(0, tr.n_snapshots)

    def src_runs(engine, runs, seed):
        init = random_initial(runs, engine.n, seed)
        inf_u, rec_u = crn(runs, engine.n_snapshots, engine.n, seed + 1)
        return simulate_sir(
            lambda x, t: engine.source_probabilities(x, t, mechanism, lam),
            init, inf_u, rec_u, engine.n_snapshots, gamma)

    traj_train = src_runs(tr, RUNS_SOURCE, seed0 + 100)
    traj_test = src_runs(te, RUNS_SOURCE, seed0 + 200)
    splits = {"train": traj_train[:100], "validation": traj_train[100:200],
              "audit": traj_train[200:300]}
    panels = {}
    for k, (name, sub) in enumerate(splits.items()):
        recs = sample_records(sub, N_RECORDS, seed0 + 300 + k)
        panels[name] = record_panel(tr, mechanism, lam, sub, recs, union)
    test_records = sample_records(traj_test, N_RECORDS, seed0 + 400)
    test_panel = record_panel(te, mechanism, lam, traj_test, test_records,
                              te.union_adjacency(0, te.n_snapshots))

    fitted = {}
    for m in LOWER:
        h, _ = fit_global_hazard(panels["train"]["features"][m],
                                 panels["train"]["source"], config)
        fitted[m] = h

    def max_err(m, panel):
        pred = SparseEngine.predict(panel["features"][m], fitted[m])
        return float(np.abs(pred - panel["source"]).max())

    val_err = {m: max_err(m, panels["validation"]) for m in LOWER}
    audit_err = {m: max_err(m, panels["audit"]) for m in LOWER}
    lbs = {m: collision_lower_bound(panels["validation"]["features"][m],
                                    panels["validation"]["source"]) for m in LOWER}
    for d in (val_err, audit_err, lbs):
        d["H"] = 0.0

    q2 = {}
    for m in ("G", "MG", "C2"):
        worst = max_err(m, test_panel)
        lb = collision_lower_bound(test_panel["features"][m], test_panel["source"])
        q2[m] = {"max_err": worst, "lb_infty": lb,
                 "ratio": (lb / worst) if worst > 1e-9 else None}
    result["q2"] = q2

    ours = {}
    for eps in BUDGETS:
        eligible = [m for m in LADDER if lbs[m] <= eps]
        m_val = next((m for m in eligible if val_err[m] <= eps), "H")
        selected = "H"
        for m in LADDER[LADDER.index(m_val):]:
            if audit_err[m] <= eps:
                selected = m
                break
        ours[eps] = {"m_val": m_val, "selected": selected,
                     "val_max_err": dict(val_err), "audit_max_err": dict(audit_err),
                     "lb": dict(lbs)}

    hs = {}
    for method, sizes in (("Lucas", external["lucas_sizes"]),
                          ("Kirkley", external["kirkley_sizes"])):
        sizes = set(int(s) for s in sizes)
        recs = sample_records(splits["train"], N_RECORDS, seed0 + 300)
        rows = []
        for step in sorted(set(int(r[1]) for r in recs)):
            runs = [int(r[0]) for r in recs if int(r[1]) == step]
            st = splits["train"][runs, step]
            x = (st == 1).astype(np.float64)
            hf = tr.hazard_feature(x, step, mechanism, lam, allowed_sizes=sizes)
            rows.append(hf[st == 0])
        feats1d = np.concatenate(rows)[:, None]
        mult, _ = fit_global_hazard(feats1d, panels["train"]["source"], config,
                                    upper_bound=50.0)
        hs[method] = {"sizes": sorted(sizes), "multiplier": float(mult[0])}

    feats_all, out_all, src_all = susceptible_transitions(
        tr, mechanism, lam, traj_train[:100], union)
    n_obs = len(out_all)
    bic = {}
    for m in LOWER:
        h_ml, info = bernoulli_ml_fit(feats_all[m], out_all, config)
        bic[m] = {"h": h_ml.tolist(), "k": len(h_ml),
                  "bic": len(h_ml) * math.log(n_obs) - 2.0 * info["log_likelihood"]}
    bic["H"] = {"h": [lam], "k": 1,
                "bic": math.log(n_obs) - 2.0 * log_likelihood(src_all, out_all)}
    bic_sel = min(LADDER, key=lambda m: bic[m]["bic"])

    models = {
        "Lucas": {"kind": "HS", "sizes": hs["Lucas"]["sizes"],
                  "multiplier": hs["Lucas"]["multiplier"]},
        "Kirkley": {"kind": "HS", "sizes": hs["Kirkley"]["sizes"],
                    "multiplier": hs["Kirkley"]["multiplier"]},
        "BIC": ({"kind": "lower", "candidate": bic_sel,
                 "h": np.asarray(bic[bic_sel]["h"])} if bic_sel != "H"
                else {"kind": "H"}),
    }
    for eps in BUDGETS:
        sel = ours[eps]["selected"]
        models["Ours(eps=%.2f)" % eps] = (
            {"kind": "lower", "candidate": sel, "h": fitted[sel]}
            if sel != "H" else {"kind": "H"})
    for m in LOWER:
        models["Always-%s" % m] = {"kind": "lower", "candidate": m, "h": fitted[m]}
    models["Always-H"] = {"kind": "H"}

    init = random_initial(RUNS_EVAL, te.n, seed0 + 500)
    inf_u, rec_u = crn(RUNS_EVAL, te.n_snapshots, te.n, seed0 + 501)
    te_union = union
    t_src = simulate_sir(
        lambda x, t: te.source_probabilities(x, t, mechanism, lam),
        init, inf_u, rec_u, te.n_snapshots, gamma)
    fs_src = final_size(t_src)
    src_storage = float(sum(s.multiplicities.sum() for s in test_snaps))
    cache, rows = {}, {}
    for name in METHOD_ORDER:
        model = models[name]
        key = model_key(model)
        if key not in cache:
            cache[key] = simulate_sir(
                lambda x, t, model=model: model_predict(te, model, x, t, mechanism,
                                                        lam, te_union),
                init, inf_u, rec_u, te.n_snapshots, gamma)
        tm = cache[key]
        w1 = float(wasserstein_distance(fs_src, final_size(tm)))
        nmax, nmean = node_errors(te, model, mechanism, lam, te_union,
                                  traj_test, test_records)
        rows[name] = {
            "method": name,
            "selected": (model.get("candidate") if model["kind"] == "lower"
                         else ("H" if model["kind"] == "H"
                               else "H_S(%s)" % ",".join(map(str, model["sizes"])))),
            "w1_final_size": w1,
            "accurate": bool(w1 <= W1_TOLERANCE),
            "exact_agreement": float(np.all(tm == t_src, axis=(1, 2)).mean()),
            "node_max_err": nmax, "node_mean_err": nmean,
            "n_params": n_params(model),
            "storage_rel_source": storage_units(model, te, test_snaps,
                                                int(union.nnz)) / src_storage,
        }
    result.update({"train_days": train_days, "ours": {"%.2f" % e: ours[e] for e in BUDGETS},
                   "order_filtered": hs, "bic_selected": bic_sel,
                   "fitted_h": {m: fitted[m].tolist() for m in LOWER},
                   "methods": rows,
                   "mean_final_size_test_source": float(fs_src.mean())})
    return result

def main():
    import pandas as pd

    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    infra = ROOT / config["paths"]["infra_results"]
    external = json.loads((infra / "external_methods.json").read_text())
    names = config["datasets"]["contact_names"] + [config["datasets"]["email_name"]]
    only = [a.split(":")[0] for a in sys.argv[1:]]  
    only_mech = {a.split(":")[0]: a.split(":")[1] for a in sys.argv[1:] if ":" in a}
    rows = []
    started = time.perf_counter()
    for di, name in enumerate(names):
        if only and name not in only:
            continue
        dataset = load_dataset(name)
        for mi, mechanism in enumerate(MECHANISMS):
            if name in only_mech and mechanism != only_mech[name]:
                continue
            out = OUT_DIR / ("setting_%s_%s.json" % (name, mechanism))
            if out.exists():
                result = json.loads(out.read_text())
            else:
                gamma = float(json.loads((A_DIR / ("setting_%s_%s.json" % (
                    name, mechanism))).read_text())["gamma"])
                ext = {"lucas_sizes": range(2, external[name]["lucas"]["max_kept_hyperedge_size"] + 1),
                       "kirkley_sizes": external[name]["kirkley"]["R_star_sizes"]}
                t0 = time.perf_counter()
                result = run_setting(dataset, mechanism, gamma, ext, config,
                                     BASE_SEED + di * 1000 + mi * 97)
                result["seconds"] = time.perf_counter() - t0
                write_json(out, result)
            acc = {m: r["accurate"] for m, r in result["methods"].items()}
            print("%s %s: lambda=%.4g size=%.3f, ours@0.01=%s acc=%s, %.0fs" % (
                name, mechanism, result["lambda"], result["calibration"]["final_size"],
                result["ours"]["0.01"]["selected"], acc["Ours(eps=0.01)"],
                result.get("seconds", 0.0)), flush=True)
            for m, r in result["methods"].items():
                rows.append({"dataset": name, "mechanism": mechanism, **r})
        del dataset
    if not only:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
        write_json(OUT_DIR / "run_info.json",
                   {"total_seconds": time.perf_counter() - started})

if __name__ == "__main__":
    main()
