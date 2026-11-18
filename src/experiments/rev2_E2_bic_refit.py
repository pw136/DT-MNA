
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    MECHANISMS,
    SparseEngine,
    bernoulli_ml_fit,
    load_config,
    load_dataset,
    simulate_batch,
    write_json,
)
from src.experiments.exp_ext_C_selection import (  
    BASE_SEED,
    LADDER,
    LOWER,
    RUNS_EVAL,
    RUNS_SOURCE,
    W1_TOLERANCE,
    collect_susceptible_panel,
    period_split,
    process_metrics,
    random_initial,
    source_trajectories,
)

OUT_DIR = ROOT / "results" / "rev2" / "E2"
C_DIR = ROOT / "results" / "external_baselines" / "C"
WORKERS = 12

def collect_hazard_panel(engine, mechanism, lam, traj, runs_idx):

    rows = []
    for step in range(traj.shape[1] - 1):
        states = traj[runs_idx, step].astype(np.float64)
        mask = states == 0
        if not mask.any():
            continue
        hfeat = engine.hazard_feature(states, step, mechanism, lam)
        rows.append(hfeat[mask])
    return np.concatenate(rows)[:, None]

def evaluate_selection(te_eng, model, mechanism, lam, seed0, union):

    te_n = te_eng.n
    eval_uniforms = np.random.RandomState(seed0 + 9000).random_sample(
        (RUNS_EVAL, te_eng.n_snapshots, te_n)
    ).astype(np.float32)
    eval_initial = random_initial(RUNS_EVAL, te_n, seed0 + 9001)

    def prob_source(states, step):
        return te_eng.source_probabilities(states, step, mechanism, lam)

    traj_src = simulate_batch(
        prob_source, eval_initial, eval_uniforms, te_eng.n_snapshots
    )

    if model["kind"] == "H_refit":
        def prob_model(states, step):
            return te_eng.source_probabilities(
                states, step, mechanism, lam, lam_scale=model["kappa"]
            )
    elif model["kind"] == "lower":

        def prob_model(states, step):
            feats = te_eng.features(
                states, step, model["candidate"], union=union
            )
            return SparseEngine.predict(feats, model["h"])
    else:
        raise KeyError(model["kind"])

    traj_m = simulate_batch(
        prob_model, eval_initial, eval_uniforms, te_eng.n_snapshots
    )
    metrics = process_metrics(traj_src, traj_m)
    metrics["accurate"] = bool(metrics["w1_attack_rate"] <= W1_TOLERANCE)
    return metrics

def run_setting(task):
    name, mechanism, lam, di, mi = task
    started = time.perf_counter()
    config = load_config()
    dataset = load_dataset(name)
    seed0 = BASE_SEED + di * 1000 + mi * 97
    stored = json.loads(
        (C_DIR / ("setting_%s_%s.json" % (name, mechanism))).read_text()
    )

    train_snaps, test_snaps, _ = period_split(dataset, config)
    tr_eng = SparseEngine(train_snaps, dataset.n_nodes)
    te_eng = SparseEngine(test_snaps, dataset.n_nodes)
    union = tr_eng.union_adjacency(0, tr_eng.n_snapshots)

    traj_train, _, _ = source_trajectories(
        tr_eng, mechanism, lam, RUNS_SOURCE, seed0
    )
    feats_all, out_all, src_all = collect_susceptible_panel(
        tr_eng, mechanism, lam, traj_train, np.arange(100), union
    )
    n_obs = len(out_all)

    bic_table = {}
    repro_diff = {}
    for m in LOWER:
        h_ml, info = bernoulli_ml_fit(feats_all[m], out_all, config)
        ll = info["log_likelihood"]
        k = len(h_ml)
        bic = k * math.log(n_obs) - 2.0 * ll
        bic_table[m] = {"k": k, "log_likelihood": ll, "bic": bic,
                        "h": h_ml.tolist()}
        st = stored["bic_table"][m]
        repro_diff[m] = {
            "ll_abs_diff": abs(ll - st["log_likelihood"]),
            "bic_abs_diff": abs(bic - st["bic"]),
        }
    from src.experiments.exp_ext_C_selection import log_likelihood

    ll_h_old = log_likelihood(src_all, out_all)
    bic_h_old = math.log(n_obs) - 2.0 * ll_h_old
    bic_table["H_norefit"] = {"k": 1, "log_likelihood": ll_h_old,
                              "bic": bic_h_old}
    repro_diff["H"] = {
        "ll_abs_diff": abs(ll_h_old - stored["bic_table"]["H"]["log_likelihood"]),
        "bic_abs_diff": abs(bic_h_old - stored["bic_table"]["H"]["bic"]),
    }

    haz = collect_hazard_panel(
        tr_eng, mechanism, lam, traj_train, np.arange(100)
    )
    kap_ml, info = bernoulli_ml_fit(haz, out_all, config)
    kappa = float(kap_ml[0])
    ll_h_new = info["log_likelihood"]
    bic_h_new = math.log(n_obs) - 2.0 * ll_h_new
    bic_table["H_refit"] = {"k": 1, "log_likelihood": ll_h_new,
                            "bic": bic_h_new, "kappa": kappa}

    cand_old = {"SG": bic_table["SG"], "G": bic_table["G"],
                "MG": bic_table["MG"], "C2": bic_table["C2"],
                "H": bic_table["H_norefit"]}
    sel_old = min(LADDER, key=lambda m: cand_old[m]["bic"])
    cand_new = dict(cand_old)
    cand_new["H"] = bic_table["H_refit"]
    sel_new = min(LADDER, key=lambda m: cand_new[m]["bic"])

    result = {
        "dataset": name, "mechanism": mechanism, "lambda": lam,
        "seed0": seed0, "n_obs": n_obs,
        "stored_bic": {m: {"log_likelihood": stored["bic_table"][m]["log_likelihood"],
                           "bic": stored["bic_table"][m]["bic"]}
                       for m in LOWER + ("H",)},
        "reproduced_bic": bic_table,
        "reproduction_max_abs_diff": {
            m: max(d["ll_abs_diff"], d["bic_abs_diff"])
            for m, d in repro_diff.items()
        },
        "kappa_hat": kappa,
        "bic_selected_stored": stored["bic_selected"],
        "bic_selected_old_recomputed": sel_old,
        "bic_selected_with_refit": sel_new,
        "selection_changed": bool(sel_new != sel_old),
    }

    if sel_new != sel_old:
        if sel_new == "H":
            model = {"kind": "H_refit", "kappa": kappa}
        else:
            model = {"kind": "lower", "candidate": sel_new,
                     "h": np.asarray(bic_table[sel_new]["h"], dtype=float)}
        metrics = evaluate_selection(te_eng, model, mechanism, lam, seed0,
                                     union)
        result["new_selection_test"] = {"selected": sel_new, **metrics}
    result["seconds"] = time.perf_counter() - started
    return result

def main():
    import pandas as pd

    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    infra = ROOT / config["paths"]["infra_results"]
    calibration = json.loads((infra / "calibration.json").read_text())
    names = (config["datasets"]["contact_names"]
             + [config["datasets"]["email_name"]])

    tasks = [
        (name, mech, float(calibration[name][mech]["lambda"]), di, mi)
        for di, name in enumerate(names)
        for mi, mech in enumerate(MECHANISMS)
    ]
    started = time.perf_counter()
    with Pool(WORKERS) as pool:
        results = pool.map(run_setting, tasks)

    rows = []
    for res in results:
        write_json(
            OUT_DIR / ("setting_%s_%s.json" % (res["dataset"], res["mechanism"])),
            res,
        )
        rows.append({
            "dataset": res["dataset"], "mechanism": res["mechanism"],
            "n_obs": res["n_obs"], "kappa_hat": res["kappa_hat"],
            "ll_H_old": res["reproduced_bic"]["H_norefit"]["log_likelihood"],
            "ll_H_refit": res["reproduced_bic"]["H_refit"]["log_likelihood"],
            "bic_H_old": res["reproduced_bic"]["H_norefit"]["bic"],
            "bic_H_refit": res["reproduced_bic"]["H_refit"]["bic"],
            "bic_selected_stored": res["bic_selected_stored"],
            "bic_selected_with_refit": res["bic_selected_with_refit"],
            "selection_changed": res["selection_changed"],
            "max_repro_diff": max(res["reproduction_max_abs_diff"].values()),
            "new_test_w1": (res.get("new_selection_test") or {}).get("w1_attack_rate"),
            "new_test_accurate": (res.get("new_selection_test") or {}).get("accurate"),
            "seconds": res["seconds"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    write_json(OUT_DIR / "run_info.json", {
        "seed_base": BASE_SEED, "workers": WORKERS,
        "total_seconds": time.perf_counter() - started,
        "max_abs_reproduction_diff": float(df["max_repro_diff"].max()),
        "n_selection_changed": int(df["selection_changed"].sum()),
    })
    print(df.to_string(index=False))
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
