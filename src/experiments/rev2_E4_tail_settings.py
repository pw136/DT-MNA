
from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import write_json  
from src.approximate.feasibility import (  
    projection_collision_lower_bound,
)
from src.approximate.mapping import (  
    fit_dynamics_aware_pairwise,
    fit_pairwise_hazard,
    pairwise_effect,
)
from src.experiments.innovation2_core import (  
    _effect_metrics,
    check_scalar_qstar_feasibility,
)

OUT_DIR = ROOT / "results" / "rev2" / "E4"
STORED = (ROOT / "results" / "summary_tables"
          / "innovation2_approximate_transformability.json")

CONFLICT_LEVELS = (0.0, 0.01, 0.03, 0.05, 0.1, 0.16)
REPLICATES = 30
SEED = 20260726  

METHODS = (
    "Direct copy",
    "Arithmetic mean",
    "Frequency weighted",
    "Hazard projection",
    "Dynamics aware",
)

def conflict_scan(tail_quantile, tail_weight):

    rng = np.random.RandomState(SEED)
    rows = []
    for conflict in CONFLICT_LEVELS:
        for replicate in range(REPLICATES):
            channel_count = 18
            exposures = rng.randint(1, 5, size=channel_count).astype(float)
            weights = np.exp(rng.normal(0.0, 0.9, size=channel_count))
            direction = rng.normal(size=channel_count)
            direction = (direction - direction.mean()) / direction.std()
            required = np.clip(
                0.06 + float(conflict) * direction, 0.002, 0.38
            )
            effects = pairwise_effect(required, exposures)
            collision_lb, _ = projection_collision_lower_bound(
                effects, tuple(int(v) for v in exposures)
            )
            beta_values = OrderedDict()
            beta_values["Direct copy"] = float(required[0])
            beta_values["Arithmetic mean"] = float(required.mean())
            beta_values["Frequency weighted"] = float(
                np.average(required, weights=weights)
            )
            beta_values["Hazard projection"] = float(
                fit_pairwise_hazard(effects, exposures, weights).parameters
            )
            beta_values["Dynamics aware"] = float(
                fit_dynamics_aware_pairwise(
                    effects, exposures, weights,
                    tail_weight=tail_weight,
                    tail_quantile=tail_quantile,
                ).parameters
            )
            for method in METHODS:
                metrics = _effect_metrics(
                    beta_values[method], effects, exposures, weights
                )
                rows.append({
                    "conflict": float(conflict), "replicate": replicate,
                    "method": method, "beta": beta_values[method],
                    "mae": metrics["mae"], "rmse": metrics["rmse"],
                    "worst": metrics["worst"],
                    "collision_lower_bound": float(collision_lb),
                })
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["conflict", "method"], sort=True)
        .agg(mean_mae=("mae", "mean"), sd_mae=("mae", "std"),
             mean_worst=("worst", "mean"),
             mean_lower_bound=("collision_lower_bound", "mean"))
        .reset_index()
    )
    return df, summary

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    df_orig, sum_orig = conflict_scan(0.90, 0.35)
    df_orig.to_csv(OUT_DIR / "rows_tail90_035.csv", index=False)
    sum_orig.to_csv(OUT_DIR / "summary_tail90_035.csv", index=False)

    stored = json.loads(STORED.read_text())["A1"]["summary"]
    stored_idx = {
        (r["conflict"], r["method"]): r for r in stored
    }
    diffs = []
    for _, row in sum_orig.iterrows():
        key = (float(row["conflict"]), row["method"])
        if key in stored_idx:
            s = stored_idx[key]
            diffs.append(max(
                abs(row["mean_mae"] - s["mean_mae"]),
                abs(row["mean_worst"] - s["mean_worst"]),
                abs(row["mean_lower_bound"] - s["mean_lower_bound"]),
            ))
    max_diff = float(max(diffs))

    df_new, sum_new = conflict_scan(0.95, 0.20)
    df_new.to_csv(OUT_DIR / "rows_tail95_020.csv", index=False)
    sum_new.to_csv(OUT_DIR / "summary_tail95_020.csv", index=False)

    write_json(OUT_DIR / "run_info.json", {
        "seed": SEED, "replicates": REPLICATES,
        "conflict_levels": list(CONFLICT_LEVELS),
        "tail_original": {"tail_quantile": 0.90, "tail_weight": 0.35},
        "tail_new": {"tail_quantile": 0.95, "tail_weight": 0.20},
        "reproduction_max_abs_diff_vs_stored": max_diff,
        "total_seconds": time.perf_counter() - started,
    })

    key16 = sum_orig[sum_orig["conflict"] == 0.16]
    key16n = sum_new[sum_new["conflict"] == 0.16]
    print("original (0.16):"); print(key16.to_string(index=False))
    print("new tail (0.16):"); print(key16n.to_string(index=False))
    print("max repro diff vs stored: %.3e" % max_diff)
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
