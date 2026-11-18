
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import triu

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import write_json  
from src.experiments.rev2_E7_group_size import (  
    SEED,
    SIZES,
    build_snapshots,
)

OUT_DIR = ROOT / "results" / "rev3" / "X4"
E7_SUMMARY = ROOT / "results" / "rev2" / "E7" / "summary.csv"

def storage_units_v2(snaps, model):

    if model == "H":
        return float(sum(s.multiplicities.sum() for s in snaps))
    if model == "G":
        return float(sum(triu(s.pair_binary, k=1).nnz for s in snaps))
    if model == "MG":
        return float(sum(triu(s.pair_weights, k=1).nnz for s in snaps))
    if model == "C2":
        return float(sum(s.cell_multiplicities.sum() for s in snaps))
    raise KeyError(model)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    e7 = pd.read_csv(E7_SUMMARY).set_index("m")

    rows = []
    for m in SIZES:
        snaps = build_snapshots(m, seed=SEED)
        row = {"m": int(m)}
        for model in ("H", "G", "MG", "C2"):
            old = float(e7.loc[m, "storage_%s" % model])
            new = storage_units_v2(snaps, model)
            row["storage_%s_old" % model] = old
            row["storage_%s" % model] = new
            row["storage_ratio_%s_H" % model] = (
                new / float(e7.loc[m, "storage_H"]))
            row["sim_s_%s" % model] = float(e7.loc[m, "sim_s_%s" % model])
            row["sim_ratio_%s_H" % model] = (
                1.0 if model == "H" else
                float(e7.loc[m, "sim_ratio_%s_H" % model]))

        for model in ("G", "MG"):
            mat = snaps[0].pair_binary if model == "G" else snaps[0].pair_weights
            assert mat.nnz == 2 * triu(mat, k=1).nnz, "pair matrix not symmetric"
        rows.append(row)
        print("m=%d  " % m + "  ".join(
            "%s=%.0f(ratio %.3f)" % (model, row["storage_%s" % model],
                                    row["storage_ratio_%s_H" % model])
            for model in ("H", "G", "MG", "C2")), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    write_json(OUT_DIR / "run_info.json", {
        "seed": SEED,
        "definition": ("undirected nonzero pairs for G/MG; generated cells "
                       "with multiplicity for C2; event multiplicities for H"),
        "runtimes_copied_from": "results/rev2/E7/summary.csv",
        "total_seconds": time.perf_counter() - started,
    })
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
