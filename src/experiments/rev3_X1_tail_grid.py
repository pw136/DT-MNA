
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import write_json  
from src.experiments.rev2_E4_tail_settings import (  
    CONFLICT_LEVELS,
    OUT_DIR as E4_DIR,
    SEED,
    STORED,
    conflict_scan,
)

OUT_DIR = ROOT / "results" / "rev3" / "X1"
FIG_DIR = ROOT / "results" / "figures"

ALPHAS = (0.95, 0.90, 0.80)
ETAS = (0.0, 0.1, 0.2, 0.35, 0.5)

def reproduction_check():
    df_orig, sum_orig = conflict_scan(0.90, 0.35)
    stored = json.loads(STORED.read_text())["A1"]["summary"]
    idx = {(float(r["conflict"]), r["method"]): r for r in stored}
    diffs = [
        max(abs(row["mean_mae"] - s["mean_mae"]),
            abs(row["mean_worst"] - s["mean_worst"]),
            abs(row["mean_lower_bound"] - s["mean_lower_bound"]))
        for _, row in sum_orig.iterrows()
        if (s := idx.get((float(row["conflict"]), row["method"]))) is not None
    ]
    df_new, sum_new = conflict_scan(0.95, 0.20)
    return float(max(diffs)), df_orig, sum_orig, df_new, sum_new

def tail_grid():
    rows = []
    summaries = []
    for alpha in ALPHAS:
        for eta in ETAS:
            df, summ = conflict_scan(alpha, eta)
            d = df[df["method"] == "Dynamics aware"].copy()
            d["alpha"] = alpha
            d["eta"] = eta
            rows.append(d)
            s = summ[summ["method"] == "Dynamics aware"].copy()
            s["alpha"] = alpha
            s["eta"] = eta
            s["worst_minus_lb"] = s["mean_worst"] - s["mean_lower_bound"]
            summaries.append(s)
    long_df = pd.concat(rows, ignore_index=True)
    grid = pd.concat(summaries, ignore_index=True)
    return long_df, grid

def rebuild_figure(sum_new):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'Arial', 'mathtext.fontset': 'stix',
        'font.size': 8.2, 'axes.labelsize': 8.5, 'axes.titlesize': 8.7,
        'legend.fontsize': 7.2, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'xtick.major.width': .6,
        'ytick.major.width': .6, 'lines.linewidth': 1.4,
        'savefig.facecolor': 'white'})
    COL = {'G-time': '#5379AF', 'MG-time': '#349F92',
           'C2-time': '#D45E78', 'H': '#27323D'}

    def title(a, l, t):
        a.set_title(t, loc='left', pad=8, fontweight='normal')
        a.text(-.15, 1.065, l, transform=a.transAxes,
               weight='bold', fontsize=10.5, va='bottom')

    def clean(a, axis='y'):
        a.grid(axis=axis, lw=.45, color='#E5E8EA')
        a.set_axisbelow(True)

    d = sum_new
    cap = pd.read_csv(
        ROOT / 'results' / 'figure_data' / 'capacity_inputs.csv')
    f, ax = plt.subplots(1, 3, figsize=(7.8, 2.85))
    f.subplots_adjust(left=.075, right=.985, bottom=.19, top=.87, wspace=.4)
    a = ax[0]
    for method, c, lab in [('Direct copy', '#75838A', 'Direct copy'),
                           ('Hazard projection', COL['MG-time'], 'Hazard fit'),
                           ('Dynamics aware', COL['C2-time'], 'Dynamics-aware')]:
        g = d[d.method == method]
        a.fill_between(g.conflict, np.maximum(0, g.mean_mae - g.sd_mae),
                       g.mean_mae + g.sd_mae, color=c, alpha=.13, lw=0)
        a.plot(g.conflict, g.mean_mae, '-o', c=c, ms=2.6, label=lab)
    a.set(xlabel='Conflict strength', ylabel='Node-probability MAE',
          xlim=(-.004, .167), ylim=(0, .34), xticks=[0, .05, .1, .15])
    clean(a); title(a, 'a', 'Accuracy under conflict')
    a.legend(fontsize=7.1, loc='upper left', frameon=False)
    a = ax[1]
    g = d[d.method == 'Dynamics aware']
    a.fill_between(g.conflict, 0, g.mean_lower_bound, color='#9EC6BB',
                   alpha=.6, lw=0, label='Collision lower bound')
    a.fill_between(g.conflict, g.mean_lower_bound, g.mean_worst,
                   color='#E8ABBA', alpha=.7, lw=0,
                   label='Error above the bound')
    a.plot(g.conflict, g.mean_worst, color=COL['C2-time'], marker='o',
           ms=2.6, label='Dynamics-aware error')
    a.set(xlabel='Conflict strength', ylabel='Mean maximum error',
          xlim=(-.004, .167), ylim=(0, .57), xticks=[0, .05, .1, .15])
    clean(a); title(a, 'b', 'Lower bound and residual')
    a.legend(fontsize=6.8, loc='upper left', frameon=False)
    a = ax[2]
    for pos, rep, label, c in [
            (3, 'G', r'$\psi_G$', COL['G-time']),
            (2, 'MG', r'$\psi_{MG}$', COL['MG-time']),
            (1, 'SC2', r'$\psi_{\mathrm{grp}}$', COL['C2-time'])]:
        g = cap[(cap.representation == rep)
                & (cap.model == 'Target-only MLP')
                ].sort_values(['conflict', 'seed'])
        yy = pos + np.linspace(-.23, .23, len(g))
        a.hlines(yy, g.collision_lower_bound, g.worst_error,
                 color=c, alpha=.32, lw=.65)
        a.scatter(g.collision_lower_bound, yy, s=10,
                  facecolor='white', edgecolor=c, lw=.6)
        a.scatter(g.worst_error, yy, s=10, color=c, lw=0)
    g = cap[(cap.representation == 'G')
            & (cap.model == 'Source-aware MLP')
            ].sort_values(['conflict', 'seed'])
    a.scatter(g.worst_error, np.linspace(-.23, .23, len(g)), s=10,
              color=COL['H'], lw=0)
    a.set(xlabel='Maximum probability error', xlim=(-.025, .85),
          ylim=(-.55, 4.35), yticks=[0, 1, 2, 3],
          yticklabels=[r'$\psi_H$', r'$\psi_{\mathrm{grp}}$',
                       r'$\psi_{MG}$', r'$\psi_G$'],
          xticks=[0, .4, .8])
    clean(a, 'x'); title(a, 'c', 'Information retained in inputs')
    h = [Line2D([], [], marker='o', mfc='white', mec=COL['H'], ls='', ms=3,
                label='Lower bound'),
         Line2D([], [], marker='o', color=COL['H'], ls='', ms=3,
                label='MLP error')]
    a.legend(handles=h, ncol=2, fontsize=6.8, loc='upper right', frameon=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f.savefig(FIG_DIR / 'images2.pdf',
              bbox_inches='tight', pad_inches=.045)
    f.savefig(FIG_DIR / 'images2.png', dpi=240,
              bbox_inches='tight', pad_inches=.045)
    plt.close(f)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    max_diff, df_orig, sum_orig, df_new, sum_new = reproduction_check()
    df_orig.to_csv(OUT_DIR / "rows_tail90_035.csv", index=False)
    sum_orig.to_csv(OUT_DIR / "summary_tail90_035.csv", index=False)
    df_new.to_csv(OUT_DIR / "rows_tail95_020.csv", index=False)
    sum_new.to_csv(OUT_DIR / "summary_tail95_020.csv", index=False)

    long_df, grid = tail_grid()
    long_df.to_csv(OUT_DIR / "tail_grid_long.csv", index=False)
    grid.to_csv(OUT_DIR / "tail_grid_summary.csv", index=False)

    rebuild_figure(sum_new)

    write_json(OUT_DIR / "run_info.json", {
        "seed": SEED, "replicates": 30,
        "conflict_levels": list(CONFLICT_LEVELS),
        "alphas": list(ALPHAS), "etas": list(ETAS),
        "reproduction_max_abs_diff_vs_stored": max_diff,
        "total_seconds": time.perf_counter() - started,
    })
    print("repro diff: %.3e" % max_diff)
    print(grid[grid.conflict == 0.16].to_string(index=False))
    print("done in %.1fs" % (time.perf_counter() - started), flush=True)

if __name__ == "__main__":
    main()
