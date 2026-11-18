
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

FS_TICK, FS_LAB, FS_TITLE, FS_LEG, FS_ANN = 7.0, 8.0, 8.5, 7.0, 6.5

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": FS_TICK,
    "axes.labelsize": FS_LAB, "axes.titlesize": FS_TITLE,
    "legend.fontsize": FS_LEG,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.4, "savefig.facecolor": "white",
})

from src.experiments.ext_baselines_common import (  
    SparseEngine, load_config, load_dataset, simulate_batch, write_json,
)
from src.experiments.exp_ext_A_temporal import (  
    BASE_SEED, RUNS, build_variant_snapshots, sim_engine_list,
)

OUT = ROOT / "results" / "figures"
RES_C = ROOT / "results" / "external_baselines" / "C"
RES_A = ROOT / "results" / "external_baselines" / "A"

INK = "#27323D"
REP_COL = {
    "SG": "#A2ABB1", "G": "#5379AF", "MG": "#349F92",
    "C2": "#D45E78", "H": "#27323D", "HS": "#8664A6",
}
REP_LAB = {
    "SG": r"$SG$", "G": r"$G(t)$", "MG": r"$MG(t)$",
    "C2": r"$C_2(t)$", "H": r"$H(t)$", "HS": r"$H_S(t)$",
}
MECH_LAB = {"M1": "M1 pairwise", "M2": "M2 superlinear",
            "M3": "M3 threshold", "M4": "M4 SCM"}
DS_SHORT = {"contact-primary-school": "prim-school",
            "contact-high-school": "high-school", "hospital-lyon": "hospital",
            "invs15": "invs15", "sfhh-conference": "sfhh",
            "email-eu": "email-eu"}
DS_COL = {"contact-primary-school": "#5379AF",
          "contact-high-school": "#349F92",
          "hospital-lyon": "#C7836B",
          "invs15": "#D45E78",
          "sfhh-conference": "#8664A6",
          "email-eu": "#27323D"}
MECH_COL = {"M1": "#5379AF", "M2": "#349F92", "M3": "#C7836B",
            "M4": "#933548"}
MATRIX_COLS = (["Lucas", "Kirkley", "BIC"]
               + ["Ours(eps=%.2f)" % e for e in (0.01, 0.02, 0.05, 0.10)])
W1_FLOOR = 5e-4

def rep_of(method_row):
    sel = method_row["selected"]
    if sel.startswith("H_S"):
        return "HS"
    return sel

def acc_mark(ax, x, y, accurate):

    if accurate:
        ax.plot(x, y, marker="o", ms=3.2, ls="", mfc="white",
                mec=INK, mew=0.5, zorder=6)
    else:
        ax.plot(x, y, marker="x", ms=4.0, ls="", color=INK,
                mew=1.6, zorder=5)
        ax.plot(x, y, marker="x", ms=3.2, ls="", color="white",
                mew=1.1, zorder=6)

def figure_C():
    df = pd.read_csv(RES_C / "summary.csv")
    mechs = ["M1", "M2", "M3", "M4"]
    datasets = [d for d in DS_SHORT if d in set(df.dataset)]
    rows = [(d, m) for m in mechs for d in datasets]

    fig = plt.figure(figsize=(7.09, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30,
                          left=0.20, right=0.985, top=0.90, bottom=0.16)
    ax = fig.add_subplot(gs[0])
    n_r, n_c = len(rows), len(MATRIX_COLS)
    for ri, (d, m) in enumerate(rows):
        if ri % 6 == 0 and ri:
            ax.axhline(ri - 0.5, color="#C8D4D9", lw=0.8)
        sub = df[(df.dataset == d) & (df.mechanism == m)]
        for ci, col in enumerate(MATRIX_COLS):
            r = sub[sub.method == col].iloc[0]
            rep = rep_of(r)
            ax.add_patch(Rectangle((ci - 0.46, ri - 0.42), 0.92, 0.84,
                                   fc=REP_COL[rep], ec="white", lw=0.7))
            if rep == "HS":
                label = r["selected"].replace("H_S(", "").replace(")", "")

                ax.text(ci, ri - 0.12, label, ha="center",
                        va="center", fontsize=FS_ANN, color="white")
                acc_mark(ax, ci, ri + 0.26, r["accurate"])
            else:
                acc_mark(ax, ci, ri, r["accurate"])

    ytr = ax.get_yaxis_transform()  
    for gi, m in enumerate(mechs):
        y0, y1 = gi * 6 - 0.5, gi * 6 + 5.5
        ym = (y0 + y1) / 2
        ax.plot([-0.26, -0.26], [y0 + 0.25, y1 - 0.25], transform=ytr,
                color="#8A97A0", lw=0.7, clip_on=False)
        for yy in (y0 + 0.25, y1 - 0.25):
            ax.plot([-0.26, -0.24], [yy, yy], transform=ytr,
                    color="#8A97A0", lw=0.7, clip_on=False)
        ax.text(-0.285, ym, MECH_LAB[m], transform=ytr, fontsize=FS_ANN,
                ha="right", va="center", color=INK)
    ax.set_xlim(-0.55, n_c - 0.45)
    ax.set_ylim(n_r - 0.5, -0.5)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(
        ["Lucas", "Kirkley", "BIC", r"Ours" + "\n" + r"$\varepsilon$=.01",
         r"$\varepsilon$=.02", r"$\varepsilon$=.05", r"$\varepsilon$=.10"],
        fontsize=FS_TICK)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels([DS_SHORT[d] for d, _ in rows], fontsize=FS_TICK)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    handles = [Line2D([], [], ls="-", lw=5, solid_capstyle="butt",
                      color=REP_COL[k], label=REP_LAB[k]) for k in REP_COL]
    handles += [
        Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=INK,
               mew=0.5, label="accurate"),
        Line2D([], [], marker="x", ls="", ms=3.6, color=INK,
               mew=1.3, label="not accurate"),
    ]
    ax.legend(handles=handles, ncol=4, fontsize=FS_LEG, loc="upper center",
              bbox_to_anchor=(0.40, -0.105), frameon=False,
              columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)

    axb = fig.add_subplot(gs[1])
    floor = W1_FLOOR
    axb.axhline(0.02, color="#933548", lw=0.9, ls="--", zorder=2)
    axb.axhline(floor, color="#9FB2BC", lw=0.7, ls=":", zorder=1)

    ref_marker = {"SG": "v", "G": "s", "MG": "^", "C2": "D", "H": "o"}
    for rep in ("SG", "G", "MG", "C2", "H"):
        sub = df[df.method == "Always-%s" % rep]
        axb.scatter(sub.storage_rel_source.clip(lower=1e-3),
                    sub.w1_attack_rate.clip(lower=floor),
                    s=11, marker=ref_marker[rep], facecolor=REP_COL[rep],
                    edgecolor="none", alpha=0.45,
                    label="always %s" % REP_LAB[rep], zorder=2)

    ours = df[df.method.str.startswith("Ours")]
    axb.scatter(ours.storage_rel_source.clip(lower=1e-3),
                ours.w1_attack_rate.clip(lower=floor),
                s=20, marker="o", facecolor=INK, edgecolor="white", lw=0.4,
                label="Ours", zorder=6)
    for name, mk, c in (("Lucas", "s", "#8664A6"),
                        ("Kirkley", "D", "#C7836B"),
                        ("BIC", "^", "#417F90")):
        sub = df[df.method == name]
        axb.scatter(sub.storage_rel_source.clip(lower=1e-3),
                    sub.w1_attack_rate.clip(lower=floor),
                    s=18, marker=mk, facecolor=c, edgecolor="white",
                    lw=0.4, label=name, zorder=5)
        sub0 = df[df.method == name + "(no-refit)"]
        if len(sub0):
            axb.scatter(sub0.storage_rel_source.clip(lower=1e-3),
                        sub0.w1_attack_rate.clip(lower=floor),
                        s=18, marker=mk, facecolor="none", edgecolor=c,
                        lw=0.9, label=name + " no-refit", zorder=5)
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(5e-2, 5.0)
    axb.text(5.7e-2, 0.0255, r"$W_1=0.02$", fontsize=FS_ANN, color="#933548")
    axb.text(4.5, floor * 1.6, r"$W_1=0$ (exact)", fontsize=FS_ANN,
             color="#5B6B75", ha="right")
    axb.set_xlabel("storage relative to $H(t)$", fontsize=FS_LAB)
    axb.set_ylabel(r"$W_1$ (final attack rate)", fontsize=FS_LAB)
    axb.tick_params(labelsize=FS_TICK)
    axb.legend(fontsize=FS_LEG, loc="upper center", frameon=False, ncol=4,
               columnspacing=0.6, handletextpad=0.25, borderaxespad=0.0,
               bbox_to_anchor=(0.5, -0.105))

    posa, posb = ax.get_position(), axb.get_position()
    title_y = max(posa.y1, posb.y1) + 0.018
    fig.text(posa.x0 - 0.475 * posa.width, title_y,
             "a  selected representation",
             fontsize=FS_TITLE, ha="left", va="bottom")
    fig.text(posb.x0, title_y, "b  error vs storage cost",
             fontsize=FS_TITLE, ha="left", va="bottom")
    fig.savefig(OUT / "ext_C_selection.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ext_C_selection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def panel_a_curves(dataset, lam, seed0):

    n, t = dataset.n_nodes, dataset.n_snapshots
    rng = np.random.RandomState(seed0)
    uniforms = rng.random_sample((RUNS, t, n)).astype(np.float32)
    initial = np.zeros((RUNS, n), dtype=np.int8)
    seeds = np.random.RandomState(seed0 + 1).randint(n, size=RUNS)
    initial[np.arange(RUNS), seeds] = 1
    curves = {}
    for variant in ("V0", "V1", "V3", "V4b", "V5"):
        snap_lists = build_variant_snapshots(dataset, variant,
                                           perm_seed=seed0 + 5)
        traj = sim_engine_list(snap_lists, n, "M2", lam, initial, uniforms)
        dens = traj.mean(axis=2)
        curves[variant] = {
            "mean": dens.mean(axis=0),
            "lo": np.percentile(dens, 5, axis=0),
            "hi": np.percentile(dens, 95, axis=0),
        }
    return curves

VAR_GROUPS = [
    ("order perturbed",
     [("V1", "reversed", 1.9), ("V2", "global\nshuffle", 1.9),
      ("V3", "in-day\nshuffle*", 1.0)]),
    ("time aggregated",
     [("V4a", "1 h /\n7 d", 2.6), ("V4b", "1 d /\n28 d", 2.6),
      ("V5", "static\nweighted", 3.6), ("V6", "static\nunweighted", 2.0)]),
    ("exact mappings (M1 only)",
     [("V7", "static\ntheory", 2.0), ("V8", "$MG(t)$", 1.0)]),
]
VARIANTS = [v for _, grp in VAR_GROUPS for v, _, _ in grp]

def w1_panel(ax, sub, title):
    xpos, xlabels = [], []
    xi = 0.0
    for gi, (gname, members) in enumerate(VAR_GROUPS):
        x0 = xi
        for v, lab, step in members:
            sv = sub[(sub.variant == v) & sub.w1.notna()]
            rng = np.random.RandomState(len(xpos) + 7)
            for _, r in sv.iterrows():
                x = xi + rng.uniform(-0.26, 0.26)
                ax.plot(x, max(r.w1, W1_FLOOR), marker="o", ms=2.5,
                        ls="", color=MECH_COL[r.mechanism], alpha=0.65,
                        mew=0)
            if len(sv):
                med = max(float(sv.w1.median()), W1_FLOOR)
                ax.plot([xi - 0.34, xi + 0.34], [med, med], color=INK,
                        lw=1.3, zorder=6)
            xpos.append(xi)
            xlabels.append(lab)
            xi += step
        last_step = members[-1][2]

        ax.text((x0 + xi - last_step) / 2, -0.22, gname,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_ANN, color="#5B6B75")
        if gi < len(VAR_GROUPS) - 1:
            ax.axvline(xi - last_step / 2, color="#DDE5EA", lw=0.6, zorder=0)
        xi += 0.7  
    ax.set_yscale("log")
    ax.set_ylim(W1_FLOOR * 0.7, 1.0)
    ax.axhline(0.02, color="#933548", lw=0.8, ls="--")
    ax.axhline(W1_FLOOR, color="#9FB2BC", lw=0.6, ls=":")
    ax.text(-0.45, W1_FLOOR * 1.6, "exact", fontsize=FS_ANN, color="#5B6B75")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=FS_TICK)
    ax.set_xlim(-0.6, xi - 1.1)
    ax.set_ylabel(r"$W_1$")
    ax.set_title(title, loc="left", fontsize=FS_TITLE)
    ax.tick_params(length=0, axis="x")

def figure_A():
    df = pd.read_csv(RES_A / "summary.csv")
    config = load_config()
    cal = json.loads(
        (ROOT / config["paths"]["infra_results"] / "calibration.json"
         ).read_text())
    ds = load_dataset("contact-high-school")
    di = config["datasets"]["contact_names"].index("contact-high-school")
    lam = float(cal["contact-high-school"]["M2"]["lambda"])
    curves = panel_a_curves(ds, lam, BASE_SEED + di * 1000 + 97)
    hours = np.arange(len(curves["V0"]["mean"])) * 300.0 / 3600.0

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.1),
                             gridspec_kw={"width_ratios": [0.9, 1.6]})
    ax = axes[0][0]
    vc = {"V0": INK, "V1": "#5379AF", "V3": "#349F92",
          "V4b": "#C7836B", "V5": "#D45E78"}
    vlab = {"V0": "original $H(t)$", "V1": "reversed",
            "V3": "in-day shuffle", "V4b": "1-day windows",
            "V5": "static weighted"}
    c0 = curves["V0"]
    ax.fill_between(hours, c0["lo"], c0["hi"], color=INK, alpha=0.13, lw=0)
    for v in ("V0", "V1", "V3", "V4b", "V5"):
        ax.plot(hours, curves[v]["mean"], color=vc[v], lw=1.2,
                label=vlab[v])
    ax.set_xlabel("active hours")
    ax.set_ylabel("attack rate")
    ax.set_title("a  contact-high-school, M2, SI", loc="left",
                 fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEG, frameon=False, loc="upper left")

    si = df[df["mode"] == "si"]
    sis = df[df["mode"] == "sis"]
    w1_panel(axes[0][1], si, "b  SI: final-attack-rate $W_1$ vs V0")
    w1_panel(axes[1][1], sis, "d  SIS: final-prevalence $W_1$ vs V0")

    axes[0][1].legend(
        handles=[Line2D([], [], marker="o", ls="", ms=3.4,
                        color=MECH_COL[m], label=m) for m in MECH_COL]
                + [Line2D([], [], ls="-", lw=1.3, color=INK, label="median")],
        fontsize=FS_LEG, frameon=False, loc="upper left", ncol=3,
        bbox_to_anchor=(0.02, 0.99),
        columnspacing=0.7, handletextpad=0.25, borderaxespad=0.2)

    ax = axes[1][0]
    res_variants = [("V0", "snapshot"), ("V4a", "1 h / 7 d"),
                    ("V4b", "1 d / 28 d"), ("V5", "static")]
    for name, subd in si[si.mechanism == "M2"].groupby("dataset"):
        ys = []
        for v, _ in res_variants:
            vv = subd[subd.variant == v]
            ys.append(float(vv.w1.iloc[0]) if len(vv) else 0.0)
        ax.plot(range(4), np.clip(ys, W1_FLOOR, None), "o-", ms=3, lw=1.1,
                color=DS_COL[name], label=DS_SHORT[name])
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels([lab for _, lab in res_variants], fontsize=FS_TICK)
    ax.set_ylabel(r"$W_1$")
    ax.set_title("c  M2 SI: temporal resolution", loc="left",
                 fontsize=FS_TITLE)

    ax.legend(fontsize=FS_LEG, frameon=False, ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), columnspacing=0.7,
              handletextpad=0.25, borderaxespad=0.3)

    fig.tight_layout(h_pad=1.3, w_pad=2.0)
    fig.savefig(OUT / "ext_A_temporal.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ext_A_temporal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

RES_B = ROOT / "results" / "external_baselines" / "B"
B_METHOD_COL = {"Source-MC": INK, "MC-G": REP_COL["G"], "MC-MG": REP_COL["MG"],
                "MC-C2": REP_COL["C2"], "IBMF": "#8664A6", "MMCA": "#C7836B"}
B_METHOD_LAB = {"Source-MC": "source MC", "MC-G": r"$G(t)$ MC",
                "MC-MG": r"$MG(t)$ MC", "MC-C2": r"$C_2(t)$ MC",
                "IBMF": "IBMF", "MMCA": "MMCA"}

def b_curves_highschool():

    from src.experiments.ext_baselines_common import load_dataset
    from src.experiments.exp_ext_B_approx import (
        SEED_MC, SEED_OURS, SEED_REF, mc_stats, make_prob_source,
        make_prob_surrogate, mean_field_run)
    config = load_config()
    cal = json.loads(
        (ROOT / config["paths"]["infra_results"] / "calibration.json"
         ).read_text())
    ds = load_dataset("contact-high-school")
    di = config["datasets"]["contact_names"].index("contact-high-school")
    lam = float(cal["contact-high-school"]["M2"]["lambda"])
    engine = SparseEngine(ds.snapshots, ds.n_nodes)
    seed0 = 20261018 + di * 1000 + 97  
    n, T = ds.n_nodes, ds.n_snapshots

    curves = {}
    ref = mc_stats(make_prob_source(engine, "M2", lam),
                   5000, n, T, seed0 + SEED_REF, 0.0, "si")
    curves["reference"] = ref["curve"]
    h_map = {}
    sj = json.loads(
        (RES_B / "setting_contact-high-school_M2_si.json").read_text())
    for cand in ("G", "MG", "C2"):
        h_map[cand] = np.asarray(sj["fitted"][cand]["h"])
    ri = 5  
    for cand, h in h_map.items():
        est = mc_stats(make_prob_surrogate(engine, cand, h),
                       2000, n, T, seed0 + SEED_OURS + ri * 101, 0.0, "si")
        curves["MC-%s" % cand] = est["curve"]
    est = mc_stats(make_prob_source(engine, "M2", lam),
                   2000, n, T, seed0 + SEED_MC + ri * 101, 0.0, "si")
    curves["Source-MC"] = est["curve"]
    for name in ("ibmf", "mmca"):
        P = mean_field_run(ds.snapshots, n, "M2", lam, 0.0, name)
        curves[name.upper()] = P.mean(axis=1)
    return curves

def b_strip(ax, df_metric, methods, metric, title):
    rng = np.random.RandomState(3)
    for xi, mname in enumerate(methods):
        if mname in ("Source-MC", "MC-G", "MC-MG", "MC-C2"):
            sv = df_metric[(df_metric.method == mname)
                           & (df_metric.R == 2000)]
        else:
            sv = df_metric[df_metric.method == mname]
        for _, r in sv.iterrows():
            v = r[metric]
            if not np.isfinite(v):
                continue
            ax.plot(xi + rng.uniform(-0.26, 0.26), max(v, W1_FLOOR),
                    marker="o", ms=2.5, ls="", color=MECH_COL[r.mechanism],
                    alpha=0.65, mew=0)
        if len(sv):
            med = max(float(sv[metric].median()), W1_FLOOR)
            ax.plot([xi - 0.34, xi + 0.34], [med, med], color=INK,
                    lw=1.3, zorder=6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([B_METHOD_LAB[m] for m in methods], fontsize=FS_TICK)
    ax.set_title(title, loc="left", fontsize=FS_TITLE)
    ax.tick_params(length=0, axis="x")

def figure_B():
    df = pd.read_csv(RES_B / "summary.csv")
    si = df[df["mode"] == "si"]
    mc_methods = ["Source-MC", "MC-G", "MC-MG", "MC-C2"]
    all_methods = mc_methods + ["IBMF", "MMCA"]

    curves = b_curves_highschool()
    hours = np.arange(len(curves["reference"])) * 300.0 / 3600.0

    fig = plt.figure(figsize=(7.09, 6.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.85],
                          hspace=0.52, wspace=0.30,
                          left=0.09, right=0.985, top=0.94, bottom=0.09)

    axa = fig.add_subplot(gs[0, 0])
    order = [("reference", INK, 2.0, "-"), ("Source-MC", "#8A97A0", 1.0, "-"),
             ("IBMF", "#8664A6", 1.2, "--"), ("MMCA", "#C7836B", 1.2, "--"),
             ("MC-G", REP_COL["G"], 1.2, "-"), ("MC-MG", REP_COL["MG"], 1.2, "-"),
             ("MC-C2", REP_COL["C2"], 1.2, "-")]
    lab = {"reference": "reference $H(t)$", "Source-MC": "source MC",
           "IBMF": "IBMF", "MMCA": "MMCA", "MC-G": "$G(t)$ MC",
           "MC-MG": "$MG(t)$ MC", "MC-C2": "$C_2(t)$ MC"}
    for k, c, lw, ls in order:
        axa.plot(hours, curves[k], color=c, lw=lw, ls=ls, label=lab[k])
    axa.set_xlabel("active hours", fontsize=FS_LAB)
    axa.set_ylabel("attack rate", fontsize=FS_LAB)
    axa.set_title("a  contact-high-school, M2, SI", loc="left",
                  fontsize=FS_TITLE)
    axa.legend(fontsize=FS_LEG, frameon=False, loc="lower right", ncol=2)
    axa.tick_params(labelsize=FS_TICK)

    axd = fig.add_subplot(gs[0, 1])
    for mname in mc_methods:
        sub = si[(si.method == mname)]
        xs, ys = [], []
        for R in (10, 30, 100, 300, 1000, 2000):
            sR = sub[sub.R == R]
            xs.append(float(sR.seconds.median()))
            ys.append(float(sR.curve_mae.median()))
        axd.plot(xs, ys, "o-", ms=3, lw=1.0, color=B_METHOD_COL[mname],
                 label=B_METHOD_LAB[mname])
        for x, y, R in zip(xs, ys, (10, 30, 100, 300, 1000, 2000)):
            if R in (100, 2000):
                axd.annotate(str(R), (x, y), fontsize=FS_ANN,
                             textcoords="offset points", xytext=(3, 2),
                             color=B_METHOD_COL[mname])
    for mname in ("IBMF", "MMCA"):
        sub = si[si.method == mname]
        axd.plot(float(sub.seconds.median()), float(sub.curve_mae.median()),
                 marker="D" if mname == "IBMF" else "^", ms=5,
                 mfc=B_METHOD_COL[mname], mec="white", mew=0.5, ls="",
                 label=B_METHOD_LAB[mname], zorder=6)
    axd.set_xscale("log")
    axd.set_yscale("log")
    axd.set_xlabel("median runtime [s]", fontsize=FS_LAB)
    axd.set_ylabel("median curve MAE", fontsize=FS_LAB)
    axd.set_title("b  cost vs accuracy (SI)", loc="left", fontsize=FS_TITLE)
    axd.tick_params(labelsize=FS_TICK)
    axd.legend(fontsize=FS_LEG, frameon=False, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, -0.25), columnspacing=0.7,
               handletextpad=0.25, borderaxespad=0.0)

    axb = fig.add_subplot(gs[1, 0])
    b_strip(axb, si, all_methods, "curve_mae",
            "c  SI: curve MAE vs 5000-run reference")

    axc = fig.add_subplot(gs[1, 1])
    b_strip(axc, si, all_methods, "marginal_rmse",
            "d  SI: node-marginal RMSE")

    FLOOR_E = 0.05
    axe = fig.add_subplot(gs[2, :])
    lb = df[df.method.str.startswith("LB-")].copy()
    lb["cand"] = lb.method.str.replace("LB-", "", regex=False)
    cands = ["G", "MG", "C2"]
    exact = (lb.cand == "MG") & (lb.mechanism == "M1")
    for xi, cand in enumerate(cands):
        for mode, marker, off in (("si", "o", -0.13), ("sis", "s", 0.13)):
            sel = ((lb.cand == cand) & (lb["mode"] == mode))
            sv = lb[sel & ~exact]
            sx = lb[sel & exact]
            rng = np.random.RandomState(xi * 7 + (0 if mode == "si" else 5))
            for _, r in sv.iterrows():
                axe.plot(xi + off + rng.uniform(-0.10, 0.10),
                         np.clip(r.share, 1e-4, 1.1),
                         marker=marker, ms=2.8, ls="",
                         color=MECH_COL[r.mechanism], alpha=0.7, mew=0)
            for _ in range(len(sx)):
                axe.plot(xi + off + rng.uniform(-0.10, 0.10), FLOOR_E,
                         marker="*", ms=4.6, ls="",
                         color=MECH_COL["M1"], alpha=0.9, mew=0)
        med = float(lb[(lb.cand == cand) & ~exact].share.median())
        axe.plot([xi - 0.36, xi + 0.36], [med, med], color=INK, lw=1.3,
                 zorder=6)
    axe.axhline(1.0, color="#933548", lw=0.8, ls="--")
    axe.text(2.42, 1.005, "share = 1", fontsize=FS_ANN, color="#933548",
             va="bottom")
    axe.axhline(FLOOR_E, color="#8A97A0", lw=0.7, ls=":")
    axe.text(1.45, FLOOR_E * 1.05, "exact (M1)", fontsize=FS_ANN,
             color=MECH_COL["M1"], va="bottom")
    axe.set_yscale("log")
    axe.set_ylim(0.04, 1.4)
    axe.set_xticks(range(3))
    axe.set_xticklabels(["$G(t)$", "$MG(t)$", "$C_2(t)$"], fontsize=FS_TICK)
    axe.set_xlim(-0.6, 2.6)
    axe.set_ylabel(r"$\mathrm{LB}_\infty$ / max error", fontsize=FS_LAB)
    axe.set_title("e  information-loss share (same records)", loc="left",
                  fontsize=FS_TITLE)
    axe.tick_params(axis="x", length=0, labelsize=FS_TICK)
    handles = ([Line2D([], [], marker="o", ls="", ms=3.0, color="#5B6B75",
                      label="SI")]
               + [Line2D([], [], marker="s", ls="", ms=3.0, color="#5B6B75",
                         label="SIS")]
               + [Line2D([], [], marker="*", ls="", ms=4.6,
                         color=MECH_COL["M1"], label="exact")]
               + [Line2D([], [], marker="o", ls="", ms=3.0,
                         color=MECH_COL[m], label=m) for m in MECH_COL]
               + [Line2D([], [], ls="-", lw=1.3, color=INK, label="median")])

    axe.legend(handles=handles, fontsize=FS_LEG, frameon=False, ncol=8,
               loc="lower center", bbox_to_anchor=(0.5, 1.12),
               columnspacing=0.7, handletextpad=0.25, borderaxespad=0.0)

    fig.savefig(OUT / "ext_B_approx.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ext_B_approx.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def method_summary_latex():
    df = pd.read_csv(RES_C / "method_summary.csv")
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & Accurate & Median $W_1$ & Storage/$H$ & Runtime/$H$ & Params \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            "%s & %.2f & %.4f & %.2f & %.2f & %.1f \\\\" % (
                r["method"].replace("_", "\\_"),
                r["fraction_accurate"], r["median_w1"],
                r["median_storage_rel"], r["median_runtime_rel"],
                r["mean_n_params"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RES_C / "method_summary.tex").write_text("\n".join(lines),
                                             encoding="utf-8")

def b_table_latex():

    df = pd.read_csv(RES_B / "summary.csv")
    df = df[~df.method.str.startswith("LB-")].copy()
    rows = []
    for mode in ("si", "sis"):
        sub = df[df["mode"] == mode]
        for m in ("Source-MC", "MC-G", "MC-MG", "MC-C2", "IBMF", "MMCA"):
            if m in ("Source-MC", "MC-G", "MC-MG", "MC-C2"):
                s = sub[(sub.method == m) & (sub.R == 2000)]
            else:
                s = sub[sub.method == m]
            rows.append({
                "mode": mode.upper(), "method": m,
                "median_curve_mae": s.curve_mae.median(),
                "median_final_error": s.final_error.median(),
                "median_marginal_rmse": s.marginal_rmse.median(),
                "median_spearman": s.spearman.median(),
                "median_seconds": s.seconds.median(),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(RES_B / "method_summary.csv", index=False)
    lines = [
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        (r"Mode & Method & Curve MAE & Final err. & Marg.\ RMSE "
         r"& Spearman & Time [s] \\"),
        r"\midrule",
    ]
    for _, r in tab.iterrows():
        lines.append(
            "%s & %s & %.4f & %.4f & %.4f & %.3f & %.2f \\\\" % (
                r["mode"], r["method"].replace("_", "\\_"),
                r["median_curve_mae"], r["median_final_error"],
                r["median_marginal_rmse"], r["median_spearman"],
                r["median_seconds"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RES_B / "method_summary.tex").write_text("\n".join(lines),
                                            encoding="utf-8")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_C()
    print("fig_C_selection written")
    figure_A()
    print("fig_A_temporal written")
    method_summary_latex()
    print("method_summary.tex written")
    figure_B()
    print("fig_B_approx written")
    b_table_latex()
    print("method_summary_B.tex written")

if __name__ == "__main__":
    main()
