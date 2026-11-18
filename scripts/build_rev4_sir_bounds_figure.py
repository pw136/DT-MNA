
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MS_FIGS = ROOT / "results" / "figures"
FS_TICK, FS_LAB, FS_TITLE, FS_LEG = 7.0, 8.0, 8.5, 7.0
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": FS_TICK, "axes.labelsize": FS_LAB, "axes.titlesize": FS_TITLE,
    "legend.fontsize": FS_LEG, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.4, "savefig.facecolor": "white",
})
COL = {"sis": "#5379AF", "sir": "#D45E78"}
LAB = {"sis": "SIS", "sir": "SIR"}
INK = "#27323D"

r = pd.DataFrame(json.load(open(ROOT / "results/rev4/sir_small/bounds.json")))
Ts = sorted(r["T"].unique())

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), constrained_layout=True)

ax = axes[0]
sub = r[(r.candidate == "C2") & r.mechanism.isin(["M2", "M4"])]
for proc in ("sis", "sir"):
    d = sub[sub.process == proc].groupby("T")
    ax.plot(Ts, d.dist_bound.median().reindex(Ts), color=COL[proc], marker="o",
            ms=3, label="%s bound" % LAB[proc])
    ax.plot(Ts, d.tv.median().reindex(Ts), color=COL[proc], ls="--", marker="s",
            ms=2.6, label="%s realized TV" % LAB[proc])
ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks(Ts)
ax.set_xticklabels([str(t) for t in Ts])
ax.set_xlabel("Horizon $T$")
ax.set_ylabel("Terminal TV distance")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="upper left", handlelength=2.0)
ax.set_title(r"$\mathbf{a}$  $C_2(t)$ under M2 and M4", loc="left")

ax = axes[1]
for proc in ("sis", "sir"):
    d = r[r.process == proc].groupby("T")
    ax.plot(Ts, d.dist_bound.apply(lambda s: (s < 1 - 1e-12).mean()).reindex(Ts),
            color=COL[proc], marker="o", ms=3, label="%s, bound $<1$" % LAB[proc])
    ax.plot(Ts, d.dist_bound.apply(lambda s: (s <= 0.2).mean()).reindex(Ts),
            color=COL[proc], ls="--", marker="s", ms=2.6,
            label=r"%s, bound $\leq 0.2$" % LAB[proc])
ax.set_xscale("log", base=2)
ax.set_xticks(Ts)
ax.set_xticklabels([str(t) for t in Ts])
ax.set_xlabel("Horizon $T$")
ax.set_ylabel("Share of comparisons")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="upper right")
ax.set_title(r"$\mathbf{b}$  All 90 comparisons", loc="left")

ax = axes[2]
for proc in ("sis", "sir"):
    d = r[(r.process == proc) & (r.candidate == "G")].groupby("T")
    ax.plot(Ts, d.source_active.median().reindex(Ts), color=COL[proc], marker="o",
            ms=3, label=LAB[proc])
ax.set_xscale("log", base=2)
ax.set_xticks(Ts)
ax.set_xticklabels([str(t) for t in Ts])
ax.set_xlabel("Horizon $T$")
ax.set_ylabel("P(source has infected nodes)")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="lower left")
ax.set_title(r"$\mathbf{c}$  Activity of the source", loc="left")

MS_FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(MS_FIGS / "sir_bounds.pdf")
fig.savefig(MS_FIGS / "sir_bounds.png", dpi=200)
print("saved")
