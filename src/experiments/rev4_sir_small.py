
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "rev4" / "sir_small"
N = 6
N_SNAP = 8
GAMMA = 0.15
T_MAX = 128
REPORT_T = (1, 2, 4, 8, 16, 32, 64, 128)
N_NET = 10
LAM = {"M1": 0.15, "M2": 0.08, "M3": 0.35, "M4": 0.06}

def hazard(mech, lam, m, chi):
    if mech == "M1":
        return lam * chi
    if mech == "M2":
        return lam * chi * chi
    if mech == "M3":
        return lam if chi / (m - 1.0) >= 0.5 else 0.0
    if mech == "M4":
        return lam * chi + 8.0 * lam * chi * (chi - 1) / 2.0
    raise KeyError(mech)

def random_network(rng, max_size):
    snaps = []
    for _ in range(N_SNAP):
        events = []
        for _ in range(rng.randint(3, 6)):
            m = rng.randint(2, max_size + 1)
            h = tuple(sorted(rng.choice(N, m, replace=False)))
            events.append((h, int(rng.randint(1, 3))))
        snaps.append(events)
    return snaps

def source_p(events, x, mech, lam):

    haz = np.zeros(x.shape)
    for h, c in events:
        idx = list(h)
        chi_h = x[:, idx].sum(axis=1)
        for i in idx:
            chi = chi_h - x[:, i]
            haz[:, i] += c * np.array([hazard(mech, lam, len(h), int(k)) for k in chi])
    return -np.expm1(-haz)

def features(events, x, cand):

    S = x.shape[0]
    if cand in ("G", "MG"):
        W = np.zeros((N, N))
        for h, c in events:
            for i, j in itertools.combinations(h, 2):
                W[i, j] += c
                W[j, i] += c
        if cand == "G":
            W = (W > 0).astype(float)
        return (x @ W.T)[:, :, None]
    z1 = np.zeros((S, N))
    z2 = np.zeros((S, N))
    for h, c in events:
        cells = [h] if len(h) <= 3 else list(itertools.combinations(h, 3))
        for cell in cells:
            idx = list(cell)
            chi_c = x[:, idx].sum(axis=1)
            for i in idx:
                chi = chi_c - x[:, i]
                z1[:, i] += c * chi
                z2[:, i] += c * chi * (chi - 1) / 2.0
    return np.stack([z1, z2], axis=2)

def predict(feat, theta):
    return -np.expm1(-np.einsum("snf,f->sn", feat, theta))

def fit_neglog(events, X2, mech, lam, cand):

    p = source_p(events, X2, mech, lam)
    f = features(events, X2, cand)
    mask = (X2 == 0) & (X2.sum(axis=1, keepdims=True) >= 1)
    A = f[mask]
    y = -np.log1p(-np.clip(p[mask], 0.0, 1.0 - 1e-12))
    if A.shape[0] == 0 or not np.any(A):
        return np.zeros(f.shape[2])
    return lsq_linear(A, y, bounds=(0.0, -math.log(0.05))).x

def kernel(states, p, process):

    k = 3 if process == "sir" else 2
    X = states.shape[0]
    v = np.zeros((X, N, k))
    sus = states == 0
    inf = states == 1
    v[:, :, 0] = np.where(sus, 1.0 - p, 0.0)
    v[:, :, 1] = np.where(sus, p, 0.0) + np.where(inf, 1.0 - GAMMA, 0.0)
    if process == "sir":
        v[:, :, 2] = np.where(inf, GAMMA, 0.0) + (states == 2)
    else:
        v[:, :, 0] += np.where(inf, GAMMA, 0.0)
    K = np.ones((X, X))
    for i in range(N):
        K *= v[:, i, :][:, states[:, i]]
    return K

def process_kernels(snaps, mech, lam, process, cand=None, theta=None):
    k = 3 if process == "sir" else 2
    states = np.array(list(itertools.product(range(k), repeat=N)), dtype=np.int64)
    x = (states == 1).astype(float)
    out = []
    for t, events in enumerate(snaps):
        p = source_p(events, x, mech, lam) if cand is None else \
            predict(features(events, x, cand), theta[t])
        out.append(kernel(states, p, process))
    return states, out

def initial(states):
    mu = ((states == 1).sum(axis=1) == 1) & ((states == 0).sum(axis=1) == N - 1)
    return mu / mu.sum()

def run_bounds(KH, KM, mu0):
    rowtv = [0.5 * np.abs(a - b).sum(axis=1) for a, b in zip(KH, KM)]
    mh, mm = mu0.copy(), mu0.copy()
    reach = mu0 > 0
    dist_b = worst_b = 0.0
    rec = {}
    for T in range(1, T_MAX + 1):
        s = (T - 1) % N_SNAP
        dist_b += float(mh @ rowtv[s])
        worst_b += float(rowtv[s][reach].max())
        mh = mh @ KH[s]
        mm = mm @ KM[s]
        reach = mh > 1e-15
        if T in REPORT_T:
            rec[T] = {"tv": float(0.5 * np.abs(mh - mm).sum()),
                      "dist_bound": min(1.0, dist_b), "worst_bound": min(1.0, worst_b),
                      "source_active": float(mh[(np.array(STATE_CACHE[len(mh)]) == 1).any(axis=1)].sum())}
    return rec

STATE_CACHE = {}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (2, 3):
        st = np.array(list(itertools.product(range(k), repeat=N)), dtype=np.int64)
        STATE_CACHE[len(st)] = st.tolist()
    X2 = np.array(list(itertools.product((0, 1), repeat=N)), dtype=float)

    exact = []
    for net in range(N_NET):
        for mech, cand, max_size in (("M1", "MG", 4), ("M4", "C2", 3)):
            rng = np.random.RandomState(1000 + 17 * net + (0 if mech == "M1" else 7))
            snaps = random_network(rng, max_size)
            lam = LAM[mech]
            if cand == "MG":
                theta = [np.array([lam]) for _ in snaps]
            else:
                theta = [np.array([lam, 8.0 * lam]) for _ in snaps]
            states, KH = process_kernels(snaps, mech, lam, "sir")
            _, KM = process_kernels(snaps, mech, lam, "sir", cand, theta)
            mu0 = initial(states)
            mh, mm = mu0.copy(), mu0.copy()
            max_row, max_dist = 0.0, 0.0
            for T in range(T_MAX):
                s = T % N_SNAP
                reach = mh > 0
                max_row = max(max_row, float(0.5 * np.abs(KH[s] - KM[s]).sum(axis=1)[reach].max()))
                mh, mm = mh @ KH[s], mm @ KM[s]
                max_dist = max(max_dist, float(0.5 * np.abs(mh - mm).sum()))
            exact.append({"net": net, "mechanism": mech, "target": cand,
                          "max_row_tv_reachable": max_row, "max_state_tv": max_dist})
    json.dump(exact, open(OUT / "exact_sir.json", "w"), indent=1)
    print("Part A: max row TV %.2e, max state-distribution TV %.2e over %d cases" % (
        max(e["max_row_tv_reachable"] for e in exact),
        max(e["max_state_tv"] for e in exact), len(exact)), flush=True)

    rows = []
    for net in range(N_NET):
        rng = np.random.RandomState(5000 + 31 * net)
        snaps = random_network(rng, 4)
        for mech in ("M2", "M3", "M4"):
            lam = LAM[mech]
            thetas = {c: [fit_neglog(ev, X2, mech, lam, c) for ev in snaps]
                      for c in ("G", "MG", "C2")}
            for process in ("sis", "sir"):
                states, KH = process_kernels(snaps, mech, lam, process)
                mu0 = initial(states)
                for cand in ("G", "MG", "C2"):
                    _, KM = process_kernels(snaps, mech, lam, process, cand, thetas[cand])
                    rec = run_bounds(KH, KM, mu0)
                    for T, r in rec.items():
                        rows.append({"net": net, "mechanism": mech, "process": process,
                                     "candidate": cand, "T": T, **r})
        print("net %d done" % net, flush=True)
    json.dump(rows, open(OUT / "bounds.json", "w"), indent=1)

if __name__ == "__main__":
    main()
