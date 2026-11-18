
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "external_baselines.json"

MECHANISMS = ("M1", "M2", "M3", "M4")
LOWER_CANDIDATES = ("SG", "G", "MG", "C2")

def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, default=_plain, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(type(obj).__name__)

class DatasetError(RuntimeError):
    pass

def _parse_timestamp(value):

    if value is None:
        raise DatasetError("dataset edge lacks a timestamp")
    if isinstance(value, (int, float)):
        return float(value)
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return (dt - datetime(1970, 1, 1)).total_seconds()

def fetch_xgi_hypergraph(name, raw_dir):

    import xgi

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / ("%s.json" % name)
    if not raw_path.exists():
        try:
            xgi.download_xgi_data(name, path=str(raw_dir))
        except Exception as exc:  
            raise DatasetError("cannot download %r: %s" % (name, exc)) from exc
    if not raw_path.exists():
        raise DatasetError("download of %r produced no local file" % name)
    return xgi.load_xgi_data(name, read=True, path=str(raw_dir), max_order=None)

def load_events(name, raw_dir):

    config = load_config()
    pre = config["preprocessing"]
    hypergraph = fetch_xgi_hypergraph(name, raw_dir)
    attrs = hypergraph.edges.attrs.asdict()
    total = 0
    events = []
    for edge_id in hypergraph.edges:
        members = tuple(sorted(hypergraph.edges.members(edge_id)))
        stamp = _parse_timestamp(attrs.get(str(edge_id), {}).get("timestamp"))
        total += 1
        if pre["min_hyperedge_size"] <= len(members) <= pre["max_hyperedge_size"]:
            events.append((stamp, members))
    if not events:
        raise DatasetError("no usable events in %r" % name)
    dropped = 1.0 - len(events) / float(total)
    events.sort(key=lambda item: item[0])
    return events, dropped

@dataclass
class SnapshotData:

    window_index: int
    day_index: int
    hyperedges: tuple  
    multiplicities: np.ndarray  
    sizes: np.ndarray  
    incidence: csr_matrix  
    pair_weights: csr_matrix  
    pair_binary: csr_matrix  
    cell_incidence: csr_matrix = None  
    cell_multiplicities: np.ndarray = None  

    @property
    def n_unique(self):
        return len(self.hyperedges)

@dataclass
class TemporalDataset:
    name: str
    nodes: tuple
    node_to_index: dict
    snapshots: list  
    delta_seconds: float
    size_min: int
    size_max: int
    first_timestamp: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_snapshots(self):
        return len(self.snapshots)

    @property
    def n_days(self):
        return len({s.day_index for s in self.snapshots})

    def to_temporal_hypergraph(self):

        from ..temporal_structures import TemporalHypergraph

        snapshots = {}
        for idx, snap in enumerate(self.snapshots):
            edges = []
            for edge, count in zip(snap.hyperedges, snap.multiplicities):
                edges.extend([tuple(self.nodes[i] for i in edge)] * int(count))
            snapshots[idx] = tuple(edges)
        return TemporalHypergraph(self.nodes, snapshots)

def _snapshot_core(index_counter, n_nodes, window_index, day_index,
                   label_hyperedges=None):

    index_hyperedges = tuple(sorted(index_counter.keys()))
    cvec = np.array(
        [index_counter[h] for h in index_hyperedges], dtype=np.float64
    )
    sizes = np.array([len(h) for h in index_hyperedges], dtype=np.int64)

    rows, cols = [], []
    for row, edge in enumerate(index_hyperedges):
        for node in edge:
            rows.append(row)
            cols.append(node)
    incidence = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(index_hyperedges), n_nodes),
        dtype=np.float64,
    )

    prow, pcol, pval = [], [], []
    for edge, count in zip(index_hyperedges, cvec):
        for a, b in combinations(edge, 2):
            prow.append(a)
            pcol.append(b)
            pval.append(float(count))
    coo = csr_matrix(
        (pval + pval, (prow + pcol, pcol + prow)),
        shape=(n_nodes, n_nodes),
        dtype=np.float64,
    )
    pair_binary = csr_matrix(
        (np.ones(coo.nnz), coo.nonzero()), shape=coo.shape, dtype=np.float64
    )

    cells = defaultdict(float)
    for edge, count in zip(index_hyperedges, cvec):
        if len(edge) <= 3:
            cells[edge] += float(count)
        else:
            for cell in combinations(edge, 3):
                cells[cell] += float(count)
    unique_cells = tuple(sorted(cells.keys()))
    cell_mult = np.array([cells[c] for c in unique_cells], dtype=np.float64)
    crows, ccols = [], []
    for row, cell in enumerate(unique_cells):
        for node in cell:
            crows.append(row)
            ccols.append(node)
    cell_incidence = csr_matrix(
        (np.ones(len(crows)), (crows, ccols)),
        shape=(len(unique_cells), n_nodes),
        dtype=np.float64,
    )

    return SnapshotData(
        window_index=window_index,
        day_index=day_index,
        hyperedges=(
            tuple(index_hyperedges) if label_hyperedges is None
            else tuple(label_hyperedges)
        ),
        multiplicities=cvec,
        sizes=sizes,
        incidence=incidence,
        pair_weights=coo,
        pair_binary=pair_binary,
        cell_incidence=cell_incidence,
        cell_multiplicities=cell_mult,
    )

def _build_snapshot(window_index, day_index, counter, node_to_index, n_nodes):
    hyperedges = tuple(sorted(counter.keys()))
    index_counter = {
        tuple(node_to_index[v] for v in h): counter[h] for h in hyperedges
    }
    return _snapshot_core(
        index_counter, n_nodes, window_index, day_index,
        label_hyperedges=hyperedges,
    )

def snapshot_index_edges(snapshot):

    return tuple(
        tuple(sorted(
            snapshot.incidence.indices[
                snapshot.incidence.indptr[r]:snapshot.incidence.indptr[r + 1]
            ].tolist()
        ))
        for r in range(snapshot.n_unique)
    )

def permuted_snapshots(snapshots, permutation):

    return [snapshots[int(i)] for i in permutation]

def aggregated_snapshots(snapshots, window_ids, n_nodes):

    window_ids = [int(w) for w in window_ids]
    groups = defaultdict(list)
    for pos, w in enumerate(window_ids):
        groups[w].append(pos)
    out = [None] * len(snapshots)
    for w, positions in groups.items():
        counter = defaultdict(float)
        for pos in positions:
            snap = snapshots[pos]
            for edge, count in zip(
                snapshot_index_edges(snap), snap.multiplicities
            ):
                counter[edge] += float(count)
        n_w = len(positions)
        per_step = {h: c / n_w for h, c in counter.items()}
        day = snapshots[positions[0]].day_index
        window_snap = _snapshot_core(
            per_step, n_nodes, window_index=w, day_index=day,
        )
        for pos in positions:
            out[pos] = window_snap
    return out

def load_dataset(name, raw_dir=None, config=None):

    config = config or load_config()
    raw_dir = raw_dir or (ROOT / config["paths"]["raw_cache"])
    is_email = name == config["datasets"]["email_name"]
    block = config["datasets"]["email" if is_email else "contact"]
    delta = float(block["snapshot_delta_seconds"])
    size_min = config["preprocessing"]["min_hyperedge_size"]
    size_max = config["preprocessing"]["max_hyperedge_size"]

    events, dropped = load_events(name, raw_dir)
    t0 = events[0][0]

    if is_email:
        horizon = t0 + config["datasets"]["email"]["first_days_from_first_timestamp"] * 86400.0
        events = [e for e in events if e[0] < horizon]
        if not events:
            raise DatasetError("no email-eu events inside the first 364 days")
    else:
        expected = float(block["expected_frame_seconds"])
        offsets = np.array([e[0] - t0 for e in events])
        residuals = np.abs(offsets / expected - np.round(offsets / expected))
        if residuals.max() > 1e-6:
            raise DatasetError(
                "%r: timestamps not on the expected %gs frame grid "
                "(max residual %.6g)" % (name, expected, residuals.max())
            )

    def day_index(ts):
        if is_email:
            return int((ts - t0) // 86400.0)
        return (datetime.utcfromtimestamp(ts).date()
                - datetime.utcfromtimestamp(t0).date()).days

    windows = defaultdict(Counter)
    window_day = {}
    for ts, members in events:
        w = int((ts - t0) // delta)
        windows[w][members] += 1
        window_day.setdefault(w, day_index(ts))

    node_set = sorted({v for _, members in events for v in members})
    node_to_index = {v: i for i, v in enumerate(node_set)}

    snapshots = [
        _build_snapshot(w, window_day[w], counter, node_to_index, len(node_set))
        for w, counter in sorted(windows.items())
        if counter  
    ]

    size_dist = Counter()
    events_kept = 0
    for snap in snapshots:
        events_kept += int(snap.multiplicities.sum())
        for size, count in zip(snap.sizes, snap.multiplicities):
            size_dist[int(size)] += int(count)

    stats = {
        "N": len(node_set),
        "snapshots": len(snapshots),
        "days": len({s.day_index for s in snapshots}),
        "events_kept": events_kept,
        "size_distribution": {
            str(s): size_dist.get(s, 0) for s in range(size_min, size_max + 1)
        },
        "dropped_fraction": float(dropped),
        "mean_distinct_hyperedges_per_snapshot": float(
            np.mean([s.n_unique for s in snapshots])
        ),
        "delta_seconds": delta,
        "is_email": is_email,
        "day_index_min": int(min(s.day_index for s in snapshots)),
        "day_index_max": int(max(s.day_index for s in snapshots)),
    }
    if is_email and not (
        stats["day_index_min"] == 0 and stats["day_index_max"] <= 363
    ):
        raise DatasetError("email-eu day indices outside 0..363")

    return TemporalDataset(
        name=name,
        nodes=tuple(node_set),
        node_to_index=node_to_index,
        snapshots=snapshots,
        delta_seconds=delta,
        size_min=size_min,
        size_max=size_max,
        first_timestamp=t0,
        stats=stats,
    )

def snapshot_data_from_temporal_hypergraph(th):

    n = len(th.nodes)
    snapshots = []
    for pos, t in enumerate(th.time_points):
        counter = Counter(
            tuple(sorted(th.node_to_index[v] for v in h))
            for h in th.active_hyperedges(t)
        )
        hyperedges = tuple(sorted(counter.keys()))
        snapshots.append(
            _snapshot_core(
                counter,
                n,
                window_index=pos,
                day_index=0,
                label_hyperedges=tuple(
                    tuple(th.nodes[i] for i in edge) for edge in hyperedges
                ),
            )
        )
    return snapshots

def hazard_value(mechanism, lam, size, chi):

    if mechanism == "M1":
        return lam * chi
    if mechanism == "M2":
        return lam * chi * chi
    if mechanism == "M3":
        return lam if chi / (size - 1.0) >= 0.5 else 0.0
    if mechanism == "M4":
        return lam * chi + 8.0 * lam * (chi * (chi - 1) / 2.0)
    raise KeyError("unknown mechanism %r" % mechanism)

def hazard_table(mechanism, lam, sizes=(2, 3, 4, 5, 6)):

    return {
        int(m): np.array(
            [hazard_value(mechanism, lam, m, chi) for chi in range(m + 1)]
        )
        for m in sizes
    }

def effect_table(mechanism, lam, sizes=(2, 3, 4, 5, 6)):

    return {m: -np.expm1(-table) for m, table in hazard_table(mechanism, lam, sizes).items()}

def lambda_rates_for_revision_engine(mechanism, lam, sizes=(2, 3, 4, 5, 6)):

    from ..dynamics import HypergraphSIParameters

    rates = {}
    for m, table in effect_table(mechanism, lam, sizes).items():
        for chi in range(m):  
            rates[(m, chi)] = float(table[chi])
    return HypergraphSIParameters(rates, nu=0.0)

class SparseEngine:

    def __init__(self, snapshots, n_nodes):
        self.snapshots = snapshots
        self.n = n_nodes
        self._union_cache = {}

    @property
    def n_snapshots(self):
        return len(self.snapshots)

    def hazard_feature(
        self, states, t, mechanism, lam, allowed_sizes=None
    ):

        states = np.atleast_2d(np.asarray(states, dtype=np.float64))
        snap = self.snapshots[t]
        chi = snap.incidence.dot(states.T)  
        table = hazard_table(
            mechanism, lam, sizes=set(int(s) for s in snap.sizes)
        )
        lam_mat = np.empty_like(chi)
        for row, size in enumerate(snap.sizes):
            lam_mat[row] = table[int(size)][chi[row].astype(int)]
        weights = snap.multiplicities.astype(np.float64)[:, None]
        if allowed_sizes is not None:
            mask = np.isin(snap.sizes, list(allowed_sizes)).astype(np.float64)
            weights = weights * mask[:, None]
        hazard = snap.incidence.T.dot(weights * lam_mat).T  
        return hazard * (1.0 - states)

    def source_probabilities(
        self, states, t, mechanism, lam, allowed_sizes=None, lam_scale=1.0
    ):

        states = np.atleast_2d(np.asarray(states, dtype=np.float64))
        hazard = self.hazard_feature(
            states, t, mechanism, lam * lam_scale, allowed_sizes
        )
        return -np.expm1(-hazard) * (1.0 - states)

    def union_adjacency(self, first, last):

        key = (int(first), int(last))
        if key not in self._union_cache:
            union = self.snapshots[first].pair_binary.copy()
            for snap in self.snapshots[first + 1 : last]:
                union = union + snap.pair_binary
            union.data[:] = 1.0
            self._union_cache[key] = union.tocsr()
        return self._union_cache[key]

    def features(self, states, t, model, union=None):

        states = np.atleast_2d(np.asarray(states, dtype=np.float64))
        snap = self.snapshots[t]
        susceptible = (1.0 - states)[:, :, None]
        if model == "SG":
            if union is None:
                union = self.union_adjacency(0, self.n_snapshots)
            z = union.dot(states.T).T
            return (z * (1.0 - states))[:, :, None]
        if model == "G":
            z = snap.pair_binary.dot(states.T).T
            return (z * (1.0 - states))[:, :, None]
        if model == "MG":
            z = snap.pair_weights.dot(states.T).T
            return (z * (1.0 - states))[:, :, None]
        if model == "C2":
            chi = snap.cell_incidence.dot(states.T)  
            mult = snap.cell_multiplicities.astype(np.float64)[:, None]
            z1 = snap.cell_incidence.T.dot(mult * chi).T
            z2 = snap.cell_incidence.T.dot(mult * (chi * (chi - 1.0) / 2.0)).T
            return np.stack([z1, z2], axis=2) * susceptible
        raise KeyError("unknown candidate %r" % model)

    @staticmethod
    def predict(features, h):

        return -np.expm1(-np.einsum("...f,f->...", features, np.asarray(h)))

def fit_global_hazard(features, targets, config=None, upper_bound=None):

    config = config or load_config()
    from .exp_results_revision import objective_gradient

    fit = config["fitting"]
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    shape = (1, features.shape[1])
    times = np.zeros(len(targets), dtype=int)
    upper = (-math.log(0.05) if upper_bound is None
             else float(upper_bound))
    bounds = [(0.0, upper)] * int(np.prod(shape))
    solutions = []
    started = time.perf_counter()
    for start in fit["initial_component_values"]:
        solutions.append(
            minimize(
                objective_gradient,
                np.full(int(np.prod(shape)), -math.log1p(-start)),
                args=(features, targets, times, shape, fit["alpha"], fit["eta"]),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "maxfun": 1500, "ftol": 1e-12,
                         "gtol": 1e-9, "maxls": 30},
            )
        )
    best = min(solutions, key=lambda s: s.fun)
    return best.x.reshape(shape)[0], {
        "objective": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
        "function_evaluations": int(sum(s.nfev for s in solutions)),
        "seconds": time.perf_counter() - started,
    }

def _bernoulli_nll(flat, features, outcomes):

    u = features @ flat

    log_p = np.log1p(-np.exp(-np.clip(u, 0.0, 700.0)))
    log_p = np.clip(log_p, -745.0, 0.0)
    nll = -np.sum(outcomes * log_p + (1.0 - outcomes) * (-u))
    exp_u = np.exp(np.clip(u, 0.0, 60.0))
    grad_coef = outcomes / np.maximum(exp_u - 1.0, 1e-300) - (1.0 - outcomes)
    grad = -features.T.dot(grad_coef)
    return float(nll), grad

def bernoulli_ml_fit(features, outcomes, config=None):

    config = config or load_config()
    features = np.asarray(features, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    upper = -math.log(0.05)
    bounds = [(0.0, upper)] * features.shape[1]
    best = None
    for start in config["fitting"]["initial_component_values"]:
        result = minimize(
            _bernoulli_nll,
            np.full(features.shape[1], -math.log1p(-start)),
            args=(features, outcomes),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
        )
        if best is None or result.fun < best.fun:
            best = result
    nll = float(best.fun)
    return best.x, {
        "nll": nll,
        "log_likelihood": -nll,
        "success": bool(best.success),
        "n_parameters": int(features.shape[1]),
        "n_observations": int(features.shape[0]),
    }

def simulate_batch(prob_fn, initial_states, uniforms, n_steps,
                   recovery_gamma=0.0, recovery_uniforms=None):

    initial_states = np.atleast_2d(np.asarray(initial_states, dtype=np.int8))
    uniforms = np.asarray(uniforms)
    if uniforms.shape[0] != initial_states.shape[0] or uniforms.shape[1] < n_steps:
        raise ValueError("uniforms must have shape (runs, >=n_steps, N)")
    if recovery_gamma > 0.0 and recovery_uniforms is None:
        recovery_uniforms = np.random.RandomState(987654321).random_sample(
            uniforms.shape
        )
    state = initial_states.copy()
    states = [state.copy()]
    for step in range(n_steps):
        probs = prob_fn(state, step)
        infected_now = (uniforms[:, step] < probs) & (state == 0)
        if recovery_gamma > 0.0:
            recovered = (recovery_uniforms[:, step] < recovery_gamma) & (state == 1)
            state = np.where(
                state == 1,
                np.where(recovered, 0, 1),
                np.where(infected_now, 1, 0),
            ).astype(np.int8)
        else:
            state = np.maximum(state, infected_now.astype(np.int8))
        states.append(state.copy())
    return np.stack(states, axis=1)

def attack_rate(engine, mechanism, lam, initial_states, uniforms):

    def prob_fn(states, step):
        return engine.source_probabilities(states, step, mechanism, lam)

    traj = simulate_batch(prob_fn, initial_states, uniforms, engine.n_snapshots)
    return float(traj[:, -1].mean())

def calibrate_lambda(engine, mechanism, config=None, uniforms=None,
                     initial_states=None):

    config = config or load_config()
    cal = config["calibration"]
    runs = int(cal["runs"])
    seed = int(cal["seed"])
    target = float(cal["target_mean_attack_rate"])
    tol = float(cal["tolerance"])
    lo_log, hi_log = (float(v) for v in cal["log_lambda_bracket"])
    max_iter = int(cal["max_bisection_iterations"])

    rng = np.random.RandomState(seed)
    if uniforms is None:
        uniforms = rng.random_sample(
            (runs, engine.n_snapshots, engine.n)
        ).astype(np.float32)
    if initial_states is None:
        initial_states = np.zeros((runs, engine.n), dtype=np.int8)
        seeds = np.random.RandomState(seed + 1).randint(engine.n, size=runs)
        initial_states[np.arange(runs), seeds] = 1

    evals = []

    def rate(log_lam):
        lam = float(math.exp(log_lam))
        value = attack_rate(engine, mechanism, lam, initial_states, uniforms)
        evals.append({"log_lambda": log_lam, "lambda": lam, "attack_rate": value})
        return value

    r_lo = rate(lo_log)
    r_hi = rate(hi_log)
    record = {
        "mechanism": mechanism,
        "target": target,
        "tolerance": tol,
        "runs": runs,
        "seed": seed,
    }
    if r_lo >= target:
        record.update(
            {"lambda": float(math.exp(lo_log)), "attack_rate": r_lo,
             "status": "unreachable_low",
             "note": "attack rate exceeds target even at the lower bracket edge"}
        )
        return record
    if r_hi <= target:
        record.update(
            {"lambda": float(math.exp(hi_log)), "attack_rate": r_hi,
             "status": "unreachable_high",
             "note": "attack rate below target even at the upper bracket edge"}
        )
        return record

    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo_log + hi_log)
        r_mid = rate(mid)
        candidate = {"log_lambda": mid, "lambda": float(math.exp(mid)),
                     "attack_rate": r_mid}
        if best is None or abs(r_mid - target) < abs(best["attack_rate"] - target):
            best = candidate
        if abs(r_mid - target) <= tol:
            break
        if r_mid < target:
            lo_log = mid
        else:
            hi_log = mid

    record.update(best)
    record["status"] = (
        "ok" if abs(best["attack_rate"] - target) <= tol else "unreachable"
    )
    record["bisection_evaluations"] = len(evals)
    if record["status"] != "ok":
        record["note"] = "target not reached within %d bisection steps" % max_iter
    return record

def training_aggregate(dataset, config=None):

    config = config or load_config()
    tr = config["training_period"]
    if dataset.stats["is_email"]:
        last_day = int(tr["email_days"])
    else:
        last_day = int(math.ceil(dataset.n_days / 2.0))
    edges = set()
    for snap in dataset.snapshots:
        if snap.day_index < last_day:
            edges.update(snap.hyperedges)
    unique = sorted(edges)
    return {
        "hyperedges": unique,
        "edge_index_tuples": [
            tuple(dataset.node_to_index[v] for v in e) for e in unique
        ],
        "training_days": last_day,
        "n_unique": len(unique),
    }

def run_kirkley(hyperedges):

    import hypergraphx as hgx
    from hypergraphx.measures.reducibility import reducibility

    hg = hgx.Hypergraph(edge_list=[tuple(e) for e in hyperedges],
                        weighted=False)
    started = time.perf_counter()
    eta, rstar = reducibility(hg, optimization="exact", entropy_method="count")
    return {
        "eta": float(eta),
        "R_star_sizes": [int(s) for s in rstar],
        "seconds": time.perf_counter() - started,
    }

def run_lucas(hyperedges, lucas_src=None):

    import xgi

    lucas_src = Path(lucas_src or (ROOT / load_config()["paths"]["lucas_repo"] / "src"))
    if not lucas_src.is_dir():
        raise RuntimeError(
            "Lucas et al. hypergraph_reducibility code not found at "
            f"{lucas_src}. Obtain it with: git clone "
            "https://github.com/maximelucas/hypergraph_reducibility "
            "third_party/hypergraph_reducibility && git -C "
            "third_party/hypergraph_reducibility checkout "
            "bd3fbedf110b4304d20d561d4f8816e1b27623c9"
        )
    if str(lucas_src) not in sys.path:
        sys.path.insert(0, str(lucas_src))
    try:
        from hypergraph_reducibility import find_charact_tau, optimization  
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import hypergraph_reducibility from {lucas_src}: {exc}"
        ) from exc

    hypergraph = xgi.Hypergraph()
    hypergraph.add_edges_from([tuple(e) for e in hyperedges])
    n_before, e_before = hypergraph.num_nodes, hypergraph.num_edges
    hypergraph.cleanup(isolates=True, singletons=False, connected=True,
                       relabel=False)
    adaptations = {
        "nodes_before_cleanup": int(n_before),
        "edges_before_cleanup": int(e_before),
        "nodes_after_cleanup": int(hypergraph.num_nodes),
        "edges_after_cleanup": int(hypergraph.num_edges),
        "cleanup": "isolates=True, singletons=False, connected=True, relabel=False",
        "xgi_version": xgi.__version__,
    }

    orders = np.array(xgi.unique_edge_sizes(hypergraph)) - 1
    weights = np.ones(len(orders))
    rescale, sparse, sparse_lap = True, False, True

    started = time.perf_counter()
    tau = find_charact_tau(hypergraph, orders, weights,
                           rescale_per_node=rescale, sparse_Lap=sparse_lap,
                           idx=-1)
    rescaling = [len(orders) / i for i in range(1, len(orders) + 1)]
    D, lZ = optimization(
        hypergraph,
        tau,
        rescaling_factors=rescaling,
        tau_per_order=False,
        rescale_per_node=rescale,
        sparse=sparse,
        sparse_Lap=sparse_lap,
        show_progress=False,
    )
    elapsed = time.perf_counter() - started

    message_length = D + lZ
    idx = int(np.argmin(message_length))
    d_star = int(orders[idx])
    return {
        "orders": [int(o) for o in orders],
        "edge_sizes": [int(o + 1) for o in orders],
        "tau_c": float(tau),
        "rescaling_factors": [float(r) for r in rescaling],
        "D": [float(v) for v in D],
        "lZ": [float(v) for v in lZ],
        "message_length": [float(v) for v in message_length],
        "d_star_order": d_star,
        "max_kept_hyperedge_size": d_star + 1,
        "seconds": float(elapsed),
        "adaptations": adaptations,
    }
