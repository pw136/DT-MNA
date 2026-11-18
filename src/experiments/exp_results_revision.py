
import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from ..approximate import adaptive_model_sequence, cvar
from ..dynamics import ExposureSemantics, HypergraphSIParameters
from ..necessity.closed_loop import build_candidate_networks
from ..theory import build_temporal_c1_lambda_rates, build_temporal_c3_lambda_rates, C3ProjectionRoute
from .conversion_validation_data import load_real_temporal
from .exp_innovation23_supplementary import _random_hypergraph
from .innovation2_core import build_conflicting_lambda_rates

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/results_revision_supplement.json"
DEFAULT_OUT = ROOT / "results/results_revision_supplement"
INDEPENDENT = ExposureSemantics.INDEPENDENT_EXPOSURES
FAIR_MODELS = ("SG-global", "SG-time", "G-time", "MG-time", "C2-time")
DYNAMIC_MODELS = ("G-time", "MG-time", "C2-time", "H")
COST = np.array([1.0, 2.0, 4.5, 10.0])

def plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=plain, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def source_parameters(network, mechanism, dynamic=False):
    points = network.time_points
    beta1 = {t: 0.024 + 0.009 * math.sin(2 * math.pi * k / len(points))
             for k, t in enumerate(points)}
    c3 = build_temporal_c3_lambda_rates(
        network, 2, 2.0, beta1, exposure_semantics=INDEPENDENT,
        route=C3ProjectionRoute.DIRECT)
    beta = {(t, 1): 0.014 for t in points}
    beta.update({(t, 2): 0.045 for t in points})
    c1 = build_temporal_c1_lambda_rates(network, 2, 2.0, beta,
                                      exposure_semantics=INDEPENDENT)
    conflict, _ = build_conflicting_lambda_rates(network, 2.0, 0.045, 0.05, 0.009)
    laws = {"pairwise": c3, "synergy": c1, "conflict": conflict}
    if not dynamic:
        return HypergraphSIParameters(laws[mechanism], 2.0)

    phases = ("pairwise", "synergy", "conflict", "synergy", "pairwise", "conflict")
    rates = {}
    for k, t in enumerate(points):
        law = laws[phases[min(5, 6 * k // len(points))]]
        rates.update({key: value for key, value in law.items() if key[0] == t})
    return HypergraphSIParameters(rates, 2.0)

class ProbabilityEngine:

    def __init__(self, network, parameters):
        self.network = network
        self.parameters = parameters
        self.points = network.time_points
        self.n = len(network.nodes)
        self.candidates = build_candidate_networks(network)
        self.groups = {}
        self.effects = {}
        self.cells = {}
        self.adjacency = {}
        for t in self.points:
            groups = [np.array([network.node_to_index[v] for v in group], dtype=int)
                      for group in network.active_hyperedges(t)]
            self.groups[t] = groups
            self.effects[t] = {
                size: np.array([0.0] + [
                    parameters.lambda_rate(t, size, m) * m ** parameters.nu
                    for m in range(1, size)])
                for size in {len(g) for g in groups}
            }
            self.cells[t] = [
                np.array([network.node_to_index[v] for v in cell], dtype=int)
                for cell in self.candidates["SC2"].active_maximal_simplices(t)
            ]
            for name in ("SG", "G", "MG"):
                proxy = self.candidates[name]
                ii, jj, ww = [], [], []
                for i, node in enumerate(network.nodes):
                    neighbors = (proxy.weighted_neighbors(node, t) if name == "MG"
                                 else [(v, 1) for v in proxy.neighbors(node, t)])
                    for other, weight in neighbors:
                        ii.append(i)
                        jj.append(network.node_to_index[other])
                        ww.append(weight)
                self.adjacency[(name, t)] = csr_matrix((ww, (ii, jj)), shape=(self.n, self.n))

    def source(self, states, t):
        states = np.atleast_2d(states)
        accum = np.zeros(states.shape, dtype=float)
        for group in self.groups[t]:
            m = np.minimum(states[:, group].sum(axis=1), len(group) - 1)
            effect = self.effects[t][len(group)][m.astype(int)]
            if np.any(effect < 0) or np.any(effect > 1):
                raise ValueError("Illegal source effect")
            with np.errstate(divide="ignore"):
                accum[:, group] += -np.log1p(-effect)[:, None]
        return -np.expm1(-accum) * (1 - states)

    def features(self, states, t, model):
        states = np.atleast_2d(states)
        if model.startswith("C2"):
            result = np.zeros(states.shape + (2,), dtype=float)
            for cell in self.cells[t]:
                m = states[:, cell].sum(axis=1)
                result[:, cell, 0] += m[:, None]
                result[:, cell, 1] += (m * (m - 1) / 2)[:, None]
            result *= (1 - states)[:, :, None]
            return result
        name = model.split("-")[0]
        result = self.adjacency[(name, t)].dot(states.T).T
        return (result * (1 - states))[:, :, None]

    def predict(self, states, t, model, parameters):
        if model == "H":
            return self.source(states, t)
        features = self.features(states, t, model)
        h = parameters[0 if model == "SG-global" else self.points.index(t)]
        return -np.expm1(-np.einsum("snf,f->sn", features, h))

def disjoint_panels(n, points, counts, seed):

    rng = np.random.RandomState(seed)
    total = sum(counts.values())
    if n <= 8:
        all_states = np.array([x for x in itertools.product((0, 1), repeat=n)
                               if 0 < sum(x) < n], dtype=np.int8)
        if total > len(all_states):
            raise ValueError("Requested too many distinct small-system states")
        rng.shuffle(all_states)
        result, offset = {}, 0
        for split, count in counts.items():
            chosen = all_states[offset:offset + count]
            result[split] = {t: chosen.copy() for t in points}
            offset += count
        return result
    occupied = set()
    result = {}
    fractions = (0.04, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80)
    for split, count in counts.items():
        result[split] = {}
        for t in points:
            states = []
            attempts = 0
            while len(states) < count:
                attempts += 1
                fraction = fractions[rng.randint(len(fractions))]
                state = (rng.random_sample(n) < fraction).astype(np.int8)
                key = state.tobytes()
                if 0 < state.sum() < n and key not in occupied:
                    states.append(state)
                    occupied.add(key)
                if attempts > 100000:
                    raise RuntimeError("Could not create disjoint state panels")
            result[split][t] = np.array(states)
    return result

def samples(engine, panel):
    output = {"time": [], "target": [], "features": {m: [] for m in FAIR_MODELS}}
    for pos, t in enumerate(engine.points):
        if t not in panel:
            continue
        states = panel[t]
        susceptible = states == 0
        output["target"].append(engine.source(states, t)[susceptible])
        output["time"].append(np.full(susceptible.sum(), pos, dtype=int))
        for model in FAIR_MODELS:
            output["features"][model].append(engine.features(states, t, model)[susceptible])
    return {
        "time": np.concatenate(output["time"]),
        "target": np.concatenate(output["target"]),
        "features": {m: np.concatenate(x) for m, x in output["features"].items()},
    }

def combine(blocks):
    return {
        "time": np.concatenate([x["time"] for x in blocks]),
        "target": np.concatenate([x["target"] for x in blocks]),
        "features": {m: np.concatenate([x["features"][m] for x in blocks])
                     for m in FAIR_MODELS},
    }

def objective_gradient(flat, features, targets, times, parameter_shape, alpha, eta):
    parameters = flat.reshape(parameter_shape)
    survival = np.exp(-np.einsum("nf,nf->n", features, parameters[times]))
    residual = 1 - survival - targets
    absolute = np.abs(residual)
    weights = 2 * residual / len(residual)
    loss = np.mean(residual ** 2)
    if eta:
        k = max(1, int(np.ceil((1 - alpha) * len(residual))))
        selected = np.argpartition(absolute, len(absolute) - k)[-k:]
        tail = absolute[selected].mean()
        loss += eta * tail ** 2
        weights[selected] += 2 * eta * tail * np.sign(residual[selected]) / k
    derivative = features * (weights * survival)[:, None]
    gradient = np.zeros(parameter_shape, dtype=float)
    for column in range(features.shape[1]):
        gradient[:, column] = np.bincount(times, derivative[:, column],
                                         minlength=parameter_shape[0])
    return float(loss), gradient.ravel()

def predict_samples(data, model, parameters):
    tt = np.zeros(len(data["time"]), dtype=int) if model == "SG-global" else data["time"]
    return -np.expm1(-np.einsum("nf,nf->n", data["features"][model], parameters[tt]))

def risk(target, prediction, alpha=0.95, eta=0.2):
    error = np.abs(prediction - target)
    return float(np.mean(error ** 2) + eta * cvar(error, alpha) ** 2)

def fit_model(train, model, eta, config, time_count):
    features = train["features"][model]
    times = np.zeros(len(train["time"]), dtype=int) if model == "SG-global" else train["time"]
    shape = (1 if model == "SG-global" else time_count, features.shape[1])
    upper = -math.log1p(-config["fit_probability_max"])
    solutions = []
    started = time.perf_counter()
    for start in config["optimizer_starts"]:
        solution = minimize(
            objective_gradient, np.full(np.prod(shape), -math.log1p(-start)),
            args=(features, train["target"], times, shape, config["alpha"], eta),
            jac=True, method="L-BFGS-B", bounds=[(0.0, upper)] * int(np.prod(shape)),
            options={"maxiter": config["optimizer_maxiter"], "maxfun": config["optimizer_maxfun"],
                     "ftol": 1e-12, "gtol": 1e-9, "maxls": 30})
        solutions.append(solution)
    best = min(solutions, key=lambda x: x.fun)
    parameters = best.x.reshape(shape)
    expected = risk(train["target"], predict_samples(train, model, parameters), config["alpha"], eta)
    if not np.isclose(best.fun, expected, atol=1e-12):
        raise AssertionError("Optimizer objective differs from code's tail loss")
    return parameters, {
        "eta": eta, "alpha": config["alpha"], "objective": float(best.fun),
        "success": bool(best.success), "message": str(best.message),
        "function_evaluations": int(sum(x.nfev for x in solutions)),
        "parameters": parameters, "sample_count": len(train["target"]),
        "tail_count": max(1, int(np.ceil((1 - config["alpha"]) * len(train["target"])))),
        "seconds": time.perf_counter() - started,
    }

def metrics(target, prediction):
    error = np.abs(prediction - target)
    return {"mae": float(error.mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "tail95": cvar(error, 0.95), "worst": float(error.max()), "samples": len(error)}

def save_network(path, engine, panels):
    write_json(path.with_suffix(".json"), {
        "nodes": engine.network.nodes,
        "snapshots": {str(t): engine.network.active_hyperedges(t) for t in engine.points},
        "lambda_rates": [[*key, value] for key, value in engine.parameters.lambda_rates.items()],
        "nu": engine.parameters.nu,
    })
    np.savez_compressed(path.with_suffix(".npz"),
                        **{f"{split}_{t}": states for split, row in panels.items()
                           for t, states in row.items()})

def real_networks(config):
    for name in config["real_datasets"]:
        yield name, load_real_temporal(name, max_events=config["real_max_events"],
                                      time_count=config["real_snapshots"],
                                      max_hyperedge_size=config["real_max_hyperedge_size"])[0]

def fair_context(context_id, engines, panels, roles, config, directory, domain, mechanism, replicate):
    train = combine([samples(e, p["train"]) for e, p, r in zip(engines, panels, roles) if r in ("train", "all")])
    validation = combine([samples(e, p["validation"]) for e, p, r in zip(engines, panels, roles) if r in ("validation", "all")])
    models, fit_records = {}, {}
    for model in FAIR_MODELS:
        options = [fit_model(train, model, eta, config, len(engines[0].points))
                   for eta in config["tail_weight_grid"]]
        validation_risks = [risk(validation["target"], predict_samples(validation, model, p),
                                 config["alpha"], config["validation_tail_weight"])
                            for p, _ in options]
        chosen = int(np.argmin(validation_risks))
        models[model] = options[chosen][0]
        fit_records[model] = {"selected_index": chosen, "validation_risks": validation_risks,
                              "options": [record for _, record in options]}
    rows = []
    prediction_arrays = {}
    for network_index, (engine, panel, role) in enumerate(zip(engines, panels, roles)):
        save_network(directory / f"network_{network_index}", engine, panel)
        if role not in ("test-id", "test-shift", "all"):
            continue
        test = samples(engine, panel["test"])
        prediction_arrays[f"target_{network_index}"] = test["target"]
        for model, parameters in models.items():
            prediction = predict_samples(test, model, parameters)
            prediction_arrays[f"{model}_{network_index}"] = prediction
            rows.append({
                "context": context_id, "domain": domain, "mechanism": mechanism,
                "replicate": replicate, "network_index": network_index, "test_role": role,
                "model": model, "selected_eta": fit_records[model]["options"][fit_records[model]["selected_index"]]["eta"],
                **metrics(test["target"], prediction),
                "fit_seconds": sum(x["seconds"] for x in fit_records[model]["options"]),
                "all_starts_converged": all(x["success"] for x in fit_records[model]["options"]),
            })
    write_json(directory / "fits.json", fit_records)
    np.savez_compressed(directory / "test_predictions.npz", **prediction_arrays)
    write_csv(directory / "results.csv", rows)
    return rows

def causal_sequence(errors, config):
    settings = config["dynamic"]
    chosen = []
    for row in errors:
        allowed = row <= settings["error_budget"]
        cost = row + settings["complexity_weight"] * COST
        if chosen:
            cost += settings["switch_cost"] * (np.arange(len(COST)) != chosen[-1])
        cost[~allowed] = np.inf
        chosen.append(int(np.argmin(cost)))
    return np.array(chosen, dtype=int)

def select_sequences(validation, audit, config):
    settings = config["dynamic"]
    checked = np.maximum(validation, audit)
    offline = adaptive_model_sequence(
        checked, COST, DYNAMIC_MODELS, error_budget=settings["error_budget"],
        complexity_weight=settings["complexity_weight"], switch_cost=settings["switch_cost"])
    if not offline.feasible:
        raise AssertionError("Full H must always be an available exact fallback")
    eligible = np.max(checked, axis=0) <= settings["error_budget"]
    fixed = int(np.flatnonzero(eligible)[0])
    return {
        "Causal-checked": causal_sequence(checked, config),
        "Causal-validation": causal_sequence(validation, config),
        "Offline-DP": np.array(offline.selected_indices),
        "Fixed-validated": np.full(len(checked), fixed),
        **{f"Fixed-{name}": np.full(len(checked), index)
           for index, name in enumerate(DYNAMIC_MODELS)},
    }

def fit_dynamic(engine, panel, config):
    models = {model: [] for model in DYNAMIC_MODELS if model != "H"}
    records = []
    for position, t in enumerate(engine.points):
        batch = samples(engine, {t: panel["train"][t]})

        keep = batch["time"] == position
        train = {
            "time": np.zeros(keep.sum(), dtype=int), "target": batch["target"][keep],
            "features": {m: f[keep] for m, f in batch["features"].items()},
        }
        for model in models:
            parameters, record = fit_model(train, model, config["dynamic"]["tail_weight"], config, 1)
            models[model].append(parameters[0])
            records.append({"time": t, "model": model, **record})
    return {m: np.array(p) for m, p in models.items()}, records

def simulate_batch(engine, models, sequence, initial, uniforms):
    state = initial.copy()
    states = [state.copy()]
    for position, t in enumerate(engine.points):
        model = DYNAMIC_MODELS[int(sequence[position])]
        probabilities = engine.predict(state, t, model, models.get(model))
        state = np.maximum(state, (uniforms[:, position] < probabilities).astype(np.int8))
        states.append(state.copy())
    return np.stack(states, axis=1)

def initial_states(engine, count, fraction, seed, panel):
    rng = np.random.RandomState(seed)
    prohibited = {state.tobytes() for split in panel.values()
                  for states in split.values() for state in states}
    states = []
    for _ in range(count):
        while True:
            state = (rng.random_sample(engine.n) < fraction).astype(np.int8)
            if 0 < state.sum() < engine.n and state.tobytes() not in prohibited:
                states.append(state)
                break
    return np.array(states)

def exact_kernel(states, probabilities):
    next_one = np.where(states == 1, 1.0, probabilities)
    return np.prod(np.where(states[None, :, :] == 1,
                            next_one[:, None, :], 1 - next_one[:, None, :]), axis=2)

def exact_process(engine, models, sequence):
    states = np.array(list(itertools.product((0, 1), repeat=engine.n)), dtype=np.int8)
    mu = (states.sum(axis=1) == 1).astype(float)
    mu /= mu.sum()
    source_mu, proxy_mu = mu.copy(), mu.copy()
    weighted_bound, worst_bound = 0.0, 0.0
    rows = []
    for position, t in enumerate(engine.points):
        source_kernel = exact_kernel(states, engine.source(states, t))
        model = DYNAMIC_MODELS[int(sequence[position])]
        proxy_kernel = exact_kernel(states, engine.predict(states, t, model, models.get(model)))
        if not np.allclose(source_kernel.sum(axis=1), 1) or not np.allclose(proxy_kernel.sum(axis=1), 1):
            raise AssertionError("Non-stochastic exact kernel")
        row_tv = np.abs(source_kernel - proxy_kernel).sum(axis=1) / 2
        weighted_bound = min(1.0, weighted_bound + source_mu @ row_tv)
        worst_bound = min(1.0, worst_bound + row_tv.max())
        source_mu = source_mu @ source_kernel
        proxy_mu = proxy_mu @ proxy_kernel
        tv = np.abs(source_mu - proxy_mu).sum() / 2
        if tv > weighted_bound + 1e-10 or tv > worst_bound + 1e-10:
            raise AssertionError("Finite-time bound violation")
        rows.append({"time": t + 1, "actual_tv": float(tv),
                     "distribution_bound": float(weighted_bound), "worst_bound": float(worst_bound)})
    return rows


def _csv_scalar(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer() and "." not in value and "e" not in value.lower():
        return int(number)
    return number


def dynamic_context(context, engine, config, seed, out, exact=False, domain="synthetic"):
    directory = out / "dynamic" / context
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "results.csv").exists():
        with (directory / "results.csv").open(encoding="utf-8-sig") as f:
            return [{k: _csv_scalar(v) for k, v in row.items()}
                    for row in csv.DictReader(f)]
    settings = config["dynamic"]
    counts = {"train": 12, "validation": 12, "audit": 12, "test": 26} if exact else {
        "train": settings["train_states"], "validation": settings["validation_states"],
        "audit": settings["audit_states"], "test": settings["test_states"]}
    panel = disjoint_panels(engine.n, engine.points, counts, seed)
    save_network(directory / "network", engine, panel)
    started = time.perf_counter()
    models, fits = fit_dynamic(engine, panel, config)
    fit_seconds = time.perf_counter() - started
    errors = {split: np.zeros((len(engine.points), len(DYNAMIC_MODELS))) for split in ("validation", "audit", "test")}
    test_means = np.zeros_like(errors["test"])
    for split in errors:
        for position, t in enumerate(engine.points):
            states = panel[split][t]
            target = engine.source(states, t)
            for model_index, model in enumerate(DYNAMIC_MODELS):
                residual = np.abs(engine.predict(states, t, model, models.get(model)) - target)[states == 0]
                errors[split][position, model_index] = residual.max()
                if split == "test":
                    test_means[position, model_index] = residual.mean()
    sequences = select_sequences(errors["validation"], errors["audit"], config)
    rows, trajectory_arrays = [], {}
    if not exact:
        initial = initial_states(engine, settings["trajectory_runs"],
                                 settings["initial_infected_fraction"], seed + 60000, panel)
        uniforms = np.random.RandomState(seed + 70000).random_sample(
            (settings["trajectory_runs"], len(engine.points), engine.n))
        reference = simulate_batch(engine, models, sequences["Fixed-H"], initial, uniforms)
        trajectory_arrays.update(initial=initial, uniforms=uniforms, source_states=reference)
    exact_rows, timeline = [], []
    for method, sequence in sequences.items():
        indexes = (np.arange(len(sequence)), sequence)
        selected_test = errors["test"][indexes]
        row = {"context": context, "domain": domain, "method": method,
               "nodes": engine.n, "snapshots": len(engine.points),
               "mean_complexity": float(COST[sequence].mean()),
               "switches": int(np.sum(sequence[1:] != sequence[:-1])),
               "test_worst": float(selected_test.max()),
               "test_stage_exceed_rate": float(np.mean(selected_test > settings["error_budget"] + 1e-12)),
               "test_probability_mae": float(test_means[indexes].mean()),
               "fit_seconds_shared": fit_seconds}
        for position, model_index in enumerate(sequence):
            timeline.append({"context": context, "method": method, "time": position,
                             "model": DYNAMIC_MODELS[model_index],
                             "validation_worst": float(errors["validation"][position, model_index]),
                             "audit_worst": float(errors["audit"][position, model_index]),
                             "test_worst": float(errors["test"][position, model_index])})
        if exact:
            values = exact_process(engine, models, sequence)
            row.update(endpoint_tv=values[-1]["actual_tv"],
                       max_time_tv=max(x["actual_tv"] for x in values),
                       final_distribution_bound=values[-1]["distribution_bound"])
            exact_rows.extend({"context": context, "method": method, **x} for x in values)
        else:
            started = time.perf_counter()
            states = simulate_batch(engine, models, sequence, initial, uniforms)
            row["simulation_seconds"] = time.perf_counter() - started
            density = states.mean(axis=2)
            source_density = reference.mean(axis=2)
            row.update(
                paired_density_mae=float(np.abs(density - source_density).mean()),
                mean_curve_mae=float(np.abs(density.mean(axis=0) - source_density.mean(axis=0)).mean()),
                final_density_mae=float(np.abs(density[:, -1] - source_density[:, -1]).mean()),
                state_mismatch_rate=float(np.mean(states != reference)),
                paired_path_match_rate=float(np.mean(np.all(states == reference, axis=(1, 2)))))
            trajectory_arrays[method + "_states"] = states
        rows.append(row)
    if not exact:
        np.savez_compressed(directory / "trajectories.npz", **trajectory_arrays)
    write_json(directory / "fits.json", fits)
    write_json(directory / "selection.json", {"sequences": sequences, "errors": errors})
    write_csv(directory / "timeline.csv", timeline)
    if exact:
        write_csv(directory / "exact_distributions.csv", exact_rows)
    write_csv(directory / "results.csv", rows)
    print(f"Completed dynamic {context}", flush=True)
    return rows

def run_dynamic(config, out, smoke=False):
    settings = config["dynamic"]
    rows, exact_rows = [], []
    for rep in range(1 if smoke else settings["synthetic_replicates"]):
        seed = config["seed"] + 800000 + 100 * rep
        network = _random_hypergraph(
            settings["nodes"], settings["snapshots"], seed,
            persistence=0.15 + 0.07 * rep, overlap=0.2 + 0.05 * rep,
            duplicate_rate=0.06 + 0.035 * rep)
        engine = ProbabilityEngine(network, source_parameters(network, "conflict", dynamic=True))
        rows.extend(dynamic_context(f"synthetic_{rep}", engine, config, seed + 1, out))
    for rep in range(1 if smoke else settings["exact_replicates"]):
        seed = config["seed"] + 900000 + rep
        network = _random_hypergraph(settings["exact_nodes"], settings["exact_snapshots"],
                                    seed, events_per_snapshot=4)
        engine = ProbabilityEngine(network, source_parameters(network, "conflict", dynamic=True))
        exact_rows.extend(dynamic_context(f"exact_{rep}", engine, config, seed + 1, out, exact=True, domain="exact"))
    if not smoke:
        for offset, (name, network) in enumerate(real_networks(config)):
            for rep in range(settings["real_replicates"]):
                engine = ProbabilityEngine(network, source_parameters(network, "conflict", dynamic=True))
                seed = config["seed"] + 1000000 + 100 * offset + rep
                rows.extend(dynamic_context(f"real_{name}_{rep}", engine, config, seed, out, domain=name))
    write_csv(out / "dynamic_results.csv", rows)
    write_csv(out / "dynamic_exact_results.csv", exact_rows)
    return rows, exact_rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--suite", choices=("fair", "dynamic", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "smoke": args.smoke,
    }
    path = args.output / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "code_sha256", "smoke"):
            if old[key] != manifest[key]:
                raise RuntimeError(f"Cannot mix result versions: {key}. Choose a fresh output directory.")
    else:
        write_json(path, manifest)
    if args.suite in ("fair", "all"):
        run_fair(config, args.output, args.smoke)
    if args.suite in ("dynamic", "all"):
        run_dynamic(config, args.output, args.smoke)

def run_fair(config, out, smoke=False):
    settings = config["fair"]
    rows = []
    repetitions = 1 if smoke else settings["synthetic_replicates"]
    for mechanism in settings["mechanisms"]:
        for rep in range(repetitions):
            context = f"synthetic_{mechanism}_{rep}"
            directory = out / "fair" / context
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "results.csv").exists():
                with (directory / "results.csv").open(encoding="utf-8-sig") as f:
                    rows.extend(list(csv.DictReader(f)))
                continue
            engines, panels = [], []
            roles = ("train", "train", "validation", "test-id", "test-shift")
            for index, role in enumerate(roles):
                seed = config["seed"] + 10000 * settings["mechanisms"].index(mechanism) + rep * 100 + index
                shifted = role == "test-shift"
                network = _random_hypergraph(
                    settings["nodes"], settings["snapshots"], seed,
                    persistence=0.65 if shifted else 0.25, overlap=0.7 if shifted else 0.35,
                    duplicate_rate=0.35 if shifted else 0.12)
                engine = ProbabilityEngine(network, source_parameters(network, mechanism))
                count = {"train": settings["train_states"]} if role == "train" else (
                    {"validation": settings["validation_states"]} if role == "validation"
                    else {"test": settings["test_states"]})
                engines.append(engine)
                panels.append(disjoint_panels(engine.n, engine.points, count, seed + 300000))
            rows.extend(fair_context(context, engines, panels, roles, config, directory,
                                     "synthetic", mechanism, rep))
            print(f"Completed fair {context}", flush=True)
    if not smoke:
        for name, network in real_networks(config):
            for rep in range(settings["real_replicates"]):
                context = f"real_{name}_{rep}"
                directory = out / "fair" / context
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "results.csv").exists():
                    with (directory / "results.csv").open(encoding="utf-8-sig") as f:
                        rows.extend(list(csv.DictReader(f)))
                    continue
                engine = ProbabilityEngine(network, source_parameters(network, "conflict"))
                panel = disjoint_panels(engine.n, engine.points, {
                    "train": settings["real_train_states"], "validation": settings["real_validation_states"],
                    "test": settings["real_test_states"]}, config["seed"] + 500000 + 100 * config["real_datasets"].index(name) + rep)
                rows.extend(fair_context(context, [engine], [panel], ["all"], config,
                                         directory, name, "conflict", rep))
                print(f"Completed fair {context}", flush=True)
    write_csv(out / "fair_results.csv", rows)
    return rows

if __name__ == "__main__":
    main()
