
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.ext_baselines_common import (  
    LOWER_CANDIDATES,
    MECHANISMS,
    DatasetError,
    SparseEngine,
    attack_rate,
    bernoulli_ml_fit,
    calibrate_lambda,
    fit_global_hazard,
    lambda_rates_for_revision_engine,
    load_config,
    load_dataset,
    run_kirkley,
    run_lucas,
    simulate_batch,
    snapshot_data_from_temporal_hypergraph,
    training_aggregate,
    write_json,
)

def env_report(config):
    import platform

    import scipy
    import xgi
    import hypergraphx

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "xgi": xgi.__version__,
        "hypergraphx": hypergraphx.__version__,
        "pypi_release_timestamps_utc": {
            "xgi": config["packages"]["xgi"]["pypi_release_timestamp_utc"],
            "hypergraphx": config["packages"]["hypergraphx"]["pypi_release_timestamp_utc"],
        },
        "min_release_age_days": config["packages"]["xgi"]["min_release_age_days"],
    }

def validate_against_revision_engine(config):

    from src.dynamics import HypergraphSIParameters  
    from src.experiments.exp_innovation23_supplementary import _random_hypergraph
    from src.experiments.exp_results_revision import ProbabilityEngine

    rng = np.random.RandomState(123)
    lam = 0.05
    report = {"lambda": lam, "hypergraphs": []}
    specs = [(8, 6, 11), (9, 5, 22), (10, 7, 33)]
    for nodes, times, seed in specs:
        th = _random_hypergraph(nodes, times, seed)
        snapshots = snapshot_data_from_temporal_hypergraph(th)
        engine = SparseEngine(snapshots, len(th.nodes))
        union = engine.union_adjacency(0, engine.n_snapshots)
        entry = {"nodes": nodes, "snapshots": times, "seed": seed,
                 "source_max_abs_diff": {}, "feature_max_abs_diff": {},
                 "m1_equals_mg": None}
        for mechanism in MECHANISMS:
            params = lambda_rates_for_revision_engine(mechanism, lam)
            reference = ProbabilityEngine(th, params)
            worst = 0.0
            for pos, t in enumerate(th.time_points):
                states = (rng.random_sample((24, len(th.nodes))) < 0.5).astype(np.int8)
                mine = engine.source_probabilities(states, pos, mechanism, lam)
                ref = reference.source(states, t)
                worst = max(worst, float(np.abs(mine - ref).max()))
            entry["source_max_abs_diff"][mechanism] = worst

        params = lambda_rates_for_revision_engine("M1", lam)
        reference = ProbabilityEngine(th, params)
        for model in LOWER_CANDIDATES:
            worst = 0.0
            for pos, t in enumerate(th.time_points):
                states = (rng.random_sample((24, len(th.nodes))) < 0.5).astype(np.int8)
                mine = engine.features(states, pos, model, union=union)
                ref = reference.features(states, t, model)
                worst = max(worst, float(np.abs(mine - ref).max()))
            entry["feature_max_abs_diff"][model] = worst

        worst = 0.0
        for pos in range(engine.n_snapshots):
            states = (rng.random_sample((24, len(th.nodes))) < 0.5).astype(np.int8)
            source = engine.source_probabilities(states, pos, "M1", lam)
            pred = engine.predict(
                engine.features(states, pos, "MG"), np.array([lam])
            )
            worst = max(worst, float(np.abs(source - pred).max()))
        entry["m1_equals_mg"] = worst
        report["hypergraphs"].append(entry)
    report["sg_definition"] = (
        "ProbabilityEngine builds SG via _static_simple_graph: the union of "
        "all pairwise-projected edges over every snapshot "
        "(src/necessity/closed_loop.py::_static_simple_graph). Our SG uses "
        "union_adjacency(0, T), i.e. the same definition."
    )
    report["m1_equals_mg_tolerance"] = config["validation"]["m1_equals_mg_tolerance"]
    report["m1_equals_mg_ok"] = all(
        h["m1_equals_mg"] <= config["validation"]["m1_equals_mg_tolerance"]
        for h in report["hypergraphs"]
    )
    return report

def bernoulli_smoke_test(config):

    from src.experiments.exp_innovation23_supplementary import _random_hypergraph

    th = _random_hypergraph(8, 6, 7)
    snapshots = snapshot_data_from_temporal_hypergraph(th)
    engine = SparseEngine(snapshots, len(th.nodes))
    rng = np.random.RandomState(5)
    features, outcomes = [], []
    lam = 0.4
    for pos in range(engine.n_snapshots):
        states = (rng.random_sample((40, len(th.nodes))) < 0.35).astype(np.int8)
        probs = engine.source_probabilities(states, pos, "M1", lam)
        feats = engine.features(states, pos, "MG")
        mask = states.ravel() == 0
        features.append(feats.reshape(-1, feats.shape[-1])[mask])
        outcomes.append(
            (rng.random_sample(probs.size) < probs.ravel())[mask].astype(float)
        )
    features = np.concatenate(features)
    outcomes = np.concatenate(outcomes)
    h, info = bernoulli_ml_fit(features, outcomes, config)
    return {"h": h.tolist(), "info": info}

def main():
    started_all = time.perf_counter()
    config = load_config()
    out_dir = ROOT / config["paths"]["infra_results"]
    out_dir.mkdir(parents=True, exist_ok=True)

    names = config["datasets"]["contact_names"] + [config["datasets"]["email_name"]]
    print("[1/6] environment report", flush=True)
    write_json(out_dir / "environment.json", env_report(config))

    print("[2/6] loading datasets", flush=True)
    datasets = {}
    stats = {}
    for name in names:
        t0 = time.perf_counter()
        try:
            datasets[name] = load_dataset(name)
            stats[name] = datasets[name].stats
            stats[name]["seconds_to_load"] = time.perf_counter() - t0
            print("  %s: N=%d T=%d days=%d" % (
                name, stats[name]["N"], stats[name]["snapshots"],
                stats[name]["days"]), flush=True)
        except DatasetError as exc:
            stats[name] = {"error": str(exc)}
            print("  %s: ERROR %s" % (name, exc), flush=True)
    write_json(out_dir / "dataset_report.json", stats)

    print("[3/6] validation against ProbabilityEngine", flush=True)
    validation = validate_against_revision_engine(config)
    validation["bernoulli_ml_smoke"] = bernoulli_smoke_test(config)
    write_json(out_dir / "validation_report.json", validation)
    print(json.dumps(validation["hypergraphs"], indent=1)[:1200], flush=True)

    print("[4/6] external reducibility methods on training aggregates", flush=True)
    external = {}
    for name, dataset in datasets.items():
        aggregate = training_aggregate(dataset, config)
        entry = {"n_unique_hyperedges": aggregate["n_unique"],
                 "training_days": aggregate["training_days"]}
        try:
            entry["kirkley"] = run_kirkley(aggregate["hyperedges"])
        except Exception as exc:
            entry["kirkley"] = {"error": repr(exc)}
        try:
            entry["lucas"] = run_lucas(aggregate["hyperedges"])
        except Exception as exc:
            entry["lucas"] = {"error": repr(exc)}
        external[name] = entry
        print("  %s: %d unique edges; kirkley=%s; lucas d*=%s" % (
            name, entry["n_unique_hyperedges"],
            entry["kirkley"].get("R_star_sizes", entry["kirkley"].get("error")),
            entry["lucas"].get("d_star_order", entry["lucas"].get("error"))),
            flush=True)
        write_json(out_dir / "external_methods.json", external)

    print("[5/6] calibration (6 datasets x 4 mechanisms)", flush=True)
    engines = {name: SparseEngine(ds.snapshots, ds.n_nodes)
               for name, ds in datasets.items()}
    calibration = {}
    for name in names:
        if name not in engines:
            continue
        calibration[name] = {}
        for mechanism in MECHANISMS:
            t0 = time.perf_counter()
            record = calibrate_lambda(engines[name], mechanism, config)
            record["seconds"] = time.perf_counter() - t0
            calibration[name][mechanism] = record
            print("  %s %s: lambda=%.4g rate=%.4f (%s, %.0fs)" % (
                name, mechanism, record["lambda"], record["attack_rate"],
                record["status"], record["seconds"]), flush=True)
        write_json(out_dir / "calibration.json", calibration)

    print("[6/6] runtime benchmark: 500-run SI on H(t), contact-high-school",
          flush=True)
    bench_cfg = config["benchmark"]
    bench_name = bench_cfg["dataset"]
    engine = engines[bench_name]
    lam = calibration.get(bench_name, {}).get("M1", {}).get("lambda", 0.01)
    runs = int(bench_cfg["runs"])
    rng = np.random.RandomState(config["calibration"]["seed"])
    uniforms = rng.random_sample((runs, engine.n_snapshots, engine.n)).astype(np.float32)
    initial = np.zeros((runs, engine.n), dtype=np.int8)
    seeds = np.random.RandomState(config["calibration"]["seed"] + 1).randint(engine.n, size=runs)
    initial[np.arange(runs), seeds] = 1
    t0 = time.perf_counter()
    rate = attack_rate(engine, "M1", lam, initial, uniforms)
    elapsed = time.perf_counter() - t0
    write_json(out_dir / "benchmark.json", {
        "dataset": bench_name, "runs": runs, "mechanism": "M1",
        "lambda": lam, "attack_rate": rate,
        "seconds": elapsed, "runs_per_second": runs / elapsed,
        "snapshots": engine.n_snapshots, "N": engine.n,
    })
    print("  benchmark: %.2fs for %d runs (%.1f runs/s)" % (
        elapsed, runs, runs / elapsed), flush=True)

    write_json(out_dir / "summary.json", {
        "total_seconds": time.perf_counter() - started_all,
        "artifacts": sorted(p.name for p in out_dir.iterdir()),
    })
    print("done in %.1fs" % (time.perf_counter() - started_all), flush=True)

if __name__ == "__main__":
    main()
