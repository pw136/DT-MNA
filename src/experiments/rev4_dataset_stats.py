
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.conversion_validation_data import REAL_PROFILES  
from src.experiments.exp_paper_core import chronological_blocks, chronological_events  

UNIT_SECONDS = {"Email-Enron": 1e-3, "NDC-classes-full": 1e-3,
                "Primary-School": 1.0, "High-School": 1.0}
LIMITS = (600, 3000, 10000)

out = {}
for name, prof in REAL_PROFILES.items():
    folder = ROOT / "data" / prof["folder"]
    sizes = np.loadtxt(folder / (prof["folder"] + "-nverts.txt"), dtype=np.int64)
    rows = chronological_events(name, 6)
    kept_sizes = np.array([len(r[2]) for r in rows])
    nodes = {v for r in rows for v in r[2]}
    span_s = (rows[-1][0] - rows[0][0]) * UNIT_SECONDS[name]
    dist = {int(m): float((kept_sizes == m).mean()) for m in range(2, 7)}
    windows = []
    for b, block in enumerate(chronological_blocks(rows, 3)):
        for limit in LIMITS:
            if len(block) < limit:
                continue
            w = block[:limit]
            windows.append({"block": b, "events": limit,
                            "nodes": len({v for r in w for v in r[2]}),
                            "span_days": (w[-1][0] - w[0][0]) * UNIT_SECONDS[name] / 86400.0,
                            "share_size_ge3": float(np.mean([len(r[2]) >= 3 for r in w]))})
    out[name] = {
        "raw_events": int(len(sizes)),
        "raw_size1": int((sizes == 1).sum()),
        "raw_size_gt6": int((sizes > 6).sum()),
        "retained_events": int(len(rows)),
        "nodes": len(nodes),
        "span_days": span_s / 86400.0,
        "size_distribution": dist,
        "share_size_ge3": float((kept_sizes >= 3).mean()),
        "windows": windows,
    }
    w = windows
    print("%-17s raw=%d kept=%d nodes=%d span=%.2f d  sizes=%s  >=3:%.3f  windows=%d nodes %d-%d span %.3g-%.3g d"
          % (name, len(sizes), len(rows), len(nodes), span_s / 86400.0,
             {k: round(v, 3) for k, v in dist.items()}, out[name]["share_size_ge3"], len(w),
             min(x["nodes"] for x in w), max(x["nodes"] for x in w),
             min(x["span_days"] for x in w), max(x["span_days"] for x in w)))
p = ROOT / "results" / "rev4" / "dataset_stats.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1))
