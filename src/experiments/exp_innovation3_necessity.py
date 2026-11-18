
from __future__ import print_function

import argparse
import json
from pathlib import Path

from .innovation3_core import run_all_innovation3
from .innovation3_outputs import (
    create_innovation3_figures,
    write_innovation3_tables,
)

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None):
    path = (
        ROOT / "configs" / "innovation3.json"
        if path is None
        else Path(path)
    )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def run_experiment(config=None):
    config = load_config() if config is None else dict(config)
    return run_all_innovation3(config)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "summary_tables"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(ROOT / "results" / "figures"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_experiment(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "innovation3_higher_order_necessity.json"
    tables = write_innovation3_tables(result, output_dir)
    figures = create_innovation3_figures(result, args.figure_dir)
    result["artifacts"] = {"tables": tables, "figures": figures}
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(str(result_path.resolve()))
    print(
        "tables=%d figures=%d overall_passed=%s runtime=%.2fs"
        % (
            len(tables),
            len(figures),
            result["acceptance"]["overall_passed"],
            result["runtime_seconds"],
        )
    )

if __name__ == "__main__":
    main()
