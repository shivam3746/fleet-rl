"""Export FleetLab and held-out replay recordings for static hosting."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .dashboard import Dashboard, _WEB
from .evaluate import BENCHMARK_SEEDS
from .operator_llm import OperatorCommand


def export_dashboard(runs: Path, output: Path) -> None:
    dashboard = Dashboard(runs)
    reports = dashboard.reports()
    if set(reports["benchmarks"]) != {"synthetic", "transfer", "manhattan"} or not reports["training"]:
        raise ValueError("The export needs the completed benchmark and training reports")
    data = output / "data"
    data.mkdir(parents=True)
    index = (_WEB / "index.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        index.replace('name="fleetlab-mode" content="local"', 'name="fleetlab-mode" content="recorded"'),
        encoding="utf-8",
    )
    for file in ("dashboard.css", "dashboard.js"):
        shutil.copyfile(_WEB / file, output / file)
    (output / ".nojekyll").touch()
    (data / "reports.json").write_text(json.dumps(reports, separators=(",", ":")), encoding="utf-8")
    for seed in BENCHMARK_SEEDS:
        frames = [dashboard.replay(OperatorCommand("start", "synthetic", str(seed)))]
        while not frames[-1]["complete"]:
            frames.append(dashboard.replay(OperatorCommand("advance", argument="1")))
        (data / f"replay-{seed}.json").write_text(json.dumps(frames, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True, help="new directory for static assets")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("the output directory already exists")
    torch.set_num_threads(1)
    export_dashboard(args.runs, args.output)
    print(f"Exported FleetLab to {args.output.resolve()}")


if __name__ == "__main__":
    main()
