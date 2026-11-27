"""Evaluate synthetic-grid GraphSAGE and the classical policies on Manhattan trips."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

from .baselines import baseline_episodes, replay_episode, summarize_episodes
from .demand import DemandTrace
from .env import FleetDispatchEnv, Observation
from .evaluate import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, REPOSITION_SEED, paired_differences
from .nyc_data import OSM_ATTRIBUTION, ROAD_URL, TRIP_URL, ZONE_URL, build_manhattan_city, load_demand_windows
from .road_city import RoadCity
from .train import load_checkpoint


def _source(path: Path, url: str) -> dict:
    with path.open("rb") as stream:
        return {"file": path.name, "source": url, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def _plot(path: Path, city: RoadCity, zones: gpd.GeoDataFrame, traces: list[DemandTrace], summary: dict) -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    figure = Figure(figsize=(10.5, 5.7), layout="constrained")
    map_axis, tradeoff = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.35]})
    pickups = Counter(city.zone_ids[request.origin] for trace in traces for request in trace.requests)
    zones.assign(sampled_pickups=zones.index.map(lambda zone: pickups[zone])).plot(
        column="sampled_pickups", cmap="YlGnBu", legend=True, ax=map_axis,
        edgecolor="#d3dce5", linewidth=0.35,
        legend_kwds={"label": "Sampled pickups", "shrink": 0.7},
    )
    map_axis.set_title("Sampled pickups by taxi zone", loc="left", fontweight="bold")
    map_axis.set_axis_off()
    map_axis.text(0.5, -0.04, OSM_ATTRIBUTION, transform=map_axis.transAxes, ha="center", fontsize=7)

    labels = ("Greedy (stay)", "Hungarian (stay)", "Hungarian (random reposition)", "GraphSAGE")
    colors = ("#595e67", "#426c9c", "#168378", "#a13862")
    markers = ("o", "s", "D", "*")
    for name, color, marker in zip(labels, colors, markers):
        metrics = summary[name]
        tradeoff.scatter(metrics["mean_empty_travel"], metrics["service_rate"], color=color,
                         marker=marker, s=160 if name == "GraphSAGE" else 70, label=name, zorder=3)
    tradeoff.set_title("Service and empty travel", loc="left", fontweight="bold")
    tradeoff.set(xlabel="Empty travel per episode (vehicle-min)", ylabel="Service rate")
    tradeoff.margins(x=0.2, y=0.22)
    tradeoff.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    tradeoff.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    tradeoff.grid(alpha=0.25)
    tradeoff.legend(loc="best", frameon=True, fontsize=8)
    figure.suptitle("Manhattan dispatch replay | January 8–14, 2024", fontweight="bold")
    figure.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="directory with OSM PBF, TLC zones ZIP, and TLC trips")
    parser.add_argument("--checkpoint", type=Path, required=True, help="validation-selected GraphSAGE checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for benchmark outputs")
    args = parser.parse_args()
    road_path = args.data_dir / "NewYork.osm.pbf"
    zone_path = args.data_dir / "taxi_zones.zip"
    trip_path = args.data_dir / "yellow_tripdata_2024-01.parquet"
    for path in (road_path, zone_path, trip_path, args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")

    torch.set_num_threads(1)
    city, zones, network = build_manhattan_city(road_path, zone_path)
    print(f"Built {city.zone_count} taxi zones from {network['road_nodes']:,} road nodes.", flush=True)
    windows = load_demand_windows(trip_path, city)
    traces = list(windows.values())
    print(f"Loaded {sum(len(trace.requests) for trace in traces):,} requests in {len(traces)} windows.", flush=True)
    episodes = baseline_episodes(city, traces, reposition_seed=REPOSITION_SEED)
    policy = load_checkpoint(args.checkpoint, city=city)
    env = FleetDispatchEnv(city, fleet_size=len(traces[0].initial_zones))

    def choose_action(observation: Observation) -> np.ndarray:
        return policy.act(observation, deterministic=True).numpy()

    episodes["GraphSAGE"] = [replay_episode(env, trace, choose_action) for trace in traces]
    summary = {}
    for name, runs in episodes.items():
        aggregate = summarize_episodes(name, runs)
        summary[name] = {
            **asdict(aggregate), "service_rate": aggregate.service_rate,
            "empty_travel_minutes": aggregate.empty_travel_minutes,
            "mean_empty_travel": aggregate.empty_travel_minutes / len(traces),
        }
        print(f"{name:30} {aggregate.service_rate:7.2%} served | "
              f"{aggregate.mean_pickup_wait:.2f} min mean wait | "
              f"{summary[name]['mean_empty_travel']:,.1f} empty vehicle-min/window", flush=True)

    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(traces), size=(BOOTSTRAP_DRAWS, len(traces)))
    report = {
        "inputs": {
            "roads": _source(road_path, ROAD_URL), "zone_boundaries": _source(zone_path, ZONE_URL),
            "metered_trips": _source(trip_path, TRIP_URL),
            "checkpoint": _source(args.checkpoint, args.checkpoint.as_posix()),
        },
        "road_network": network,
        "window_starts_local": list(windows),
        "fleet_size": len(traces[0].initial_zones),
        "window_minutes": 120,
        "sample_fraction": 0.04,
        "pickup_patience_minutes": 5,
        "reposition_seed": REPOSITION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "summary": summary,
        "graphsage_minus_reference": {
            name: paired_differences(episodes["GraphSAGE"], runs, draws)
            for name, runs in episodes.items() if name != "GraphSAGE"
        },
        "episodes": {
            name: [{"window": start, **asdict(episode)} for start, episode in zip(windows, runs)]
            for name, runs in episodes.items()
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manhattan_evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot(args.output_dir / "manhattan_comparison.png", city, zones, traces, summary)
    print(f"Artifacts saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
