"""Evaluate validation-selected PPO checkpoints on shared held-out traces."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .baselines import EpisodeResult, baseline_episodes, replay_episode, summarize_episodes
from .city import GridCity
from .demand import generate_trace
from .env import FleetDispatchEnv, Observation
from .train import load_checkpoint

BENCHMARK_SEEDS = tuple(range(1000, 1020))
REPOSITION_SEED = 0
BOOTSTRAP_SEED = 6
BOOTSTRAP_DRAWS = 10_000


def paired_differences(
    graph: list[EpisodeResult], reference: list[EpisodeResult], draws: np.ndarray
) -> dict[str, dict[str, float | list[float] | None]]:
    """Estimate GraphSAGE minus reference with episode-paired resampling."""
    if not graph or [episode.requests for episode in graph] != [episode.requests for episode in reference]:
        raise ValueError("Policies must have the same ordered episode request counts")

    def metrics(episodes: list[EpisodeResult], indices: np.ndarray) -> tuple[np.ndarray, ...]:
        requests = np.array([episode.requests for episode in episodes])[indices].sum(axis=-1)
        served = np.array([len(episode.pickup_waits) for episode in episodes])[indices].sum(axis=-1)
        waits = np.array([sum(episode.pickup_waits) for episode in episodes])[indices].sum(axis=-1)
        empty_travel = np.array([episode.empty_travel_minutes for episode in episodes])[indices].mean(axis=-1)
        rate = np.divide(100.0 * served, requests, out=np.full(requests.shape, np.nan), where=requests != 0)
        mean_wait = np.divide(waits, served, out=np.full(served.shape, np.nan), where=served != 0)
        return rate, mean_wait, empty_travel

    indices = np.concatenate((np.arange(len(graph))[None, :], draws), axis=0)
    names = ("service_rate_pp", "mean_pickup_wait", "mean_empty_travel")
    differences = {}
    for name, learned, baseline in zip(names, metrics(graph, indices), metrics(reference, indices)):
        delta = learned - baseline
        valid = delta[1:][np.isfinite(delta[1:])]
        differences[name] = {
            "difference": float(delta[0]) if np.isfinite(delta[0]) else None,
            "ci_95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
        }
    return differences


def _evaluate_grid(city: GridCity, checkpoints: dict[str, Path]) -> dict:
    traces = tuple(generate_trace(city, seed=seed) for seed in BENCHMARK_SEEDS)
    episodes_by_policy = baseline_episodes(city, traces, reposition_seed=REPOSITION_SEED)
    for name, path in checkpoints.items():
        policy = load_checkpoint(path, city=city)
        env = FleetDispatchEnv(city, fleet_size=len(traces[0].initial_zones))

        def choose_action(observation: Observation) -> np.ndarray:
            return policy.act(observation, deterministic=True).numpy()

        episodes_by_policy[name] = [replay_episode(env, trace, choose_action) for trace in traces]

    summary = {}
    for name, episodes in episodes_by_policy.items():
        result = summarize_episodes(name, episodes)
        summary[name] = {
            **asdict(result),
            "service_rate": result.service_rate,
            "empty_travel_minutes": result.empty_travel_minutes,
            "mean_empty_travel": result.empty_travel_minutes / len(episodes),
        }
    count = len(BENCHMARK_SEEDS)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(count, size=(BOOTSTRAP_DRAWS, count))
    return {
        "city": asdict(city),
        "summary": summary,
        "graphsage_minus_reference": {
            name: paired_differences(episodes_by_policy["GraphSAGE"], episodes, draws)
            for name, episodes in episodes_by_policy.items()
            if name != "GraphSAGE"
        },
        "episodes": {
            name: [{"seed": seed, **asdict(episode)} for seed, episode in zip(BENCHMARK_SEEDS, episodes)]
            for name, episodes in episodes_by_policy.items()
        },
    }


def _plot_tradeoffs(path: Path, suites: dict[str, dict]) -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    figure = Figure(figsize=(11, 7), layout="constrained")
    axes = figure.subplots(2, 2, sharex="row", sharey=True)
    styles = {
        "Greedy (stay)": ("#595e67", "o"),
        "Hungarian (stay)": ("#426c9c", "s"),
        "Hungarian (random reposition)": ("#168378", "D"),
        "GraphSAGE": ("#a13862", "*"),
        "MLP": ("#b75b20", "^"),
    }
    for column, suite in enumerate(suites.values()):
        city = suite["city"]
        title = f'{city["rows"] * city["columns"]}-zone grid'
        if column:
            title += " (no retraining)"
        axes[0, column].set_title(title, loc="left", fontsize=12, fontweight="bold")
        for row, (metric, label) in enumerate((
            ("mean_pickup_wait", "Mean pickup wait (min)"),
            ("mean_empty_travel", "Empty travel per episode (vehicle-min)"),
        )):
            axis = axes[row, column]
            for name, summary in suite["summary"].items():
                color, marker = styles[name]
                axis.scatter(summary[metric], summary["service_rate"], color=color, marker=marker,
                             s=170 if name == "GraphSAGE" else 72, label=name, zorder=3)
            axis.set(xlabel=label, ylabel="Service rate")
            axis.margins(x=0.12, y=0.09)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            axis.grid(alpha=0.25)
            if row == 1:
                axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1, 0].set_xlim(left=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("Fleet dispatch on shared held-out demand traces", fontweight="bold")
    figure.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="PPO run with the validation-selected checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for evaluation artifacts")
    args = parser.parse_args()
    checkpoints = {name: args.run_dir / model / "best.pt" for name, model in (("GraphSAGE", "graph"), ("MLP", "mlp"))}
    for path in checkpoints.values():
        if not path.is_file():
            parser.error(f"missing checkpoint: {path}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")

    torch.set_num_threads(1)
    suites = {
        "training_grid": _evaluate_grid(GridCity(), checkpoints),
        "larger_grid": _evaluate_grid(GridCity(rows=6, columns=12), {"GraphSAGE": checkpoints["GraphSAGE"]}),
    }
    report = {
        "demand_seeds": BENCHMARK_SEEDS,
        "reposition_seed": REPOSITION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "checkpoint_selection": "best validation return; deterministic legal actions",
        "checkpoints": {
            name: {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in checkpoints.items()
        },
        "suites": suites,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot_tradeoffs(args.output_dir / "held_out_comparison.png", suites)
    for suite in suites.values():
        city = suite["city"]
        print(f'\n{city["rows"]} x {city["columns"]} grid')
        for name, summary in suite["summary"].items():
            print(f'{name:30} {summary["service_rate"]:7.2%} served | '
                  f'{summary["mean_pickup_wait"]:.2f} min mean wait | '
                  f'{summary["mean_empty_travel"]:,.1f} empty vehicle-min/episode')
    print(f"\nEvaluation saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
