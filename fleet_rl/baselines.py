"""Replay shared demand traces through the classical fleet policies."""

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .city import City, GridCity, Move
from .demand import DemandTrace, generate_trace
from .dispatch import greedy_assignments, hungarian_assignments
from .env import FleetDispatchEnv, Observation


@dataclass(frozen=True)
class EpisodeResult:
    requests: int
    pickup_waits: tuple[int, ...]
    pickup_travel_minutes: int
    reposition_travel_minutes: int

    @property
    def empty_travel_minutes(self) -> int:
        return self.pickup_travel_minutes + self.reposition_travel_minutes


@dataclass(frozen=True)
class BaselineResult:
    policy: str
    requests: int
    served_requests: int
    mean_pickup_wait: float | None
    p90_pickup_wait: float | None
    pickup_travel_minutes: int
    reposition_travel_minutes: int

    @property
    def service_rate(self) -> float | None:
        return self.served_requests / self.requests if self.requests else None

    @property
    def empty_travel_minutes(self) -> int:
        return self.pickup_travel_minutes + self.reposition_travel_minutes


def replay_episode(
    env: FleetDispatchEnv, trace: DemandTrace, choose_action: Callable[[Observation], np.ndarray]
) -> EpisodeResult:
    observation, _ = env.reset(options={"trace": trace})
    while True:
        observation, _, terminated, _, info = env.step(choose_action(observation))
        if info["masked_actions"]:
            raise RuntimeError("The evaluated policy selected a masked action")
        if terminated:
            metrics = info["episode_metrics"]
            return EpisodeResult(
                metrics["requests"], metrics["pickup_waits"],
                metrics["pickup_travel_minutes"], metrics["reposition_travel_minutes"],
            )


def summarize_episodes(name: str, episodes: list[EpisodeResult]) -> BaselineResult:
    pickup_waits = [wait for episode in episodes for wait in episode.pickup_waits]
    return BaselineResult(
        policy=name,
        requests=sum(episode.requests for episode in episodes),
        served_requests=len(pickup_waits),
        mean_pickup_wait=float(np.mean(pickup_waits)) if pickup_waits else None,
        p90_pickup_wait=float(np.percentile(pickup_waits, 90)) if pickup_waits else None,
        pickup_travel_minutes=sum(episode.pickup_travel_minutes for episode in episodes),
        reposition_travel_minutes=sum(episode.reposition_travel_minutes for episode in episodes),
    )


def baseline_episodes(
    city: City, traces: Iterable[DemandTrace], *, reposition_seed: int = 0
) -> dict[str, list[EpisodeResult]]:
    """Replay identical traces for every baseline and preserve episode order."""
    traces = tuple(traces)
    if not traces:
        raise ValueError("Provide at least one demand trace")
    episode_seeds = np.random.SeedSequence(reposition_seed).spawn(len(traces))
    policies = (
        ("Greedy (stay)", greedy_assignments, False),
        ("Hungarian (stay)", hungarian_assignments, False),
        ("Hungarian (random reposition)", hungarian_assignments, True),
    )
    results = {}
    for name, dispatcher, random_reposition in policies:
        episodes = []
        for trace, episode_seed in zip(traces, episode_seeds):
            fleet_size = len(trace.initial_zones)
            env = FleetDispatchEnv(city, fleet_size=fleet_size, dispatcher=dispatcher)
            stay = np.full(fleet_size, Move.STAY, dtype=np.int64)
            if random_reposition:
                env.action_space.seed(int(episode_seed.generate_state(1, dtype=np.uint64)[0]))

            def choose_action(observation: Observation) -> np.ndarray:
                if random_reposition:
                    return env.action_space.sample(mask=tuple(observation["action_mask"]))
                return stay

            episodes.append(replay_episode(env, trace, choose_action))
        results[name] = episodes
    return results


def evaluate_baselines(
    city: City, traces: Iterable[DemandTrace], *, reposition_seed: int = 0
) -> list[BaselineResult]:
    """Summarize baselines using pooled waits over served requests."""
    results = baseline_episodes(city, traces, reposition_seed=reposition_seed)
    return [summarize_episodes(name, episodes) for name, episodes in results.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1000, help="first demand seed")
    parser.add_argument("--episodes", type=int, default=20, help="number of consecutive demand seeds")
    parser.add_argument("--reposition-seed", type=int, default=0, help="seed for random repositioning")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or args.reposition_seed < 0:
        parser.error("provide a positive episode count and nonnegative seeds")

    city = GridCity()
    seed_end = args.seed_start + args.episodes
    traces = (generate_trace(city, seed=seed) for seed in range(args.seed_start, seed_end))
    results = evaluate_baselines(city, traces, reposition_seed=args.reposition_seed)
    print(f"Demand seeds {args.seed_start} through {seed_end - 1}, reposition seed {args.reposition_seed}.")
    print()
    print("| Policy | Served / requests | Service rate | Mean wait (min) | P90 wait (min) | Empty travel (min) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        service_rate = result.service_rate
        rate = "n/a" if service_rate is None else f"{service_rate:.2%}"
        mean = "n/a" if result.mean_pickup_wait is None else f"{result.mean_pickup_wait:.2f}"
        p90 = "n/a" if result.p90_pickup_wait is None else f"{result.p90_pickup_wait:.2f}"
        print(
            f"| {result.policy} | {result.served_requests:,} / {result.requests:,} | "
            f"{rate} | {mean} | {p90} | {result.empty_travel_minutes:,} |"
        )


if __name__ == "__main__":
    main()
