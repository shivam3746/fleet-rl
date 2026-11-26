"""Replay shared demand traces through the classical fleet policies."""

import argparse
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .city import GridCity, Move
from .demand import DemandTrace, generate_trace
from .dispatch import greedy_assignments, hungarian_assignments
from .env import FleetDispatchEnv


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


def evaluate_baselines(
    city: GridCity, traces: Iterable[DemandTrace], *, reposition_seed: int = 0
) -> list[BaselineResult]:
    """Replay identical traces for each policy and pool waits over served requests."""
    traces = tuple(traces)
    if not traces:
        raise ValueError("Provide at least one demand trace")
    episode_seeds = np.random.SeedSequence(reposition_seed).spawn(len(traces))
    policies = (
        ("Greedy (stay)", greedy_assignments, False),
        ("Hungarian (stay)", hungarian_assignments, False),
        ("Hungarian (random reposition)", hungarian_assignments, True),
    )
    results = []
    for name, dispatcher, random_reposition in policies:
        requests = pickup_minutes = reposition_minutes = 0
        pickup_waits = []
        for trace, episode_seed in zip(traces, episode_seeds):
            fleet_size = len(trace.initial_zones)
            env = FleetDispatchEnv(city, fleet_size=fleet_size, dispatcher=dispatcher)
            observation, _ = env.reset(options={"trace": trace})
            stay = np.full(fleet_size, Move.STAY, dtype=np.int64)
            if random_reposition:
                env.action_space.seed(int(episode_seed.generate_state(1, dtype=np.uint64)[0]))

            terminated = False
            while not terminated:
                action = stay
                if random_reposition:
                    action = env.action_space.sample(mask=tuple(observation["action_mask"]))
                observation, _, terminated, _, info = env.step(action)

            metrics = info["episode_metrics"]
            requests += metrics["requests"]
            pickup_waits.extend(metrics["pickup_waits"])
            pickup_minutes += metrics["pickup_travel_minutes"]
            reposition_minutes += metrics["reposition_travel_minutes"]

        results.append(BaselineResult(
            policy=name,
            requests=requests,
            served_requests=len(pickup_waits),
            mean_pickup_wait=float(np.mean(pickup_waits)) if pickup_waits else None,
            p90_pickup_wait=float(np.percentile(pickup_waits, 90)) if pickup_waits else None,
            pickup_travel_minutes=pickup_minutes,
            reposition_travel_minutes=reposition_minutes,
        ))
    return results


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
