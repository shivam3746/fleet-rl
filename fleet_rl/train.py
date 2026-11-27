"""Train fleet policies with PPO and write checkpoints and validation curves."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .city import City, GridCity
from .env import FleetDispatchEnv
from .policy import FleetPolicy, GraphFleetPolicy, MLPFleetPolicy
from .ppo import TRAIN_SEED_START, PPOConfig, RolloutCollector, ppo_update

POLICIES = {"graph": GraphFleetPolicy, "mlp": MLPFleetPolicy}
VALIDATION_SEEDS = tuple(range(2000, 2004))


@torch.no_grad()
def validate_policy(policy: FleetPolicy, env: FleetDispatchEnv) -> dict[str, float | None]:
    """Evaluate greedy action selection on the fixed validation traces."""
    returns = []
    pickup_waits = []
    total_requests = empty_minutes = 0
    for seed in VALIDATION_SEEDS:
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        while True:
            action = policy.act(observation, deterministic=True).cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break

        metrics = info["episode_metrics"]
        returns.append(episode_return)
        total_requests += metrics["requests"]
        pickup_waits.extend(metrics["pickup_waits"])
        empty_minutes += metrics["empty_travel_minutes"]

    return {
        "mean_return": float(np.mean(returns)),
        "service_rate": len(pickup_waits) / total_requests if total_requests else None,
        "mean_pickup_wait": float(np.mean(pickup_waits)) if pickup_waits else None,
        "p90_pickup_wait": float(np.percentile(pickup_waits, 90)) if pickup_waits else None,
        "mean_empty_travel": empty_minutes / len(VALIDATION_SEEDS),
    }


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu", *, city: City | None = None
) -> FleetPolicy:
    """Reload a policy for inference, optionally transferring GraphSAGE to a new grid."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    policy_type = POLICIES[checkpoint["policy"]]
    trained_city = GridCity(**checkpoint["city"])
    if city is not None and city != trained_city and policy_type is MLPFleetPolicy:
        raise ValueError("The MLP checkpoint requires its training grid")
    policy = policy_type(trained_city if city is None else city, hidden_size=checkpoint["hidden_size"]).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    return policy.eval()


def _plot_curves(path: Path, curves: dict[str, list[dict]]) -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(10, 6), layout="constrained")
    axes = figure.subplots(2, 2).flat
    panels = (
        ("mean_return", "Mean episode return"),
        ("service_rate", "Service rate"),
        ("mean_pickup_wait", "Mean pickup wait (min)"),
        ("mean_empty_travel", "Empty travel per episode (vehicle-min)"),
    )
    for axis, (metric, title) in zip(axes, panels):
        for name, records in curves.items():
            axis.plot(
                [record["transitions"] for record in records],
                [record["validation"][metric] for record in records],
                marker=".",
                label="GraphSAGE" if name == "graph" else "MLP",
            )
        axis.set(title=title, xlabel="Training transitions per policy")
        axis.grid(alpha=0.25)
        if metric == "service_rate":
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    figure.axes[0].legend()
    figure.suptitle("PPO validation on shared synthetic demand traces")
    figure.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for training artifacts")
    parser.add_argument("--policy", choices=("both", *POLICIES), default="both")
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7, help="model initialization and action sampling seed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.updates, args.num_envs, args.rollout_steps, args.eval_every, args.hidden_size) < 1 or args.seed < 0:
        parser.error("updates, environment count, rollout length, evaluation interval and hidden size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True)
    city = GridCity()
    config = PPOConfig(rollout_steps=args.rollout_steps)
    names = tuple(POLICIES) if args.policy == "both" else (args.policy,)
    run_config = {
        "policies": names,
        "city": asdict(city),
        "hidden_size": args.hidden_size,
        "updates": args.updates,
        "num_envs": args.num_envs,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "train_seed_start": TRAIN_SEED_START,
        "validation_seeds": VALIDATION_SEEDS,
        "ppo": asdict(config),
    }
    (args.output_dir / "run.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    curves: dict[str, list[dict]] = {name: [] for name in names}
    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for name in names:
            torch.manual_seed(args.seed)
            policy = POLICIES[name](city, hidden_size=args.hidden_size).to(args.device)
            optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
            collector = RolloutCollector([FleetDispatchEnv(city) for _ in range(args.num_envs)])
            validation_env = FleetDispatchEnv(city)
            best_return = float("-inf")
            policy_path = args.output_dir / name
            policy_path.mkdir()

            for update in range(args.updates + 1):
                transitions = update * args.num_envs * config.rollout_steps
                record = {"policy": name, "update": update, "transitions": transitions}
                if update:
                    rollout = collector.collect(policy, config)
                    record["mean_training_step_reward"] = rollout.mean_step_reward
                    record.update(ppo_update(policy, optimizer, rollout, config))

                if update % args.eval_every == 0 or update == args.updates:
                    validation = validate_policy(policy, validation_env)
                    record["validation"] = validation
                    curves[name].append(record)
                    mean_return = validation["mean_return"]
                    print(f"{name:5}  {transitions:>7,} transitions  validation return {mean_return:>8.2f}", flush=True)
                    checkpoint = {
                        "policy": name,
                        "city": asdict(city),
                        "hidden_size": args.hidden_size,
                        "update": update,
                        "transitions": transitions,
                        "validation": validation,
                        "state_dict": policy.state_dict(),
                    }
                    if update and mean_return > best_return:
                        torch.save(checkpoint, policy_path / "best.pt")
                        best_return = mean_return
                    if update == args.updates:
                        torch.save(checkpoint, policy_path / "final.pt")

                log.write(json.dumps(record, allow_nan=False) + "\n")

    _plot_curves(args.output_dir / "learning_curves.png", curves)
    print(f"Artifacts saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
