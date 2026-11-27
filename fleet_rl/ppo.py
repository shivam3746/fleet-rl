"""Rollout collection and clipped PPO updates for fleet policies."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from .env import FleetDispatchEnv
from .policy import FleetPolicy

TRAIN_SEED_START = 10_000


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1


@dataclass(frozen=True)
class Rollout:
    observations: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    mean_step_reward: float


def generalized_advantages(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    bootstrap_value: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    """Calculate GAE for time-major rollouts and cut it at episode boundaries."""
    advantages = torch.empty_like(rewards)
    next_value = bootstrap_value
    next_advantage = torch.zeros_like(bootstrap_value)
    for step in reversed(range(len(rewards))):
        continuing = ~terminated[step]
        delta = rewards[step] + gamma * next_value * continuing - values[step]
        next_advantage = delta + gamma * gae_lambda * continuing * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    return advantages


class RolloutCollector:
    """Run independent environments with matching seed schedules across policies."""

    def __init__(self, environments: list[FleetDispatchEnv]) -> None:
        if not environments:
            raise ValueError("Provide at least one training environment")
        self._envs = environments
        self._next_seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + len(environments)))
        self._observations = [self._reset(index) for index in range(len(environments))]

    def _reset(self, index: int) -> dict[str, np.ndarray]:
        observation, _ = self._envs[index].reset(seed=self._next_seeds[index])
        self._next_seeds[index] += len(self._envs)
        return observation

    def _batch(self, device: torch.device) -> dict[str, Tensor]:
        return {
            key: torch.as_tensor(np.stack([obs[key] for obs in self._observations]), device=device)
            for key in self._observations[0]
        }

    @torch.no_grad()
    def collect(self, policy: FleetPolicy, config: PPOConfig) -> Rollout:
        device = next(policy.parameters()).device
        observations = []
        actions = []
        log_probs = []
        values = []
        rewards = np.empty((config.rollout_steps, len(self._envs)), dtype=np.float32)
        terminated = np.empty(rewards.shape, dtype=bool)
        for step in range(config.rollout_steps):
            batch = self._batch(device)
            logits, value = policy(batch)
            distribution = Categorical(logits=logits, validate_args=False)
            action = distribution.sample()
            observations.append(batch)
            actions.append(action)
            log_probs.append(distribution.log_prob(action).sum(dim=-1))
            values.append(value)

            for index, (env, moves) in enumerate(zip(self._envs, action.cpu().numpy())):
                observation, reward, done, _, _ = env.step(moves)
                rewards[step, index] = reward
                terminated[step, index] = done
                self._observations[index] = self._reset(index) if done else observation

        _, bootstrap_value = policy(self._batch(device))
        value_tensor = torch.stack(values)
        advantages = generalized_advantages(
            torch.as_tensor(rewards, device=device) * config.reward_scale,
            value_tensor,
            torch.as_tensor(terminated, device=device),
            bootstrap_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        return Rollout(
            observations={key: torch.cat([batch[key] for batch in observations]) for key in observations[0]},
            actions=torch.cat(actions),
            old_log_probs=torch.cat(log_probs),
            advantages=advantages.flatten(),
            returns=(advantages + value_tensor).flatten(),
            mean_step_reward=float(rewards.mean()),
        )


def ppo_update(
    policy: FleetPolicy, optimizer: torch.optim.Optimizer, rollout: Rollout, config: PPOConfig
) -> dict[str, float]:
    """Optimize the joint action ratio for one fleet transition at a time."""
    advantages = rollout.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = len(advantages)
    totals = torch.zeros(4, device=advantages.device)
    for _ in range(config.epochs):
        order = torch.randperm(sample_count, device=advantages.device)
        for indices in order.split(config.minibatch_size):
            observation = {key: batch[indices] for key, batch in rollout.observations.items()}
            logits, value = policy(observation)
            distribution = Categorical(logits=logits, validate_args=False)
            log_ratio = distribution.log_prob(rollout.actions[indices]).sum(dim=-1) - rollout.old_log_probs[indices]
            ratio = log_ratio.exp()
            clipped_ratio = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
            batch_advantages = advantages[indices]
            policy_loss = -torch.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            value_loss = 0.5 * (value - rollout.returns[indices]).square().mean()
            choosing = (observation["action_mask"].sum(dim=-1) > 1).sum(dim=-1).clamp_min(1)
            entropy = (distribution.entropy().sum(dim=-1) / choosing).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            approx_kl = (ratio - 1 - log_ratio).mean()
            totals += torch.stack((policy_loss, value_loss, entropy, approx_kl)).detach() * len(indices)

    policy_loss, value_loss, entropy, approx_kl = (totals / (sample_count * config.epochs)).tolist()
    return {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy, "approx_kl": approx_kl}
