import unittest

import torch

from fleet_rl.city import GridCity
from fleet_rl.env import FleetDispatchEnv
from fleet_rl.policy import GraphFleetPolicy, MLPFleetPolicy
from fleet_rl.ppo import PPOConfig, RolloutCollector, generalized_advantages, ppo_update


class PPOTests(unittest.TestCase):
    def test_advantages_stop_at_termination_and_bootstrap_at_rollout_end(self):
        rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        values = torch.tensor([[0.5, 1.0], [0.7, 1.5], [0.9, 2.0]])
        terminated = torch.tensor([[False, True], [True, False], [False, False]])

        actual = generalized_advantages(
            rewards, values, terminated, torch.tensor([1.2, 2.5]), gamma=0.9, gae_lambda=0.8
        )

        expected = torch.tensor([[2.786, 1.0], [2.3, 8.8], [5.18, 6.25]])
        torch.testing.assert_close(actual, expected)

    def test_rollout_uses_legal_actions_and_ppo_updates_both_models(self):
        city = GridCity(rows=2, columns=3)
        config = PPOConfig(rollout_steps=8, epochs=1, minibatch_size=8)
        for policy_type in (GraphFleetPolicy, MLPFleetPolicy):
            with self.subTest(policy=policy_type.__name__):
                torch.manual_seed(7)
                policy = policy_type(city, hidden_size=8)
                optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
                collector = RolloutCollector([
                    FleetDispatchEnv(city, fleet_size=2, arrival_cutoff=3, patience=2)
                    for _ in range(2)
                ])
                rollout = collector.collect(policy, config)

                self.assertEqual(rollout.actions.shape, (16, 2))
                self.assertEqual(rollout.returns.shape, (16,))
                allowed = rollout.observations["action_mask"].gather(-1, rollout.actions.unsqueeze(-1))
                self.assertTrue(allowed.bool().all().item())
                self.assertTrue(torch.isfinite(rollout.old_log_probs).all().item())
                before = [parameter.detach().clone() for parameter in policy.parameters()]
                metrics = ppo_update(policy, optimizer, rollout, config)

                self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all().item())
                for parameters in (policy.actor.parameters(), policy.critic.parameters()):
                    parameter_ids = {id(parameter) for parameter in parameters}
                    self.assertTrue(any(
                        id(parameter) in parameter_ids and not torch.equal(parameter, previous)
                        for parameter, previous in zip(policy.parameters(), before)
                    ))


if __name__ == "__main__":
    unittest.main()
