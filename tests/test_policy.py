import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from fleet_rl.city import GridCity, Move
from fleet_rl.demand import DemandTrace
from fleet_rl.env import FleetDispatchEnv
from fleet_rl.policy import GraphFleetPolicy, MLPFleetPolicy
from fleet_rl.simulation import Request


class FleetPolicyTests(unittest.TestCase):
    def setUp(self):
        self.city = GridCity(rows=2, columns=3)
        self.env = FleetDispatchEnv(self.city, fleet_size=3)
        trace = DemandTrace((0, 2, 4), (Request(0, 0, 3, 1),), 4)
        self.obs, _ = self.env.reset(options={"trace": trace})

    def test_policies_mask_busy_vehicles_and_boundary_moves(self):
        mask = torch.from_numpy(self.obs["action_mask"]).bool()
        self.assertEqual(self.obs["vehicle_zones"].tolist(), [-1, 2, 4])
        self.assertEqual(mask[1].tolist(), [True, False, True, False, True])

        for policy_type in (GraphFleetPolicy, MLPFleetPolicy):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(self.city, hidden_size=8)
                logits, value = policy(self.obs)
                torch.testing.assert_close(torch.isfinite(logits), mask)
                self.assertEqual(value.shape, ())
                self.assertTrue(torch.isfinite(value).item())
                for deterministic in (False, True):
                    action = policy.act(self.obs, deterministic=deterministic)
                    self.assertEqual(action[0].item(), Move.STAY)
                    self.assertTrue(mask[torch.arange(3), action].all().item())

    def test_batched_forward_matches_separate_observations(self):
        next_obs, *_ = self.env.step(np.array([Move.STAY, Move.SOUTH, Move.NORTH]))
        batch = {
            key: torch.from_numpy(np.stack((self.obs[key], next_obs[key])))
            for key in self.obs
        }
        for policy_type in (GraphFleetPolicy, MLPFleetPolicy):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(self.city, hidden_size=8)
                logits, values = policy(batch)
                expected_logits, expected_values = zip(policy(self.obs), policy(next_obs))
                torch.testing.assert_close(logits, torch.stack(expected_logits))
                torch.testing.assert_close(values, torch.stack(expected_values))

    def test_actor_and_critic_receive_finite_gradients(self):
        for policy_type in (GraphFleetPolicy, MLPFleetPolicy):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(self.city, hidden_size=8)
                logits, value = policy(self.obs)
                distribution = Categorical(logits=logits)
                stay = torch.zeros(3, dtype=torch.long)
                loss = -distribution.log_prob(stay).sum() - 0.01 * distribution.entropy().sum()
                loss = loss + (value - 1).square()
                loss.backward()

                self.assertTrue(torch.isfinite(loss).item())
                for parameter in policy.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_policies_handle_no_idle_vehicles(self):
        city = GridCity(rows=1, columns=2)
        env = FleetDispatchEnv(city, fleet_size=1)
        trace = DemandTrace((0,), (Request(0, 0, 1, 1),), 4)
        observation, _ = env.reset(options={"trace": trace})

        for policy_type in (GraphFleetPolicy, MLPFleetPolicy):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(city, hidden_size=8)
                action = policy.act(observation)
                self.assertEqual(action.tolist(), [Move.STAY])
                _, _, _, _, info = env.step(action.numpy())
                self.assertEqual(info["masked_actions"], 0)
                observation, _ = env.reset(options={"trace": trace})

    def test_graph_weights_load_on_a_larger_grid(self):
        city = GridCity(rows=6, columns=12)
        env = FleetDispatchEnv(city)
        observation, _ = env.reset(seed=12)
        source = GraphFleetPolicy(GridCity(), hidden_size=8)
        target = GraphFleetPolicy(city, hidden_size=8)
        target.load_state_dict(source.state_dict())

        action = target.act(observation).numpy()
        self.assertTrue(env.action_space.contains(action))
        _, _, _, _, info = env.step(action)
        self.assertEqual(info["masked_actions"], 0)

    def test_single_zone_and_empty_fleet_produce_empty_actions(self):
        city = GridCity(rows=1, columns=1)
        env = FleetDispatchEnv(city, fleet_size=0)
        observation, _ = env.reset(seed=12)
        for policy_type in (GraphFleetPolicy, MLPFleetPolicy):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(city, hidden_size=8)
                action = policy.act(observation).numpy()
                self.assertTrue(env.action_space.contains(action))


if __name__ == "__main__":
    unittest.main()
