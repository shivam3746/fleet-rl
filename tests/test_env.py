import unittest

import numpy as np
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from fleet_rl.city import GridCity, Move
from fleet_rl.demand import DemandTrace, generate_trace
from fleet_rl.env import FleetDispatchEnv
from fleet_rl.simulation import Request


class FleetDispatchEnvTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        env = FleetDispatchEnv()
        check_env(env, skip_render_check=True, skip_close_check=True)
        self.assertEqual(env.edge_index.shape, (2, 170))
        self.assertEqual(tuple(map(tuple, env.edge_index.T)), env.city.directed_edges())

    def test_reset_seed_matches_generated_trace(self):
        env = FleetDispatchEnv()
        trace = generate_trace(env.city, seed=42)
        generated, _ = env.reset(seed=42)
        replayed, _ = env.reset(options={"trace": trace})

        for key in generated:
            np.testing.assert_array_equal(generated[key], replayed[key])

    def test_scripted_episode_reproduces_rewards_and_final_metrics(self):
        env = FleetDispatchEnv(GridCity(rows=1, columns=6), fleet_size=1, patience=3)
        trace = DemandTrace((0,), (
            Request(0, 1, 2, 1),
            Request(0, 5, 0, 2),
            Request(2, 4, 5, 4),
        ), arrival_cutoff=3)
        observation, _ = env.reset(options={"trace": trace})
        np.testing.assert_array_equal(observation["clock"], [0, 3])
        np.testing.assert_array_equal(observation["zone_features"][2], [0, 0, 0, 1])
        np.testing.assert_array_equal(observation["zone_features"][5], [0, 1, 0, 0])
        np.testing.assert_array_equal(observation["vehicle_zones"], [-1])
        np.testing.assert_array_equal(observation["action_mask"], [[1, 0, 0, 0, 0]])

        for minute, expected_reward in enumerate((-2.0, -7.0, -1.0, -1.0), start=1):
            observation, reward, terminated, truncated, info = env.step(np.array([Move.EAST]))
            self.assertEqual(reward, expected_reward)
            self.assertEqual(info["masked_actions"], 1)
            self.assertEqual(terminated, minute == 4)
            self.assertFalse(truncated)
            self.assertTrue(env.observation_space.contains(observation))
            if minute == 1:
                np.testing.assert_array_equal(observation["zone_features"][5], [0, 1, 1, 0])
            if minute == 2:
                self.assertEqual(info["expired_requests"], 1)
                self.assertEqual(observation["zone_features"][5, 1], 0)

        np.testing.assert_array_equal(observation["clock"], [4, 0])
        self.assertEqual(info["episode_metrics"], {
            "requests": 3,
            "served_requests": 2,
            "expired_requests": 1,
            "service_rate": 2 / 3,
            "mean_pickup_wait": 1.5,
            "p90_pickup_wait": 1.9,
            "pickup_waits": (1, 2),
            "waiting_passenger_minutes": 5,
            "pickup_travel_minutes": 3,
            "reposition_travel_minutes": 0,
            "empty_travel_minutes": 3,
        })
        with self.assertRaises(ResetNeeded):
            env.step(np.array([Move.STAY]))

    def test_valid_repositioning_and_boundary_mask(self):
        env = FleetDispatchEnv(GridCity(rows=1, columns=3), fleet_size=2, arrival_cutoff=1, arrival_rate=0)
        observation, _ = env.reset(options={"trace": DemandTrace((0, 2), (), 1)})
        np.testing.assert_array_equal(observation["action_mask"], [[1, 0, 0, 1, 0], [1, 0, 0, 0, 1]])
        observation, reward, terminated, _, info = env.step(np.array([Move.WEST, Move.WEST]))

        np.testing.assert_array_equal(observation["vehicle_zones"], [0, 1])
        np.testing.assert_array_equal(observation["zone_features"][:, 0], [1, 1, 0])
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertEqual(info["masked_actions"], 1)
        self.assertEqual(info["episode_metrics"]["empty_travel_minutes"], 1)
        self.assertIsNone(info["episode_metrics"]["service_rate"])
        self.assertIsNone(info["episode_metrics"]["mean_pickup_wait"])

    def test_zero_patience_expiry_is_charged_once_with_empty_fleet(self):
        env = FleetDispatchEnv(fleet_size=0, expiry_penalty=4)
        trace = DemandTrace((), (Request(0, 0, 1, 0),), 2)
        observation, info = env.reset(options={"trace": trace})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["initial_expiries"], 1)
        _, reward, terminated, _, info = env.step(env.action_space.sample())
        self.assertEqual((reward, terminated, info["expired_requests"]), (-4.0, False, 1))
        _, reward, terminated, _, info = env.step(env.action_space.sample())
        self.assertEqual((reward, terminated, info["expired_requests"]), (0.0, True, 0))
        self.assertEqual(info["episode_metrics"]["expired_requests"], 1)

    def test_malformed_actions_do_not_advance_the_clock(self):
        env = FleetDispatchEnv(fleet_size=1, arrival_rate=0)
        with self.assertRaises(ResetNeeded):
            env.step(np.array([Move.STAY]))
        env.reset(seed=2)
        for action in ([1.0], [5], [[0]], [0, 0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        observation, *_ = env.step(np.array([Move.STAY]))
        self.assertEqual(observation["clock"][0], 1)


if __name__ == "__main__":
    unittest.main()
