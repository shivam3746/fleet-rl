import unittest

import numpy as np

from fleet_rl.baselines import EpisodeResult
from fleet_rl.evaluate import paired_differences


class HeldOutEvaluationTests(unittest.TestCase):
    def test_paired_intervals_pool_pickups_by_passenger(self):
        graph = [EpisodeResult(2, (1, 2), 6, 2), EpisodeResult(6, (3,) * 4, 10, 4)]
        baseline = [EpisodeResult(2, (4,), 5, 0), EpisodeResult(6, (5,) * 5, 12, 0)]
        draws = np.array([[0, 0], [1, 1], [0, 1]])

        deltas = paired_differences(graph, baseline, draws)

        self.assertEqual(deltas["service_rate_pp"]["difference"], 0)
        self.assertAlmostEqual(deltas["mean_pickup_wait"]["difference"], 15 / 6 - 29 / 6)
        self.assertEqual(deltas["mean_empty_travel"]["difference"], 2.5)
        np.testing.assert_allclose(deltas["mean_empty_travel"]["ci_95"], [2.025, 2.975])

    def test_requires_paired_traces(self):
        left = [EpisodeResult(2, (1,), 1, 0)]
        right = [EpisodeResult(3, (1,), 1, 0)]
        with self.assertRaises(ValueError):
            paired_differences(left, right, np.array([[0]]))

    def test_no_pickups_leaves_wait_difference_undefined(self):
        episode = EpisodeResult(2, (), 0, 0)
        delta = paired_differences([episode], [episode], np.array([[0]]))
        self.assertIsNone(delta["mean_pickup_wait"]["difference"])
        self.assertIsNone(delta["mean_pickup_wait"]["ci_95"])
        self.assertEqual(delta["service_rate_pp"]["difference"], 0)


if __name__ == "__main__":
    unittest.main()
