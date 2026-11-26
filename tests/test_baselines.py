import unittest

from fleet_rl.baselines import evaluate_baselines
from fleet_rl.city import GridCity
from fleet_rl.demand import DemandTrace
from fleet_rl.dispatch import greedy_assignments
from fleet_rl.simulation import FleetSimulation, Request


class BaselineTests(unittest.TestCase):
    def test_greedy_uses_arrival_order_then_nearest_feasible_vehicle(self):
        city = GridCity(rows=1, columns=3)
        requests = [Request(1, 0, 1, 1), Request(0, 1, 2, 3)]
        sim = FleetSimulation(city, [0, 2], requests, arrival_cutoff=2)
        sim.dispatch()
        sim.advance()

        assignments = greedy_assignments(sim)

        # Request 1 arrived first. Both vehicles are equally close to it.
        self.assertEqual(assignments, {0: 1})
        self.assertEqual(sim.dispatch(assignments), 1)

    def test_stationary_policies_replay_shared_traces_and_pool_passenger_waits(self):
        city = GridCity(rows=1, columns=5)
        traces = (
            DemandTrace((0, 4), (Request(0, 0, 2, 4), Request(0, 1, 2, 1)), 1),
            DemandTrace((0,), (Request(0, 0, 0, 2),) * 3, 1),
        )
        greedy, hungarian, random = evaluate_baselines(city, iter(traces))

        self.assertEqual((greedy.requests, hungarian.requests, random.requests), (5, 5, 5))
        self.assertEqual((greedy.served_requests, hungarian.served_requests), (4, 5))
        self.assertEqual((greedy.service_rate, hungarian.service_rate), (0.8, 1.0))
        self.assertAlmostEqual(greedy.mean_pickup_wait, 0.75)
        self.assertAlmostEqual(hungarian.mean_pickup_wait, 1.6)
        self.assertAlmostEqual(greedy.p90_pickup_wait, 1.7)
        self.assertAlmostEqual(hungarian.p90_pickup_wait, 3.2)
        self.assertEqual((greedy.empty_travel_minutes, hungarian.empty_travel_minutes), (0, 5))

    def test_random_repositioning_is_seeded_and_records_empty_travel(self):
        city = GridCity(rows=1, columns=2)
        traces = (DemandTrace((0,), (), 12),)
        first = evaluate_baselines(city, traces, reposition_seed=9)
        replay = evaluate_baselines(city, traces, reposition_seed=9)

        self.assertEqual(first, replay)
        greedy, hungarian, random = first
        self.assertEqual((greedy.empty_travel_minutes, hungarian.empty_travel_minutes), (0, 0))
        self.assertGreater(random.reposition_travel_minutes, 0)
        self.assertEqual(random.empty_travel_minutes, random.reposition_travel_minutes)
        self.assertIsNone(random.service_rate)
        self.assertIsNone(random.mean_pickup_wait)
        self.assertIsNone(random.p90_pickup_wait)


if __name__ == "__main__":
    unittest.main()
