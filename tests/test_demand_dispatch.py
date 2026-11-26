import unittest

from fleet_rl.city import GridCity
from fleet_rl.demand import generate_trace
from fleet_rl.dispatch import hungarian_assignments
from fleet_rl.simulation import FleetSimulation, Request


class DemandTests(unittest.TestCase):
    def test_seed_replays_fleet_and_demand_independently(self):
        city = GridCity()
        trace = generate_trace(city, seed=42)
        smaller_fleet = generate_trace(city, seed=42, fleet_size=10)

        self.assertEqual(trace, generate_trace(city, seed=42))
        self.assertNotEqual(trace.requests, generate_trace(city, seed=43).requests)
        self.assertEqual(smaller_fleet.requests, trace.requests)
        self.assertEqual(smaller_fleet.initial_zones, trace.initial_zones[:10])
        for request in trace.requests:
            self.assertLess(request.arrival_minute, trace.arrival_cutoff)
            self.assertEqual(request.pickup_deadline - request.arrival_minute, 5)
            self.assertNotEqual(request.origin, request.destination)

    def test_hotspot_switches_corners_and_repeats(self):
        city = GridCity(rows=1, columns=6)
        trace = generate_trace(
            city, seed=8, arrival_cutoff=9, shift_minutes=3, arrival_rate=10, hotspot_share=1
        )
        origins_by_block = [set(), set(), set()]
        for request in trace.requests:
            origins_by_block[request.arrival_minute // 3].add(request.origin)

        for block, allowed in enumerate(({0, 1, 2}, {3, 4, 5}, {0, 1, 2})):
            self.assertTrue(origins_by_block[block])
            self.assertLessEqual(origins_by_block[block], allowed)

    def test_empty_demand_and_single_zone_maps(self):
        city = GridCity(rows=1, columns=1)
        self.assertEqual(generate_trace(city, seed=1, arrival_rate=0).requests, ())
        trace = generate_trace(city, seed=1, arrival_cutoff=3, arrival_rate=5)
        self.assertTrue(trace.requests)
        self.assertTrue(all(request.origin == request.destination == 0 for request in trace.requests))


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.city = GridCity(rows=1, columns=6)

    def test_more_feasible_pickups_take_priority_over_a_zero_cost_pickup(self):
        requests = [Request(0, 0, 2, 4), Request(0, 1, 2, 1)]
        sim = FleetSimulation(self.city, [0, 4], requests, arrival_cutoff=1)

        assignments = hungarian_assignments(sim)

        self.assertEqual(assignments, {0: 1, 1: 0})
        self.assertEqual(sim.dispatch(assignments), 0)

    def test_minimizes_pickup_cost_for_equal_size_matchings(self):
        requests = [Request(0, 3, 0, 5), Request(0, 1, 0, 5), Request(0, 5, 0, 5)]
        sim = FleetSimulation(self.city, [0, 3], requests, arrival_cutoff=1)

        self.assertEqual(hungarian_assignments(sim), {0: 1, 1: 0})

    def test_ignores_busy_vehicles_and_assigned_requests_and_uses_current_minute(self):
        requests = [
            Request(0, 2, 0, 3),
            Request(1, 3, 0, 2),
            Request(1, 5, 0, 2),
            Request(1, 0, 5, 1),
        ]
        sim = FleetSimulation(self.city, [0, 4, 1], requests, arrival_cutoff=2)
        sim.dispatch({0: 0})
        sim.advance()

        assignments = hungarian_assignments(sim)

        self.assertEqual(len(assignments), 1)
        self.assertIn(assignments[1], (1, 2))
        self.assertEqual(sim.dispatch(assignments), 1)

    def test_empty_or_infeasible_matching_leaves_fleet_idle(self):
        request = Request(0, 5, 0, 1)
        for zones, requests in (([], [request]), ([0], []), ([0], [request])):
            with self.subTest(zones=zones, requests=requests):
                sim = FleetSimulation(self.city, zones, requests, arrival_cutoff=1)
                self.assertEqual(hungarian_assignments(sim), {})

    def test_generated_trace_replays_to_completion_with_stay_actions(self):
        city = GridCity()
        trace = generate_trace(city, seed=42)

        def run_episode():
            sim = FleetSimulation(city, trace.initial_zones, trace.requests, trace.arrival_cutoff)
            waits = expiries = empty_minutes = 0
            while True:
                expiries += sim.dispatch(hungarian_assignments(sim))
                if sim.terminated:
                    break
                counts = sim.advance()
                waits += counts.waiting_passenger_minutes
                empty_minutes += counts.pickup_travel_minutes
                self.assertEqual(counts.reposition_travel_minutes, 0)
                self.assertLess(sim.minute, trace.arrival_cutoff + 5)

            served = sum(state.picked_up_at is not None for state in sim.requests.values())
            self.assertGreater(served, 0)
            self.assertEqual(served + expiries, len(trace.requests))
            return served, expiries, waits, empty_minutes

        self.assertEqual(run_episode(), run_episode())


if __name__ == "__main__":
    unittest.main()
