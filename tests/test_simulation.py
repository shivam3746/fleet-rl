import unittest

from fleet_rl.city import GridCity, Move
from fleet_rl.simulation import FleetSimulation, LegKind, MinuteCounts, Request


class FleetSimulationTests(unittest.TestCase):
    def setUp(self):
        self.city = GridCity(rows=1, columns=6)

    def test_assigned_passenger_waits_until_pickup_at_deadline(self):
        sim = FleetSimulation(self.city, [0], [Request(0, 2, 5, 2)], arrival_cutoff=1)

        self.assertEqual(sim.dispatch({0: 0}), 0)
        self.assertEqual(sim.advance(), MinuteCounts(1, 1, 0))
        self.assertEqual(sim.dispatch(), 0)
        self.assertFalse(sim.terminated)
        self.assertEqual(sim.advance(), MinuteCounts(1, 1, 0))
        self.assertEqual(sim.dispatch(), 0)

        self.assertEqual(sim.requests[0].picked_up_at, 2)
        self.assertIsNone(sim.requests[0].expired_at)
        self.assertEqual(sim.vehicles[0].leg.kind, LegKind.PASSENGER)
        self.assertTrue(sim.terminated)
        with self.assertRaises(RuntimeError):
            sim.advance()

    def test_reposition_enables_same_zone_pickup_before_expiry(self):
        requests = [Request(0, 2, 5, 2), Request(0, 5, 0, 2)]
        sim = FleetSimulation(self.city, [0], requests, arrival_cutoff=1)

        sim.dispatch()
        self.assertEqual(sim.advance({0: Move.EAST}), MinuteCounts(2, 0, 1))
        sim.dispatch()
        self.assertEqual(sim.advance({0: Move.EAST}), MinuteCounts(2, 0, 1))
        self.assertEqual(sim.vehicles[0].zone, 2)
        self.assertIsNone(sim.vehicles[0].leg)
        self.assertEqual(sim.dispatch({0: 0}), 1)

        self.assertEqual(sim.requests[0].picked_up_at, 2)
        self.assertIsNone(sim.requests[0].expired_at)
        self.assertEqual(sim.requests[1].expired_at, 2)
        self.assertIsNone(sim.requests[1].picked_up_at)
        self.assertEqual(len(sim.pending_requests), 0)
        self.assertTrue(sim.terminated)

    def test_trip_completion_makes_vehicle_available_for_last_arrival(self):
        requests = [Request(0, 0, 2, 1), Request(2, 2, 4, 3)]
        sim = FleetSimulation(self.city, [0], requests, arrival_cutoff=3)

        self.assertNotIn(1, sim.requests)
        sim.dispatch({0: 0})
        self.assertEqual(sim.requests[0].picked_up_at, 0)
        self.assertEqual(sim.advance(), MinuteCounts(0, 0, 0))
        sim.dispatch()
        self.assertEqual(sim.advance(), MinuteCounts(0, 0, 0))
        self.assertEqual(sim.vehicles[0].zone, 2)
        sim.dispatch({0: 1})
        self.assertEqual(sim.requests[1].picked_up_at, 2)
        self.assertFalse(sim.terminated)
        sim.advance()
        sim.dispatch()
        self.assertTrue(sim.terminated)

    def test_invalid_assignments_leave_boundary_open_for_retry(self):
        requests = [Request(0, 0, 1, 2), Request(0, 5, 3, 1)]
        sim = FleetSimulation(self.city, [0, 1], requests, arrival_cutoff=1)

        with self.assertRaises(ValueError):
            sim.dispatch({0: 0, 1: 0})
        with self.assertRaises(ValueError):
            sim.dispatch({0: 0, 1: 1})
        self.assertIsNone(sim.requests[0].assigned_vehicle)
        self.assertIsNone(sim.vehicles[0].leg)
        self.assertEqual(sim.dispatch({0: 0}), 0)

    def test_invalid_moves_leave_simulation_at_same_minute(self):
        sim = FleetSimulation(self.city, [0, 1], [Request(0, 2, 5, 2)], arrival_cutoff=1)
        sim.dispatch({1: 0})

        with self.assertRaises(ValueError):
            sim.advance({0: Move.EAST, 1: Move.EAST})
        with self.assertRaises(ValueError):
            sim.advance({0: Move.WEST})
        self.assertEqual(sim.minute, 0)
        self.assertEqual(sim.vehicles[0].zone, 0)
        self.assertIsNone(sim.vehicles[0].leg)
        self.assertEqual(sim.advance({0: Move.STAY}), MinuteCounts(1, 1, 0))

    def test_zero_length_passenger_trip_releases_vehicle_immediately(self):
        sim = FleetSimulation(self.city, [0], [Request(0, 0, 0, 0)], arrival_cutoff=1)

        self.assertEqual(sim.dispatch({0: 0}), 0)
        self.assertIsNone(sim.vehicles[0].leg)
        self.assertEqual(sim.advance({0: Move.EAST}), MinuteCounts(0, 0, 1))

    def test_dispatch_and_advance_must_alternate(self):
        sim = FleetSimulation(self.city, [0], [], arrival_cutoff=1)

        with self.assertRaises(RuntimeError):
            sim.advance()
        sim.dispatch()
        with self.assertRaises(RuntimeError):
            sim.dispatch()
        sim.advance()
        sim.dispatch()
        self.assertTrue(sim.terminated)


if __name__ == "__main__":
    unittest.main()
