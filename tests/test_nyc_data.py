import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fleet_rl.city import Move
from fleet_rl.demand import DemandTrace
from fleet_rl.env import FleetDispatchEnv
from fleet_rl.nyc_data import load_demand_windows
from fleet_rl.road_city import RoadCity
from fleet_rl.simulation import FleetSimulation, MinuteCounts, Request


class RoadBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.city = RoadCity.from_routes(
            (10, 20, 30), np.array([[0, 0], [1, 0], [2, 0]]),
            np.array([[0, 190, 150], [40, 0, 70], [61, 100, 0]]),
        )

    def test_cardinal_move_uses_fastest_directed_route(self):
        self.assertEqual(self.city.destination(0, Move.EAST), 2)
        self.assertIsNone(self.city.destination(0, Move.WEST))
        self.assertEqual(self.city.travel_minutes(0, 2), 3)
        self.assertEqual(self.city.travel_minutes(2, 0), 2)

    def test_repositioning_commits_fleet_and_counts_all_three_minutes(self):
        sim = FleetSimulation(self.city, [0], [], arrival_cutoff=4)
        sim.dispatch()
        self.assertEqual(sim.advance({0: Move.EAST}), MinuteCounts(0, 0, 1))
        for _ in range(2):
            sim.dispatch()
            self.assertIsNotNone(sim.vehicles[0].leg)
            self.assertEqual(sim.advance(), MinuteCounts(0, 0, 1))
        self.assertIsNone(sim.vehicles[0].leg)
        self.assertEqual(sim.vehicles[0].zone, 2)

    def test_meter_duration_controls_same_zone_trip_and_idle_forecast(self):
        request = Request(0, 0, 0, 5, trip_minutes=6)
        env = FleetDispatchEnv(self.city, fleet_size=1)
        observation, _ = env.reset(options={"trace": DemandTrace((0,), (request,), 8)})
        self.assertEqual(observation["vehicle_zones"].tolist(), [-1])
        self.assertEqual(observation["zone_features"][0, 3], 0)
        observation, *_ = env.step(np.array([Move.STAY]))
        self.assertEqual(observation["zone_features"][0, 3], 1)

    def test_tlc_windows_filter_bad_records_and_replay_metered_durations(self):
        data = pd.DataFrame({
            "tpep_pickup_datetime": pd.to_datetime([
                "2024-01-08 08:00:45", "2024-01-08 09:59:30", "2024-01-08 10:00:00",
                "2024-01-08 08:10:00", "2024-01-08 17:10:00", "2024-01-08 17:20:00",
            ]),
            "tpep_dropoff_datetime": pd.to_datetime([
                "2024-01-08 08:03:00", "2024-01-08 10:14:25", "2024-01-08 10:10:00",
                "2024-01-08 08:09:00", "2024-01-08 17:16:00", "2024-01-08 17:26:00",
            ]),
            "PULocationID": [10, 20, 10, 20, 30, 20],
            "DOLocationID": [10, 30, 20, 10, 999, 10],
            "trip_distance": [1, 2, 3, 1, 1, 0],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trips.parquet"
            data.to_parquet(path)
            traces = load_demand_windows(path, self.city, first_day=date(2024, 1, 8), days=1, sample_fraction=1)

        morning, evening = traces.values()
        self.assertEqual(morning.requests, (Request(0, 0, 0, 5, 3), Request(119, 1, 2, 124, 15)))
        self.assertEqual(evening.requests, ())


if __name__ == "__main__":
    unittest.main()
