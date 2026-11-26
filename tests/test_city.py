import unittest

from fleet_rl.city import GridCity, Move


class GridCityTests(unittest.TestCase):
    def test_default_grid_has_bidirectional_roads(self):
        city = GridCity()
        directed_edges = city.directed_edges()
        edges = set(directed_edges)

        self.assertEqual(city.zone_count, 50)
        self.assertEqual(len(directed_edges), 170)
        self.assertEqual(len(edges), 170)
        self.assertEqual(edges, {(destination, origin) for origin, destination in edges})
        for origin, destination in edges:
            self.assertEqual(city.travel_minutes(origin, destination), 1)

    def test_moves_do_not_wrap_at_grid_boundaries(self):
        city = GridCity()

        self.assertIsNone(city.destination(0, Move.NORTH))
        self.assertIsNone(city.destination(0, Move.WEST))
        self.assertIsNone(city.destination(9, Move.EAST))
        self.assertIsNone(city.destination(10, Move.WEST))
        self.assertIsNone(city.destination(49, Move.SOUTH))
        self.assertEqual(city.destination(0, Move.SOUTH), 10)
        self.assertEqual(city.destination(0, Move.EAST), 1)
        self.assertEqual(city.destination(49, Move.STAY), 49)

    def test_travel_time_counts_row_and_column_changes(self):
        city = GridCity()

        self.assertEqual(city.travel_minutes(0, 49), 13)
        self.assertEqual(city.travel_minutes(9, 10), 10)
        self.assertEqual(city.travel_minutes(22, 22), 0)

    def test_single_zone_grid_only_allows_staying(self):
        city = GridCity(rows=1, columns=1)

        self.assertEqual(city.directed_edges(), ())
        self.assertEqual(city.destination(0, Move.STAY), 0)
        with self.assertRaises(ValueError):
            city.travel_minutes(0, -1)


if __name__ == "__main__":
    unittest.main()
