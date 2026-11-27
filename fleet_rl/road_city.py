"""A zone graph backed by directed road travel times."""

from dataclasses import dataclass

import numpy as np

from .city import Move


@dataclass(frozen=True)
class RoadCity:
    zone_ids: tuple[int, ...]
    _minutes: tuple[tuple[int, ...], ...]
    _destinations: tuple[tuple[int | None, ...], ...]
    max_travel_minutes: int

    @classmethod
    def from_routes(
        cls, zone_ids: tuple[int, ...], coordinates: np.ndarray, route_seconds: np.ndarray
    ) -> "RoadCity":
        """Build moves from projected (east, north) coordinates and directed routes.

        In each compass sector, the fastest destination is legal. Ties use
        straight-line distance and then the destination's zone ID.
        """
        count = len(zone_ids)
        if not count or len(set(zone_ids)) != count:
            raise ValueError("Road cities require unique, nonempty zone IDs")
        if coordinates.shape != (count, 2) or route_seconds.shape != (count, count):
            raise ValueError("Coordinates and travel times must match the zones")
        if not np.isfinite(coordinates).all() or not np.isfinite(route_seconds).all():
            raise ValueError("All zones must have finite coordinates and be mutually reachable")
        if np.any(np.diag(route_seconds) != 0) or np.any(route_seconds[~np.eye(count, dtype=bool)] <= 0):
            raise ValueError("Travel between distinct zones must be positive and within-zone time zero")

        raw_minutes = np.ceil(route_seconds / 60).astype(int)
        destinations = []
        for origin, (x, y) in enumerate(coordinates):
            legal: list[int | None] = [None] * len(Move)
            legal[Move.STAY] = origin
            best = {}
            for target, (target_x, target_y) in enumerate(coordinates):
                if target == origin:
                    continue
                east, north = target_x - x, target_y - y
                if abs(east) >= abs(north):
                    direction = Move.EAST if east > 0 else Move.WEST
                else:
                    direction = Move.NORTH if north > 0 else Move.SOUTH
                score = (route_seconds[origin, target], east**2 + north**2, zone_ids[target])
                if direction not in best or score < best[direction]:
                    best[direction] = score
                    legal[direction] = target
            destinations.append(tuple(legal))
        return cls(zone_ids, tuple(tuple(map(int, row)) for row in raw_minutes),
                   tuple(destinations), int(raw_minutes.max()))

    @property
    def zone_count(self) -> int:
        return len(self.zone_ids)

    def destination(self, zone: int, move: Move) -> int | None:
        self._require_zone(zone)
        return self._destinations[zone][Move(move)]

    def directed_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (origin, destination)
            for origin, moves in enumerate(self._destinations)
            for destination in moves
            if destination is not None and origin != destination
        )

    def travel_minutes(self, origin: int, destination: int) -> int:
        self._require_zone(origin)
        self._require_zone(destination)
        return self._minutes[origin][destination]

    def _require_zone(self, zone: int) -> None:
        if not 0 <= zone < self.zone_count:
            raise ValueError(f"Unknown zone {zone}")
