"""Topology and travel times for the synthetic city."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol


class Move(IntEnum):
    STAY = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class City(Protocol):
    @property
    def zone_count(self) -> int: ...

    @property
    def max_travel_minutes(self) -> int: ...

    def destination(self, zone: int, move: Move) -> int | None: ...

    def directed_edges(self) -> tuple[tuple[int, int], ...]: ...

    def travel_minutes(self, origin: int, destination: int) -> int: ...


_OFFSETS = {
    Move.STAY: (0, 0),
    Move.NORTH: (-1, 0),
    Move.SOUTH: (1, 0),
    Move.EAST: (0, 1),
    Move.WEST: (0, -1),
}


@dataclass(frozen=True)
class GridCity:
    """Rectangular grid with row-major zone IDs and one minute per edge."""

    rows: int = 5
    columns: int = 10

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("Grid dimensions must be positive")

    @property
    def zone_count(self) -> int:
        return self.rows * self.columns

    @property
    def max_travel_minutes(self) -> int:
        return self.rows + self.columns - 2

    def destination(self, zone: int, move: Move) -> int | None:
        """Return the destination zone, or None if the move leaves the grid."""
        row, column = self._coordinates(zone)
        row_change, column_change = _OFFSETS[Move(move)]
        row += row_change
        column += column_change
        if 0 <= row < self.rows and 0 <= column < self.columns:
            return row * self.columns + column
        return None

    def directed_edges(self) -> tuple[tuple[int, int], ...]:
        """Return ordered origin-destination pairs, with both directions per road."""
        edges = []
        for origin in range(self.zone_count):
            for move in (Move.NORTH, Move.SOUTH, Move.EAST, Move.WEST):
                destination = self.destination(origin, move)
                if destination is not None:
                    edges.append((origin, destination))
        return tuple(edges)

    def travel_minutes(self, origin: int, destination: int) -> int:
        """Return the shortest travel time along the grid roads."""
        origin_row, origin_column = self._coordinates(origin)
        destination_row, destination_column = self._coordinates(destination)
        return abs(origin_row - destination_row) + abs(origin_column - destination_column)

    def _coordinates(self, zone: int) -> tuple[int, int]:
        if not 0 <= zone < self.zone_count:
            raise ValueError(f"Unknown zone {zone}")
        return divmod(zone, self.columns)
