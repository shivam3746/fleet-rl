"""Minute-by-minute fleet simulation driven by scripted requests and assignments."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from .city import City, Move


@dataclass(frozen=True)
class Request:
    arrival_minute: int
    origin: int
    destination: int
    pickup_deadline: int
    trip_minutes: int | None = None


@dataclass
class RequestState:
    request: Request
    assigned_vehicle: int | None = None
    picked_up_at: int | None = None
    expired_at: int | None = None


class LegKind(Enum):
    PICKUP = auto()
    PASSENGER = auto()
    REPOSITION = auto()


@dataclass(frozen=True)
class Leg:
    kind: LegKind
    destination: int
    arrival_minute: int
    request_id: int | None = None


@dataclass
class Vehicle:
    # While traveling, zone is the last zone reached at a leg boundary.
    zone: int
    leg: Leg | None = None


@dataclass(frozen=True)
class MinuteCounts:
    waiting_passenger_minutes: int
    pickup_travel_minutes: int
    reposition_travel_minutes: int


class FleetSimulation:
    """Alternate dispatch at a minute boundary with advancement to the next one.

    Vehicle and request IDs are their positions in the input sequences. Dispatch
    takes vehicle-to-request assignments and returns the number of expirations
    at that boundary. Advance takes vehicle-to-move actions and returns counts
    for the elapsed minute. Treat the exposed records as read-only.
    """

    def __init__(
        self,
        city: City,
        initial_zones: Iterable[int],
        requests: Iterable[Request],
        arrival_cutoff: int,
    ) -> None:
        if arrival_cutoff < 0:
            raise ValueError("Arrival cutoff must be nonnegative")
        self.city = city
        self.arrival_cutoff = arrival_cutoff
        self.minute = 0
        self.vehicles = tuple(Vehicle(zone) for zone in initial_zones)
        for vehicle in self.vehicles:
            city.destination(vehicle.zone, Move.STAY)

        self._arrivals: dict[int, list[tuple[int, Request]]] = {}
        for request_id, request in enumerate(requests):
            if not 0 <= request.arrival_minute < arrival_cutoff:
                raise ValueError(f"Request {request_id} arrives outside the demand horizon")
            if request.pickup_deadline < request.arrival_minute:
                raise ValueError(f"Request {request_id} has a deadline before its arrival")
            if request.trip_minutes is not None and request.trip_minutes < 1:
                raise ValueError(f"Request {request_id} has a nonpositive passenger trip duration")
            city.travel_minutes(request.origin, request.destination)
            self._arrivals.setdefault(request.arrival_minute, []).append((request_id, request))

        self._requests: dict[int, RequestState] = {}
        self._pending: dict[int, RequestState] = {}
        self.requests = MappingProxyType(self._requests)
        self.pending_requests = MappingProxyType(self._pending)
        self._dispatched = False
        self._admit_arrivals()

    @property
    def terminated(self) -> bool:
        return self._dispatched and self.minute >= self.arrival_cutoff and not self._pending

    def dispatch(self, assignments: Mapping[int, int] | None = None) -> int:
        """Apply assignments, resolve same-zone pickups, then expire requests."""
        if self._dispatched:
            raise RuntimeError(f"Dispatch already completed at minute {self.minute}")

        pickups = []
        assigned_requests = set()
        for vehicle_id, request_id in (assignments or {}).items():
            vehicle = self._idle_vehicle(vehicle_id)
            request = self._pending.get(request_id)
            if request is None or request.assigned_vehicle is not None or request_id in assigned_requests:
                raise ValueError(f"Request {request_id} is unavailable for assignment")
            arrival = self.minute + self.city.travel_minutes(vehicle.zone, request.request.origin)
            if arrival > request.request.pickup_deadline:
                raise ValueError(f"Vehicle {vehicle_id} cannot meet request {request_id}'s deadline")
            pickups.append((vehicle_id, request_id, arrival))
            assigned_requests.add(request_id)

        for vehicle_id, request_id, arrival in pickups:
            vehicle = self.vehicles[vehicle_id]
            request = self._pending[request_id]
            request.assigned_vehicle = vehicle_id
            if arrival == self.minute:
                self._pick_up(vehicle, request_id)
            else:
                vehicle.leg = Leg(LegKind.PICKUP, request.request.origin, arrival, request_id)

        expired = [
            request_id for request_id, request in self._pending.items()
            if request.request.pickup_deadline == self.minute
        ]
        for request_id in expired:
            self._pending.pop(request_id).expired_at = self.minute
        self._dispatched = True
        return len(expired)

    def advance(self, moves: Mapping[int, Move] | None = None) -> MinuteCounts:
        """Apply idle moves and advance one minute. Omitted vehicles stay put."""
        if not self._dispatched:
            raise RuntimeError("Dispatch must be completed before advancing")
        if self.terminated:
            raise RuntimeError("The episode has ended")

        repositioning = []
        for vehicle_id, move in (moves or {}).items():
            vehicle = self._idle_vehicle(vehicle_id)
            destination = self.city.destination(vehicle.zone, move)
            if destination is None:
                raise ValueError(f"Vehicle {vehicle_id}'s move is unavailable from zone {vehicle.zone}")
            if destination != vehicle.zone:
                arrival = self.minute + self.city.travel_minutes(vehicle.zone, destination)
                repositioning.append((vehicle, Leg(LegKind.REPOSITION, destination, arrival)))

        for vehicle, leg in repositioning:
            vehicle.leg = leg

        pickup_minutes = sum(
            vehicle.leg is not None and vehicle.leg.kind is LegKind.PICKUP
            for vehicle in self.vehicles
        )
        reposition_minutes = sum(
            vehicle.leg is not None and vehicle.leg.kind is LegKind.REPOSITION
            for vehicle in self.vehicles
        )
        counts = MinuteCounts(len(self._pending), pickup_minutes, reposition_minutes)

        self.minute += 1
        for vehicle in self.vehicles:
            leg = vehicle.leg
            if leg is not None and leg.arrival_minute == self.minute:
                vehicle.zone = leg.destination
                vehicle.leg = None
                if leg.request_id is not None:
                    self._pick_up(vehicle, leg.request_id)

        self._dispatched = False
        self._admit_arrivals()
        return counts

    def _admit_arrivals(self) -> None:
        for request_id, request in self._arrivals.pop(self.minute, []):
            state = RequestState(request)
            self._requests[request_id] = state
            self._pending[request_id] = state

    def _idle_vehicle(self, vehicle_id: int) -> Vehicle:
        if not 0 <= vehicle_id < len(self.vehicles):
            raise ValueError(f"Unknown vehicle {vehicle_id}")
        vehicle = self.vehicles[vehicle_id]
        if vehicle.leg is not None:
            raise ValueError(f"Vehicle {vehicle_id} is already traveling")
        return vehicle

    def passenger_minutes(self, request: Request) -> int:
        if request.trip_minutes is not None:
            return request.trip_minutes
        return self.city.travel_minutes(request.origin, request.destination)

    def _pick_up(self, vehicle: Vehicle, request_id: int) -> None:
        state = self._pending.pop(request_id)
        state.picked_up_at = self.minute
        destination = state.request.destination
        travel_minutes = self.passenger_minutes(state.request)
        if travel_minutes:
            vehicle.leg = Leg(LegKind.PASSENGER, destination, self.minute + travel_minutes)
