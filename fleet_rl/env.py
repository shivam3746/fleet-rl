"""Gymnasium interface for fleet repositioning."""

from collections.abc import Callable, Mapping
from math import isfinite

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from .city import GridCity, Move
from .demand import DemandTrace, generate_trace
from .dispatch import hungarian_assignments
from .simulation import FleetSimulation

Observation = dict[str, np.ndarray]
ZONE_FEATURES = (
    "idle_vehicles",
    "unassigned_requests",
    "oldest_unassigned_age",
    "vehicles_idle_within_five_minutes",
)


class FleetDispatchEnv(gym.Env[Observation, np.ndarray]):
    """Advance one minute per step with one repositioning choice per vehicle.

    Observations are produced after dispatch at the current minute. Masked
    choices have the same effect as staying, so samples from the unmasked
    action space are safe. The static graph is available as ``edge_index``.
    Pass a ``DemandTrace`` as ``reset(options={"trace": trace})`` to replay it.
    """

    def __init__(
        self,
        city: GridCity | None = None,
        *,
        fleet_size: int = 20,
        arrival_cutoff: int = 120,
        arrival_rate: float = 2.0,
        patience: int = 5,
        shift_minutes: int = 30,
        hotspot_share: float = 0.85,
        expiry_penalty: float | None = None,
        dispatcher: Callable[[FleetSimulation], Mapping[int, int]] = hungarian_assignments,
    ) -> None:
        super().__init__()
        if fleet_size < 0 or arrival_cutoff < 1:
            raise ValueError("Fleet size must be nonnegative and arrival cutoff must be positive")
        if patience < 1:
            raise ValueError("Patience must be positive")
        self.expiry_penalty = float(2 * patience if expiry_penalty is None else expiry_penalty)
        if not isfinite(self.expiry_penalty) or self.expiry_penalty < 0:
            raise ValueError("Expiry penalty must be finite and nonnegative")

        self.city = GridCity() if city is None else city
        self._dispatcher = dispatcher
        self._fleet_size = fleet_size
        self._demand_settings = dict(
            fleet_size=fleet_size,
            arrival_cutoff=arrival_cutoff,
            arrival_rate=arrival_rate,
            patience=patience,
            shift_minutes=shift_minutes,
            hotspot_share=hotspot_share,
        )
        self._simulation: FleetSimulation | None = None

        self.edge_index = np.asarray(self.city.directed_edges(), dtype=np.int64).reshape(-1, 2).T
        self.edge_index.setflags(write=False)
        self._zone_move_mask = np.asarray([
            [self.city.destination(zone, move) is not None for move in Move]
            for zone in range(self.city.zone_count)
        ], dtype=np.int8)
        count_limit = np.iinfo(np.int64).max - 1
        self.action_space = spaces.MultiDiscrete(np.full(fleet_size, len(Move), dtype=np.int64))
        self.observation_space = spaces.Dict({
            "zone_features": spaces.Box(
                0, count_limit, (self.city.zone_count, len(ZONE_FEATURES)), np.int64
            ),
            "clock": spaces.Box(0, count_limit, (2,), np.int64),
            "vehicle_zones": spaces.Box(-1, self.city.zone_count - 1, (fleet_size,), np.int64),
            "action_mask": spaces.Box(0, 1, (fleet_size, len(Move)), np.int8),
        })

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        super().reset(seed=seed)
        trace = None if options is None else options.get("trace")
        if trace is None:
            trace_seed = int(self.np_random.integers(2**63)) if seed is None else seed
            trace = generate_trace(self.city, seed=trace_seed, **self._demand_settings)
        if not isinstance(trace, DemandTrace):
            raise TypeError("The trace option must be a DemandTrace")
        if len(trace.initial_zones) != self._fleet_size or trace.arrival_cutoff < 1:
            raise ValueError("Trace must match the environment's fleet size and have a positive cutoff")

        sim = FleetSimulation(self.city, trace.initial_zones, trace.requests, trace.arrival_cutoff)
        self._initial_expiries = sim.dispatch(self._dispatcher(sim))
        self._simulation = sim
        self._waiting_minutes = 0
        self._pickup_travel_minutes = 0
        self._reposition_travel_minutes = 0
        return self._observe(sim), {"initial_expiries": self._initial_expiries}

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict]:
        sim = self._simulation
        if sim is None or sim.terminated:
            raise ResetNeeded("Call reset before stepping the environment")
        action = np.asarray(action)
        if not np.issubdtype(action.dtype, np.integer) or not self.action_space.contains(action):
            raise ValueError("Expected one integer move in the action space per vehicle")

        moves = {}
        masked_actions = 0
        for vehicle_id, choice in enumerate(action):
            if choice == Move.STAY:
                continue
            vehicle = sim.vehicles[vehicle_id]
            if vehicle.leg is None and self._zone_move_mask[vehicle.zone, choice]:
                moves[vehicle_id] = Move(choice)
            else:
                masked_actions += 1

        counts = sim.advance(moves)
        expiries = sim.dispatch(self._dispatcher(sim))
        # A scripted zero-patience request can expire at reset, which has no reward.
        expiries += self._initial_expiries
        self._initial_expiries = 0

        self._waiting_minutes += counts.waiting_passenger_minutes
        self._pickup_travel_minutes += counts.pickup_travel_minutes
        self._reposition_travel_minutes += counts.reposition_travel_minutes
        reward = -(counts.waiting_passenger_minutes + self.expiry_penalty * expiries)
        info = {
            "waiting_passenger_minutes": counts.waiting_passenger_minutes,
            "expired_requests": expiries,
            "pickup_travel_minutes": counts.pickup_travel_minutes,
            "reposition_travel_minutes": counts.reposition_travel_minutes,
            "masked_actions": masked_actions,
        }
        terminated = sim.terminated
        if terminated:
            info["episode_metrics"] = self._episode_metrics(sim)
        return self._observe(sim), reward, terminated, False, info

    def _observe(self, sim: FleetSimulation) -> Observation:
        zone_features = np.zeros((self.city.zone_count, len(ZONE_FEATURES)), dtype=np.int64)
        vehicle_zones = np.full(self._fleet_size, -1, dtype=np.int64)
        action_mask = np.zeros((self._fleet_size, len(Move)), dtype=np.int8)
        action_mask[:, Move.STAY] = 1

        for vehicle_id, vehicle in enumerate(sim.vehicles):
            leg = vehicle.leg
            if leg is None:
                vehicle_zones[vehicle_id] = vehicle.zone
                zone_features[vehicle.zone, 0] += 1
                action_mask[vehicle_id] = self._zone_move_mask[vehicle.zone]
                continue

            idle_zone = leg.destination
            idle_minute = leg.arrival_minute
            if leg.request_id is not None:
                request = sim.requests[leg.request_id].request
                idle_zone = request.destination
                idle_minute += self.city.travel_minutes(leg.destination, idle_zone)
            if idle_minute - sim.minute <= 5:
                zone_features[idle_zone, 3] += 1

        for state in sim.pending_requests.values():
            if state.assigned_vehicle is None:
                request = state.request
                zone = request.origin
                zone_features[zone, 1] += 1
                zone_features[zone, 2] = max(zone_features[zone, 2], sim.minute - request.arrival_minute)

        return {
            "zone_features": zone_features,
            "clock": np.array([sim.minute, max(0, sim.arrival_cutoff - sim.minute)], dtype=np.int64),
            "vehicle_zones": vehicle_zones,
            "action_mask": action_mask,
        }

    def _episode_metrics(self, sim: FleetSimulation) -> dict:
        pickup_waits = [
            state.picked_up_at - state.request.arrival_minute
            for state in sim.requests.values()
            if state.picked_up_at is not None
        ]
        requests = len(sim.requests)
        served = len(pickup_waits)
        return {
            "requests": requests,
            "served_requests": served,
            "expired_requests": requests - served,
            "service_rate": served / requests if requests else None,
            "mean_pickup_wait": float(np.mean(pickup_waits)) if served else None,
            "p90_pickup_wait": float(np.percentile(pickup_waits, 90)) if served else None,
            "pickup_waits": tuple(pickup_waits),
            "waiting_passenger_minutes": self._waiting_minutes,
            "pickup_travel_minutes": self._pickup_travel_minutes,
            "reposition_travel_minutes": self._reposition_travel_minutes,
            "empty_travel_minutes": self._pickup_travel_minutes + self._reposition_travel_minutes,
        }
