"""Deadline-aware vehicle-to-request matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .simulation import FleetSimulation


def hungarian_assignments(simulation: FleetSimulation) -> dict[int, int]:
    """Maximize feasible assignments, then minimize their total pickup travel."""
    idle_vehicles = [
        (vehicle_id, vehicle.zone)
        for vehicle_id, vehicle in enumerate(simulation.vehicles)
        if vehicle.leg is None
    ]
    waiting_requests = [
        (request_id, state.request)
        for request_id, state in sorted(simulation.pending_requests.items())
        if state.assigned_vehicle is None
    ]
    if not idle_vehicles or not waiting_requests:
        return {}

    vehicle_count = len(idle_vehicles)
    request_count = len(waiting_requests)
    city = simulation.city
    diameter = city.travel_minutes(0, city.zone_count - 1)

    # One dummy column per vehicle allows any number of vehicles to stay idle.
    # Its cost exceeds the highest possible total travel cost of real matches.
    idle_cost = min(vehicle_count, request_count) * diameter + 1
    costs = np.full((vehicle_count, request_count + vehicle_count), np.inf)
    costs[:, request_count:] = idle_cost
    for row, (_, zone) in enumerate(idle_vehicles):
        for column, (_, request) in enumerate(waiting_requests):
            pickup_minutes = city.travel_minutes(zone, request.origin)
            if simulation.minute + pickup_minutes <= request.pickup_deadline:
                costs[row, column] = pickup_minutes

    rows, columns = linear_sum_assignment(costs)
    return {
        idle_vehicles[row][0]: waiting_requests[column][0]
        for row, column in zip(rows, columns)
        if column < request_count
    }
