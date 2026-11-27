"""Deadline-aware vehicle-to-request matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .simulation import FleetSimulation


def greedy_assignments(simulation: FleetSimulation) -> dict[int, int]:
    """Visit requests oldest first and give each the nearest available vehicle."""
    idle_vehicles = {
        vehicle_id: vehicle.zone
        for vehicle_id, vehicle in enumerate(simulation.vehicles)
        if vehicle.leg is None
    }
    waiting_requests = sorted(
        (state.request.arrival_minute, request_id)
        for request_id, state in simulation.pending_requests.items()
        if state.assigned_vehicle is None
    )
    assignments = {}
    for _, request_id in waiting_requests:
        if not idle_vehicles:
            break
        request = simulation.pending_requests[request_id].request
        pickup_minutes, vehicle_id = min(
            (simulation.city.travel_minutes(zone, request.origin), vehicle_id)
            for vehicle_id, zone in idle_vehicles.items()
        )
        if simulation.minute + pickup_minutes <= request.pickup_deadline:
            assignments[vehicle_id] = request_id
            del idle_vehicles[vehicle_id]
    return assignments


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
    costs = np.full((vehicle_count, request_count + vehicle_count), np.inf)
    for row, (_, zone) in enumerate(idle_vehicles):
        for column, (_, request) in enumerate(waiting_requests):
            pickup_minutes = city.travel_minutes(zone, request.origin)
            if simulation.minute + pickup_minutes <= request.pickup_deadline:
                costs[row, column] = pickup_minutes

    # Each vehicle can stay idle. Its dummy cost dominates any real travel savings.
    costs[:, request_count:] = min(vehicle_count, request_count) * city.max_travel_minutes + 1

    rows, columns = linear_sum_assignment(costs)
    return {
        idle_vehicles[row][0]: waiting_requests[column][0]
        for row, column in zip(rows, columns)
        if column < request_count
    }
