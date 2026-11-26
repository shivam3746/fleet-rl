"""Seeded synthetic demand and initial fleet placement."""

from dataclasses import dataclass
from math import isfinite

from numpy.random import SeedSequence, default_rng

from .city import GridCity
from .simulation import Request


@dataclass(frozen=True)
class DemandTrace:
    initial_zones: tuple[int, ...]
    requests: tuple[Request, ...]
    arrival_cutoff: int


def generate_trace(
    city: GridCity,
    *,
    seed: int,
    arrival_cutoff: int = 120,
    fleet_size: int = 20,
    arrival_rate: float = 2.0,
    patience: int = 5,
    shift_minutes: int = 30,
    hotspot_share: float = 0.85,
) -> DemandTrace:
    """Generate Poisson arrivals around a hotspot that alternates by time block.

    With probability hotspot_share, an origin is sampled uniformly from zones
    closest to the active corner. Otherwise it is sampled from the whole city.
    Destinations are uniform among other zones, or the origin on a one-zone map.
    Fleet placement uses its own random stream so fleet size can vary while
    keeping the same demand trace.
    """
    if arrival_cutoff < 0 or fleet_size < 0:
        raise ValueError("Arrival cutoff and fleet size must be nonnegative")
    if not isfinite(arrival_rate) or arrival_rate < 0:
        raise ValueError("Arrival rate must be finite and nonnegative")
    if patience < 1 or shift_minutes < 1:
        raise ValueError("Patience and shift duration must be positive")
    if not 0 <= hotspot_share <= 1:
        raise ValueError("Hotspot share must be between zero and one")

    fleet_seed, demand_seed = SeedSequence(seed).spawn(2)
    fleet_rng = default_rng(fleet_seed)
    demand_rng = default_rng(demand_seed)
    initial_zones = tuple(int(zone) for zone in fleet_rng.integers(city.zone_count, size=fleet_size))

    last_zone = city.zone_count - 1
    diameter = city.travel_minutes(0, last_zone)
    first_region = []
    second_region = []
    for zone in range(city.zone_count):
        distance_from_first = city.travel_minutes(0, zone)
        if 2 * distance_from_first <= diameter:
            first_region.append(zone)
        if 2 * distance_from_first >= diameter:
            second_region.append(zone)
    regions = (first_region, second_region)

    requests = []
    for minute, count in enumerate(demand_rng.poisson(arrival_rate, size=arrival_cutoff)):
        active_region = regions[(minute // shift_minutes) % len(regions)]
        for _ in range(count):
            if demand_rng.random() < hotspot_share:
                origin = active_region[demand_rng.integers(len(active_region))]
            else:
                origin = int(demand_rng.integers(city.zone_count))
            destination = origin
            if city.zone_count > 1:
                # Map a uniform draw over N - 1 slots onto zones other than origin.
                destination = int(demand_rng.integers(last_zone))
                if destination >= origin:
                    destination += 1
            requests.append(Request(minute, origin, destination, minute + patience))

    return DemandTrace(initial_zones, tuple(requests), arrival_cutoff)
