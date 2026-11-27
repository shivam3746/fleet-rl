"""Prepare Manhattan taxi-zone routes and reproducible TLC demand traces."""

from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .demand import DemandTrace
from .road_city import RoadCity
from .simulation import Request

ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ROAD_URL = "https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf"
OSM_ATTRIBUTION = "© OpenStreetMap contributors, ODbL | https://www.openstreetmap.org/copyright"
_BOUNDS = (-74.034, 40.694, -73.902, 40.883)
_PROJECTED_CRS = "EPSG:32618"
_ROADS = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
})
_PICKUP = "tpep_pickup_datetime"
_DROPOFF = "tpep_dropoff_datetime"
_ORIGIN = "PULocationID"
_DESTINATION = "DOLocationID"


def _road_graph(path: Path) -> tuple[nx.MultiDiGraph, str]:
    import osmium

    west, south, east, north = _BOUNDS
    processor = (osmium.FileProcessor(path).with_locations()
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                 .with_filter(osmium.filter.KeyFilter("highway")))
    header = processor.header
    timestamp = header.get("osmosis_replication_timestamp") or "unavailable"
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for way in processor:
        tags = way.tags
        highway = tags.get("highway")
        if highway not in _ROADS or tags.get("area") == "yes":
            continue
        access = next((tags[key] for key in ("motorcar", "motor_vehicle", "vehicle", "access")
                       if key in tags), None)
        if access in {"no", "private"}:
            continue
        oneway = tags.get("oneway:motorcar") or tags.get("oneway")
        reverse_only = oneway in {"-1", "reverse"}
        forward_only = oneway in {"yes", "true", "1"} or (
            oneway is None and (highway == "motorway" or tags.get("junction") == "roundabout")
        )
        previous = None
        for node in way.nodes:
            if not node.location.valid() or not (west <= node.lon <= east and south <= node.lat <= north):
                previous = None
                continue
            graph.add_node(node.ref, x=node.lon, y=node.lat)
            if previous is not None and previous != node.ref:
                if not reverse_only:
                    graph.add_edge(previous, node.ref, highway=highway,
                                   maxspeed=tags.get("maxspeed:forward") or tags.get("maxspeed"))
                if not forward_only:
                    graph.add_edge(node.ref, previous, highway=highway,
                                   maxspeed=tags.get("maxspeed:backward") or tags.get("maxspeed"))
            previous = node.ref

    if not graph.number_of_edges():
        raise ValueError("OpenStreetMap extract contains no drivable roads in the benchmark bounds")
    largest = max(nx.strongly_connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()
    graph = ox.distance.add_edge_lengths(graph)
    graph = ox.routing.add_edge_speeds(graph, fallback=30)
    graph = ox.routing.add_edge_travel_times(graph)
    return ox.project_graph(graph, to_crs=_PROJECTED_CRS), timestamp


def build_manhattan_city(
    road_pbf: Path, zone_zip: Path
) -> tuple[RoadCity, gpd.GeoDataFrame, dict]:
    """Snap each connected Manhattan TLC zone to a road node inside its boundary."""
    graph, timestamp = _road_graph(road_pbf)
    zone_path = f"/vsizip/{zone_zip.resolve().as_posix()}/taxi_zones/taxi_zones.shp"
    zones = gpd.read_file(zone_path)
    zones = zones.loc[zones["borough"] == "Manhattan", ["LocationID", "geometry"]].to_crs(_PROJECTED_CRS)
    zones = zones.dissolve(by="LocationID").sort_index()
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)[["geometry"]]
    candidates = gpd.sjoin(nodes, zones.reset_index(), how="inner", predicate="within")
    if candidates.empty:
        raise ValueError("No Manhattan zone covers a node in the connected road graph")

    centroids = zones.geometry.centroid
    anchors = {}
    for zone_id, points in candidates.groupby("LocationID", sort=True):
        center = centroids.loc[zone_id]
        anchors[int(zone_id)] = min((point.distance(center), int(node_id)) for node_id, point in points.geometry.items())[1]

    zone_ids = tuple(anchors)
    anchor_nodes = tuple(anchors.values())
    coordinates = np.array([(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in anchor_nodes])
    seconds = []
    for origin in anchor_nodes:
        route_cost = nx.single_source_dijkstra_path_length(graph, origin, weight="travel_time")
        seconds.append([route_cost[target] for target in anchor_nodes])
    city = RoadCity.from_routes(zone_ids, coordinates, np.asarray(seconds))
    metadata = {
        "osm_base_timestamp": timestamp,
        "osm_attribution": OSM_ATTRIBUTION,
        "road_nodes": graph.number_of_nodes(),
        "directed_road_segments": graph.number_of_edges(),
        "zone_edges": len(city.directed_edges()),
        "zone_to_osm_node": anchors,
        "excluded_manhattan_zone_ids": sorted(set(map(int, zones.index)) - set(zone_ids)),
    }
    return city, zones.loc[list(zone_ids)], metadata


def load_demand_windows(
    trip_parquet: Path, city: RoadCity, *, first_day: date = date(2024, 1, 8),
    days: int = 7, fleet_size: int = 20, sample_fraction: float = 0.04,
) -> dict[str, DemandTrace]:
    """Sample 08:00 and 17:00 local two-hour windows from metered trips."""
    if days < 1 or fleet_size < 0 or not 0 < sample_fraction <= 1:
        raise ValueError("Use positive days, a nonnegative fleet size, and a sample fraction in (0, 1]")
    zone_index = {zone_id: index for index, zone_id in enumerate(city.zone_ids)}
    records = pd.read_parquet(trip_parquet, columns=[_PICKUP, _DROPOFF, _ORIGIN, _DESTINATION, "trip_distance"])
    pickup, dropoff = pd.to_datetime(records[_PICKUP]), pd.to_datetime(records[_DROPOFF])
    duration = (dropoff - pickup).dt.total_seconds() / 60
    earliest = pd.Timestamp(first_day)
    latest = earliest + pd.Timedelta(days=days)
    eligible = (
        pickup.between(earliest, latest, inclusive="left") & records[_ORIGIN].isin(zone_index)
        & records[_DESTINATION].isin(zone_index) & duration.between(1, 120) & (records["trip_distance"] > 0)
    )
    records = records.loc[eligible, [_ORIGIN, _DESTINATION]].assign(
        pickup=pickup[eligible], trip_minutes=np.ceil(duration[eligible]).astype(int),
    )

    traces = {}
    for day_offset in range(days):
        day = first_day + timedelta(days=day_offset)
        for hour in (8, 17):
            start = pd.Timestamp(day).replace(hour=hour)
            finish = start + pd.Timedelta(minutes=120)
            window = records.loc[records.pickup.between(start, finish, inclusive="left")]
            fleet_seed, demand_seed = np.random.SeedSequence(int(start.strftime("%Y%m%d%H"))).spawn(2)
            fleet = np.random.default_rng(fleet_seed).integers(city.zone_count, size=fleet_size)
            selected = window.loc[np.random.default_rng(demand_seed).random(len(window)) < sample_fraction]
            requests = []
            ordered_rows = selected.sort_values("pickup", kind="stable").itertuples(index=False, name=None)
            for origin, destination, pickup_time, trip_minutes in ordered_rows:
                minute = int((pickup_time - start).total_seconds() // 60)
                requests.append(Request(minute, zone_index[origin], zone_index[destination], minute + 5, trip_minutes))
            traces[start.isoformat()] = DemandTrace(tuple(map(int, fleet)), tuple(requests), 120)
    return traces
