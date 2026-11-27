"""Operate the trained fleet policy through a local natural-language terminal."""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch

from .city import City, GridCity, Move
from .demand import DemandTrace, generate_trace
from .dispatch import hungarian_assignments
from .env import FleetDispatchEnv
from .operator_llm import DEFAULT_SERVER_URL, LocalCommandModel, OperatorCommand
from .policy import FleetPolicy, GraphFleetPolicy
from .road_city import RoadCity
from .simulation import FleetSimulation
from .train import load_checkpoint

_EXAMPLES = (
    "Try starting a GraphSAGE replay, advancing a few minutes, asking for status, "
    "explaining the latest decision, or comparing the saved benchmarks."
)
_PREFIX = r"^(?:(?:please|could you|can you|would you)\s+)*"
_CONTROL_VERBS = {
    "start": re.compile(_PREFIX + r"(?:start|launch|begin|open|replay|run|load|set up)\b", re.I),
    "advance": re.compile(_PREFIX + r"(?:advance|fast[- ]forward|step|continue|run|simulate|skip|finish|take|move)\b", re.I),
}


@dataclass(frozen=True)
class AssignmentEvidence:
    vehicle: int
    request: int
    origin: int
    pickup_zone: int
    pickup_minutes: int


class DispatchAudit:
    def __call__(self, sim: FleetSimulation) -> dict[int, int]:
        self.simulation = sim
        assignments = hungarian_assignments(sim)
        self.assignments = []
        for vehicle, request_id in assignments.items():
            origin = sim.vehicles[vehicle].zone
            pickup_zone = sim.pending_requests[request_id].request.origin
            self.assignments.append(AssignmentEvidence(
                vehicle, request_id, origin, pickup_zone, sim.city.travel_minutes(origin, pickup_zone),
            ))
        return assignments


@dataclass(frozen=True)
class MoveEvidence:
    vehicle: int
    origin: int
    destination: int
    choice: Move
    score: float
    destination_queue: int


class Replay:
    def __init__(self, scenario: str, label: str, city: City, trace: DemandTrace, policy: FleetPolicy) -> None:
        self.scenario = scenario
        self.label = label
        self.city = city
        self.policy = policy
        self.audit = DispatchAudit()
        self.env = FleetDispatchEnv(city, fleet_size=len(trace.initial_zones), dispatcher=self.audit)
        self.observation, _ = self.env.reset(options={"trace": trace})
        self.terminated = False
        self.last_reward: float | None = None
        self.total_reward = 0.0
        self.empty_travel = 0
        self.moves: list[MoveEvidence] = []
        self.last_info: dict = {}

    @property
    def minute(self) -> int:
        return self.audit.simulation.minute

    def advance(self, minutes: int) -> int:
        first = self.minute
        for _ in range(minutes):
            if self.terminated:
                break
            self._step()
        return self.minute - first

    @torch.no_grad()
    def _step(self) -> None:
        logits, _ = self.policy(self.observation)
        scores = logits.cpu().numpy()
        choices = scores.argmax(axis=-1)
        moves = []
        for vehicle, origin in enumerate(self.observation["vehicle_zones"]):
            if origin < 0:
                continue
            choice = Move(choices[vehicle])
            destination = self.city.destination(int(origin), choice)
            if destination is None:
                raise RuntimeError("The trained policy selected an unavailable move")
            moves.append(MoveEvidence(
                vehicle, int(origin), destination, choice, float(scores[vehicle, choice]),
                int(self.observation["zone_features"][destination, 1]),
            ))

        observation, reward, terminated, truncated, info = self.env.step(choices)
        self.observation = observation
        self.terminated = terminated or truncated
        self.moves = moves
        self.last_info = info
        self.last_reward = reward
        self.total_reward += reward
        self.empty_travel += info["pickup_travel_minutes"] + info["reposition_travel_minutes"]

    def status(self) -> str:
        sim = self.audit.simulation
        served = sum(state.picked_up_at is not None for state in sim.requests.values())
        expired = sum(state.expired_at is not None for state in sim.requests.values())
        idle = int((self.observation["vehicle_zones"] >= 0).sum())
        state = "complete" if self.terminated else "active"
        return (
            f"{self.label} | GraphSAGE | minute {self.minute} | {state}\n"
            f"Simulator evidence: {len(sim.requests)} arrivals, {served} served, {expired} expired, "
            f"{len(sim.pending_requests)} awaiting pickup, {idle} idle vehicles. "
            f"Cumulative reward {self.total_reward:g}, empty travel {self.empty_travel} vehicle-min."
        )

    def _zone(self, index: int) -> str:
        if isinstance(self.city, RoadCity):
            return f"TLC zone {self.city.zone_ids[index]}"
        return f"zone {index}"

    def explain(self, vehicle: int | None) -> str:
        if vehicle is not None and not 0 <= vehicle < len(self.audit.simulation.vehicles):
            raise ValueError("Vehicle ID is outside this replay's fleet")
        assignments = [row for row in self.audit.assignments if vehicle is None or row.vehicle == vehicle]
        moves = [row for row in self.moves if vehicle is None or row.vehicle == vehicle]
        moves.sort(key=lambda row: row.choice is Move.STAY)
        scope = "" if vehicle is None else f" for vehicle {vehicle}"
        lines = [f"Simulator evidence{scope} at minute {self.minute}."]
        if self.last_reward is not None:
            info = self.last_info
            lines.append(f"Minute {self.minute - 1} to {self.minute} produced reward {self.last_reward:g}, "
                         f"with {info['waiting_passenger_minutes']} passenger-minutes of wait, "
                         f"{info['expired_requests']} expiries, {info['pickup_travel_minutes']} pickup vehicle-min "
                         f"and {info['reposition_travel_minutes']} repositioning vehicle-min.")
        lines.append(f"Hungarian dispatch at minute {self.minute} made {len(assignments)} matching assignments{scope}. "
                     "It maximizes feasible matches, then minimizes total pickup travel.")
        for row in assignments[:4]:
            lines.append(f"  Vehicle {row.vehicle} was assigned request {row.request}, "
                         f"from {self._zone(row.origin)} to {self._zone(row.pickup_zone)} "
                         f"in {row.pickup_minutes} pickup minutes.")
        if self.last_reward is None:
            lines.append("Advance the replay to see repositioning decisions.")
        else:
            lines.append(f"GraphSAGE scored choices for {len(moves)} idle vehicles{scope} at minute {self.minute - 1}. "
                         "Each chose the highest-scoring legal move.")
            for row in moves[:4]:
                lines.append(f"  Vehicle {row.vehicle} chose {row.choice.name.lower()} "
                             f"from {self._zone(row.origin)} to {self._zone(row.destination)} "
                             f"(score {row.score:.3f}, destination queue {row.destination_queue} at decision time).")
        return "\n".join(lines)


class FleetOperator:
    def __init__(self, checkpoint: Path, nyc_data: Path, synthetic_report: Path, manhattan_report: Path) -> None:
        self._checkpoint = checkpoint
        self._nyc_data = nyc_data
        self._reports = {"synthetic": synthetic_report, "transfer": synthetic_report, "manhattan": manhattan_report}
        self._manhattan: tuple[RoadCity, dict[str, DemandTrace]] | None = None
        self.replay: Replay | None = None

    def ask(self, model: LocalCommandModel, message: str) -> str:
        try:
            command = model.interpret(message)
            control = _CONTROL_VERBS.get(command.action)
            if control is not None and not control.search(message.strip()):
                return _EXAMPLES
            return self.execute(command)
        except (OSError, KeyError, ValueError, requests.RequestException) as exc:
            return f"Operator request could not be completed. {exc}"

    def execute(self, command: OperatorCommand) -> str:
        action, scenario, argument = command.action, command.scenario, command.argument
        if action == "start":
            return self._start(scenario, argument)
        if action == "compare" and not argument:
            return self._compare(scenario)
        if action not in {"advance", "status", "explain"} or scenario != "current":
            return _EXAMPLES
        if self.replay is None:
            return "Start a synthetic or Manhattan replay first."
        if action == "advance":
            minutes = 150 if argument == "end" else self._number(argument or "1", 150)
            if minutes == 0:
                raise ValueError("Advance by at least one minute")
            elapsed = self.replay.advance(minutes)
            return f"Advanced {elapsed} simulated minute(s).\n{self.replay.status()}"
        if action == "explain":
            vehicle = self._number(argument, 9999) if argument else None
            return self.replay.explain(vehicle)
        return self.replay.status() if not argument else _EXAMPLES

    def _start(self, scenario: str, argument: str) -> str:
        city: City
        if scenario in {"synthetic", "current"}:
            seed = self._number(argument or "1000", 2**32 - 1)
            city = GridCity()
            trace = generate_trace(city, seed=seed)
            label = f"Synthetic seed {seed}"
            scenario = "synthetic"
        elif scenario == "manhattan":
            if argument and not re.fullmatch(r"\d{4}-\d{2}-\d{2}T(?:08|17):00(?::00)?", argument):
                raise ValueError("Use a Manhattan window such as 2024-01-08T08:00")
            window = datetime.fromisoformat(argument).isoformat() if argument else "2024-01-08T08:00:00"
            if not "2024-01-08T08:00:00" <= window <= "2024-01-14T17:00:00":
                raise ValueError("Choose a Manhattan window between January 8 and January 14, 2024")
            city, traces = self._manhattan_inputs()
            trace = traces[window]
            label = f"Manhattan window {window} local"
        else:
            return _EXAMPLES
        policy = load_checkpoint(self._checkpoint, city=city)
        if not isinstance(policy, GraphFleetPolicy):
            raise ValueError("Operator replays require a GraphSAGE checkpoint")
        replay = Replay(scenario, label, city, trace, policy)
        self.replay = replay
        return replay.status()

    def _manhattan_inputs(self) -> tuple[RoadCity, dict[str, DemandTrace]]:
        if self._manhattan is None:
            from .nyc_data import build_manhattan_city, load_demand_windows

            pbf = self._nyc_data / "NewYork.osm.pbf"
            zones = self._nyc_data / "taxi_zones.zip"
            trips = self._nyc_data / "yellow_tripdata_2024-01.parquet"
            for path in (pbf, zones, trips):
                if not path.is_file():
                    raise FileNotFoundError(f"Manhattan input file missing: {path}")
            city, _, _ = build_manhattan_city(pbf, zones)
            self._manhattan = city, load_demand_windows(trips, city)
        return self._manhattan

    def _compare(self, scenario: str) -> str:
        if scenario == "current":
            scenario = self.replay.scenario if self.replay is not None else "synthetic"
        if scenario not in self._reports:
            return _EXAMPLES
        source = self._reports[scenario]
        report = json.loads(source.read_text(encoding="utf-8"))
        if scenario == "manhattan":
            summary = report["summary"]
            title = f"Manhattan benchmark, {len(report['window_starts_local'])} replay windows"
        else:
            suite_key = "larger_grid" if scenario == "transfer" else "training_grid"
            suite = report["suites"][suite_key]
            summary = suite["summary"]
            zone_count = suite["city"]["rows"] * suite["city"]["columns"]
            title = f"{zone_count}-zone benchmark, {len(report['demand_seeds'])} held-out traces"
        lines = [title, f"Evidence source: {source.as_posix()}",
                 "Policy | Served | Mean/P90 pickup wait | Empty vehicle-min per episode"]
        for name, row in summary.items():
            lines.append(f"{name} | {row['served_requests']:,}/{row['requests']:,} ({row['service_rate']:.2%}) | "
                         f"{row['mean_pickup_wait']:.2f}/{row['p90_pickup_wait']:.2f} min | "
                         f"{row['mean_empty_travel']:,.2f}")
        return "\n".join(lines)

    @staticmethod
    def _number(text: str, maximum: int) -> int:
        if not re.fullmatch(r"[0-9]{1,10}", text) or int(text) > maximum:
            raise ValueError(f"Enter a whole number between 0 and {maximum}")
        return int(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"model server origin (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--model", default="fleet-operator")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/step5-ppo/graph/best.pt"))
    parser.add_argument("--nyc-data", type=Path, default=Path("runs/step7-data"))
    parser.add_argument("--synthetic-report", type=Path, default=Path("runs/step6-evaluation/evaluation.json"))
    parser.add_argument("--manhattan-report", type=Path, default=Path("runs/step7-manhattan/manhattan_evaluation.json"))
    parser.add_argument("--ask", action="append", help="execute one turn; repeat to share a replay across turns")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = LocalCommandModel(args.server, args.model)
    operator = FleetOperator(args.checkpoint, args.nyc_data, args.synthetic_report, args.manhattan_report)
    if args.ask:
        for message in args.ask:
            print(f"> {message}\n{operator.ask(model, message)}\n", flush=True)
        return
    print(f"Fleet operator\n{_EXAMPLES}\nEnter 'exit' to close.")
    while True:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.casefold() in {"exit", "quit"}:
            break
        if message:
            print(operator.ask(model, message))


if __name__ == "__main__":
    main()
