"""Serve the FleetLab dashboard and its local simulator controls."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import requests
import torch

from .city import GridCity, Move
from .operator import FleetOperator
from .operator_llm import DEFAULT_SERVER_URL, LocalCommandModel, OperatorCommand

_WEB = Path(__file__).with_name("web")
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


class Dashboard:
    def __init__(self, runs: Path) -> None:
        self.runs = runs
        self._lock = Lock()
        self.operator = FleetOperator(
            runs / "step5-ppo/graph/best.pt", runs / "step7-data",
            runs / "step6-evaluation/evaluation.json", runs / "step7-manhattan/manhattan_evaluation.json",
        )

    def reports(self) -> dict:
        result = {"benchmarks": {}, "training": {}}
        synthetic = self.runs / "step6-evaluation/evaluation.json"
        if synthetic.is_file():
            report = json.loads(synthetic.read_text(encoding="utf-8"))
            for name, key in (("synthetic", "training_grid"), ("transfer", "larger_grid")):
                suite = report["suites"][key]
                result["benchmarks"][name] = {
                    "episodes": len(report["demand_seeds"]), "summary": suite["summary"],
                    "paired": suite["graphsage_minus_reference"],
                }
        manhattan = self.runs / "step7-manhattan/manhattan_evaluation.json"
        if manhattan.is_file():
            report = json.loads(manhattan.read_text(encoding="utf-8"))
            result["benchmarks"]["manhattan"] = {
                "episodes": len(report["window_starts_local"]), "summary": report["summary"],
                "paired": report["graphsage_minus_reference"],
            }
        training = self.runs / "step5-ppo/metrics.jsonl"
        if training.is_file():
            for line in training.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if "validation" in row:
                    result["training"].setdefault(row["policy"], []).append(row)
        return result

    def replay(self, command: OperatorCommand | None = None) -> dict:
        with self._lock:
            if self.operator.replay is None and command is None:
                self.operator.execute(OperatorCommand("start", "synthetic", "1007"))
                self.operator.execute(OperatorCommand("advance", argument="5"))
            elif command is not None:
                self.operator.execute(command)
            replay = self.operator.replay
            if replay is None:
                raise ValueError("Start a replay before advancing")
            sim = replay.audit.simulation
            waiting = Counter(state.request.origin for state in sim.pending_requests.values())
            pickup_waits = [state.picked_up_at - state.request.arrival_minute for state in sim.requests.values()
                            if state.picked_up_at is not None]
            city = replay.city
            if not isinstance(city, GridCity):
                raise ValueError("Dashboard playback requires the synthetic grid")
            zones = [{
                "id": index, "idle": int(features[0]), "waiting": waiting[index],
                "unassigned": int(features[1]), "inbound": int(features[3]),
            } for index, features in enumerate(replay.observation["zone_features"])]
            return {
                "label": replay.label, "grid": asdict(city), "minute": replay.minute,
                "arrival_cutoff": sim.arrival_cutoff, "complete": replay.terminated,
                "arrivals": len(sim.requests), "served": len(pickup_waits),
                "expired": sum(state.expired_at is not None for state in sim.requests.values()),
                "waiting": len(sim.pending_requests), "idle": sum(zone["idle"] for zone in zones),
                "mean_wait": sum(pickup_waits) / len(pickup_waits) if pickup_waits else None,
                "empty_travel": replay.empty_travel, "zones": zones,
                "assignments": [asdict(row) for row in replay.audit.assignments],
                "moves": [{**asdict(row), "choice": row.choice.name.lower()}
                          for row in replay.moves if row.choice is not Move.STAY],
                "explanation": replay.explain(None),
            }

    def ask(self, message: str) -> str:
        try:
            command = LocalCommandModel(DEFAULT_SERVER_URL, "fleet-operator").interpret(message)
        except requests.ConnectionError as exc:
            raise ValueError("The local language model is stopped. Replay controls and decision evidence are still available.") from exc
        if command.action not in {"status", "explain", "compare"}:
            return "Ask about the current run, its latest decisions, or saved benchmarks. Use the replay controls to change the run."
        with self._lock:
            return self.operator.execute(command)


def request_handler(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, body: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(body).encode("utf-8"))

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            try:
                if path in _STATIC:
                    name, media = _STATIC[path]
                    self._send(HTTPStatus.OK, media, (_WEB / name).read_bytes())
                elif path == "/api/reports":
                    self._json(dashboard.reports())
                elif path == "/api/replay":
                    self._json(dashboard.replay())
                else:
                    self._json({"error": "Page not found"}, HTTPStatus.NOT_FOUND)
            except FileNotFoundError:
                self._json({"error": "The replay checkpoint is missing from the local runs folder."}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path not in {"/api/replay", "/api/operator"}:
                self._json({"error": "Page not found"}, HTTPStatus.NOT_FOUND)
                return
            if self.headers.get_content_type() != "application/json":
                self._json({"error": "Expected application/json"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if path == "/api/operator":
                    if set(data) != {"message"} or not isinstance(data["message"], str):
                        raise ValueError("Enter a question for the operator")
                    self._json({"answer": dashboard.ask(data["message"])})
                elif set(data) == {"action", "seed"} and data["action"] == "start" and type(data["seed"]) is int:
                    self._json(dashboard.replay(OperatorCommand("start", "synthetic", str(data["seed"]))))
                elif set(data) == {"action", "minutes"} and data["action"] == "advance" and type(data["minutes"]) is int:
                    self._json(dashboard.replay(OperatorCommand("advance", argument=str(data["minutes"]))))
                else:
                    raise ValueError("Use a replay start or advance command")
            except (OSError, ValueError, requests.RequestException) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="directory containing saved experiment artifacts")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with ThreadingHTTPServer(("127.0.0.1", args.port), request_handler(Dashboard(args.runs))) as server:
        print(f"FleetLab dashboard available at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
