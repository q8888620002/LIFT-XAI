#!/usr/bin/env python3
"""Simple backend for collecting clinician rating submissions.

Run:
    python tools/ratings_server.py --host 0.0.0.0 --port 8000

Then submit to:
    POST /api/ratings
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "unknown"


def _is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", value))


class RatingsAPIHandler(BaseHTTPRequestHandler):
    submissions_dir: Path

    def _set_headers(self, status: HTTPStatus, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        # Allow static frontend to call this API from another origin/port.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._set_headers(status)
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs concise but visible in terminal.
        super().log_message(fmt, *args)

    def do_OPTIONS(self) -> None:
        self._set_headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/ratings":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        length_header = self.headers.get("Content-Length")
        if not length_header:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing Content-Length"})
            return

        try:
            length = int(length_header)
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return

        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON"})
            return

        required = ["cohort", "method", "expertise", "specialty", "rater_email", "ratings"]
        missing = [key for key in required if key not in data]
        if missing:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Missing required keys: {', '.join(missing)}"},
            )
            return

        if not _is_valid_email(str(data.get("rater_email", ""))):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid rater_email"})
            return

        submission_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        payload = {
            "submission_id": submission_id,
            "received_at": now_iso,
            **data,
        }

        cohort = _safe_slug(str(data.get("cohort", "unknown")))
        specialty = _safe_slug(str(data.get("specialty", "unknown")))
        filename = f"ratings_{cohort}_{specialty}_{_utc_stamp()}_{submission_id[:8]}.json"

        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.submissions_dir / filename
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self._write_json(
            HTTPStatus.CREATED,
            {
                "status": "saved",
                "submission_id": submission_id,
                "file": str(out_path.as_posix()),
            },
        )


def build_handler(submissions_dir: Path) -> type[RatingsAPIHandler]:
    class _Handler(RatingsAPIHandler):
        pass

    _Handler.submissions_dir = submissions_dir
    return _Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ratings collection backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--out-dir",
        default="data/ratings_submissions",
        help="Directory where submissions are stored",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()

    server = ThreadingHTTPServer((args.host, args.port), build_handler(out_dir))
    print(f"Ratings backend listening on http://{args.host}:{args.port}")
    print(f"Saving submissions to: {out_dir}")
    print("Health check: GET /api/health")
    print("Submit endpoint: POST /api/ratings")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ratings backend...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
