#!/usr/bin/env python3
"""Index and browse an incrementally flushed selection candidate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    frame_id INTEGER,
    parent_frame_id INTEGER,
    demand_id INTEGER,
    candidate_id INTEGER,
    source_candidate_id INTEGER,
    multiplicity INTEGER,
    depth INTEGER,
    position INTEGER,
    rank INTEGER,
    token_id INTEGER,
    piece TEXT,
    prefix TEXT,
    context TEXT,
    completion TEXT,
    status TEXT,
    reason TEXT,
    local_logit REAL,
    local_log_probability REAL,
    observer_score REAL,
    backed_score REAL,
    aggregate_before REAL,
    aggregate_after REAL,
    scale_scores TEXT NOT NULL,
    raw TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind_event ON events(kind, event_id);
CREATE INDEX IF NOT EXISTS events_frame_event ON events(frame_id, event_id);
CREATE INDEX IF NOT EXISTS events_candidate_event ON events(candidate_id, event_id);
CREATE INDEX IF NOT EXISTS events_depth_event ON events(depth, event_id);
CREATE INDEX IF NOT EXISTS events_backed_score ON events(backed_score);
"""


class LedgerIndex:
    def __init__(self, ledger_path: Path, database_path: Path):
        self.ledger_path = ledger_path.resolve()
        self.database_path = database_path.resolve()
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def _metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _identity(self) -> tuple[str, int]:
        with self.ledger_path.open("rb") as stream:
            first = stream.readline()
        return hashlib.sha256(first).hexdigest(), self.ledger_path.stat().st_size

    def _reset(self) -> None:
        self.connection.execute("DELETE FROM events")
        self.connection.execute("DELETE FROM metadata")
        self.connection.commit()

    def refresh(self) -> int:
        with self.lock:
            first_hash, size = self._identity()
            indexed_path = self._metadata("ledger_path")
            indexed_hash = self._metadata("first_line_sha256")
            offset = int(self._metadata("offset") or "0")
            if (
                indexed_path != str(self.ledger_path)
                or indexed_hash != first_hash
                or size < offset
            ):
                self._reset()
                offset = 0

            inserted = 0
            with self.ledger_path.open("rb") as stream:
                stream.seek(offset)
                while True:
                    line_offset = stream.tell()
                    raw_line = stream.readline()
                    if not raw_line:
                        break
                    if not raw_line.endswith(b"\n"):
                        stream.seek(line_offset)
                        break
                    record = json.loads(raw_line)
                    if record.get("schema") != "llama2.selection-ledger":
                        raise ValueError(f"unexpected ledger schema at byte {line_offset}")
                    self._insert(record, raw_line.decode("utf-8").rstrip("\n"))
                    inserted += 1
                offset = stream.tell()

            self._set_metadata("ledger_path", str(self.ledger_path))
            self._set_metadata("first_line_sha256", first_hash)
            self._set_metadata("offset", str(offset))
            self.connection.commit()
            return inserted

    def _insert(self, record: dict[str, Any], raw: str) -> None:
        columns = (
            "event_id", "kind", "frame_id", "parent_frame_id", "demand_id",
            "candidate_id", "source_candidate_id", "multiplicity", "depth",
            "position", "rank", "token_id", "piece", "prefix", "context", "completion",
            "status", "reason", "local_logit", "local_log_probability",
            "observer_score", "backed_score", "aggregate_before",
            "aggregate_after",
        )
        values = [record.get(column) for column in columns]
        values.extend(
            [
                json.dumps(record.get("scale_scores", []), separators=(",", ":")),
                raw,
            ]
        )
        placeholders = ",".join("?" for _ in values)
        self.connection.execute(
            f"INSERT OR REPLACE INTO events({','.join(columns)},scale_scores,raw) "
            f"VALUES ({placeholders})",
            values,
        )

    def summary(self) -> dict[str, Any]:
        self.refresh()
        totals = self.connection.execute(
            "SELECT COUNT(*) AS events, "
            "COUNT(DISTINCT candidate_id) AS candidates, "
            "COUNT(DISTINCT frame_id) AS frames, "
            "MAX(event_id) AS last_event FROM events"
        ).fetchone()
        kinds = self.connection.execute(
            "SELECT kind, COUNT(*) AS count FROM events "
            "GROUP BY kind ORDER BY count DESC, kind"
        ).fetchall()
        run = self.connection.execute(
            "SELECT raw FROM events WHERE kind = 'run_start' ORDER BY event_id LIMIT 1"
        ).fetchone()
        return {
            **dict(totals),
            "kinds": [dict(row) for row in kinds],
            "run": None if run is None else json.loads(run["raw"]),
        }

    def events(self, query: dict[str, list[str]]) -> dict[str, Any]:
        self.refresh()
        where: list[str] = []
        values: list[Any] = []

        def integer_filter(parameter: str, column: str) -> None:
            if parameter in query and query[parameter][0] != "":
                where.append(f"{column} = ?")
                values.append(int(query[parameter][0]))

        integer_filter("frame", "frame_id")
        integer_filter("depth", "depth")
        integer_filter("position", "position")
        integer_filter("rank", "rank")
        integer_filter("token", "token_id")

        if query.get("kind", [""])[0]:
            requested = [part for part in query["kind"][0].split(",") if part]
            where.append("kind IN (" + ",".join("?" for _ in requested) + ")")
            values.extend(requested)
        if query.get("status", [""])[0]:
            where.append("status = ?")
            values.append(query["status"][0])
        if query.get("text", [""])[0]:
            where.append(
                "(COALESCE(piece,'') || COALESCE(prefix,'') || "
                "COALESCE(context,'') || "
                "COALESCE(completion,'')) LIKE ?"
            )
            values.append(f"%{query['text'][0]}%")
        if query.get("min_score", [""])[0]:
            where.append("COALESCE(backed_score, observer_score) >= ?")
            values.append(float(query["min_score"][0]))
        if query.get("max_score", [""])[0]:
            where.append("COALESCE(backed_score, observer_score) <= ?")
            values.append(float(query["max_score"][0]))

        limit = min(max(int(query.get("limit", ["200"])[0]), 1), 2000)
        offset = max(int(query.get("offset", ["0"])[0]), 0)
        clause = "" if not where else " WHERE " + " AND ".join(where)
        total = self.connection.execute(
            "SELECT COUNT(*) FROM events" + clause, values
        ).fetchone()[0]
        rows = self.connection.execute(
            "SELECT * FROM events"
            + clause
            + " ORDER BY event_id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "events": [self._row(row) for row in rows],
        }

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["scale_scores"] = json.loads(result["scale_scores"])
        result.pop("raw", None)
        return result


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selection candidate ledger</title>
<style>
:root { color-scheme: light dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
body { margin: 0; }
header { position: sticky; top: 0; z-index: 2; padding: 12px; background: Canvas; border-bottom: 1px solid GrayText; }
form { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
label { display: grid; gap: 3px; font-size: 12px; }
input { width: 9em; }
#text { width: 24em; }
#summary { margin-top: 8px; font-size: 12px; }
main { padding: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border-bottom: 1px solid color-mix(in srgb, CanvasText 20%, transparent); padding: 5px; text-align: left; vertical-align: top; }
th { position: sticky; top: 112px; background: Canvas; }
tr:hover { background: color-mix(in srgb, Highlight 18%, transparent); cursor: pointer; }
.text { white-space: pre-wrap; max-width: 40em; overflow-wrap: anywhere; }
.number { text-align: right; font-variant-numeric: tabular-nums; }
dialog { width: min(1000px, 92vw); }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
</head>
<body>
<header>
  <form id="filters">
    <label>Kind<input id="kind" placeholder="candidate,backup"></label>
    <label>Text<input id="text" placeholder="decoded text"></label>
    <label>Frame<input id="frame" type="number"></label>
    <label>Depth<input id="depth" type="number"></label>
    <label>Position<input id="position" type="number"></label>
    <label>Rank<input id="rank" type="number"></label>
    <label>Min score<input id="min_score" type="number" step="any"></label>
    <label>Max score<input id="max_score" type="number" step="any"></label>
    <button>Apply</button>
    <button type="button" id="previous">Previous</button>
    <button type="button" id="next">Next</button>
  </form>
  <div id="summary">Loading…</div>
</header>
<main><table>
<thead><tr>
<th>event</th><th>kind</th><th>frame</th><th>depth</th><th>rank</th>
<th>token</th><th>decoded candidate</th><th>local log p</th>
<th>observer</th><th>backed</th><th>multiplicity</th><th>status/reason</th>
</tr></thead>
<tbody id="rows"></tbody>
</table></main>
<dialog id="detail"><button id="close">Close</button><pre></pre></dialog>
<script>
const pageSize = 200;
let offset = 0;
const fields = ["kind","text","frame","depth","position","rank","min_score","max_score"];
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
const score = value => value == null ? "" : Number(value).toPrecision(8);
async function load() {
  const p = new URLSearchParams({limit: pageSize, offset: offset});
  for (const id of fields) {
    const value = document.getElementById(id).value;
    if (value) p.set(id, value);
  }
  const data = await fetch("/api/events?" + p).then(r => r.json());
  document.getElementById("summary").textContent =
    data.total.toLocaleString() + " matching events; showing " +
    (data.offset + 1) + "–" + Math.min(data.offset + data.limit, data.total);
  const body = document.getElementById("rows");
  body.innerHTML = "";
  for (const e of data.events) {
    const tr = document.createElement("tr");
    const decoded = e.completion ?? e.context ?? e.prefix ?? e.piece ?? "";
    tr.innerHTML = "<td class=number>" + e.event_id + "</td><td>" + esc(e.kind) + "</td>" +
      "<td class=number>" + (e.frame_id ?? "") + "</td><td class=number>" + (e.depth ?? "") + "</td>" +
      "<td class=number>" + (e.rank ?? "") + "</td><td>" + (e.token_id ?? "") + " " + esc(e.piece ?? "") + "</td>" +
      "<td class=text>" + esc(decoded) + "</td><td class=number>" + score(e.local_log_probability) + "</td>" +
      "<td class=number>" + score(e.observer_score) + "</td><td class=number>" + score(e.backed_score) + "</td>" +
      "<td class=number>" + (e.multiplicity ?? "") + "</td>" +
      "<td>" + esc([e.status,e.reason].filter(Boolean).join(" / ")) + "</td>";
    tr.onclick = () => {
      document.querySelector("#detail pre").textContent = JSON.stringify(e, null, 2);
      document.getElementById("detail").showModal();
    };
    body.appendChild(tr);
  }
}
document.getElementById("filters").onsubmit = event => {
  event.preventDefault(); offset = 0; load();
};
document.getElementById("previous").onclick = () => {
  offset = Math.max(0, offset - pageSize); load();
};
document.getElementById("next").onclick = () => {
  offset += pageSize; load();
};
document.getElementById("close").onclick = () => document.getElementById("detail").close();
load();
setInterval(load, 2000);
</script>
</body></html>
"""


def handler_for(index: LedgerIndex) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send("text/html; charset=utf-8", PAGE.encode())
                    return
                if parsed.path == "/api/summary":
                    self._json(index.summary())
                    return
                if parsed.path == "/api/events":
                    self._json(index.events(urllib.parse.parse_qs(parsed.query)))
                    return
                self.send_error(404)
            except (ValueError, OSError, sqlite3.Error) as error:
                self._json({"error": str(error)}, status=500)

        def _json(self, value: Any, status: int = 200) -> None:
            self._send(
                "application/json; charset=utf-8",
                json.dumps(value, ensure_ascii=False).encode(),
                status,
            )

        def _send(self, content_type: str, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *arguments: object) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if not args.ledger.is_file():
        parser.error(f"ledger does not exist: {args.ledger}")
    database = args.database or args.ledger.with_suffix(args.ledger.suffix + ".sqlite3")
    index = LedgerIndex(args.ledger, database)
    index.refresh()
    if args.summary:
        json.dump(index.summary(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    server = ThreadingHTTPServer((args.host, args.port), handler_for(index))
    print(f"http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
