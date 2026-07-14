"""Backtest run persistence and exact-input AI decision cache."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pa_agent.config.paths import BACKTEST_CACHE_DIR, BACKTEST_RUNS_DIR, PROJECT_ROOT
from pa_agent.records.schema import AnalysisRecord


class BacktestRunStore:
    """Persist partial and complete run evidence as it is produced."""

    def __init__(self, config: Any, *, root: Path | None = None) -> None:
        self.run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.run_dir = (root or BACKTEST_RUNS_DIR) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.db_path = self.run_dir / "run.sqlite"
        self.manifest_path = self.run_dir / "manifest.json"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE decisions (
                decision_id TEXT PRIMARY KEY,
                decision_time_ms INTEGER NOT NULL,
                cache_key TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE equity (
                timestamp_ms INTEGER PRIMARY KEY,
                equity REAL NOT NULL
            );
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        manifest = {
            "run_id": self.run_id,
            "created_at_ms": int(time.time() * 1000),
            "status": "preparing",
            "git_commit": git_commit_hash(),
            "config": config.as_dict() if hasattr(config, "as_dict") else _jsonable(config),
        }
        self._write_manifest(manifest)

    def update_manifest(self, **updates: Any) -> None:
        current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        current.update(_jsonable(updates))
        self._write_manifest(current)

    def add_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO events(timestamp_ms, kind, payload_json) VALUES (?, ?, ?)",
            (int(time.time() * 1000), kind, _dumps(payload)),
        )
        self._conn.commit()

    def add_decision(
        self,
        decision_id: str,
        decision_time_ms: int,
        cache_key: str,
        cache_hit: bool,
        record: AnalysisRecord,
    ) -> None:
        self._conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
            (
                decision_id,
                int(decision_time_ms),
                cache_key,
                1 if cache_hit else 0,
                _dumps(record.model_dump()),
            ),
        )
        self._conn.commit()

    def replace_equity(self, points: list[tuple[int, float]]) -> None:
        self._conn.execute("DELETE FROM equity")
        self._conn.executemany(
            "INSERT INTO equity(timestamp_ms, equity) VALUES (?, ?)",
            ((int(ts), float(value)) for ts, value in points),
        )
        self._conn.commit()

    def replace_trades(self, trades: list[Any]) -> None:
        self._conn.execute("DELETE FROM trades")
        self._conn.executemany(
            "INSERT INTO trades(trade_id, payload_json) VALUES (?, ?)",
            ((str(trade.trade_id), _dumps(asdict(trade))) for trade in trades),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _write_manifest(self, value: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class DecisionCache:
    """SQLite cache keyed by exact dataset/frame/prompt/model/source hashes."""

    def __init__(self, path: Path | None = None) -> None:
        cache_root = BACKTEST_CACHE_DIR
        cache_root.mkdir(parents=True, exist_ok=True)
        self.path = path or cache_root / "decisions.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    cache_key TEXT PRIMARY KEY,
                    created_at_ms INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> AnalysisRecord | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT record_json FROM decisions WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return AnalysisRecord.model_validate(json.loads(row[0]))

    def put(self, cache_key: str, record: AnalysisRecord) -> None:
        if record.exception is not None or record.stage2_decision is None:
            return
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)",
                (cache_key, int(time.time() * 1000), _dumps(record.model_dump())),
            )

    def delete(self, cache_key: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM decisions WHERE cache_key=?", (cache_key,))


class MemoryPendingWriter:
    """Orchestrator writer that keeps backtest analyses out of live records."""

    def __init__(self) -> None:
        self.record: AnalysisRecord | None = None
        self.partial_reason: str | None = None

    def save_full(self, record: AnalysisRecord) -> Path:
        self.record = record
        self.partial_reason = None
        return Path("<backtest-memory>")

    def save_partial(self, record: AnalysisRecord, reason: str) -> Path:
        self.record = record
        self.partial_reason = reason
        return Path("<backtest-memory>")


def build_cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_dumps(payload).encode("utf-8")).hexdigest()


def prompt_source_hash() -> str:
    hasher = hashlib.sha256()
    roots = [PROJECT_ROOT / "prompt_engineering", PROJECT_ROOT / "pa_agent" / "ai"]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".txt", ".md"}:
                continue
            hasher.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def record_preparation_failure(
    request: dict[str, Any],
    error: BaseException,
    *,
    status: str = "failed",
) -> Path:
    """Persist failures that happen before a frozen dataset can create a run store."""
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = BACKTEST_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "run_id": run_id,
        "created_at_ms": int(time.time() * 1000),
        "status": status,
        "stage": "preparing",
        "git_commit": git_commit_hash(),
        "request": request,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def _dumps(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value
