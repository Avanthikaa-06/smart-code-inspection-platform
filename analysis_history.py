"""
analysis_history.py
--------------------
Small SQLite-backed persistent store for per-user analysis history,
per-analysis AI chat history, and PDF report file organization.

This is NOT Streamlit session state — it's the durable store that must
survive logout and app restarts. Every read/write is scoped to a
`user_id`: the authenticated user's stable identifier (see auth.py,
which uses the lowercase username — including `gh_<login>` for
GitHub-authenticated users — as the users_db.json key, so that same
value is reused here as the primary key for history rows, chat
messages, and report directories).

Nothing in here ever stores a Gemini API key, a password, or an auth
token — see `_strip_sensitive`, which is applied to every payload
(including chat message text) before it's serialized.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "analysis_history.db")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Any dict key that looks like a secret is dropped before persisting,
# regardless of what the caller passes in — defense in depth on top of
# the fact that callers should never pass these into this module at all.
_SENSITIVE_KEYS = {
    "api_key", "gemini_api_key", "password", "password_hash", "salt",
    "token", "access_token", "client_secret",
}

_VALID_CHAT_ROLES = {"user", "assistant"}


def _strip_sensitive(obj):
    if isinstance(obj, dict):
        return {
            k: _strip_sensitive(v)
            for k, v in obj.items()
            if k.lower() not in _SENSITIVE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_sensitive(v) for v in obj]
    return obj


def _safe_id(value: str) -> str:
    """Filesystem-safe version of an id/username, for use in paths."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value)) or "unknown"


# ==============================================================================
# SCHEMA
# ==============================================================================
def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                username TEXT,
                file_name TEXT NOT NULL,
                language TEXT,
                analyzed_at TEXT NOT NULL,
                security_score INTEGER,
                total_findings INTEGER,
                critical_count INTEGER DEFAULT 0,
                high_count INTEGER DEFAULT 0,
                medium_count INTEGER DEFAULT 0,
                low_count INTEGER DEFAULT 0,
                analysis_json TEXT,
                report_path TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON analysis_history(user_id)")

        # Per-analysis AI chat log. One row per message (not per exchange),
        # so a Q/A turn is two rows — this keeps ordering trivial (id or
        # created_at ASC) and makes partial-failure states (e.g. the answer
        # never came back) representable instead of losing the question too.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (record_id) REFERENCES analysis_history(id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_record_user "
            "ON analysis_chat_history(record_id, user_id)"
        )
        conn.commit()


# ==============================================================================
# REPORT FILE ORGANIZATION — reports/<user_id>/<analysis_id>/report.pdf
# ==============================================================================
def new_analysis_id() -> str:
    return uuid.uuid4().hex[:12]


def user_reports_dir(user_id: str) -> str:
    path = os.path.join(REPORTS_DIR, _safe_id(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def report_path_for(user_id: str, analysis_id: str) -> str:
    return os.path.join(user_reports_dir(user_id), _safe_id(analysis_id), "report.pdf")


def report_belongs_to_user(report_path: Optional[str], user_id: str) -> bool:
    """Ownership check every caller must go through before ever serving
    report bytes back — confirms the stored path actually resolves to
    somewhere inside this user's own reports directory, not just that a
    history row happened to reference it."""
    if not report_path:
        return False
    try:
        user_dir = os.path.realpath(user_reports_dir(user_id))
        target = os.path.realpath(report_path)
    except OSError:
        return False
    return target == user_dir or target.startswith(user_dir + os.sep)


# ==============================================================================
# OWNERSHIP HELPER (shared by history + chat lookups)
# ==============================================================================
def _record_belongs_to_user(conn: sqlite3.Connection, record_id, user_id: str) -> bool:
    """Lightweight existence+ownership check against analysis_history,
    without pulling the (potentially large) analysis_json column. Used
    by the chat functions so a record_id alone is never sufficient —
    same rule as get_history_record / delete_history_record below."""
    if record_id is None or not user_id:
        return False
    row = conn.execute(
        "SELECT 1 FROM analysis_history WHERE id = ? AND user_id = ?",
        (record_id, user_id),
    ).fetchone()
    return row is not None


# ==============================================================================
# READ / WRITE — ANALYSIS HISTORY
# ==============================================================================
def save_history_record(
    user_id: str,
    username: str,
    file_name: str,
    language: str,
    security_score: int,
    total_findings: int,
    critical_count: int,
    high_count: int,
    medium_count: int,
    low_count: int,
    analysis_payload: dict,
    report_path: Optional[str] = None,
) -> int:
    """
    Inserts one history row for the given user and returns its new id.
    `analysis_payload` should be a dict like {"deep_result": ..., "result": ...} —
    everything needed to reconstruct the "View Analysis" screen later. It is
    recursively stripped of any key that looks like a secret before being
    serialized (see _strip_sensitive) — API keys and passwords never reach
    this table even if a caller accidentally included them upstream.
    """
    payload = _strip_sensitive(analysis_payload)
    analyzed_at = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            """
            INSERT INTO analysis_history (
                user_id, username, file_name, language, analyzed_at,
                security_score, total_findings, critical_count, high_count,
                medium_count, low_count, analysis_json, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, username, file_name, language, analyzed_at,
                security_score, total_findings, critical_count, high_count,
                medium_count, low_count, json.dumps(payload), report_path,
            ),
        )
        conn.commit()
        return cur.lastrowid


def list_history_for_user(user_id: str) -> list[dict]:
    """Summary rows only (no analysis_json — that's loaded on demand by
    get_history_record) — every query is filtered by user_id, never global."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, user_id, username, file_name, language, analyzed_at,
                   security_score, total_findings, critical_count, high_count,
                   medium_count, low_count, report_path
            FROM analysis_history
            WHERE user_id = ?
            ORDER BY analyzed_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_history_record(record_id, user_id: str) -> Optional[dict]:
    """
    Returns the full record (including the parsed analysis_payload) ONLY
    if it belongs to user_id. This is the ownership check that stands
    between "View Analysis" / "PDF Report" and another user's data — a
    record_id alone is never sufficient to read a row.
    """
    if record_id is None or not user_id:
        return None
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM analysis_history WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["analysis_payload"] = json.loads(record.get("analysis_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            record["analysis_payload"] = {}
        return record


def delete_history_record(record_id, user_id: str) -> bool:
    """
    Deletes exactly one history row — ONLY if it belongs to user_id — and
    returns whether a row was actually removed. Mirrors the same ownership
    check used everywhere else in this module (get_history_record,
    report_belongs_to_user): a record_id alone is never sufficient, it must
    also match user_id in the WHERE clause.

    This also deletes that analysis's AI chat history (see
    clear_chat_history) so a deleted analysis never leaves an orphaned
    conversation behind, and — best-effort — that row's own report
    directory under this user's reports folder. Deleting one entry can
    never touch any other user's data or any other history entry for the
    same user.
    """
    if record_id is None or not user_id:
        return False

    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT report_path FROM analysis_history WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
        if not row:
            return False

        cur = conn.execute(
            "DELETE FROM analysis_history WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        conn.execute(
            "DELETE FROM analysis_chat_history WHERE record_id = ? AND user_id = ?",
            (record_id, user_id),
        )
        conn.commit()
        deleted = cur.rowcount > 0

    if deleted:
        report_path = row["report_path"] if row else None
        if report_path and report_belongs_to_user(report_path, user_id):
            try:
                report_dir = os.path.dirname(report_path)
                if os.path.isdir(report_dir):
                    shutil.rmtree(report_dir, ignore_errors=True)
            except OSError:
                pass

    return deleted


# ==============================================================================
# READ / WRITE — PER-ANALYSIS AI CHAT HISTORY
# ==============================================================================
def get_chat_history(record_id, user_id: str) -> list[dict]:
    """
    Returns this analysis's chat log as a list of
    {"role": "user"|"assistant", "message": str, "created_at": str}
    dicts in chronological order — ONLY if the analysis belongs to
    user_id. Returns [] both when the analysis has no messages yet and
    when the analysis doesn't exist / isn't owned by user_id, so callers
    can render the same "no previous conversation" empty state either
    way without leaking whether a record_id exists for someone else.
    """
    if record_id is None or not user_id:
        return []
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        if not _record_belongs_to_user(conn, record_id, user_id):
            return []
        rows = conn.execute(
            """
            SELECT role, message, created_at
            FROM analysis_chat_history
            WHERE record_id = ? AND user_id = ?
            ORDER BY id ASC
            """,
            (record_id, user_id),
        ).fetchall()
        return [dict(r) for r in rows]


def append_chat_message(record_id, user_id: str, role: str, message: str) -> Optional[int]:
    """
    Appends one message to this analysis's chat log and returns its new
    row id, or None if the analysis doesn't exist / isn't owned by
    user_id (same ownership rule as everywhere else in this module) or
    role isn't "user"/"assistant". Message text is run through
    _strip_sensitive in case a caller ever passes a dict/list payload
    instead of plain text; a plain string passes through unchanged.

    Call this twice per turn — once for the user's question, once for
    the assistant's answer — so a mid-turn failure still preserves the
    question that was actually asked.
    """
    if record_id is None or not user_id or role not in _VALID_CHAT_ROLES:
        return None
    clean_message = _strip_sensitive(message) if isinstance(message, (dict, list)) else message
    if not clean_message:
        return None
    created_at = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if not _record_belongs_to_user(conn, record_id, user_id):
            return None
        cur = conn.execute(
            """
            INSERT INTO analysis_chat_history (record_id, user_id, role, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (record_id, user_id, role, clean_message, created_at),
        )
        conn.commit()
        return cur.lastrowid


def clear_chat_history(record_id, user_id: str) -> bool:
    """
    Deletes all chat messages for one analysis — ONLY if it belongs to
    user_id — without touching the analysis_history row itself. Exposed
    separately from delete_history_record (which already calls the
    equivalent DELETE) in case the UI ever wants a "clear conversation,
    keep the analysis" action.
    """
    if record_id is None or not user_id:
        return False
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if not _record_belongs_to_user(conn, record_id, user_id):
            return False
        cur = conn.execute(
            "DELETE FROM analysis_chat_history WHERE record_id = ? AND user_id = ?",
            (record_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    
