"""SQLite persistence layer for the Smart Campus AI Assistant.

Tables
------
locations : campus places and their block
routes    : weighted edges of the campus map
feedback  : student feedback and its analysed sentiment
queries   : log of questions asked to the assistant
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "campus_data.db",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    block       TEXT
);
CREATE TABLE IF NOT EXISTS routes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    destination TEXT NOT NULL,
    distance    REAL NOT NULL,
    UNIQUE (source, destination)
);
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    sentiment   TEXT NOT NULL,
    score       REAL NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class Database:
    """Thin, safe wrapper around SQLite (parameterised queries only)."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or DEFAULT_DB
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- seeding -----------------------------------------------------------
    def seed(self, graph: Dict[str, Dict[str, float]]) -> None:
        cur = self.conn.cursor()
        for name in graph:
            cur.execute("INSERT OR IGNORE INTO locations (name, block) VALUES (?, ?)", (name, "Main"))
            for dest, distance in graph[name].items():
                cur.execute(
                    "INSERT OR IGNORE INTO routes (source, destination, distance) VALUES (?, ?, ?)",
                    (name, dest, float(distance)),
                )
        self.conn.commit()

    # -- reads -------------------------------------------------------------
    def load_graph(self) -> Dict[str, Dict[str, float]]:
        graph: Dict[str, Dict[str, float]] = {}
        for row in self.conn.execute("SELECT name FROM locations"):
            graph[row["name"]] = {}
        for row in self.conn.execute("SELECT source, destination, distance FROM routes"):
            graph.setdefault(row["source"], {})[row["destination"]] = row["distance"]
        return graph

    def recent_feedback(self, limit: int = 5) -> List[Tuple[str, str, float]]:
        rows = self.conn.execute(
            "SELECT text, sentiment, score FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(r["text"], r["sentiment"], r["score"]) for r in rows]

    def sentiment_summary(self) -> Dict[str, int]:
        rows = self.conn.execute(
            "SELECT sentiment, COUNT(*) AS total FROM feedback GROUP BY sentiment"
        ).fetchall()
        return {r["sentiment"]: r["total"] for r in rows}

    def query_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM queries").fetchone()["c"])

    # -- writes ------------------------------------------------------------
    def save_feedback(self, text: str, sentiment: str, score: float) -> None:
        self.conn.execute(
            "INSERT INTO feedback (text, sentiment, score, created_at) VALUES (?, ?, ?, ?)",
            (text, sentiment, float(score), datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def log_query(self, question: str, answer: str) -> None:
        self.conn.execute(
            "INSERT INTO queries (question, answer, created_at) VALUES (?, ?, ?)",
            (question, answer, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
