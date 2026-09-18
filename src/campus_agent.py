"""The intelligent agent that coordinates every AI capability.

Agent cycle: perceive (user input) -> reason (knowledge base / search / ML /
NLP) -> act (return a structured result to the terminal UI).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from database.database import Database
from knowledge.knowledge_base import KnowledgeBase
from ml.student_model import StudentModel
from nlp.sentiment import analyze
from search.astar import astar
from search.bfs import bfs


class CampusAgent:
    """Goal-based intelligent agent for campus services."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.kb = KnowledgeBase()
        self.db = db if db is not None else Database()
        self.db.seed(self.kb.graph)
        self.model = StudentModel()

    # -- Module 1: knowledge-based question answering ----------------------
    def ask(self, question: str) -> str:
        answer = self.kb.answer(question)
        self.db.log_query(question, answer)
        return answer

    # -- Module 2: navigation ---------------------------------------------
    def find_route(self, source: str, destination: str, algorithm: str = "astar") -> Dict[str, object]:
        start = self.kb.normalise_location(source)
        goal = self.kb.normalise_location(destination)

        if start is None or goal is None:
            return {"ok": False, "message": "Unknown location. Please choose from the list."}
        if start == goal:
            return {"ok": True, "algorithm": "none", "path": [start], "cost": 0.0, "expanded": 1}

        if algorithm.lower() == "bfs":
            path, cost, expanded = bfs(self.kb.graph, start, goal)
            name = "BFS (uninformed search)"
        else:
            path, cost, expanded = astar(
                self.kb.graph, start, goal, self.kb.straight_line_distance
            )
            name = "A* (informed search)"

        if path is None:
            return {"ok": False, "message": f"No route found between {start} and {goal}."}
        return {"ok": True, "algorithm": name, "path": path, "cost": cost, "expanded": expanded}

    def compare_algorithms(self, source: str, destination: str) -> List[Dict[str, object]]:
        return [self.find_route(source, destination, a) for a in ("bfs", "astar")]

    # -- Module 3: ML recommendation ---------------------------------------
    def recommend(
        self, study_hours: float, attendance: float, marks: float, preference: str = "Morning"
    ) -> Dict[str, object]:
        return self.model.recommend(study_hours, attendance, marks, preference)

    # -- Module 4: sentiment analysis --------------------------------------
    def analyze_feedback(self, text: str, persist: bool = True) -> Dict[str, object]:
        result = analyze(text)
        if persist and text.strip():
            self.db.save_feedback(text.strip(), str(result["label"]), float(result["score"]))
        return result

    # -- Module 5: reports --------------------------------------------------
    def report(self) -> Dict[str, object]:
        return {
            "locations": len(self.kb.locations()),
            "questions_asked": self.db.query_count(),
            "sentiment_summary": self.db.sentiment_summary(),
            "recent_feedback": self.db.recent_feedback(),
            "model": self.model.backend,
            "model_accuracy": self.model.accuracy,
        }

    def close(self) -> None:
        self.db.close()
