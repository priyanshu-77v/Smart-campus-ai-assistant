"""Unit tests for the Smart Campus AI Assistant."""

from __future__ import annotations

import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from agents.campus_agent import CampusAgent  # noqa: E402
from database.database import Database  # noqa: E402
from knowledge.knowledge_base import KnowledgeBase  # noqa: E402
from ml.student_model import StudentModel, label_for  # noqa: E402
from nlp.sentiment import analyze, tokenize  # noqa: E402
from search.astar import astar  # noqa: E402
from search.bfs import bfs, path_cost  # noqa: E402


@pytest.fixture()
def agent() -> CampusAgent:
    a = CampusAgent(db=Database(":memory:"))
    yield a
    a.close()


# -- Module 1: knowledge base ------------------------------------------------

def test_answer_location_question():
    kb = KnowledgeBase()
    assert "Block B" in kb.answer("Where is the library?")


def test_answer_timing_intent():
    kb = KnowledgeBase()
    assert "open" in kb.answer("What are the canteen timings?").lower()


def test_unknown_location_is_handled():
    kb = KnowledgeBase()
    assert "do not have information" in kb.answer("Where is the swimming pool?")


def test_empty_question_does_not_crash():
    assert KnowledgeBase().answer("   ").startswith("Please type")


# -- Module 2: search --------------------------------------------------------

def test_bfs_finds_a_path():
    kb = KnowledgeBase()
    path, cost, expanded = bfs(kb.graph, "Hostel", "Library")
    assert path[0] == "Hostel" and path[-1] == "Library"
    assert cost == path_cost(kb.graph, path)
    assert expanded >= 1


def test_astar_returns_optimal_cost():
    kb = KnowledgeBase()
    path, cost, _ = astar(kb.graph, "Hostel", "Library", kb.straight_line_distance)
    assert path == ["Hostel", "Academic Block", "Library"]
    assert cost == 6


def test_astar_cost_never_worse_than_bfs():
    kb = KnowledgeBase()
    _, bfs_cost, _ = bfs(kb.graph, "Medical Centre", "Admin Office")
    _, astar_cost, _ = astar(kb.graph, "Medical Centre", "Admin Office", kb.straight_line_distance)
    assert astar_cost <= bfs_cost


def test_unknown_node_returns_none():
    assert astar(KnowledgeBase().graph, "Mars", "Library")[0] is None


# -- Module 3: machine learning ---------------------------------------------

def test_model_trains_with_reasonable_accuracy():
    model = StudentModel()
    assert model.accuracy >= 0.8


def test_strong_student_is_not_at_risk():
    assert StudentModel().predict(6, 95, 88) == "Good"


def test_weak_student_is_at_risk():
    assert StudentModel().predict(0.5, 45, 35) == "At Risk"


def test_recommendation_contains_plan():
    result = StudentModel().recommend(5, 82, 74, "Morning")
    assert result["category"] in {"At Risk", "Average", "Good"}
    assert result["tips"]
    assert "AM" in result["preferred_slot"]


def test_label_rule_boundaries():
    assert label_for(8, 100, 100) == "Good"
    assert label_for(0, 40, 30) == "At Risk"


# -- Module 4: sentiment -----------------------------------------------------

def test_positive_feedback():
    assert analyze("The campus library is very clean and helpful.")["label"] == "POSITIVE"


def test_negative_feedback():
    assert analyze("The canteen service is slow and the food quality is poor.")["label"] == "NEGATIVE"


def test_negation_flips_sentiment():
    assert analyze("The hostel rooms are not clean.")["label"] == "NEGATIVE"


def test_neutral_feedback():
    assert analyze("I went to the academic block today.")["label"] == "NEUTRAL"


def test_tokenizer():
    assert tokenize("Hello, World!") == ["hello", "world"]


# -- Database and agent integration -----------------------------------------

def test_database_round_trip():
    db = Database(":memory:")
    db.seed(KnowledgeBase().graph)
    assert "Library" in db.load_graph()
    db.save_feedback("Great library", "POSITIVE", 1.0)
    assert db.sentiment_summary()["POSITIVE"] == 1
    db.close()


def test_agent_ask_logs_query(agent):
    agent.ask("Where is the hostel?")
    assert agent.db.query_count() == 1


def test_agent_route_accepts_aliases(agent):
    result = agent.find_route("dorm", "books")
    assert result["ok"] is True
    assert result["path"][-1] == "Library"


def test_agent_rejects_unknown_location(agent):
    assert agent.find_route("Moon", "Library")["ok"] is False


def test_agent_report(agent):
    agent.analyze_feedback("The staff is helpful and friendly.")
    report = agent.report()
    assert report["sentiment_summary"]["POSITIVE"] == 1
    assert report["locations"] == 7
