"""Knowledge representation for the Smart Campus AI Assistant.

Campus information is stored as facts (location -> attributes) and the campus
map is stored as a weighted graph with coordinates used as an A* heuristic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Facts: knowledge about each campus location
# ---------------------------------------------------------------------------

FACTS: Dict[str, Dict[str, str]] = {
    "Library": {
        "location": "The library is located in Block B, ground and first floor.",
        "timing": "The library is open from 8:00 AM to 10:00 PM on all working days.",
        "facilities": "The library provides reading areas, digital resources and computer facilities.",
    },
    "Canteen": {
        "location": "The canteen is next to the Academic Block, near the main lawn.",
        "timing": "The canteen is open from 7:30 AM to 9:00 PM.",
        "facilities": "The canteen offers breakfast, lunch, snacks and beverages.",
    },
    "Hostel": {
        "location": "The hostel blocks are at the north side of the campus.",
        "timing": "Hostel gates close at 10:30 PM.",
        "facilities": "The hostel provides Wi-Fi, laundry, a common room and a gym.",
    },
    "Academic Block": {
        "location": "The Academic Block is at the centre of the campus, opposite the library.",
        "timing": "Classes run from 8:30 AM to 5:30 PM.",
        "facilities": "The Academic Block has lecture halls, labs and faculty rooms.",
    },
    "Sports Complex": {
        "location": "The sports complex is beside the hostel, at the east gate.",
        "timing": "The sports complex is open from 6:00 AM to 8:00 PM.",
        "facilities": "The sports complex has a football ground, courts and an indoor hall.",
    },
    "Admin Office": {
        "location": "The admin office is on the ground floor of the Academic Block.",
        "timing": "The admin office works from 9:00 AM to 5:00 PM.",
        "facilities": "The admin office handles fees, documents and student records.",
    },
    "Medical Centre": {
        "location": "The medical centre is near the hostel entrance.",
        "timing": "The medical centre is available 24 hours.",
        "facilities": "The medical centre provides first aid, a doctor on call and an ambulance.",
    },
}

# Synonyms so free-text queries can be matched to a known location.
LOCATION_ALIASES: Dict[str, str] = {
    "library": "Library",
    "books": "Library",
    "reading room": "Library",
    "canteen": "Canteen",
    "cafeteria": "Canteen",
    "mess": "Canteen",
    "food": "Canteen",
    "hostel": "Hostel",
    "dorm": "Hostel",
    "room": "Hostel",
    "academic block": "Academic Block",
    "class": "Academic Block",
    "classroom": "Academic Block",
    "lab": "Academic Block",
    "sports": "Sports Complex",
    "sports complex": "Sports Complex",
    "ground": "Sports Complex",
    "gym": "Sports Complex",
    "admin": "Admin Office",
    "admin office": "Admin Office",
    "office": "Admin Office",
    "fees": "Admin Office",
    "medical": "Medical Centre",
    "medical centre": "Medical Centre",
    "hospital": "Medical Centre",
    "doctor": "Medical Centre",
    "clinic": "Medical Centre",
}

# Intent keywords -> attribute stored in FACTS
INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "location": ("where", "located", "location", "reach", "find", "place"),
    "timing": ("time", "timing", "open", "close", "hours", "when", "schedule"),
    "facilities": ("facility", "facilities", "available", "service", "offer", "provide", "what"),
}

# ---------------------------------------------------------------------------
# Campus map: weighted graph + coordinates for the A* heuristic
# ---------------------------------------------------------------------------

CAMPUS_GRAPH: Dict[str, Dict[str, float]] = {
    "Hostel": {"Academic Block": 4, "Sports Complex": 2, "Medical Centre": 1},
    "Academic Block": {"Hostel": 4, "Library": 2, "Canteen": 1, "Admin Office": 1},
    "Library": {"Academic Block": 2, "Canteen": 3},
    "Canteen": {"Academic Block": 1, "Library": 3, "Sports Complex": 4},
    "Sports Complex": {"Hostel": 2, "Canteen": 4},
    "Admin Office": {"Academic Block": 1},
    "Medical Centre": {"Hostel": 1},
}

# (x, y) grid positions used only by the heuristic function.
COORDINATES: Dict[str, Tuple[float, float]] = {
    "Hostel": (0, 0),
    "Medical Centre": (1, 0),
    "Sports Complex": (0, 2),
    "Academic Block": (3, 1),
    "Admin Office": (4, 1),
    "Canteen": (3, 2),
    "Library": (4, 3),
}


class KnowledgeBase:
    """Stores campus facts and the campus map, and answers simple queries."""

    def __init__(
        self,
        facts: Optional[Dict[str, Dict[str, str]]] = None,
        graph: Optional[Dict[str, Dict[str, float]]] = None,
        coordinates: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.facts = facts if facts is not None else FACTS
        self.graph = graph if graph is not None else CAMPUS_GRAPH
        self.coordinates = coordinates if coordinates is not None else COORDINATES

    # -- locations ---------------------------------------------------------
    def locations(self) -> List[str]:
        return sorted(self.graph.keys())

    def normalise_location(self, text: str) -> Optional[str]:
        """Map free text such as 'lib' or 'the canteen' to a known location."""
        if not text:
            return None
        cleaned = text.strip().lower()
        for name in self.graph:
            if cleaned == name.lower():
                return name
        # Longest aliases first so "academic block" beats "block".
        for alias in sorted(LOCATION_ALIASES, key=len, reverse=True):
            if alias in cleaned:
                return LOCATION_ALIASES[alias]
        for name in self.graph:
            if name.lower().startswith(cleaned) and len(cleaned) >= 3:
                return name
        return None

    # -- query answering ---------------------------------------------------
    def detect_intent(self, question: str) -> str:
        text = question.lower()
        scores = {
            intent: sum(1 for word in words if word in text)
            for intent, words in INTENT_KEYWORDS.items()
        }
        best = max(scores, key=lambda key: scores[key])
        return best if scores[best] > 0 else "location"

    def answer(self, question: str) -> str:
        """Return an answer for a natural-language campus question."""
        if not question or not question.strip():
            return "Please type a question, for example: Where is the library?"

        place = self.normalise_location(question)
        if place is None:
            return (
                "I do not have information about that yet. "
                "Known places: " + ", ".join(self.locations()) + "."
            )

        intent = self.detect_intent(question)
        entry = self.facts.get(place, {})
        return entry.get(intent) or entry.get("location", "No information available.")

    # -- map helpers -------------------------------------------------------
    def neighbours(self, node: str) -> Iterable[Tuple[str, float]]:
        return self.graph.get(node, {}).items()

    def straight_line_distance(self, node: str, goal: str) -> float:
        """Admissible heuristic: Euclidean distance between grid positions."""
        if node not in self.coordinates or goal not in self.coordinates:
            return 0.0
        (x1, y1), (x2, y2) = self.coordinates[node], self.coordinates[goal]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
