"""Supervised learning module: classifies student performance.

Features : study_hours, attendance (%), previous_marks (%)
Labels   : "At Risk", "Average", "Good"

A DecisionTreeClassifier from scikit-learn is trained on a generated dataset
and validated with a train/test split. If scikit-learn is unavailable, an
equivalent rule-based fallback keeps the application runnable (NFR3).
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

LABELS = ("At Risk", "Average", "Good")

RECOMMENDATIONS: Dict[str, List[str]] = {
    "At Risk": [
        "Increase daily study time to at least 3 hours.",
        "Attend all classes; attendance strongly affects your result.",
        "Start with AI Search Algorithms and solve past questions.",
        "Meet your faculty advisor for a weekly review.",
    ],
    "Average": [
        "Keep a steady 2 hours of focused study each day.",
        "Revise weak topics such as Machine Learning basics.",
        "Practise numerical problems weekly.",
    ],
    "Good": [
        "Maintain your current routine of 1-2 hours revision.",
        "Attempt advanced problems and mini projects.",
        "Help peers; teaching reinforces your own understanding.",
    ],
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def label_for(study_hours: float, attendance: float, marks: float) -> str:
    """Ground-truth rule used to generate the training dataset."""
    score = 0.35 * min(study_hours, 8) / 8 * 100 + 0.30 * attendance + 0.35 * marks
    if score >= 72:
        return "Good"
    if score >= 55:
        return "Average"
    return "At Risk"


def generate_dataset(n: int = 400, seed: int = 42) -> Tuple[List[List[float]], List[str]]:
    """Create a synthetic but realistic training dataset."""
    rng = random.Random(seed)
    features: List[List[float]] = []
    labels: List[str] = []
    for _ in range(n):
        study = round(rng.uniform(0, 8), 1)
        attendance = round(rng.uniform(40, 100), 1)
        marks = round(rng.uniform(30, 98), 1)
        features.append([study, attendance, marks])
        labels.append(label_for(study, attendance, marks))
    return features, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StudentModel:
    """Trains (or falls back to rules) and predicts a student's category."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = None
        self.accuracy: float = 0.0
        self.backend: str = "rule-based"
        self.train()

    def train(self) -> None:
        features, labels = generate_dataset(seed=self.seed)
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            self.model = None
            self.backend = "rule-based"
            self.accuracy = 1.0
            return

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=self.seed, stratify=labels
        )
        # max_depth limits complexity, reducing overfitting (bias/variance trade-off).
        model = DecisionTreeClassifier(max_depth=5, random_state=self.seed)
        model.fit(x_train, y_train)
        self.model = model
        self.accuracy = round(float(model.score(x_test, y_test)), 3)
        self.backend = "scikit-learn DecisionTreeClassifier"

    def predict(self, study_hours: float, attendance: float, marks: float) -> str:
        if self.model is None:
            return label_for(study_hours, attendance, marks)
        return str(self.model.predict([[study_hours, attendance, marks]])[0])

    def recommend(
        self, study_hours: float, attendance: float, marks: float, preference: str = "Morning"
    ) -> Dict[str, object]:
        """Return the predicted category plus a concrete study plan."""
        category = self.predict(study_hours, attendance, marks)
        tips: Sequence[str] = RECOMMENDATIONS[category]
        target_hours = {"At Risk": 3.0, "Average": 2.0, "Good": 1.5}[category]
        slot = "6:00 AM - 8:00 AM" if preference.strip().lower().startswith("morning") else "8:00 PM - 10:00 PM"
        return {
            "category": category,
            "suggested_hours": target_hours,
            "preferred_slot": slot,
            "tips": list(tips),
            "model": self.backend,
            "validation_accuracy": self.accuracy,
        }
