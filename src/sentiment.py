"""Rule-based NLP sentiment analyser for student feedback.

Pipeline: lowercase -> tokenise -> remove stop words -> score tokens against a
sentiment lexicon, handling negation ("not clean") and intensifiers ("very").
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

POSITIVE_WORDS = {
    "good", "great", "excellent", "clean", "helpful", "friendly", "fast",
    "nice", "amazing", "comfortable", "useful", "quiet", "affordable",
    "tasty", "well", "best", "happy", "satisfied", "organised", "organized",
    "supportive", "efficient", "love", "like", "wonderful", "safe",
}

NEGATIVE_WORDS = {
    "bad", "poor", "slow", "dirty", "rude", "worst", "noisy", "expensive",
    "crowded", "unhelpful", "terrible", "awful", "late", "broken", "hate",
    "dislike", "problem", "issue", "delay", "unhygienic", "boring", "unsafe",
    "difficult", "disappointing",
}

NEGATIONS = {"not", "no", "never", "nor", "cannot", "dont", "doesnt", "isnt", "wasnt"}
INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "really": 1.5, "so": 1.3, "too": 1.3}
STOP_WORDS = {"the", "is", "are", "was", "a", "an", "and", "of", "in", "at", "to", "it"}

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens (apostrophes removed)."""
    return [t.replace("'", "") for t in TOKEN_RE.findall(text.lower())]


def remove_stop_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def analyze(text: str) -> Dict[str, object]:
    """Return sentiment label, score, confidence and the matched keywords."""
    tokens = tokenize(text or "")
    score = 0.0
    matched: List[Tuple[str, float]] = []

    for index, token in enumerate(tokens):
        value = 0.0
        if token in POSITIVE_WORDS:
            value = 1.0
        elif token in NEGATIVE_WORDS:
            value = -1.0
        if value == 0.0:
            continue

        window = tokens[max(0, index - 2): index]
        for prev in window:
            if prev in INTENSIFIERS:
                value *= INTENSIFIERS[prev]
            if prev in NEGATIONS:
                value *= -1.0

        score += value
        matched.append((token, value))

    if score > 0.3:
        label = "POSITIVE"
    elif score < -0.3:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    magnitude = sum(abs(v) for _w, v in matched)
    confidence = round(min(1.0, abs(score) / magnitude), 2) if magnitude else 0.0

    return {
        "label": label,
        "score": round(score, 2),
        "confidence": confidence,
        "keywords": [word for word, _v in matched],
        "tokens": remove_stop_words(tokens),
    }
