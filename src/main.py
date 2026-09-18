"""Smart Campus AI Assistant — terminal user interface.

Run:  python src/main.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.campus_agent import CampusAgent  # noqa: E402

LINE = "=" * 44


def header(title: str) -> None:
    print("\n" + LINE)
    print(title.center(44))
    print(LINE)


def ask_float(prompt: str, low: float, high: float) -> float:
    """Read a number, re-prompting until it is valid (NFR3, NFR6)."""
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print(f"  ! Please enter a number between {low} and {high}.")
            continue
        if not low <= value <= high:
            print(f"  ! Value must be between {low} and {high}.")
            continue
        return value


def show_locations(agent: CampusAgent) -> None:
    print("\nAvailable locations:")
    for index, name in enumerate(agent.kb.locations(), start=1):
        print(f"  {index}. {name}")


def print_route(result: dict) -> None:
    if not result.get("ok"):
        print(f"\n  ! {result.get('message')}")
        return
    print(f"\nAlgorithm : {result['algorithm']}")
    print("Route     :")
    for step, node in enumerate(result["path"]):
        prefix = "            " if step else "            "
        print(f"{prefix}{node}")
        if step < len(result["path"]) - 1:
            print("              |")
    print(f"Total cost: {result['cost']}")
    print(f"Nodes expanded: {result['expanded']}")


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def module_assistant(agent: CampusAgent) -> None:
    header("INTELLIGENT CAMPUS ASSISTANT")
    print("Ask a campus question (type 'back' to return).")
    print("Examples: Where is the library? / Canteen timing?")
    while True:
        question = input("\nYou > ").strip()
        if question.lower() in {"back", "exit", "q"}:
            return
        print("AI  > " + agent.ask(question))


def module_navigation(agent: CampusAgent) -> None:
    header("AI CAMPUS NAVIGATION")
    show_locations(agent)
    source = input("\nEnter source      : ")
    destination = input("Enter destination : ")
    print("\nChoose algorithm: 1) A*  2) BFS  3) Compare both")
    choice = input("Choice [1] : ").strip() or "1"

    if choice == "2":
        print_route(agent.find_route(source, destination, "bfs"))
    elif choice == "3":
        for result in agent.compare_algorithms(source, destination):
            print_route(result)
    else:
        print_route(agent.find_route(source, destination, "astar"))


def module_recommendation(agent: CampusAgent) -> None:
    header("STUDENT RECOMMENDATION")
    study = ask_float("Enter daily study hours (0-12) : ", 0, 12)
    attendance = ask_float("Enter attendance %       (0-100): ", 0, 100)
    marks = ask_float("Enter previous marks %   (0-100): ", 0, 100)
    preference = input("Study preference (Morning/Evening) [Morning]: ").strip() or "Morning"

    result = agent.recommend(study, attendance, marks, preference)
    print(f"\nModel      : {result['model']}")
    print(f"Validation accuracy : {result['validation_accuracy']}")
    print(f"Prediction : {result['category']}")
    print("\nRecommended study plan:")
    print(f"  Suggested daily study time : {result['suggested_hours']} hours")
    print(f"  Preferred slot             : {result['preferred_slot']}")
    for tip in result["tips"]:
        print(f"  - {tip}")


def module_sentiment(agent: CampusAgent) -> None:
    header("FEEDBACK SENTIMENT ANALYSER")
    text = input("Enter your feedback:\n> ").strip()
    if not text:
        print("  ! No feedback entered.")
        return
    result = agent.analyze_feedback(text)
    print(f"\nSentiment  : {result['label']}")
    print(f"Score      : {result['score']}")
    print(f"Confidence : {result['confidence']}")
    print(f"Keywords   : {', '.join(result['keywords']) or 'none detected'}")
    print("Feedback saved to the database.")


def module_report(agent: CampusAgent) -> None:
    header("CAMPUS REPORT")
    data = agent.report()
    print(f"Locations in knowledge base : {data['locations']}")
    print(f"Questions asked             : {data['questions_asked']}")
    print(f"ML model                    : {data['model']} (accuracy {data['model_accuracy']})")
    summary = data["sentiment_summary"] or {}
    print("\nFeedback sentiment summary:")
    if not summary:
        print("  No feedback recorded yet.")
    for label, count in summary.items():
        print(f"  {label:<9}: {'#' * count} ({count})")
    recent = data["recent_feedback"]
    if recent:
        print("\nRecent feedback:")
        for text, label, score in recent:
            snippet = text if len(text) <= 50 else text[:47] + "..."
            print(f"  [{label}] {snippet} (score {score})")


MENU = """
1. Ask Campus Assistant
2. Find Campus Route
3. Student Recommendation
4. Sentiment Analysis
5. Reports
6. Exit
"""


def main() -> None:
    header("SMART CAMPUS AI ASSISTANT")
    print("CSA2001 - Fundamentals in AI and ML")
    agent = CampusAgent()

    actions = {
        "1": module_assistant,
        "2": module_navigation,
        "3": module_recommendation,
        "4": module_sentiment,
        "5": module_report,
    }

    try:
        while True:
            print(MENU)
            choice = input("Enter your choice: ").strip()
            if choice == "6":
                print("\nThank you for using Smart Campus AI Assistant.")
                return
            action = actions.get(choice)
            if action is None:
                print("  ! Invalid choice. Please enter a number from 1 to 6.")
                continue
            try:
                action(agent)
            except Exception as error:  # NFR6: never crash on a module error
                print(f"  ! Something went wrong: {error}")
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting. Goodbye!")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
