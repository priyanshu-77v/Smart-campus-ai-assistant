# Smart Campus AI Assistant

A terminal-based AI application for CSA2001 (Fundamentals in AI and ML).

## Modules

| Module | AI concept |
| --- | --- |
| Intelligent Campus Assistant | Intelligent agent + knowledge representation |
| AI Campus Navigation | BFS (uninformed) and A* (informed) search |
| Student Recommendation | Supervised learning / classification (scikit-learn) |
| Sentiment Analyzer | NLP / text classification |

## Project structure

```
smart-campus-ai/
├── README.md
├── statement.md
├── requirements.txt
├── .gitignore
├── data/                 # campus_data.db is created on first run
├── src/
│   ├── main.py
│   ├── agents/campus_agent.py
│   ├── search/bfs.py
│   ├── search/astar.py
│   ├── knowledge/knowledge_base.py
│   ├── ml/student_model.py
│   ├── nlp/sentiment.py
│   └── database/database.py
└── tests/test_project.py
```

## Setup

```bash
cd smart-campus-ai
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Tests

```bash
python -m pytest tests -v
```

## Notes

- scikit-learn is optional. If it is not installed, the recommendation module
  falls back to an equivalent rule-based classifier so the app still runs.
- The SQLite database in `data/` is created and seeded automatically.
