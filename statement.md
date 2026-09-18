# Project Statement — Smart Campus AI Assistant

## 1. Problem Statement

Students frequently need information about campus facilities, locations and
services. Finding this information manually is slow, especially for new
students. The Smart Campus AI Assistant provides a single terminal-based AI
system where a student can ask campus questions, find a route between two
campus locations, receive a study recommendation, and analyse feedback text.

## 2. Objectives

- Develop an AI-based campus assistant for students.
- Use knowledge representation to store and retrieve campus information.
- Implement AI search algorithms (BFS and A*) for campus navigation.
- Apply a supervised machine learning technique for student recommendations.
- Use NLP-based sentiment analysis on student feedback.
- Demonstrate AI/ML techniques applied to a real-world problem.

## 3. Scope

In scope: terminal interface, campus knowledge base, campus map graph search,
student performance classification, sentiment analysis of feedback, SQLite
persistence of campus data and feedback history.

Out of scope: web/mobile interface, real-time GPS, multi-campus support,
authentication.

## 4. Target Users

- **Student (primary)** — asks questions, finds routes, gets recommendations,
  submits feedback.
- **Administrator (secondary)** — maintains campus locations, facts and map data.

## 5. Functional Requirements

| ID | Requirement |
| --- | --- |
| FR1 | The system shall accept campus-related queries from users. |
| FR2 | The system shall retrieve relevant information from its knowledge base. |
| FR3 | The system shall find a route between two campus locations using an AI search algorithm. |
| FR4 | The system shall accept student information for recommendation. |
| FR5 | The system shall classify student input and generate a recommendation. |
| FR6 | The system shall analyse text feedback and determine its sentiment. |
| FR7 | The system shall handle invalid user input without terminating unexpectedly. |

## 6. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR1 Usability | Simple, clearly labelled terminal menu. |
| NFR2 Performance | Search and prediction return results in under one second. |
| NFR3 Reliability | Invalid input never crashes the application. |
| NFR4 Maintainability | Each AI capability lives in its own module. |
| NFR5 Resource Efficiency | Runs on a standard laptop with no GPU. |
| NFR6 Error Handling | Meaningful messages instead of raw tracebacks. |

## 7. High-Level Architecture

```text
              USER
                |
        +-----------------+
        | Terminal UI     |  src/main.py
        +--------+--------+
                 |
       +---------+----------+
       |         |          |
    AI Agent   Search      ML / NLP
       |         |          |
       +---------+----------+
                 |
           Knowledge Base
                 |
              SQLite (data/campus_data.db)
```

## 8. Workflow

```text
START -> Main Menu -> Select AI Service -> Input Data
      -> Process using AI Algorithm -> Generate Result
      -> Display Result -> Continue? -> END
```
