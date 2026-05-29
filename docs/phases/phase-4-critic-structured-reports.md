# Phase 4: Critic Loop and Structured Reports

## Goal

Add a research-quality reflection loop and upgrade writing from raw markdown to structured reports with inline citations.

New flow:

```text
User Goal -> Planner -> Human Approval -> Memory Retrieval -> Parallel Research -> Critic -> Targeted Research Loop -> Writer -> Memory Storage -> Final Report
```

## What Changed

- Added `CriticFinding` and `CriticReport` schemas.
- Added `Citation`, `ReportSection`, and `FinalReport` schemas.
- Added `CriticAgent`.
- Added `critic_node` and `targeted_research_node`.
- Added a max 3-iteration reflection loop.
- Extended `ResearchAgent` with `targeted_research()`.
- Rewrote `WriterAgent` to produce `FinalReport` JSON.
- Added confidence score calculation.
- Added final report markdown rendering.
- Added report API endpoints:
  - `GET /workflows/{run_id}/report`
  - `GET /workflows/{run_id}/report.md`
- Extended Streamlit with critic progress, findings, structured report rendering, references, and confidence badge.
- Added `backend/tests/test_phase4.py`.

## Graph

```text
START -> planner -> human_approval -> memory_retrieval -> parallel_research -> critic -> [targeted_research -> critic]* -> writer -> memory_storage -> END
```

The critic loop exits when the report passes or when `critic_iteration` reaches 3.

## Phase 4 Status

Implementation is in place. Python syntax and Phase 4 import checks pass.
