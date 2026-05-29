# Phase 3: Persistent Memory

## Goal

Add project-level persistent memory so Nexus can reuse prior research and outputs across workflow runs.

New flow:

```text
User Goal -> Planner -> Human Approval -> Memory Retrieval -> Parallel Research -> Writer -> Memory Storage -> Final Output
```

## What Changed

### Vector Memory

- Added pgvector migration.
- Enabled `vector` extension.
- Added `memory_chunks` table for chunked research content.
- Added `project_summaries` table for run-level summaries.
- Added ivfflat cosine indexes for vector search.

### Chunking

- Added `backend/memory/chunker.py`.
- Added token-aware chunking with a default 512-token chunk size and 64-token overlap.
- Uses `tiktoken` when the `cl100k_base` encoding is locally cached.
- Falls back to a local tokenizer when the encoding is unavailable, avoiding network download hangs in restricted environments.

### Embeddings

- Added `backend/memory/embeddings.py`.
- Uses OpenAI `text-embedding-3-small`.
- Batches up to 100 texts per API call.
- Validates 1536-dimensional vectors.

### Memory Store

- Added `backend/memory/store.py`.
- Stores research results by chunking content, embedding chunks, and inserting rows into `memory_chunks`.
- Retrieves memory with pgvector cosine similarity scoped to `project_id`.
- Reranks retrieved chunks with the configured OpenRouter model.

### Memory Agent

- Added `backend/agents/memory_agent.py`.
- Retrieves relevant past context before new research begins.
- Summarises final workflow output and stores the summary embedding in `project_summaries`.

### Graph

- Extended graph to:

```text
START -> planner -> human_approval -> memory_retrieval -> parallel_research -> writer -> memory_storage -> END
```

- `memory_retrieval_node` retrieves project memory for the goal.
- `parallel_research_node` passes memory context into each `ResearchAgent`.
- `memory_storage_node` persists research chunks and final-output summary.

### Projects API

- Added `backend/api/routes/projects.py`.
- Added:
  - `GET /projects`
  - `POST /projects`
  - `GET /projects/{id}/memory?query=...`
  - `GET /projects/{id}/runs`

### UI

- Added project sidebar.
- Added project selector.
- Added past run list.
- Added memory search.
- Workflow creation can now run inside an existing project.

### Tests

- Added `backend/tests/test_phase3.py`.
- Tests cover:
  - chunking and token limits
  - embedding vector shape
  - memory store/retrieve round trip with mocked DB and embeddings
  - reranking behavior
  - graph completion storing memory with mocked external boundaries

## Important Files

- `backend/db/migrations/002_vector_memory.sql`
- `backend/memory/chunker.py`
- `backend/memory/embeddings.py`
- `backend/memory/store.py`
- `backend/agents/memory_agent.py`
- `backend/api/routes/projects.py`
- `backend/graphs/research_graph.py`
- `frontend/app.py`
- `backend/tests/test_phase3.py`

## Phase 3 Status

Implementation is in place. Fast syntax/import checks pass. Full pytest was not completed in this environment because earlier runs were blocked by `tiktoken` network-download behavior; the chunker has since been patched to avoid that path unless explicitly enabled.
