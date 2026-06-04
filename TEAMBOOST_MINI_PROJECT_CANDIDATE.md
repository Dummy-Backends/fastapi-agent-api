# TeamBoostAI — Technical Assessment (Take-Home)

**Role:** Junior Software / AI Engineer  
**Submission deadline:** 7 days

---

## Project Title

**Mini System-Aware PR Bottleneck Investigator**

A lightweight prototype that simulates how TeamBoostAI detects when work is stalled, maps task dependencies, identifies who is blocking whom, and produces actionable notifications for engineers.

### In one sentence

> **You'll use a dependency graph to find who's blocking whom, embeddings to find semantically related tasks, and an agent that calls both as tools before the LLM writes the engineer notification.**

This is **not** a single chatbot prompt. The graph gives **structural truth** (dependencies and blockers). Embeddings give **semantic context** (similar work). The agent **orchestrates** tools; the LLM **classifies** (router) and **explains** (notification).

---

## What You Are Building

When a **Pull Request fails** (or a task is marked blocked), an **agentic workflow** should:

1. **Ingest** the event (webhook simulation or CLI is fine).
2. **Investigate** task dependencies and history using a **mock database** (not a real GitHub/Jira integration required).
3. **Determine** the bottleneck type — e.g. dependency on another engineer, self-stall, review wait, external blocker.
4. **Route** through a structured decision step (router) with a **strict machine-readable output**.
5. **Search** for semantically related tasks using **embeddings** and a **vector store** (with top-k, similarity threshold, and metadata filters).
6. **Generate** a human-readable notification for the affected engineer(s) via an **LLM** (local model or API — your choice).
7. **Expose** a simple API so a UI or script can trigger investigation and read results.

You may use AI coding assistants, but you must **understand and explain** your design in the write-up. We will ask about your architecture in a follow-up interview.

---

## Core Scenario (Must Work End-to-End)

```
Engineer A owns Task T1 (in progress, overdue).
Task T2 (owned by Engineer B) depends on T1.
PR #42 for T1 fails CI.

→ System investigates → concludes B is blocked by A's delay on T1
→ Notifies A with a concrete message
→ Optionally notifies B that they are waiting on T1
```

Your prototype must handle **at least**:

| Scenario | Expected outcome |
|----------|------------------|
| **Dependency stall** | Engineer B blocked because Engineer A has not finished prerequisite task |
| **Self-stall** | Engineer has no open blockers but task is overdue / PR failing with no external dependency |
| **Review / external wait** | Task blocked on review or non-engineer dependency (mock data) |

---

## Functional Requirements

### 1. Mock data layer

Provide a **seeded database** (PostgreSQL, SQLite,mongodb or JSON loaded at startup) with:

- **Engineers** — `id`, `name`, `email`, `team`
- **Tasks** — `id`, `title`, `description`, `owner_id`, `status`, `due_date`, `related_pr_id` (optional), `sprint_id` / `team` (recommended for embedding filters)
- **Dependencies** — directed edges: `task_id` → `depends_on_task_id` (form a DAG; no cycles)
- **PR events** — `pr_id`, `task_id`, `status` (`open` / `failed` / `merged`), `timestamp`
- **Activity log** (recommended) — comments, status changes for historical context

**Starter seed (optional):** We provide `teamboost-assessment-starter/db/seed.json` with engineers, tasks, dependencies, and PR events for all three grading scenarios. **You may use it as-is** (load into your DB on startup) **or create your own synthetic seed from scratch** — as long as your README documents the data and the **dependency-stall scenario** (PR fails on a task → another engineer blocked on a dependent task) still works end-to-end. If you use our file, **do not change** IDs `task-001`, `task-002`, `engineer-a`, `engineer-b`, or `pr-042` so your demo matches our review checklist.

Include a **README section** describing the seed data and how it maps to the core scenario above.

### 2. Task dependency graph

- Build an in-memory or queryable **directed graph** from dependencies.
- Implement:
  - **Upstream blockers** — what must finish before task X?
  - **Downstream impact** — who is waiting on task X?
  - **Graph hop search** — tasks connected within N hops on the dependency DAG (structural, not semantic).

### 3. Semantic search with embeddings (required)

You must implement **vector-based** related-task search. Keyword-only search is **not sufficient** for a pass.

#### What to embed

Embed at minimum **task title + description** for every task in seed data. Document what text you concatenate and why.

#### Vector store

Use one of: **pgvector**, **Chroma**, **FAISS**, **Qdrant**, **Pinecone**, or similar. Store vectors in Docker if possible.

#### `search_related_tasks` tool (required — used by the agent)

The agent must call this tool during investigation. It must:

| Requirement | Detail |
|-------------|--------|
| **top-k** | Retrieve top-k neighbors (you choose k; document default, e.g. k=5) |
| **Similarity threshold** | Drop results below a cutoff (e.g. cosine similarity < 0.75); document how you picked it |
| **Metadata filters** | Apply **at least 2** filters before or after vector search — e.g. `team`, `status`, `sprint_id`, `owner_id`, `exclude_task_id` |
| **Hybrid behavior** | Combine **graph/blocker facts** with **semantic neighbors** in the investigation (e.g. mention related tasks in notification or tool trace) |

#### Documentation (required in README)

Explain in a dedicated subsection **"Embeddings & retrieval"**:

1. Embedding model used (name + dimension if known).
2. Chunking strategy (one task = one vector, or chunks — justify).
3. Your **top-k** value and **threshold** and what happens when nothing passes the threshold.
4. Which **metadata filters** you support and an example query.
5. One **concrete example**: given task T1, list the top 3 related tasks returned and why they make sense.

#### API endpoint (required)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/tasks/{id}/related` | Query params: `query` (optional text), `k`, `min_score`. Returns ranked tasks with **similarity scores** and applied filters. |

### 4. Agentic workflow (required)

Implement a real **plan → tool → observe** loop, not a single prompt.

**Minimum tools** (each must be a real function the agent invokes):

| Tool | Purpose |
|------|---------|
| `get_task` | Fetch task + owner metadata |
| `get_dependency_graph` | Upstream/downstream for a task |
| `get_blocked_engineers` | Who is waiting on this task |
| `get_engineer_workload` | Open/overdue tasks for an engineer |
| `get_pr_history` | Recent PR events for a task |
| `search_related_tasks` | **Embedding search** — top-k + threshold + metadata filters (see §3) |

**Router node** (required):

- Input: investigation context (failed PR, task id, graph summary).
- Output: **exactly one** classification string from this set (no extra prose):

  ```
  DEPENDENCY_BLOCK | SELF_STALL | REVIEW_WAIT | EXTERNAL_BLOCK | UNKNOWN
  ```

**Notification node** (required):

- Input: router classification + tool results.
- Output: **actionable message** for the engineer (2–4 sentences, specific names and task ids).

You may use **LangChain, LangGraph, custom Python,** — justify your choice in the write-up.

### 5. HTTP API (required)

Expose at minimum:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/investigate` | Body: `{ "pr_id": "..." }` or `{ "task_id": "..." }` — starts investigation, returns `investigation_id` |
| `GET` | `/api/investigations/{id}` | Status + router classification + tools trace + final notification |
| `GET` | `/api/tasks/{id}/graph` | Dependency graph for debugging (no LLM required) |
| `GET` | `/api/tasks/{id}/related` | Semantic related tasks with scores (see §3) |
| `GET` | `/api/health` | Health check |

Responses must be **JSON**. Include example `curl` commands in your README.

### 6. LLM configuration (required — document your choices)

You must configure **different inference settings** for:

- **Router node** — strict, deterministic classification.
- **Notification node** — natural language suitable for a developer.

In your README, explain **why** you chose `temperature`, `do_sample`, and any other parameters for each node. If you use an API (OpenAI, etc.), document the exact settings.

### 7. Deliverables

Submit a **Git repository** (GitHub/GitLab link or zip) containing:

| Item | Required |
|------|----------|
| Source code | Yes |
| `README.md` — setup, architecture, design decisions | Yes |
| `ARCHITECTURE.md` — diagrams showing **graph + embeddings + agent** flow | Yes |
| Seed data / migrations | Yes |
| Example run (screenshot or terminal log) | Yes |
| `docker-compose.yml` (app + DB) | Recommended |
| Tests (unit or integration) | At least 5 meaningful tests (include **≥2** for graph and **≥1** for embedding retrieval) |
| Embedding index build script or startup hook | Yes — document how vectors are created/refreshed |

**Do not** submit secrets. Use `.env.example`.

---

## Stretch Goals (Optional — Strong Differentiators)

Not required for a pass. Implement any subset if you want to stand out.

| Stretch | What it shows |
|---------|----------------|
| **Async job queue** | `POST /investigate` returns immediately; worker processes via Redis/RabbitMQ; document why |
| **Engineer “ping” simulation** | Tool `draft_engineer_message(recipient, context)` + mock send log |
| **PDF / Markdown report** | Manager summary aggregating multiple stalled tasks |
| **Spring Boot** | Java orchestration layer + Python AI worker (two services) |
| **Observability** | Structured logs per agent step; tool latency |

---

## Constraints

- **Scope:** This is a **mini** prototype. Prefer clarity over feature count.
- **No production GitHub/Jira integration** required — mock data only.
- **LLM:** API keys or local model (Ollama, etc.) — document setup.
- **Timebox:** If you run out of time, document what you would add next in `README.md` under **Future work**.
- **Not required:** Full Graph RAG, Neo4j, or real GitHub/Jira integration — keep scope on the hybrid design above.

---

## Evaluation Criteria (What We Look For)

We will score your submission on:

1. **Correct end-to-end flow** for the dependency stall scenario.
2. **Real agentic design** — tools, loop, termination, not one-shot chat.
3. **Router vs notification separation** — appropriate LLM settings.
4. **Graph logic correctness** — upstream/downstream/blockers.
5. **Embeddings & retrieval** — top-k, threshold, metadata filters, agent uses `search_related_tasks`, documented example.
6. **API clarity** — usable, documented endpoints (including `/related`).
7. **Architecture write-up** — honest tradeoffs, scalability notes.
8. **Code quality** — readable, runnable, not over-engineered.

---

## Submission

Send to: _adam ngazzou_

Include:

- Repository URL
- Candidate name
- Hours spent (honest estimate)
- LLM provider used (if any)

---

## Follow-Up Interview

Be prepared to:

- Walk through your agent loop on a whiteboard.
- Explain router vs notification temperature settings.
- Defend one scalability decision (queue, DB choice, vector index rebuild).
- Live-debug a seeded scenario we pick from your mock data.

---

*TeamBoostAI — Internal hiring assessment. Do not publish this brief publicly without permission.*
