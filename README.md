# TeamBoostAI — Mini System-Aware PR Bottleneck Investigator

This is a lightweight system prototype that simulates how TeamBoostAI detects when work is stalled, maps task dependencies, identifies who is blocking whom, and produces actionable notifications for engineers.

---

## 🚀 Quick Start (Local Run)

Follow these steps to run the application locally outside of Docker.

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`. By default, it is configured to use the `mock` LLM provider:
```bash
cp .env.example .env
```
*Note: If you want to run with live LLM calls, change `LLM_PROVIDER` to `gemini` or `openai` and fill in `GEMINI_API_KEY` or `OPENAI_API_KEY` respectively.*

### 3. Run the Application
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
On startup, the system automatically:
1. Initializes the SQLite database (`recruitment.db`) and seeds it from `seed.json`.
2. Automatically generates task embeddings and builds the **ChromaDB** index in `./chroma_db`.

---

## 🐳 Docker Deployment

To spin up the service in a containerized environment, simply run:
```bash
docker-compose up --build
```
This builds the python application image and runs it on `http://localhost:8000`.

---

## 🛠️ API Documentation & Example Curl Commands

Once the server is running, the interactive Swagger docs are accessible at `http://localhost:8000/docs`.

### 1. Health Check
Checks service readiness.
```bash
curl -X GET http://localhost:8000/api/health
```

### 2. Trigger Bottleneck Investigation
Starts an asynchronous background agent loop to investigate a task or failing PR. Returns an `investigation_id` immediately.
```bash
curl -X POST http://localhost:8000/api/investigate \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task-001"}'
```
Or trigger by PR:
```bash
curl -X POST http://localhost:8000/api/investigate \
     -H "Content-Type: application/json" \
     -d '{"pr_id": "pr-042"}'
```

### 3. Check Investigation Status & Result
Query the background task status, agent step trace, and final notification text.
```bash
# Replace <id> with the investigation_id returned in the previous step
curl -X GET http://localhost:8000/api/investigations/<id>
```

### 4. Fetch Dependency Graph (Debugging)
View structural upstream blockers and downstream impacts.
```bash
curl -X GET http://localhost:8000/api/tasks/task-001/graph
```

### 5. Fetch Semantically Related Tasks
Retrieve related tasks using embeddings.
```bash
curl -X GET "http://localhost:8000/api/tasks/task-001/related?k=3&min_score=0.70"
```

---

## 📊 Mock Seed Data & Core Scenarios

The system is seeded with mock data containing engineers, tasks, dependencies, failing PRs, and logs representing these core scenarios:

### 1. Dependency Stall (`task-001` / `pr-042`)
*   **Context:** Engineer A (`engineer-a` - Alex Martin) owns `task-001` ("Implement auth middleware for public API"), which has a failing PR (`pr-042`).
*   **Impact:** `task-002` ("Expose user profile endpoints"), owned by Engineer B (`engineer-b` - Samira Chen), depends on `task-001`. Because `task-001` is red, Samira is blocked.
*   **Result:** The agent identifies `DEPENDENCY_BLOCK`, traces that Alex's failed PR is blocking Samira, and generates a notification prompting Alex to fix the CI failure.

### 2. Self-Stall (`task-003` / `pr-017`)
*   **Context:** Engineer C (`engineer-c` - Jordan Lee) owns `task-003` ("Optimize checkout query performance"). The task has no upstream task dependencies but its PR is failing, and the task is overdue.
*   **Result:** Classified as `SELF_STALL`. The agent notifies Jordan to resolve the performance bottlenecks and complete the index migration.

### 3. Review Wait (`task-004` / `pr-031`)
*   **Context:** Engineer D (`engineer-d` - Priya Nair) owns `task-004` ("Payment webhook handler refactor"), status `in_review`. The comments show she is waiting for staff review.
*   **Result:** Classified as `REVIEW_WAIT`. The agent prompts Priya to ping the staff reviewer.

---

## 🧠 Embeddings & Retrieval

### 1. Model & Vector Store
*   **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions) via ChromaDB's default local ONNX integration.
*   **Vector Store:** ChromaDB running in persistent mode (`PersistentClient`).
*   **Chunking Strategy:** Single task = one document. We concatenate task title and description into the format:
    ```
    Title: {title}
    Description: {description}
    ```
    Since tasks are relatively short (under 150 words), chunking is unnecessary. Concatenating title and description ensures the index retains the overall goal and technical specifics.

### 2. Retrieval Parameters
*   **Top-K:** Defaults to `5`.
*   **Similarity Threshold:** Cosine Similarity cutoff `0.70` (calculated as `1.0 - cosine_distance`).
*   **Metadata Filters:** We support filtering results by: `team`, `status`, `sprint_id`, and `owner_id`. We also support excluding the query task itself (`exclude_task_id`) to avoid matching a task with itself.

### 3. Retrieval Example (`task-001`)
*   **Input Task:** `task-001` ("Implement auth middleware for public API")
*   **Top 3 Matches Returned:**
    1.  `task-005` ("Fix OAuth token refresh race condition") - *Score: 0.771*
        *   **Why it makes sense:** Both tasks center around authentication, tokens, API gateway security, and OAuth.
    2.  `task-006` ("Add rate limiting to edge gateway") - *Score: 0.732*
        *   **Why it makes sense:** Connects semantically to API gateway, security routing, and middleware logic.
    3.  `task-002` ("Expose user profile endpoints") - *Score: 0.712*
        *   **Why it makes sense:** Specifically mentions "auth middleware" in its description.

---

## 🤖 LLM Configuration

The system defines separate agent node parameters to handle different generation requirements:

### Router Node
*   **Temperature:** `0.0`
*   **Purpose:** Deterministic classification. It is given strict instructions to output *only* one of the labels (`DEPENDENCY_BLOCK`, `SELF_STALL`, `REVIEW_WAIT`, `EXTERNAL_BLOCK`, `UNKNOWN`).

### Notification Node
*   **Temperature:** `0.7`
*   **Purpose:** Creative and natural language. Generates a supportive, professional, and clear 2-4 sentence developer-friendly notification containing details like names and IDs.
