import os
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import db
import graph
import vector_store
from agent import BottleneckInvestigatorAgent

app = FastAPI(
    title="TeamBoostAI PR Bottleneck Investigator API",
    description="A prototype that detects task bottlenecks using dependency graphs, embeddings, and LLM agents.",
    version="1.0.0"
)

# Global in-memory store for investigations status/results
INVESTIGATIONS: Dict[str, Dict[str, Any]] = {}

class InvestigateRequest(BaseModel):
    pr_id: Optional[str] = None
    task_id: Optional[str] = None

# Startup hook to initialize database and build embeddings index
@app.on_event("startup")
def startup_event():
    print("Starting up TeamBoostAI Bottleneck Investigator...")
    # Initialize SQLite database and seed data
    db.init_db("seed.json")
    # Build vector embeddings in ChromaDB
    vector_store.index_all_tasks()
    print("Startup steps completed successfully.")

@app.get("/api/health")
def health_check():
    """Simple API health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "vector_store": "ready"
    }

def run_background_investigation(investigation_id: str, task_id: str):
    """Background worker task to run the agent pipeline."""
    try:
        agent = BottleneckInvestigatorAgent()
        result = agent.investigate(task_id)
        
        if result.get("status") == "failed":
            INVESTIGATIONS[investigation_id].update({
                "status": "failed",
                "error": result.get("error", "Unknown error")
            })
        else:
            INVESTIGATIONS[investigation_id].update({
                "status": "completed",
                "classification": result["classification"],
                "trace": result["trace"],
                "notification": result["notification"]
            })
    except Exception as e:
        INVESTIGATIONS[investigation_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.post("/api/investigate", status_code=202)
def start_investigation(request: InvestigateRequest, background_tasks: BackgroundTasks):
    """
    Trigger a bottleneck investigation for a given pr_id or task_id.
    Runs asynchronously and returns an investigation_id immediately.
    """
    task_id = request.task_id
    
    if not task_id and not request.pr_id:
        raise HTTPException(status_code=400, detail="Must provide either pr_id or task_id.")
        
    # Resolve pr_id to task_id if needed
    if request.pr_id:
        task = db.fetch_task_by_pr(request.pr_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"No task found associated with PR {request.pr_id}.")
        task_id = task["id"]
    else:
        # Check if task exists
        task = db.fetch_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    # Create investigation ticket
    investigation_id = str(uuid.uuid4())
    INVESTIGATIONS[investigation_id] = {
        "id": investigation_id,
        "task_id": task_id,
        "status": "pending",
        "classification": None,
        "trace": [],
        "notification": None
    }
    
    # Enqueue job asynchronously (Stretch Goal: Async job queue)
    background_tasks.add_task(run_background_investigation, investigation_id, task_id)
    
    return {
        "investigation_id": investigation_id,
        "task_id": task_id,
        "status": "pending",
        "message": "Investigation started in background."
    }

@app.get("/api/investigations/{id}")
def get_investigation_result(id: str):
    """Retrieve the status and results of a scheduled investigation."""
    if id not in INVESTIGATIONS:
        raise HTTPException(status_code=404, detail=f"Investigation {id} not found.")
    return INVESTIGATIONS[id]

@app.get("/api/tasks/{id}/graph")
def get_task_graph(id: str):
    """Get the dependency graph node connections for a specific task (upstream and downstream)."""
    task = db.fetch_task(id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {id} not found.")
        
    upstream = graph.get_upstream_blockers(id)
    downstream = graph.get_downstream_impact(id)
    
    return {
        "task_id": id,
        "task_title": task["title"],
        "status": task["status"],
        "owner": task["owner_name"],
        "upstream_blockers": [
            {"id": t["id"], "title": t["title"], "status": t["status"], "owner": t["owner_name"]}
            for t in upstream
        ],
        "downstream_impacted": [
            {"id": t["id"], "title": t["title"], "status": t["status"], "owner": t["owner_name"]}
            for t in downstream
        ]
    }

@app.get("/api/tasks/{id}/related")
def get_related_tasks(
    id: str,
    query: Optional[str] = Query(None, description="Optional search text. Defaults to task title + description."),
    k: int = Query(5, description="Number of results to retrieve."),
    min_score: float = Query(0.30, description="Minimum similarity threshold score.")
):
    """Get semantically related tasks for a specific task using embeddings."""
    task = db.fetch_task(id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {id} not found.")
        
    # If query is not provided, use the task's title + description
    search_query = query if query else f"Title: {task['title']}\nDescription: {task['description']}"
    
    results = vector_store.search_related_tasks(
        query_text=search_query,
        top_k=k,
        min_score=min_score,
        exclude_task_id=id
    )
    
    return {
        "query_task_id": id,
        "query_used": search_query,
        "min_score": min_score,
        "results": results
    }
