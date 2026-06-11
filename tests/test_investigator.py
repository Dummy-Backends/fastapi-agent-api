import os
import pytest
import sqlite3
import db
import graph
import vector_store
from agent import BottleneckInvestigatorAgent

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Sets up the test environment, initializing database and vector store."""
    # Ensure we use seed.json for tests
    db.init_db("seed.json")
    vector_store.index_all_tasks()
    yield

def test_db_fetching():
    # Test fetch single task
    task = db.fetch_task("task-001")
    assert task is not None
    assert task["id"] == "task-001"
    assert task["owner_name"] == "Alex Martin"
    
    # Test fetch non-existent task
    assert db.fetch_task("invalid-task") is None

def test_graph_upstream_blockers():
    # task-002 depends on task-001
    blockers = graph.get_upstream_blockers("task-002")
    assert len(blockers) >= 1
    blocker_ids = [b["id"] for b in blockers]
    assert "task-001" in blocker_ids

def test_graph_downstream_impact():
    # task-001 has two downstream dependencies: task-002 and task-006
    impacted = graph.get_downstream_impact("task-001")
    assert len(impacted) >= 2
    impacted_ids = [t["id"] for t in impacted]
    assert "task-002" in impacted_ids
    assert "task-006" in impacted_ids

def test_graph_connected_hops():
    # Check 1 hop from task-001
    connected_1 = graph.get_connected_nodes("task-001", hops=1)
    connected_1_ids = [c["id"] for c in connected_1]
    assert "task-002" in connected_1_ids
    assert "task-006" in connected_1_ids
    
    # Check that distance is included
    for node in connected_1:
        assert node["graph_distance"] == 1

def test_vector_search():
    # Search tasks related to auth middleware
    results = vector_store.search_related_tasks(
        query_text="auth middleware validation",
        top_k=2,
        min_score=0.50
    )
    assert len(results) > 0
    # The top result should be task-001 or task-005 (auth related)
    top_ids = [r["id"] for r in results]
    assert "task-001" in top_ids or "task-005" in top_ids

def test_agent_offline_investigation():
    agent = BottleneckInvestigatorAgent()
    result = agent.investigate("task-001")
    
    assert result["status"] == "completed"
    assert result["task_id"] == "task-001"
    assert result["classification"] == "DEPENDENCY_BLOCK"
    assert "Samira" in result["notification"]
    assert "Alex" in result["notification"]
    assert len(result["trace"]) > 0
