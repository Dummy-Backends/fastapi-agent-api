from typing import List, Dict, Set, Any, Tuple
import db

def build_dependency_graph() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Build adjacency lists for upstream and downstream dependencies.
    Returns:
        - upstream: mapping of task_id -> list of depends_on_task_ids
        - downstream: mapping of task_id -> list of task_ids that depend on it
    """
    dependencies = db.fetch_dependencies()
    upstream = {}
    downstream = {}
    
    # Initialize dictionary for all tasks
    tasks = db.fetch_all_tasks()
    for task in tasks:
        tid = task["id"]
        upstream[tid] = []
        downstream[tid] = []
        
    for dep in dependencies:
        tid = dep["task_id"]
        dep_id = dep["depends_on_task_id"]
        
        # Ensure we don't crash if seed/db contains tasks not in task list
        if tid not in upstream:
            upstream[tid] = []
        if dep_id not in downstream:
            downstream[dep_id] = []
            
        upstream[tid].append(dep_id)
        downstream[dep_id].append(tid)
        
    return upstream, downstream

def get_upstream_blockers(task_id: str) -> List[Dict[str, Any]]:
    """
    Find all active (non-merged) tasks that the given task_id depends on, transitively.
    """
    upstream, _ = build_dependency_graph()
    if task_id not in upstream:
        return []
        
    visited: Set[str] = set()
    blockers: List[Dict[str, Any]] = []
    
    def dfs(curr_id: str):
        for parent_id in upstream.get(curr_id, []):
            if parent_id not in visited:
                visited.add(parent_id)
                task_data = db.fetch_task(parent_id)
                if task_data and task_data["status"] != "merged":
                    blockers.append(task_data)
                dfs(parent_id)
                
    dfs(task_id)
    return blockers

def get_downstream_impact(task_id: str) -> List[Dict[str, Any]]:
    """
    Find all tasks (and their owners) that transitively depend on the completion of task_id.
    """
    _, downstream = build_dependency_graph()
    if task_id not in downstream:
        return []
        
    visited: Set[str] = set()
    impacted: List[Dict[str, Any]] = []
    
    def dfs(curr_id: str):
        for child_id in downstream.get(curr_id, []):
            if child_id not in visited:
                visited.add(child_id)
                task_data = db.fetch_task(child_id)
                if task_data:
                    impacted.append(task_data)
                dfs(child_id)
                
    dfs(task_id)
    return impacted

def get_connected_nodes(task_id: str, hops: int = 1) -> List[Dict[str, Any]]:
    """
    Find tasks within N hops on the dependency DAG, treating the graph as undirected.
    """
    upstream, downstream = build_dependency_graph()
    
    # Build undirected adjacency list
    undirected: Dict[str, Set[str]] = {}
    all_nodes = set(list(upstream.keys()) + list(downstream.keys()))
    for node in all_nodes:
        undirected[node] = set()
        
    for child, parents in upstream.items():
        for parent in parents:
            undirected[child].add(parent)
            if parent not in undirected:
                undirected[parent] = set()
            undirected[parent].add(child)
            
    if task_id not in undirected:
        return []
        
    # BFS up to N hops
    visited = {task_id}
    queue = [(task_id, 0)]
    results = []
    
    while queue:
        curr, dist = queue.pop(0)
        
        # Don't include the starting node in the results
        if dist > 0:
            task_data = db.fetch_task(curr)
            if task_data:
                # Add distance info to the returned metadata
                task_copy = dict(task_data)
                task_copy["graph_distance"] = dist
                results.append(task_copy)
                
        if dist < hops:
            for neighbor in undirected.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
    return results
