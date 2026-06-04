import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import db
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
COLLECTION_NAME = "tasks_collection"

# Setup the embedding function using Chroma's default local ONNX MiniLM-L6-v2
# This model has 384 dimensions and runs 100% locally.
embedding_function = embedding_functions.DefaultEmbeddingFunction()

def get_chroma_client() -> chromadb.PersistentClient:
    """Initialize and return the persistent ChromaDB client."""
    os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

def get_tasks_collection(client: Optional[chromadb.PersistentClient] = None):
    """Retrieve or create the tasks collection with cosine space configuration."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"} # Use cosine distance to easily compute similarity
    )

def index_all_tasks():
    """
    Fetch all tasks from the SQLite database, construct embedding documents,
    and index them in ChromaDB.
    """
    client = get_chroma_client()
    # Delete collection if it exists to ensure clean index on rebuild/startup
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    collection = get_tasks_collection(client)
    tasks = db.fetch_all_tasks()
    
    if not tasks:
        print("No tasks found in SQLite database to index.")
        return
        
    ids = []
    documents = []
    metadatas = []
    
    for task in tasks:
        # Concatenate title and description
        doc_text = f"Title: {task['title']}\nDescription: {task['description']}"
        
        ids.append(task["id"])
        documents.append(doc_text)
        metadatas.append({
            "id": task["id"],
            "title": task["title"],
            "owner_id": task["owner_id"],
            "status": task["status"],
            "sprint_id": task["sprint_id"],
            "team": task["team"]
        })
        
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Indexed {len(tasks)} tasks into ChromaDB.")

def search_related_tasks(
    query_text: str,
    top_k: int = 5,
    min_score: float = 0.70,
    filters: Optional[Dict[str, Any]] = None,
    exclude_task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search in ChromaDB.
    - query_text: Text to search for (or the title+description of a task).
    - top_k: Max results to return.
    - min_score: Minimum similarity score (cosine similarity = 1.0 - cosine_distance).
    - filters: Dictionary of metadata filters (e.g. {'team': 'platform', 'status': 'todo'}).
    - exclude_task_id: Task ID to exclude from search results.
    """
    collection = get_tasks_collection()
    
    # Construct Chroma where query
    where_clause = {}
    
    # Process metadata filters
    filter_list = []
    if filters:
        for k, v in filters.items():
            filter_list.append({k: {"$eq": v}})
            
    if exclude_task_id:
        filter_list.append({"id": {"$ne": exclude_task_id}})
        
    if len(filter_list) > 1:
        where_clause = {"$and": filter_list}
    elif len(filter_list) == 1:
        where_clause = filter_list[0]
        
    # Query Chroma
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k + (1 if exclude_task_id else 0), # Fetch extra if we exclude
        where=where_clause if where_clause else None
    )
    
    # Parse results
    if not results or not results["ids"] or len(results["ids"][0]) == 0:
        return []
        
    matched_tasks = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    
    for i in range(len(ids)):
        tid = ids[i]
        dist = distances[i]
        meta = metadatas[i]
        doc = documents[i]
        
        # Calculate similarity score: cosine similarity is 1.0 - cosine_distance
        similarity = 1.0 - dist
        
        if similarity >= min_score:
            matched_tasks.append({
                "id": tid,
                "title": meta.get("title", ""),
                "description": doc.replace(f"Title: {meta.get('title', '')}\nDescription: ", ""),
                "owner_id": meta.get("owner_id", ""),
                "status": meta.get("status", ""),
                "sprint_id": meta.get("sprint_id", ""),
                "team": meta.get("team", ""),
                "similarity_score": round(similarity, 4)
            })
            
    # Return top_k
    return matched_tasks[:top_k]
