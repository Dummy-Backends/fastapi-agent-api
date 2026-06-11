import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import db
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
COLLECTION_NAME = "tasks_collection"

# Use Chroma's default local ONNX embedding function (all-MiniLM-L6-v2)
# This model outputs 384-dimensional dense vectors and runs 100% locally.
local_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def get_chroma_client() -> chromadb.PersistentClient:
    """Initialize and return the persistent ChromaDB client."""
    os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

def get_tasks_collection(client: Optional[chromadb.PersistentClient] = None):
    """Retrieve or create the tasks collection configured with Cosine distance."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=local_embedding_function,
        metadata={"hnsw:space": "cosine"} # Direct cosine similarity search support
    )

def index_all_tasks():
    """
    Fetch all tasks from the database, compute 384-dimensional local embeddings
    via ChromaDB, and persist them.
    """
    client = get_chroma_client()
    
    # Delete old collection to ensure a clean index on startup/rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    collection = get_tasks_collection(client)
    tasks = db.fetch_all_tasks()
    
    if not tasks:
        print("No tasks found in database to index.")
        return
        
    ids = []
    documents = []
    metadatas = []
    
    for task in tasks:
        # Concatenate title + description to construct the semantic representation
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
    print(f"Indexed {len(tasks)} tasks into local ChromaDB collection (384 dimensions).")

def search_related_tasks(
    query_text: str,
    top_k: int = 5,
    min_score: float = 0.50, # Standard lower threshold to accommodate real high-dim cosine similarities
    filters: Optional[Dict[str, Any]] = None,
    exclude_task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query the ChromaDB collection using semantic embedding vector search.
    - query_text: Text query or task content.
    - top_k: Maximum number of matches to return.
    - min_score: Minimum similarity threshold (Cosine Similarity = 1.0 - Cosine Distance).
    - filters: Metadata pre-filtering dictionary (e.g. {'team': 'platform'}).
    - exclude_task_id: Option to filter out the query task itself.
    """
    collection = get_tasks_collection()
    
    # Construct ChromaDB where metadata clauses
    where_clauses = []
    if filters:
        for k, v in filters.items():
            where_clauses.append({k: {"$eq": v}})
            
    if exclude_task_id:
        where_clauses.append({"id": {"$ne": exclude_task_id}})
        
    where_query = None
    if len(where_clauses) > 1:
        where_query = {"$and": where_clauses}
    elif len(where_clauses) == 1:
        where_query = where_clauses[0]
        
    # Query Chroma (n_results needs to pull slightly extra to compensate for manual exclusions)
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k + (1 if exclude_task_id else 0),
        where=where_query if where_query else None
    )
    
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
        
        # Cosine Similarity is 1.0 - Cosine Distance
        similarity = 1.0 - dist
        
        if similarity >= min_score:
            matched_tasks.append({
                "id": tid,
                "title": meta.get("title", ""),
                # Extract original description from stored doc representation
                "description": doc.replace(f"Title: {meta.get('title', '')}\nDescription: ", ""),
                "owner_id": meta.get("owner_id", ""),
                "status": meta.get("status", ""),
                "sprint_id": meta.get("sprint_id", ""),
                "team": meta.get("team", ""),
                "similarity_score": round(similarity, 4)
            })
            
    # Re-fetch full task metadata from SQLite DB to ensure consistency (joins owner details, etc.)
    final_results = []
    for matched in matched_tasks[:top_k]:
        task_info = db.fetch_task(matched["id"])
        if task_info:
            task_info["similarity_score"] = matched["similarity_score"]
            final_results.append(task_info)
            
    return final_results
