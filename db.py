import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

DATABASE_FILE = "recruitment.db"

def get_db_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_tables(conn: sqlite3.Connection):
    """Create database tables if they do not exist."""
    cursor = conn.cursor()
    
    # Engineers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS engineers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        team TEXT NOT NULL
    );
    """)
    
    # Tasks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        status TEXT NOT NULL,
        due_date TEXT NOT NULL,
        related_pr_id TEXT,
        sprint_id TEXT NOT NULL,
        team TEXT NOT NULL,
        FOREIGN KEY (owner_id) REFERENCES engineers (id)
    );
    """)
    
    # Dependencies table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dependencies (
        task_id TEXT NOT NULL,
        depends_on_task_id TEXT NOT NULL,
        type TEXT NOT NULL,
        PRIMARY KEY (task_id, depends_on_task_id),
        FOREIGN KEY (task_id) REFERENCES tasks (id),
        FOREIGN KEY (depends_on_task_id) REFERENCES tasks (id)
    );
    """)
    
    # PR Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pr_events (
        pr_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        ci_message TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks (id)
    );
    """)
    
    # Activity Log table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity_log (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        engineer_id TEXT NOT NULL,
        type TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks (id),
        FOREIGN KEY (engineer_id) REFERENCES engineers (id)
    );
    """)
    
    conn.commit()

def load_seed_data(conn: sqlite3.Connection, seed_file_path: str):
    """Load data from seed.json if tables are empty."""
    if not os.path.exists(seed_file_path):
        print(f"Seed file not found at {seed_file_path}")
        return

    with open(seed_file_path, 'r') as f:
        data = json.load(f)

    cursor = conn.cursor()

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM engineers")
    if cursor.fetchone()[0] > 0:
        print("Database already contains data, skipping seed.")
        return

    print(f"Seeding database from {seed_file_path}...")
    
    # Insert Engineers
    for eng in data.get("engineers", []):
        cursor.execute(
            "INSERT INTO engineers (id, name, email, team) VALUES (?, ?, ?, ?)",
            (eng["id"], eng["name"], eng["email"], eng["team"])
        )
        
    # Insert Tasks
    for task in data.get("tasks", []):
        cursor.execute(
            """INSERT INTO tasks (id, title, description, owner_id, status, due_date, related_pr_id, sprint_id, team) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task["id"], task["title"], task["description"], task["owner_id"], 
             task["status"], task["due_date"], task.get("related_pr_id"), task["sprint_id"], task["team"])
        )
        
    # Insert Dependencies
    for dep in data.get("dependencies", []):
        cursor.execute(
            "INSERT INTO dependencies (task_id, depends_on_task_id, type) VALUES (?, ?, ?)",
            (dep["task_id"], dep["depends_on_task_id"], dep["type"])
        )
        
    # Insert PR Events
    for pr in data.get("pr_events", []):
        cursor.execute(
            "INSERT INTO pr_events (pr_id, task_id, status, timestamp, ci_message) VALUES (?, ?, ?, ?, ?)",
            (pr["pr_id"], pr["task_id"], pr["status"], pr["timestamp"], pr.get("ci_message"))
        )
        
    # Insert Activity Log
    for act in data.get("activity_log", []):
        cursor.execute(
            "INSERT INTO activity_log (id, task_id, engineer_id, type, message, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (act["id"], act["task_id"], act["engineer_id"], act["type"], act["message"], act["timestamp"])
        )
        
    conn.commit()
    print("Database seeding completed.")

def init_db(seed_file_path: str = "seed.json"):
    """Initialize the database by creating tables and loading seed data."""
    conn = get_db_connection()
    try:
        create_tables(conn)
        load_seed_data(conn, seed_file_path)
    finally:
        conn.close()

# Helper Functions for data retrieval
def fetch_task(task_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.*, e.name as owner_name, e.email as owner_email, e.team as owner_team
            FROM tasks t
            JOIN engineers e ON t.owner_id = e.id
            WHERE t.id = ?
        """, (task_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def fetch_task_by_pr(pr_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.*, e.name as owner_name, e.email as owner_email, e.team as owner_team
            FROM tasks t
            JOIN engineers e ON t.owner_id = e.id
            WHERE t.related_pr_id = ?
        """, (pr_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def fetch_all_tasks() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def fetch_dependencies() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dependencies")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def fetch_blocked_engineers(task_id: str) -> List[Dict[str, Any]]:
    """Find engineers who own tasks that depend on the given task_id."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT e.id, e.name, e.email, e.team, t.id as task_id, t.title as task_title
            FROM dependencies d
            JOIN tasks t ON d.task_id = t.id
            JOIN engineers e ON t.owner_id = e.id
            WHERE d.depends_on_task_id = ?
        """, (task_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def fetch_engineer_workload(engineer_id: str) -> List[Dict[str, Any]]:
    """Retrieve active tasks (todo, in_progress, blocked, in_review) for an engineer."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks
            WHERE owner_id = ? AND status != 'merged'
        """, (engineer_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def fetch_pr_history(task_id: str) -> List[Dict[str, Any]]:
    """Get PR events history for a task."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM pr_events
            WHERE task_id = ?
            ORDER BY timestamp DESC
        """, (task_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def fetch_activity_log(task_id: str) -> List[Dict[str, Any]]:
    """Get activity logs for a task."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, e.name as engineer_name
            FROM activity_log a
            JOIN engineers e ON a.engineer_id = e.id
            WHERE a.task_id = ?
            ORDER BY a.timestamp DESC
        """, (task_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
