import os
import re
import json
import db
import graph
import vector_store
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Define Tools
def tool_get_task(task_id: str) -> str:
    """Fetch task details and owner metadata."""
    t = db.fetch_task(task_id)
    return json.dumps(t) if t else f"Task {task_id} not found."

def tool_get_dependency_graph(task_id: str) -> str:
    """Fetch upstream blockers and downstream impact for a task."""
    upstream = graph.get_upstream_blockers(task_id)
    downstream = graph.get_downstream_impact(task_id)
    
    # Simple representation
    upstream_ids = [t["id"] for t in upstream]
    downstream_ids = [t["id"] for t in downstream]
    
    return json.dumps({
        "task_id": task_id,
        "upstream_blockers": upstream_ids,
        "downstream_impacted": downstream_ids
    })

def tool_get_blocked_engineers(task_id: str) -> str:
    """Find engineers who are waiting on this task."""
    blocked = db.fetch_blocked_engineers(task_id)
    return json.dumps(blocked)

def tool_get_engineer_workload(engineer_id: str) -> str:
    """Fetch active tasks for an engineer."""
    workload = db.fetch_engineer_workload(engineer_id)
    return json.dumps(workload)

def tool_get_pr_history(task_id: str) -> str:
    """Get PR events history for a task."""
    history = db.fetch_pr_history(task_id)
    return json.dumps(history)

def tool_search_related_tasks(
    query: str,
    top_k: int = 5,
    min_score: float = 0.70,
    filters: Optional[Dict[str, Any]] = None,
    exclude_task_id: Optional[str] = None
) -> str:
    """Semantic search for related tasks using embeddings."""
    results = vector_store.search_related_tasks(
        query_text=query,
        top_k=top_k,
        min_score=min_score,
        filters=filters,
        exclude_task_id=exclude_task_id
    )
    return json.dumps(results)

# Map of tool names to functions
TOOLS = {
    "get_task": tool_get_task,
    "get_dependency_graph": tool_get_dependency_graph,
    "get_blocked_engineers": tool_get_blocked_engineers,
    "get_engineer_workload": tool_get_engineer_workload,
    "get_pr_history": tool_get_pr_history,
    "search_related_tasks": tool_search_related_tasks
}

# API Call Helpers
def call_llm(system_prompt: str, prompt: str, temperature: float = 0.7) -> str:
    """Dispatch LLM request to configured provider or fallback to simulated."""
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return call_gemini_api(system_prompt, prompt, temperature)
    elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return call_openai_api(system_prompt, prompt, temperature)
    else:
        # Fallback to simulated behavior or raise warning
        return "SIMULATED_LLM_RESPONSE"

def call_gemini_api(system_prompt: str, prompt: str, temperature: float) -> str:
    import httpx
    # Simple direct API call to Google GenAI REST API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Format request body
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"System Instruction:\n{system_prompt}\n\nUser Request:\n{prompt}"}]
            }
        ],
        "generationConfig": {
            "temperature": temperature
        }
    }
    
    try:
        response = httpx.post(url, json=data, headers=headers, timeout=30.0)
        response.raise_for_status()
        res_json = response.json()
        return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "SIMULATED_LLM_RESPONSE"

def call_openai_api(system_prompt: str, prompt: str, temperature: float) -> str:
    from openai import OpenAI
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "SIMULATED_LLM_RESPONSE"


def call_simulated_agent_brain(task: Dict[str, Any], trace: List[Dict[str, Any]]) -> str:
    """
    Simulates a dynamic LLM planning response based on which tools have been executed.
    Ensures that simulated/offline runs execute a true feedback loop rather than a hardcoded pipeline.
    """
    task_id = task["id"]
    task_title = task["title"]
    
    # Analyze the actions executed in previous trace steps
    completed_actions = [t["action"] for t in trace if t["type"] == "tool_call"]
    
    has_get_task = any("get_task" in act for act in completed_actions)
    has_get_pr_history = any("get_pr_history" in act for act in completed_actions)
    has_get_dependency_graph = any("get_dependency_graph" in act for act in completed_actions)
    has_get_blocked_engineers = any("get_blocked_engineers" in act for act in completed_actions)
    has_search_related_tasks = any("search_related_tasks" in act for act in completed_actions)
    
    # Dynamically select the next logical tool request
    if not has_get_task:
        return f'TOOL_CALL: get_task(task_id="{task_id}")'
    elif not has_get_pr_history:
        return f'TOOL_CALL: get_pr_history(task_id="{task_id}")'
    elif not has_get_dependency_graph:
        return f'TOOL_CALL: get_dependency_graph(task_id="{task_id}")'
    elif not has_get_blocked_engineers:
        return f'TOOL_CALL: get_blocked_engineers(task_id="{task_id}")'
    elif not has_search_related_tasks:
        return f'TOOL_CALL: search_related_tasks(query="{task_title}", top_k=3, min_score=0.50, exclude_task_id="{task_id}")'
    else:
        return f"INVESTIGATION_COMPLETE: Simulated investigation completed for task {task_id}. Retrieved details, PR events, blockers, and semantic relations."


class BottleneckInvestigatorAgent:
    def __init__(self):
        self.trace = []

    def log_step(self, step_type: str, action: str, observation: str):
        self.trace.append({
            "step": len(self.trace) + 1,
            "type": step_type,
            "action": action,
            "observation": observation
        })

    def run_agent_loop(self, task: Dict[str, Any]) -> str:
        """Runs the LLM plan-tool-observe loop to gather context."""
        task_id = task["id"]
        system_prompt = f"""You are an AI Agent investigating a development bottleneck for task {task_id}.
Your goal is to gather all context about this task and its status.
You have access to the following tools:
1. `get_task(task_id: str)`: Fetch task and owner details.
2. `get_dependency_graph(task_id: str)`: Fetch upstream blockers and downstream impact task IDs.
3. `get_blocked_engineers(task_id: str)`: Fetch list of engineers waiting on this task.
4. `get_engineer_workload(engineer_id: str)`: Fetch active tasks for an engineer.
5. `get_pr_history(task_id: str)`: Fetch PR events for a task.
6. `search_related_tasks(query: str, top_k: int, min_score: float, filters: dict, exclude_task_id: str)`: Search semantically similar tasks.

You must choose one of two outputs:
- To call a tool, output exactly: TOOL_CALL: tool_name(arg_name="value", ...)
- When you have enough information, output exactly: INVESTIGATION_COMPLETE: <summary of findings>

Do not write code blocks or markdown backticks for the output. Follow this plan:
1. Fetch the main task details.
2. Check the PR history for this task.
3. Fetch the dependency graph to check who is blocked and what is blocking it.
4. If there is a blocked task, look up the blocked engineer and their details.
5. Search for related tasks to find similar context.
6. Finish and output INVESTIGATION_COMPLETE.
"""
        prompt = f"Start investigation for task {task_id} ({task['title']})."
        
        # Max steps to prevent infinite loop
        max_steps = 8
        for step in range(max_steps):
            # Construct agent context
            history_str = ""
            for t in self.trace:
                history_str += f"Step {t['step']}: {t['type']} -> {t['action']}\nObservation: {t['observation']}\n\n"
                
            current_prompt = f"{prompt}\n\nHere is what you have done so far:\n{history_str}What is your next action?"
            
            # Fetch decision: Use dynamic simulated brain in simulated mode or when API keys are absent
            if LLM_PROVIDER == "simulated" or not (GEMINI_API_KEY or OPENAI_API_KEY):
                response = call_simulated_agent_brain(task, self.trace)
            else:
                response = call_llm(system_prompt, current_prompt, temperature=0.3)
                if response == "SIMULATED_LLM_RESPONSE":
                    response = call_simulated_agent_brain(task, self.trace)
            
            if response.startswith("TOOL_CALL:"):
                # Parse tool name and args
                match = re.match(r"TOOL_CALL:\s*(\w+)\((.*)\)", response)
                if not match:
                    observation = "Error parsing tool call. Ensure format is: TOOL_CALL: tool_name(arg_name=\"value\", ...)"
                    self.log_step("invalid_tool_call", response, observation)
                    continue
                    
                tool_name = match.group(1)
                args_str = match.group(2)
                
                # Simple parser for keyword arguments like key="value" or key=number
                args = {}
                # Match key="val" or key='val' or key=val
                for kv in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s,]+))', args_str):
                    k = kv.group(1)
                    v = kv.group(2) or kv.group(3) or kv.group(4)
                    # Try to parse json/numbers/booleans if needed
                    if v == "None":
                        v = None
                    elif v == "True":
                        v = True
                    elif v == "False":
                        v = False
                    elif v.startswith("{") and v.endswith("}"):
                        try:
                            v = json.loads(v.replace("'", '"'))
                        except:
                            pass
                    else:
                        try:
                            if "." in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            pass
                    args[k] = v
                
                if tool_name not in TOOLS:
                    observation = f"Tool {tool_name} does not exist."
                    self.log_step("tool_call", response, observation)
                    continue
                    
                # Run tool
                try:
                    observation = TOOLS[tool_name](**args)
                except Exception as e:
                    observation = f"Error executing tool {tool_name}: {str(e)}"
                    
                self.log_step("tool_call", response, observation)
                
            elif response.startswith("INVESTIGATION_COMPLETE:"):
                summary = response.replace("INVESTIGATION_COMPLETE:", "").strip()
                self.log_step("complete", "Finish investigation", summary)
                return summary
            else:
                # If LLM didn't format output correctly, force complete or log it
                self.log_step("unexpected_output", response, "LLM returned non-standard format.")
                return f"Raw LLM output: {response}"
                
        return "Agent reached maximum steps without completing."

    def route_bottleneck(self, task: Dict[str, Any], context: str) -> str:
        """Classifies the bottleneck type strictly. (Router Node)"""
        system_prompt = """You are a classification router.
You classify developer task bottleneck situations into exactly one of these labels:
- DEPENDENCY_BLOCK (the task or dependent downstream tasks are blocked by a separate engineering task that is overdue/failing/delayed)
- SELF_STALL (the task is overdue or failing CI, but has no upstream blockers or waiting reviews)
- REVIEW_WAIT (the task is in_review or comments show it is waiting for review/sign-off)
- EXTERNAL_BLOCK (the task is blocked on external non-engineer dependencies, like external APIs, hardware, or PM input)
- UNKNOWN (if you cannot determine)

Your response must contain EXACTLY the label and NOTHING else. No explanation, no intro, no punctuation.
"""
        prompt = f"""
Task Details: {json.dumps(task)}
Investigation Context Trace:
{json.dumps(self.trace)}

Classify this bottleneck.
"""
        response = call_llm(system_prompt, prompt, temperature=0.0) # strict/deterministic
        
        if response == "SIMULATED_LLM_RESPONSE" or LLM_PROVIDER == "simulated":
            task_id = task["id"]
            
            # Fetch dependency graph, comments, and PR data
            upstream = graph.get_upstream_blockers(task_id)
            downstream = graph.get_downstream_impact(task_id)
            activity = db.fetch_activity_log(task_id)
            pr_history = db.fetch_pr_history(task_id)
            
            active_upstream_blockers = [b for b in upstream if b["status"] != "merged"]
            
            # Check review request signals
            has_review_comment = any(
                any(word in act["message"].lower() for word in ["review", "reviewer", "approv", "sign-off"])
                for act in activity
            )
            
            # Check external mention keywords
            has_external_mention = any(
                any(word in text.lower() for word in ["external", "api limit", "vendor", "partner", "hardware"])
                for text in [task["description"], task["title"]] + [act["message"] for act in activity]
            )

            has_failed_pr = any(pr["status"] == "failed" for pr in pr_history)

            # Algorithmic classification decision tree
            if len(active_upstream_blockers) > 0:
                return "DEPENDENCY_BLOCK"
            elif len(downstream) > 0 and (has_failed_pr or task["status"] == "blocked"):
                return "DEPENDENCY_BLOCK"
            elif task["status"] == "in_review" or has_review_comment:
                return "REVIEW_WAIT"
            elif has_external_mention:
                return "EXTERNAL_BLOCK"
            elif has_failed_pr or task["status"] == "in_progress":
                return "SELF_STALL"
            else:
                return "UNKNOWN"
                
        # Clean response in case LLM added extra words or punctuation
        for label in ["DEPENDENCY_BLOCK", "SELF_STALL", "REVIEW_WAIT", "EXTERNAL_BLOCK", "UNKNOWN"]:
            if label in response:
                return label
        return "UNKNOWN"

    def generate_notification(self, task: Dict[str, Any], classification: str) -> str:
        """Generates developer notification text. (Notification Node)"""
        system_prompt = """You are a developer relations assistant at TeamBoostAI.
Your job is to write a concise, friendly, and actionable Slack/Teams notification (2-4 sentences) to the engineer.
Make it highly specific, mentioning the engineer's name, the task ID, task title, PR number (if applicable), and what they need to do.
- If it's a DEPENDENCY_BLOCK: explain that their work is blocking another engineer, or that they are blocked by someone else.
- If it's a SELF_STALL: nudge them about the overdue date or failing tests and what migration/task needs completion.
- If it's a REVIEW_WAIT: advise them to ping the reviewer or follow up.
Ensure your tone is supportive and developer-focused.
"""
        prompt = f"""
Task details: {json.dumps(task)}
Bottleneck Classification: {classification}
Investigation Trace: {json.dumps(self.trace)}

Generate the notification message.
"""
        response = call_llm(system_prompt, prompt, temperature=0.7) # creative/natural
        
        if response == "SIMULATED_LLM_RESPONSE" or LLM_PROVIDER == "simulated":
            name = task["owner_name"]
            task_id = task["id"]
            title = task["title"]
            
            if classification == "DEPENDENCY_BLOCK":
                upstream = graph.get_upstream_blockers(task_id)
                active_upstream = [b for b in upstream if b["status"] != "merged"]
                
                if len(active_upstream) > 0:
                    # Blocked by another task
                    blocker = active_upstream[0]
                    return (
                        f"Hi {name}, your task {task_id} ('{title}') is currently blocked by "
                        f"task {blocker['id']} ('{blocker['title']}') owned by {blocker['owner_name']}. "
                        f"We will notify you once that prerequisite task is completed."
                    )
                else:
                    # Delay is blocking others downstream
                    blocked = db.fetch_blocked_engineers(task_id)
                    blocked_names = list(dict.fromkeys([b["name"] for b in blocked]))
                    blocked_str = ", ".join(blocked_names) if blocked_names else "other team members"
                    
                    pr_history = db.fetch_pr_history(task_id)
                    pr_num = ""
                    if pr_history:
                        pr_num = f" (PR #{pr_history[0]['pr_id'].replace('pr-', '')})"
                        
                    return (
                        f"Hi {name}, your task {task_id} ('{title}'){pr_num} has failed CI or is delayed, "
                        f"which is blocking {blocked_str} from starting their downstream work. "
                        f"Please prioritize checking the logs and moving this forward."
                    )
            elif classification == "SELF_STALL":
                pr_history = db.fetch_pr_history(task_id)
                pr_str = ""
                if pr_history:
                    pr_str = f" and PR #{pr_history[0]['pr_id'].replace('pr-', '')} is failing tests"
                
                return (
                    f"Hi {name}, task {task_id} ('{title}') is currently overdue (due {task['due_date']}){pr_str}. "
                    f"Since there are no external dependencies blocking you, please resolve the remaining test failures "
                    f"or finalize migrations to complete this task."
                )
            elif classification == "REVIEW_WAIT":
                return (
                    f"Hi {name}, task {task_id} ('{title}') is in review. "
                    f"Please follow up with your design reviewers or ping the staff engineer to secure a design sign-off."
                )
            else:
                return (
                    f"Hi {name}, friendly nudge regarding task {task_id} ('{title}'). "
                    f"Please check if there are any technical blocks hindering its completion."
                )
                
        return response

    def investigate(self, task_id: str) -> Dict[str, Any]:
        """Runs the end-to-end investigation pipeline."""
        task = db.fetch_task(task_id)
        if not task:
            return {
                "status": "failed",
                "error": f"Task {task_id} not found"
            }
            
        summary = self.run_agent_loop(task)
        classification = self.route_bottleneck(task, summary)
        notification = self.generate_notification(task, classification)
        
        return {
            "status": "completed",
            "task_id": task_id,
            "classification": classification,
            "trace": self.trace,
            "notification": notification
        }
