import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
# Initialize your LLM here (e.g., ChatOpenAI, ChatGoogleGenerativeAI, etc.)

import google.generativeai as genai

# Retrieve the API key from secrets
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)

llm_flash = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key = api_key)
llm= llm_flash

# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
  messages: List[BaseMessage]
  next_step: Optional[str]
  code: Optional[str]
  report: Optional[str]
  manager_command: Optional[str]

# ==========================================
# 3. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
  """Execute python code and return the standard output or error trace."""
  if not isinstance(code, str):
    code = str(code)
  clean_code = code.replace('```python', '').replace('```', '').strip()
  old_stdout = sys.stdout
  new_stdout = io.StringIO()
  sys.stdout = new_stdout

  try:
    local_scope = {}
    exec(clean_code, {}, local_scope)
    result = new_stdout.getvalue()
  except Exception as e:
    result = f"Execution Error:\n{traceback.format_exc()}"
  finally:
    sys.stdout = old_stdout
  return result.strip() if result.strip() else "Success (no terminal output)"

@tool
def generate_test_cases(task_description: str) -> str:
   """Generate specific test scenarios for a given coding task."""
   prompt = (
     f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
     f"for the following coding task: '{task_description}'.\n"
     f"Include standard cases and edge cases. Return them as a numbered list."
   )
   response = llm.invoke(prompt)
   return response.content if hasattr(response, 'content') else str(response)

# Initialize the test tool with the LLM
# (Uncomment this once your LLM is initialized above)
# generate_test_cases = create_test_generator_tool(llm_flash)
# ==========================================
# 4. GRAPH NODES
# ==========================================

def task_input_node(state: CrewState):
    return {"next_step": "developer"}

def real_time_developer(state: CrewState):
  print("\n[Developer] Writing dynamic code using LLM...")
  # Get the latest task description
  task = state['messages'][-1].content
  dev_prompt = f"Write a clean Python script to solve this: {task}. Only return the code, no explanation or markdown formatting."

  # Ensure LLM is initialized before calling
  if llm_flash is None:
    raise ValueError("LLM is not initialized. Please set up 'llm_flash'.")

  response = llm_flash.invoke(dev_prompt)
  # --- FIX: Safely parse Gemini's content format ---

  content = response.content

  if isinstance(content, list):
    # Extract the text from the first dictionary in the list
    code_str = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
  else:
    # It's already a standard string
    code_str = str(content)

  print(code_str)
  return {
      "code": code_str,
      "next_step": "tester"
  }

def real_time_tester(state: CrewState):
  print("\n[Tester] Generating dynamic tests and executing code...")
  task = state['messages'][-1].content

  # Generate tests
  test_cases = generate_test_cases.invoke(task)
  content = test_cases
  if isinstance(content, list):
    # Extract the text from the first dictionary in the list
    cases_str = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
  else:
    # It's already a standard string
    cases_str = str(content)
  # Execute code
  execution_result = run_python_code.invoke({"code": state['code']})

  # Compile Report
  report = f"### EXECUTION OUTPUT:\n{execution_result}\n\n### TEST SCENARIOS EVALUATED:\n{cases_str}"
  return {
      "report": report,
      "next_step": "manager_decision"
  }

def manager_decision_node(state: CrewState):
    user_input = state.get('manager_command', 'store').lower().strip()
    if user_input == 'store':
        return {"next_step": "archiver"}
    else:
        return {"next_step": "task_input"}

def archiver_node(state: CrewState):
  print("\n[Archiver] Task stored successfully. Closing workflow.")
  return {"next_step": "exit"}

# ==========================================
# 5. GRAPH CONSTRUCTION & ROUTING
# ==========================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("manager_decision", manager_decision_node)
rt_workflow.add_node("archiver", archiver_node)
rt_workflow.add_edge(START, "task_input")

def route_from_input(state):
  if state.get('next_step') == "exit":
    return END
  return "developer"

rt_workflow.add_conditional_edges("task_input", route_from_input)

# Sequential flow
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", "manager_decision")

def route_from_decision(state):
  if state.get('next_step') == "archiver":
    return "archiver"
  return "task_input"

rt_workflow.add_conditional_edges("manager_decision", route_from_decision)
rt_workflow.add_edge("archiver", END)
rt_app = rt_workflow.compile()

print("Interactive pipeline compiled and ready for live execution.")

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
app = FastAPI()
class TaskRequest(BaseModel):
    task: str
    manager_command: str = "store"

@app.post("/execute-task")
def execute_task(req: TaskRequest):
    initial_state = {
        "messages": [HumanMessage(content=req.task)],
        "manager_command": req.manager_command
    }
    return rt_app.invoke(initial_state, config={"recursion_limit": 50})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
