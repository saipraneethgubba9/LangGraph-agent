import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

# gemini-3.6-flash is the current GA flash-tier model (as of Aug 2026).
# Google deprecates model versions periodically - if this 404s again,
# run genai.list_models() with your key to see what's currently valid.
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=GOOGLE_API_KEY,
)
llm = llm_flash

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    stage: str                 # <-- tracks which node is CURRENTLY active
    next_step: Optional[str]   # <-- used ONLY for routing decisions
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
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
    return response.content if hasattr(response, "content") else str(response)


# ==========================================
# 4. GRAPH NODES
# ==========================================
def task_input_node(state: CrewState):
    print("--- STAGE: task_input ---")
    return {"stage": "task_input", "next_step": "developer"}


def real_time_developer(state: CrewState):
    print("--- STAGE: developer ---")
    task = state["messages"][-1].content
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)
    content = response.content
    if isinstance(content, list):
        code_str = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        code_str = str(content)

    print(code_str)
    return {"code": code_str, "stage": "developer", "next_step": "tester"}


def real_time_tester(state: CrewState):
    print("--- STAGE: tester ---")
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    content = test_cases
    if isinstance(content, list):
        cases_str = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        cases_str = str(content)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{cases_str}"
    )
    return {"report": report, "stage": "tester", "next_step": "archiver"}


def archiver_node(state: CrewState):
    print("--- STAGE: archiver ---")
    return {"stage": "archiver", "next_step": "exit"}


# ==========================================
# 5. GRAPH CONSTRUCTION (linear pipeline, single pass per request)
# ==========================================
rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("archiver", archiver_node)

rt_workflow.add_edge(START, "task_input")
rt_workflow.add_edge("task_input", "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", "archiver")
rt_workflow.add_edge("archiver", END)

rt_app = rt_workflow.compile()
print("Dev/Test crew graph compiled and ready.")


# ==========================================
# 6. LANGSERVE WRAPPER
# ==========================================
class AgentInput(BaseModel):
    input: str = Field(description="The coding task you want written and tested")


def format_for_graph(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {
        "messages": [HumanMessage(content=user_input)],
        "stage": "task_input",
        "next_step": None,
        "code": None,
        "report": None,
    }


def extract_response(state: dict) -> str:
    if not isinstance(state, dict):
        return str(state)
    return state.get("report") or "No report generated."


formatted_chain = (
    RunnableLambda(format_for_graph)
    | rt_app
    | RunnableLambda(extract_response)
).with_types(input_type=AgentInput, output_type=str)


# ==========================================
# 7. FASTAPI APP
# ==========================================
app = FastAPI(
    title="Dev-Test Crew Agent",
    version="1.0",
    description="A LangGraph pipeline (developer -> tester -> archiver) served via LangServe.",
)


@app.get("/")
def root():
    return {"message": "Server is running. Visit /crew/playground/ to chat, or /docs for the API"}


add_routes(app, formatted_chain, path="/crew")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
