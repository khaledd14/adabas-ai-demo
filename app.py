import os
from typing import TypedDict, List, Dict, Any, Optional
import streamlit as st
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ==========================================
# 1. Page Configuration & Title
# ==========================================
st.set_page_config(
    page_title="Adabas & Natural Agentic AI Gateway",
    page_icon="🤖",
    layout="wide"
)

st.title("Adabas & Natural Agentic AI Gateway")
st.caption("Powered by Groq Llama-3.1, LangGraph, Streamlit, and Pydantic")

# ==========================================
# 2. Helper Functions & Key Resolution
# ==========================================
def get_groq_key() -> str:
    """Retrieve Groq API key from environment variables or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if not key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
    return key or ""

groq_key = get_groq_key()

if not groq_key:
    st.error("🔑 GROQ_API_KEY not found! Please set it in your Environment Variables or Streamlit Secrets.")
    st.stop()

# ==========================================
# 3. Pydantic Models & State Definition
# ==========================================
class ActionIntent(BaseModel):
    action_type: str = Field(description="Action to perform: e.g., 'apply_credit', 'update_record', 'query'")
    target_account_type: str = Field(description="Target account type: e.g., 'commercial', 'individual'")
    zip_code_range: str = Field(description="Zip code range or target region, e.g. '70001-75000'")
    amount: float = Field(description="Numeric value or credit amount specified in request")

class AgentState(TypedDict):
    raw_prompt: str
    parsed_intent: Optional[Dict[str, Any]]
    discovered_resources: Optional[Dict[str, Any]]
    simulation_result: Optional[Dict[str, Any]]
    human_approved: Optional[bool]
    commit_status: Optional[str]
    audit_log: List[str]

# ==========================================
# 4. LangGraph Agent Nodes
# ==========================================
def parse_intent_node(state: AgentState) -> Dict[str, Any]:
    """Stage 1: Intent Parsing using Groq Structured Output"""
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=groq_key
    )
    
    structured_llm = llm.with_structured_output(ActionIntent)
    prompt = f"Parse this natural language request into structured Adabas parameters: '{state['raw_prompt']}'"
    
    parsed: ActionIntent = structured_llm.invoke(prompt)
    parsed_dict = parsed.model_dump()
    
    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 1: PARSE] Successfully parsed intent into structured parameters.")
    
    return {
        "parsed_intent": parsed_dict,
        "audit_log": logs
    }

def discover_resources_node(state: AgentState) -> Dict[str, Any]:
    """Stage 2: Resource Discovery (Predict API / CONNX Mock)"""
    intent = state.get("parsed_intent", {})
    
    # Mocking metadata discovery for Predict API & CONNX SQL
    discovered = {
        "predict_dictionary": "ADABAS_FILE_102_CUSTOMER",
        "connx_sql_view": "VW_COMMERCIAL_ACCOUNTS",
        "affected_records_count": 142,
        "target_zip_range": intent.get("zip_code_range", "N/A"),
        "account_type": intent.get("target_account_type", "N/A")
    }
    
    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 2: DISCOVER] Found Predict file ADABAS_FILE_102 matching query criteria.")
    
    return {
        "discovered_resources": discovered,
        "audit_log": logs
    }

def simulate_execution_node(state: AgentState) -> Dict[str, Any]:
    """Stage 3: Adabas Simulator Execution"""
    discovered = state.get("discovered_resources", {})
    intent = state.get("parsed_intent", {})
    
    simulation = {
        "status": "SUCCESS",
        "impact_summary": f"Would update {discovered.get('affected_records_count', 0)} records in ADABAS_FILE_102.",
        "applied_credit_amount": intent.get("amount", 0.0),
        "estimated_cpu_cost": "0.02 SU",
        "risk_level": "LOW"
    }
    
    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 3: SIMULATE] Simulation complete. Impacted records: {discovered.get('affected_records_count', 0)}.")
    
    return {
        "simulation_result": simulation,
        "audit_log": logs
    }

def commit_node(state: AgentState) -> Dict[str, Any]:
    """Stage 5: Commit Execution via Natural RPC Mock"""
    approved = state.get("human_approved", False)
    logs = state.get("audit_log", [])
    
    if approved:
        status = "COMMITTED: Natural RPC execution succeeded. Adabas transaction finalized."
        logs.append("[STAGE 5: COMMIT] Human operator APPROVED transaction. Executed successfully.")
    else:
        status = "ABORTED: Human operator REJECTED transaction."
        logs.append("[STAGE 5: COMMIT] Human operator REJECTED transaction. No changes saved.")
        
    return {
        "commit_status": status,
        "audit_log": logs
    }

# ==========================================
# 5. Build Graph Workflow
# ==========================================
@st.cache_resource
def get_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse", parse_intent_node)
    workflow.add_node("discover", discover_resources_node)
    workflow.add_node("simulate", simulate_execution_node)
    workflow.add_node("commit", commit_node)
    
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "discover")
    workflow.add_edge("discover", "simulate")
    workflow.add_edge("simulate", "commit")
    workflow.add_edge("commit", END)
    
    return workflow.compile()

app_graph = get_graph()

# ==========================================
# 6. Streamlit User Interface
# ==========================================
user_prompt = st.text_input(
    "Command Prompt:",
    value="Apply 100 credit to commercial accounts ZIP 70001-75000"
)

if st.button("Start Workflow", type="primary"):
    with st.spinner("Processing through Agentic AI Gateway..."):
        initial_state: AgentState = {
            "raw_prompt": user_prompt,
            "parsed_intent": None,
            "discovered_resources": None,
            "simulation_result": None,
            "human_approved": True,  # Default to auto-approve for initial demonstration
            "commit_status": None,
            "audit_log": []
        }
        
        final_state = app_graph.invoke(initial_state)
        st.session_state["agent_results"] = final_state

# Display Results
if "agent_results" in st.session_state:
    results = st.session_state["agent_results"]
    
    st.subheader("Workflow Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Parsed Intent")
        st.json(results.get("parsed_intent", {}))
        
        st.markdown("### 2. Discovered Resources")
        st.json(results.get("discovered_resources", {}))
        
    with col2:
        st.markdown("### 3. Simulation Result")
        st.json(results.get("simulation_result", {}))
        
        st.markdown("### 4. Final Commit Status")
        st.success(results.get("commit_status", ""))
        
    st.markdown("---")
    st.subheader("Audit Log & Trail")
    for log in results.get("audit_log", []):
        st.code(log, language="text")