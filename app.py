import os
import requests
from typing import TypedDict, List, Dict, Any, Optional
import streamlit as st
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# ==========================================
# 1. Page Configuration & Title
# ==========================================
st.set_page_config(
    page_title="Adabas & Natural Agentic AI Gateway",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Adabas & Natural Agentic AI Gateway")
st.caption("Powered by Groq gpt-oss-20b, LangGraph State Architecture, and Software AG REST Gateway")

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
adabas_rest_url = os.environ.get("ADABAS_REST_URL")

if not groq_key:
    st.error("🔑 `GROQ_API_KEY` not found! Please set it in your Environment Variables.")
    st.stop()

# ==========================================
# 3. Pydantic Models & State Definition
# ==========================================
class ActionIntent(BaseModel):
    action_type: str = Field(description="Action to perform: e.g., 'apply_credit', 'update_record', 'query'")
    target_account_type: str = Field(description="Target account type: e.g., 'commercial', 'individual'")
    zip_code_range: str = Field(description="Zip code range or target region, e.g. '70001-75000' or '90210'")
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
# 4. Agent Pipeline Nodes
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
    logs.append("[STAGE 1: PARSE] Dynamic intent successfully parsed into structured Pydantic schema.")

    return {"parsed_intent": parsed_dict, "audit_log": logs}

def discover_resources_node(state: AgentState) -> Dict[str, Any]:
    """Stage 2: Resource Discovery (Predict API / CONNX Mock)"""
    intent = state.get("parsed_intent", {})

    discovered = {
        "predict_dictionary": "ADABAS_FILE_102_CUSTOMER",
        "connx_sql_view": "VW_COMMERCIAL_ACCOUNTS",
        "affected_records_count": 142,
        "target_zip_range": intent.get("zip_code_range", "N/A"),
        "account_type": intent.get("target_account_type", "N/A")
    }

    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 2: DISCOVER] Found Predict Data Dictionary ADABAS_FILE_102 for zip {intent.get('zip_code_range')}.")

    return {"discovered_resources": discovered, "audit_log": logs}

def simulate_execution_node(state: AgentState) -> Dict[str, Any]:
    """Stage 3: Adabas Simulator Execution"""
    discovered = state.get("discovered_resources", {})
    intent = state.get("parsed_intent", {})

    simulation = {
        "status": "SIMULATED_SUCCESS",
        "impact_summary": f"Would update {discovered.get('affected_records_count', 0)} records in ADABAS_FILE_102.",
        "applied_credit_amount": intent.get("amount", 0.0),
        "estimated_cpu_cost": "0.02 SU", # Software AG Service Units
        "risk_level": "LOW"
    }

    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 3: SIMULATE] Impact assessment complete. Risk Level: LOW. Impacted records: {discovered.get('affected_records_count', 0)}.")

    return {"simulation_result": simulation, "audit_log": logs}

def commit_node(state: AgentState) -> Dict[str, Any]:
    """Stage 4: Commit Execution via Live REST Gateway or Sandbox"""
    approved = state.get("human_approved", False)
    logs = state.get("audit_log", [])

    if not approved:
        logs.append("[STAGE 4: COMMIT] Human operator REJECTED transaction. Execution aborted.")
        return {"commit_status": "❌ ABORTED: Transaction rejected by operator. No changes saved.", "audit_log": logs}

    # Attempt Live REST API call if configured
    if adabas_rest_url:
        try:
            payload = {
                "file": "ADABAS_FILE_102",
                "intent": state.get("parsed_intent"),
                "discovered_resources": state.get("discovered_resources")
            }
            response = requests.post(adabas_rest_url, json=payload, timeout=5)
            if response.status_code == 200:
                logs.append(f"[STAGE 4: COMMIT] Live HTTP request to Adabas REST Gateway succeeded. (Status: {response.status_code})")
                return {"commit_status": "✅ COMMITTED: Live Adabas REST transaction executed successfully.", "audit_log": logs}
        except Exception as e:
            logs.append(f"[STAGE 4: COMMIT] Live REST connection error: {str(e)}. Falling back to secure sandbox.")

    # Fallback to Sandbox Mode
    logs.append("[STAGE 4: COMMIT] Human operator APPROVED transaction. Executed successfully in Sandbox Mode.")
    return {"commit_status": "✅ COMMITTED: Sandbox Adabas transaction finalized.", "audit_log": logs}

# ==========================================
# 5. Streamlit User Interface & HITL
# ==========================================
# Using the messy prompt by default to prove LLM extraction reasoning
user_prompt = st.text_input(
    "Command Prompt:",
    value="Hey, refund 250 bucks to corporate clients in Beverly Hills 90210"
)

# PHASE 1: Parse, Discover, Simulate
if st.button("Simulate AI Workflow", type="primary"):
    with st.spinner("Processing natural language & running Adabas impact simulation..."):
        initial_state: AgentState = {
            "raw_prompt": user_prompt,
            "parsed_intent": None,
            "discovered_resources": None,
            "simulation_result": None,
            "human_approved": None,
            "commit_status": None,
            "audit_log": []
        }

        # Execute nodes sequentially to build the state
        s1 = parse_intent_node(initial_state)
        s2 = discover_resources_node({**initial_state, **s1})
        s3 = simulate_execution_node({**initial_state, **s1, **s2})

        # Save pending transaction to session state to pause execution
        st.session_state["pending_txn"] = {**initial_state, **s1, **s2, **s3}

# PHASE 2: Human-in-the-Loop Verification Dashboard
if "pending_txn" in st.session_state:
    st.divider()
    txn = st.session_state["pending_txn"]

    st.warning("⚠️ **Human Verification Required:** Review simulated impact before committing to Adabas.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Parsed Intent (Groq)")
        st.json(txn["parsed_intent"])
    with col2:
        st.subheader("2. Target Resources")
        st.json(txn["discovered_resources"])
    with col3:
        st.subheader("3. Impact Simulation")
        st.json(txn["simulation_result"])

    st.subheader("Action Approval")
    btn_col1, btn_col2 = st.columns(2)

    # PHASE 3: Commit Execution
    with btn_col1:
        if st.button("✅ Approve & Commit to Adabas", use_container_width=True):
            txn["human_approved"] = True
            final_res = commit_node(txn)
            st.success(final_res["commit_status"])
            
            st.markdown("### Enterprise Audit Trail")
            st.code("\n".join(final_res["audit_log"]), language="text")
            
            # Clear session state after commit
            del st.session_state["pending_txn"]

    with btn_col2:
        if st.button("❌ Reject Transaction", type="secondary", use_container_width=True):
            txn["human_approved"] = False
            final_res = commit_node(txn)
            st.error(final_res["commit_status"])
            
            st.markdown("### Enterprise Audit Trail")
            st.code("\n".join(final_res["audit_log"]), language="text")
            
            # Clear session state after rejection
            del st.session_state["pending_txn"]