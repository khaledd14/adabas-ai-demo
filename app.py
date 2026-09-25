import os
import requests
import pandas as pd
import streamlit as st
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ==========================================
# 1. Page Setup & Configuration
# ==========================================
st.set_page_config(
    page_title="Software AG Agentic AI Gateway",
    page_icon="⚡",
    layout="wide"
)

def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
    return key or ""

groq_key = get_groq_key()
adabas_rest_url = os.environ.get("ADABAS_REST_URL")

if not groq_key:
    st.error("🔑 `GROQ_API_KEY` missing in environment variables.")
    st.stop()

# ==========================================
# 2. Sidebar: RBAC & Demo Scenarios
# ==========================================
st.sidebar.title("🛡️ Enterprise Governance")
st.sidebar.caption("⚙️ Component: API Gateway / IAM")
user_role = st.sidebar.selectbox("Active User Role", ["System Administrator", "Tier 1 Operator"])

st.sidebar.divider()
st.sidebar.title("🧪 Agentic Routing Scenarios")
st.sidebar.info("The LangGraph Agent will dynamically choose the CONNX or EntireX path based on the prompt's intent.")

preset_choice = st.sidebar.radio(
    "Load Test Prompt:",
    [
        "Read-Only Analytics (Routes to CONNX)",
        "Transactional Update (Routes to EntireX)",
        "Security Violation (Blocked)"
    ]
)

preset_prompts = {
    "Read-Only Analytics (Routes to CONNX)": "Show me the credit balances for commercial accounts in ZIP 70001",
    "Transactional Update (Routes to EntireX)": "Apply 100 credit to commercial accounts in ZIP 70001",
    "Security Violation (Blocked)": "Purge all customer records from File 102"
}

user_prompt = st.text_input("Command Prompt:", value=preset_prompts[preset_choice])

# ==========================================
# 3. Pydantic Models & State Definition
# ==========================================
class ActionIntent(BaseModel):
    action_type: str = Field(description="Action: 'query', 'apply_credit', 'update_record', or 'delete_purge'")
    target_account_type: str = Field(description="Account category: 'commercial', 'individual', or 'all'")
    zip_code_range: str = Field(description="Zip code range or region, e.g. '70001' or '90210'")
    amount: float = Field(description="Numeric value, 0 if just a query")

class AgentState(TypedDict):
    raw_prompt: str
    parsed_intent: Optional[Dict[str, Any]]
    compliance_passed: Optional[bool]
    compliance_reason: Optional[str]
    predict_metadata: Optional[Dict[str, Any]]
    routing_decision: Optional[str]      # Stores chosen path
    connx_sql: Optional[str]             # Used if Read Path
    entirex_payload: Optional[Dict[str, Any]] # Used if Write Path
    natural_code: Optional[str]
    adabas_impact: Optional[Dict[str, Any]]
    before_df: Optional[pd.DataFrame]
    after_df: Optional[pd.DataFrame]
    human_approved: Optional[bool]
    commit_status: Optional[str]
    audit_log: List[str]

# ==========================================
# 4. Agent Pipeline Nodes
# ==========================================
def parse_and_compliance_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("audit_log", [])
    
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_key)
    structured_llm = llm.with_structured_output(ActionIntent)
    parsed: ActionIntent = structured_llm.invoke(f"Parse this request: '{state['raw_prompt']}'")
    intent = parsed.model_dump()
    logs.append(f"[STAGE 1: PARSE] Intent parsed via Groq LLM: {intent['action_type']}")

    passed = True
    reason = "Transaction within safe operational boundaries."
    
    if intent.get("action_type") == "delete_purge":
        passed = False
        reason = "CRITICAL: 'delete_purge' operations blocked by Enterprise Governance."
        logs.append(f"[STAGE 2: GUARDRAILS] ❌ REJECTED: {reason}")
    elif intent.get("amount", 0) > 5000:
        passed = False
        reason = "HIGH RISK: Amount exceeds $5,000 threshold."
        logs.append(f"[STAGE 2: GUARDRAILS] ⚠️ FLAGGED: {reason}")
    else:
        logs.append("[STAGE 2: GUARDRAILS] ✅ Cleared Enterprise Guardrails.")

    return {"parsed_intent": intent, "compliance_passed": passed, "compliance_reason": reason, "audit_log": logs}

def predict_lookup_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("audit_log", [])
    predict_catalog = {
        "Predict_File_Number": "102",
        "Predict_Entity_Name": "CUSTOMER-MASTER",
        "Field_Mappings": {"AA": "CUSTOMER-ID", "AB": "ZIP-CODE", "AC": "CREDIT-BALANCE"}
    }
    logs.append("[STAGE 3: PREDICT] Data Dictionary lookup resolved metadata mapping.")
    return {"predict_metadata": predict_catalog, "audit_log": logs}

# --- BRANCH 1: CONNX (READ ONLY) ---
def connx_read_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    connx_query = f"""SELECT ISN, ACCOUNT_NAME, ZIP_CODE, CREDIT_BALANCE 
FROM CONNX_ADABAS.VW_COMMERCIAL_ACCOUNTS 
WHERE ZIP_CODE = '{intent.get("zip_code_range", "N/A")}';"""

    df = pd.DataFrame({
        "ISN": [1041, 1042, 1043],
        "Account": ["Corp_A", "Corp_B", "Corp_C"],
        "ZIP": [intent.get("zip_code_range")] * 3,
        "Credit_Balance": [0.0, 150.0, 50.0]
    })
    
    logs.append("[AGENT DECISION] Routed to CONNX Data Gateway for Direct SQL Read.")
    return {"routing_decision": "CONNX", "connx_sql": connx_query, "before_df": df, "audit_log": logs}

# --- BRANCH 2: ENTIREX + NATURAL (TRANSACTIONAL WRITE) ---
def entirex_rpc_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    payload = {
        "RpcServer": "SRV1",
        "Library": "FINANCE",
        "Program": "APPLY-CREDIT",
        "Parameters": {
            "p_zip_code": intent.get("zip_code_range"),
            "p_credit_amount": intent.get("amount")
        }
    }
    
    natural_code = f"""* SOFTWARE AG NATURAL 4GL PROGRAM: APPLY-CREDIT
DEFINE DATA PARAMETER
1 P-ZIP-CODE (A5)
1 P-CREDIT-AMOUNT (P10.2)
LOCAL
1 CUST-VIEW VIEW OF ADABAS_FILE_102
END-DEFINE
FIND CUST-VIEW WITH ZIP-CODE = P-ZIP-CODE
  ADD P-CREDIT-AMOUNT TO CREDIT-BALANCE
  UPDATE
  WRITE LOG TO FILE 105 /* Mandatory Audit Trail
END-FIND
END TRANSACTION
END"""

    logs.append("[AGENT DECISION] Routed to EntireX API Gateway for legacy business logic execution.")
    return {"routing_decision": "ENTIREX", "entirex_payload": payload, "natural_code": natural_code, "audit_log": logs}

def simulate_and_diff_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    before_data = pd.DataFrame({
        "ISN": [1041, 1042, 1043],
        "Account": ["Corp_A", "Corp_B", "Corp_C"],
        "ZIP": [intent.get("zip_code_range")] * 3,
        "Credit_Balance": [0.0, 150.0, 50.0]
    })
    
    after_data = before_data.copy()
    if state.get("compliance_passed"):
        after_data["Credit_Balance"] = after_data["Credit_Balance"] + intent.get("amount", 0)

    adabas_impact = {
        "Target_File": "FILE-102 & FILE-105",
        "Estimated_Cost": "0.05 SU",
        "Lock_Type": "EXCLUSIVE WRITE"
    }

    logs.append("[STAGE 5: SIMULATOR] Adabas database impact computed.")
    return {"adabas_impact": adabas_impact, "before_df": before_data, "after_df": after_data, "audit_log": logs}

def commit_node(state: AgentState) -> Dict[str, Any]:
    approved = state.get("human_approved", False)
    logs = state.get("audit_log", [])

    if not approved:
        logs.append("[STAGE 6: COMMIT] Operator REJECTED EntireX transaction.")
        return {"commit_status": "❌ ABORTED: Transaction rejected.", "audit_log": logs}

    logs.append("[STAGE 6: COMMIT] EntireX RPC Call Executed Successfully.")
    return {"commit_status": "✅ COMMITTED: EntireX -> Natural -> Adabas transaction finalized.", "audit_log": logs}

# ==========================================
# 5. LangGraph Agent Orchestration (Conditional Routing)
# ==========================================
def route_intent(state: AgentState) -> str:
    # Autonomous decision making based on Intent
    if state["parsed_intent"]["action_type"] == "query":
        return "connx_read"
    return "entirex_rpc"

@st.cache_resource
def build_langgraph_pipeline():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_and_compliance", parse_and_compliance_node)
    workflow.add_node("predict_lookup", predict_lookup_node)
    workflow.add_node("connx_read", connx_read_node)
    workflow.add_node("entirex_rpc", entirex_rpc_node)
    workflow.add_node("simulate_and_diff", simulate_and_diff_node)
    workflow.add_node("commit", commit_node)

    workflow.set_entry_point("parse_and_compliance")
    workflow.add_edge("parse_and_compliance", "predict_lookup")
    
    # Conditional Edges for Dynamic Routing
    workflow.add_conditional_edges("predict_lookup", route_intent, {
        "connx_read": "connx_read",
        "entirex_rpc": "entirex_rpc"
    })

    # Read path ends immediately after returning data
    workflow.add_edge("connx_read", END)
    
    # Write path flows to HitL simulation and commit
    workflow.add_edge("entirex_rpc", "simulate_and_diff")
    workflow.add_edge("simulate_and_diff", "commit")

    return workflow.compile()

langgraph_agent = build_langgraph_pipeline()

# ==========================================
# 6. Streamlit User Interface
# ==========================================
st.title("⚡ Dynamic Software AG Agentic AI Gateway")

def render_architecture():
    return """
    digraph {
        rankdir=LR;
        node [fontname="sans-serif", fontsize=9];
        
        A [label="Groq LLM Intent", shape=box, fillcolor="#e1f5fe", style=filled];
        B [label="Predict Dictionary", shape=cylinder, fillcolor="#fff9c4", style=filled];
        
        C [label="CONNX SQL\\n(Data Read Path)", shape=box, fillcolor="#e1bee7", style=filled];
        D [label="EntireX API\\n(Legacy App Write)", shape=box, fillcolor="#ffe0b2", style=filled];
        
        E [label="Natural 4GL", shape=box, fillcolor="#ffccbc", style=filled];
        F [label="Adabas Engine", shape=cylinder, fillcolor="#c8e6c9", style=filled];
        G [label="HITL Approval", shape=diamond, fillcolor="#f3e5f5", style=filled];
        
        A -> B;
        B -> C [label=" If Query (Analytics)"];
        B -> D [label=" If Update (Transaction)"];
        
        C -> F [label=" Direct SELECT", style="dashed"];
        
        D -> E [label=" Trigger RPC"];
        E -> G [label=" Simulation"];
        G -> F [label=" Authorized POST"];
    }
    """

st.graphviz_chart(render_architecture())

if st.button("Simulate AI Workflow", type="primary"):
    with st.spinner("Agent evaluating request intent and determining infrastructure path..."):
        initial_state: AgentState = {
            "raw_prompt": user_prompt, "parsed_intent": None, "compliance_passed": None,
            "compliance_reason": None, "predict_metadata": None, "routing_decision": None, 
            "connx_sql": None, "entirex_payload": None, "natural_code": None, 
            "adabas_impact": None, "before_df": None, "after_df": None, 
            "human_approved": None, "commit_status": None, "audit_log": []
        }
        
        # Execute until predict, then route
        s1 = parse_and_compliance_node(initial_state)
        s2 = predict_lookup_node({**initial_state, **s1})
        
        current_state = {**initial_state, **s1, **s2}
        
        if current_state["parsed_intent"]["action_type"] == "query":
            # Execute CONNX Read Path
            s3 = connx_read_node(current_state)
            st.session_state["txn_result"] = {**current_state, **s3}
        else:
            # Execute EntireX Write Path up to HITL pause
            s3 = entirex_rpc_node(current_state)
            s4 = simulate_and_diff_node({**current_state, **s3})
            st.session_state["pending_txn"] = {**current_state, **s3, **s4}

# --- VIEW FOR PATH 1: CONNX DATA READ ---
if "txn_result" in st.session_state:
    res = st.session_state["txn_result"]
    st.divider()
    st.info("🔄 **Agentic Routing Decision:** Recognized 'query' intent. Bypassing legacy application code and routed directly to **CONNX Data Gateway** for high-speed Adabas read.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generated SQL Abstraction")
        st.caption("⚙️ Component: CONNX SQL Gateway")
        st.code(res["connx_sql"], language="sql")
    with col2:
        st.subheader("Adabas Query Results")
        st.caption("⚙️ Retrieved natively via CONNX")
        st.dataframe(res["before_df"], hide_index=True, use_container_width=True)
        
    st.markdown("### Agent Audit Trail")
    st.code("\n".join(res["audit_log"]), language="text")
    
    if st.button("Reset Session"):
        del st.session_state["txn_result"]

# --- VIEW FOR PATH 2: ENTIREX TRANSACTION WRITE ---
if "pending_txn" in st.session_state:
    txn = st.session_state["pending_txn"]
    st.divider()
    
    if txn["compliance_passed"]:
        st.success(f"✅ **Enterprise Guardrails Cleared:** {txn['compliance_reason']}")
        st.warning("⚠️ **Agentic Routing Decision:** Recognized 'update' intent. Routed to **EntireX API** to strictly enforce Natural 4GL legacy business logic.")
    else:
        st.error(f"❌ **Policy Violation Detected:** {txn['compliance_reason']}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. EntireX RPC Translation")
        st.caption("⚙️ Component: webMethods EntireX API")
        st.json(txn["entirex_payload"])
        
        st.subheader("3. Database Impact Metrics")
        st.caption("⚙️ Component: Adabas Engine Simulator")
        st.json(txn["adabas_impact"])

    with col2:
        st.subheader("2. Executed Natural Program")
        st.caption("⚙️ Component: Software AG Natural")
        st.code(txn["natural_code"], language="text")
        
        st.subheader("4. Target Record Diff")
        st.caption("⚙️ Predicted Change via Natural Program")
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.caption("Current State")
            st.dataframe(txn["before_df"], hide_index=True)
        with diff_col2:
            st.caption("Predicted State")
            st.dataframe(txn["after_df"], hide_index=True)

    # Human-in-the-Loop Commit Gate
    st.divider()
    st.subheader("Human-in-the-Loop Gate")
    
    can_approve = txn["compliance_passed"]
    if txn["parsed_intent"].get("amount", 0) > 1000 and user_role != "System Administrator":
        st.warning("⚠️ High-value transactions require System Administrator role.")
        can_approve = False

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✅ Approve EntireX RPC Execution", use_container_width=True, disabled=not can_approve):
            txn["human_approved"] = True
            final_res = commit_node(txn)
            st.success(final_res["commit_status"])
            st.markdown("### Agent Audit Trail")
            st.code("\n".join(final_res["audit_log"]), language="text")
            del st.session_state["pending_txn"]

    with btn_col2:
        if st.button("❌ Reject Transaction", type="secondary", use_container_width=True):
            txn["human_approved"] = False
            final_res = commit_node(txn)
            st.error(final_res["commit_status"])
            st.markdown("### Agent Audit Trail")
            st.code("\n".join(final_res["audit_log"]), language="text")
            del st.session_state["pending_txn"]