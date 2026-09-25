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
    page_title="Adabas & Natural AI Gateway",
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
st.sidebar.caption("⚙️ Component: Software AG API Gateway")
user_role = st.sidebar.selectbox("Active User Role", ["System Administrator", "Tier 1 Operator"])

st.sidebar.divider()
st.sidebar.title("🧪 Demo Scenarios")
preset_choice = st.sidebar.radio(
    "Load Test Prompt:",
    [
        "Standard Credit Request",
        "High-Risk / Limit Violation",
        "Security Guardrail (Malicious)",
        "Multilingual (Spanish)"
    ]
)

preset_prompts = {
    "Standard Credit Request": "Apply 100 credit to commercial accounts in ZIP 70001",
    "High-Risk / Limit Violation": "Apply 10000 credit to commercial accounts in Beverly Hills 90210",
    "Security Guardrail (Malicious)": "Purge and delete all customer records from File 102",
    "Multilingual (Spanish)": "Aplica 250 de crédito a las cuentas corporativas en el código postal 90210"
}

user_prompt = st.text_input("Command Prompt:", value=preset_prompts[preset_choice])

# ==========================================
# 3. Pydantic Models & State Definition
# ==========================================
class ActionIntent(BaseModel):
    action_type: str = Field(description="Action: 'apply_credit', 'update_record', 'query', or 'delete_purge'")
    target_account_type: str = Field(description="Account category: 'commercial', 'individual', or 'all'")
    zip_code_range: str = Field(description="Zip code range or region, e.g. '70001' or '90210'")
    amount: float = Field(description="Numeric value associated with action")

class AgentState(TypedDict):
    raw_prompt: str
    parsed_intent: Optional[Dict[str, Any]]
    compliance_passed: Optional[bool]
    compliance_reason: Optional[str]
    predict_metadata: Optional[Dict[str, Any]]
    connx_sql: Optional[str]
    natural_code: Optional[str]
    adabas_impact: Optional[Dict[str, Any]]
    before_df: Optional[pd.DataFrame]
    after_df: Optional[pd.DataFrame]
    human_approved: Optional[bool]
    commit_status: Optional[str]
    audit_log: List[str]

# ==========================================
# 4. Agent Pipeline Nodes (Software AG Components)
# ==========================================
def parse_and_compliance_node(state: AgentState) -> Dict[str, Any]:
    """Stage 1: Intent Parsing (Groq AI Gateway) & Policy Check"""
    logs = state.get("audit_log", [])
    
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_key)
    structured_llm = llm.with_structured_output(ActionIntent)
    parsed: ActionIntent = structured_llm.invoke(f"Parse this request: '{state['raw_prompt']}'")
    intent = parsed.model_dump()
    logs.append("[STAGE 1: PARSE] Intent parsed via Groq AI Gateway.")

    passed = True
    reason = "Transaction within safe operational boundaries."
    
    if intent.get("action_type") == "delete_purge":
        passed = False
        reason = "CRITICAL: 'delete_purge' operations blocked by Enterprise Data Governance."
        logs.append(f"[STAGE 2: GUARDRAILS] ❌ REJECTED: {reason}")
    elif intent.get("amount", 0) > 5000:
        passed = False
        reason = "HIGH RISK: Amount exceeds $5,000 threshold."
        logs.append(f"[STAGE 2: GUARDRAILS] ⚠️ FLAGGED: {reason}")
    else:
        logs.append("[STAGE 2: GUARDRAILS] ✅ Cleared Enterprise Guardrails.")

    return {"parsed_intent": intent, "compliance_passed": passed, "compliance_reason": reason, "audit_log": logs}

def predict_lookup_node(state: AgentState) -> Dict[str, Any]:
    """Stage 2: Metadata Catalog Lookup (Software AG Predict)"""
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    # Software AG Predict Data Dictionary Mapping
    predict_catalog = {
        "Predict_File_Number": "102",
        "Predict_Entity_Name": "CUSTOMER-MASTER",
        "Field_Mappings": {
            "AA": "CUSTOMER-ID (Numeric)",
            "AB": "ZIP-CODE (Alpha, 5-char)",
            "AC": "CREDIT-BALANCE (Numeric, Packed)"
        },
        "Access_Control": "READ/UPDATE-ALLOWED"
    }
    logs.append("[STAGE 3: PREDICT] Data Dictionary lookup resolved File 102 & Field Mappings (AA, AB, AC).")
    
    return {"predict_metadata": predict_catalog, "audit_log": logs}

def connx_sql_node(state: AgentState) -> Dict[str, Any]:
    """Stage 3: SQL Abstraction (CONNX SQL Gateway)"""
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    connx_query = f"""SELECT ISN, ACCOUNT_NAME, ZIP_CODE, CREDIT_BALANCE 
FROM CONNX_ADABAS.VW_COMMERCIAL_ACCOUNTS 
WHERE ZIP_CODE = '{intent.get("zip_code_range", "N/A")}' 
FOR UPDATE OF CREDIT_BALANCE;"""

    logs.append("[STAGE 4: CONNX] Generated ANSI SQL query based on Predict metadata.")
    
    return {"connx_sql": connx_query, "audit_log": logs}

def simulate_and_diff_node(state: AgentState) -> Dict[str, Any]:
    """Stage 4: Natural Code Gen & Adabas Database Impact Simulator"""
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    # Software AG Natural 4GL RPC Code
    natural_code = f"""* SOFTWARE AG NATURAL 4GL PROGRAM: UPDATE-CREDIT
DEFINE DATA LOCAL
1 CUST-VIEW VIEW OF ADABAS_FILE_102
  2 ZIP-CODE (A5)
  2 CREDIT-BALANCE (P10.2)
END-DEFINE
FIND CUST-VIEW WITH ZIP-CODE = '{intent.get("zip_code_range")}'
  ADD {intent.get("amount")} TO CREDIT-BALANCE
  UPDATE
END-FIND
END TRANSACTION
END"""

    # Adabas Simulator Output
    before_data = pd.DataFrame({
        "ISN (Adabas ID)": [1041, 1042, 1043],
        "Account": ["Corp_A", "Corp_B", "Corp_C"],
        "ZIP": [intent.get("zip_code_range")] * 3,
        "Credit_Balance": [0.0, 150.0, 50.0]
    })
    
    after_data = before_data.copy()
    if state.get("compliance_passed"):
        after_data["Credit_Balance"] = after_data["Credit_Balance"] + intent.get("amount", 0)

    adabas_impact = {
        "Target_Database_ID": "DBID-012 (Production Mainframe)",
        "Target_File": "FILE-102 (CUSTOMER-MASTER)",
        "Affected_ISN_Count": 3,
        "Estimated_CPU_Cost": "0.03 Service Units (SU)",
        "Lock_Type": "SHARED READ -> EXCLUSIVE WRITE"
    }
    logs.append("[STAGE 5: ADABAS SIMULATOR] Calculated database impact & generated Natural 4GL code.")

    return {
        "adabas_impact": adabas_impact, 
        "natural_code": natural_code, 
        "before_df": before_data, 
        "after_df": after_data, 
        "audit_log": logs
    }

def commit_node(state: AgentState) -> Dict[str, Any]:
    """Stage 5: Commit Execution via Adabas REST Gateway"""
    approved = state.get("human_approved", False)
    logs = state.get("audit_log", [])

    if not approved:
        logs.append("[STAGE 6: COMMIT] Human Operator REJECTED transaction.")
        return {"commit_status": "❌ ABORTED: Transaction rejected by operator.", "audit_log": logs}

    if adabas_rest_url:
        try:
            payload = {
                "file": "ADABAS_FILE_102",
                "predict": state.get("predict_metadata"),
                "intent": state.get("parsed_intent")
            }
            response = requests.post(adabas_rest_url, json=payload, timeout=5)
            if response.status_code == 200:
                logs.append("[STAGE 6: COMMIT] Live HTTP execution via Software AG Adabas REST Gateway succeeded.")
                return {"commit_status": "✅ COMMITTED: Live Adabas REST HTTP transaction successful.", "audit_log": logs}
        except Exception as e:
            logs.append(f"[STAGE 6: COMMIT] REST error: {str(e)}. Falling back to Sandbox.")

    logs.append("[STAGE 6: COMMIT] Sandbox Adabas execution finalized.")
    return {"commit_status": "✅ COMMITTED: Sandbox Adabas transaction finalized.", "audit_log": logs}

# ==========================================
# 5. LangGraph Workflow Compilation
# ==========================================
@st.cache_resource
def build_langgraph_pipeline():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_and_compliance", parse_and_compliance_node)
    workflow.add_node("predict_lookup", predict_lookup_node)
    workflow.add_node("connx_sql", connx_sql_node)
    workflow.add_node("simulate_and_diff", simulate_and_diff_node)
    workflow.add_node("commit", commit_node)

    workflow.set_entry_point("parse_and_compliance")
    workflow.add_edge("parse_and_compliance", "predict_lookup")
    workflow.add_edge("predict_lookup", "connx_sql")
    workflow.add_edge("connx_sql", "simulate_and_diff")
    workflow.add_edge("simulate_and_diff", "commit")
    workflow.add_edge("commit", END)

    return workflow.compile()

langgraph_agent = build_langgraph_pipeline()

# ==========================================
# 6. Streamlit User Interface
# ==========================================
st.title("⚡ Adabas & Natural Agentic AI Gateway")

# Precise Software AG Architecture Flow Diagram
def render_architecture():
    return """
    digraph {
        rankdir=LR;
        node [fontname="sans-serif", fontsize=9];
        edge [fontname="sans-serif", fontsize=8];
        
        A [label="Groq LLM\\n(AI Gateway)", shape=box, style=filled, fillcolor="#e1f5fe"];
        B [label="Software AG\\nPredict", shape=cylinder, style=filled, fillcolor="#fff9c4"];
        C [label="CONNX SQL\\nGateway", shape=box, style=filled, fillcolor="#e1bee7"];
        D [label="Software AG\\nNatural 4GL", shape=box, style=filled, fillcolor="#ffe0b2"];
        E [label="HITL Approval\\nGate", shape=diamond, style=filled, fillcolor="#f3e5f5"];
        F [label="Adabas REST\\nServer", shape=cylinder, style=filled, fillcolor="#c8e6c9"];
        
        A -> B [label=" Parse Intent"];
        B -> C [label=" Dictionary Lookup"];
        C -> D [label=" SQL Abstraction"];
        D -> E [label=" 4GL Simulation"];
        E -> F [label=" Approved REST POST"];
    }
    """

st.graphviz_chart(render_architecture())

# PHASE 1: Execution
if st.button("Simulate AI Workflow", type="primary"):
    with st.spinner("Running Software AG Pipeline through LangGraph..."):
        initial_state: AgentState = {
            "raw_prompt": user_prompt, "parsed_intent": None, "compliance_passed": None,
            "compliance_reason": None, "predict_metadata": None, "connx_sql": None,
            "natural_code": None, "adabas_impact": None, "before_df": None,
            "after_df": None, "human_approved": None, "commit_status": None, "audit_log": []
        }
        
        s1 = parse_and_compliance_node(initial_state)
        s2 = predict_lookup_node({**initial_state, **s1})
        s3 = connx_sql_node({**initial_state, **s1, **s2})
        s4 = simulate_and_diff_node({**initial_state, **s1, **s2, **s3})
        
        st.session_state["pending_txn"] = {**initial_state, **s1, **s2, **s3, **s4}

# PHASE 2: Human-in-the-Loop Software AG Component Grid
if "pending_txn" in st.session_state:
    txn = st.session_state["pending_txn"]
    st.divider()
    
    if txn["compliance_passed"]:
        st.success(f"✅ **Enterprise Guardrails Cleared:** {txn['compliance_reason']}")
    else:
        st.error(f"❌ **Policy Violation Detected:** {txn['compliance_reason']}")

    # Clear 2x2 Grid for Software AG Components
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. AI Intent Extraction")
        st.caption("⚙️ **Component:** Groq LLM (`openai/gpt-oss-20b`)")
        st.json(txn["parsed_intent"])
        
        st.subheader("3. SQL Abstraction Query")
        st.caption("⚙️ **Component:** CONNX SQL Gateway")
        st.code(txn["connx_sql"], language="sql")

    with col2:
        st.subheader("2. Data Dictionary Lookup")
        st.caption("⚙️ **Component:** Software AG Predict Catalog")
        st.json(txn["predict_metadata"])
        
        st.subheader("4. Generated 4GL Transaction Logic")
        st.caption("⚙️ **Component:** Software AG Natural")
        st.code(txn["natural_code"], language="text")

    st.divider()
    st.subheader("5. Database Impact & Record Diff")
    st.caption("⚙️ **Component:** Software AG Adabas Engine & Simulator")
    
    adabas_col1, adabas_col2 = st.columns([1, 2])
    with adabas_col1:
        st.markdown("**System Impact Metrics**")
        st.json(txn["adabas_impact"])
    with adabas_col2:
        st.markdown("**Target Record State (Before vs Predicted After)**")
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.caption("Current Record State")
            st.dataframe(txn["before_df"], hide_index=True, use_container_width=True)
        with diff_col2:
            st.caption("Predicted Post-Commit")
            st.dataframe(txn["after_df"], hide_index=True, use_container_width=True)

    # PHASE 3: Commit Gate
    st.divider()
    st.subheader("Action Approval Gate")
    st.caption("⚙️ **Component:** Software AG API Gateway / IAM")
    
    can_approve = True
    if not txn["compliance_passed"]:
        st.warning("⚠️ Transaction violates governance policy. Approval disabled.")
        can_approve = False
    elif txn["parsed_intent"].get("amount", 0) > 1000 and user_role != "System Administrator":
        st.warning("⚠️ High-value transactions require **System Administrator** role.")
        can_approve = False

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✅ Approve & Commit to Adabas REST Gateway", use_container_width=True, disabled=not can_approve):
            txn["human_approved"] = True
            final_res = commit_node(txn)
            st.success(final_res["commit_status"])
            st.markdown("### Enterprise Audit Trail")
            st.code("\n".join(final_res["audit_log"]), language="text")
            del st.session_state["pending_txn"]

    with btn_col2:
        if st.button("❌ Reject Transaction", type="secondary", use_container_width=True):
            txn["human_approved"] = False
            final_res = commit_node(txn)
            st.error(final_res["commit_status"])
            st.markdown("### Enterprise Audit Trail")
            st.code("\n".join(final_res["audit_log"]), language="text")
            del st.session_state["pending_txn"]
