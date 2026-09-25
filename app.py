import os
import requests
import pandas as pd
import graphviz
import streamlit as st
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

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
# 2. Sidebar: RBAC & Demo Preset Prompts
# ==========================================
st.sidebar.title("🛡️ Enterprise Governance")
st.sidebar.caption("⚙️ Component: API Gateway / IAM Layer")
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
    discovered_resources: Optional[Dict[str, Any]]
    simulation_result: Optional[Dict[str, Any]]
    natural_code: Optional[str]
    before_df: Optional[pd.DataFrame]
    after_df: Optional[pd.DataFrame]
    human_approved: Optional[bool]
    commit_status: Optional[str]
    audit_log: List[str]

# ==========================================
# 4. Agent Pipeline Nodes
# ==========================================
def parse_and_compliance_node(state: AgentState) -> Dict[str, Any]:
    """Stage 1: Intent Parsing (Groq) & Stage 2: Compliance (Guardrails)"""
    logs = state.get("audit_log", [])
    
    # 1. Parse via LLM
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_key)
    structured_llm = llm.with_structured_output(ActionIntent)
    parsed: ActionIntent = structured_llm.invoke(f"Parse this request: '{state['raw_prompt']}'")
    intent = parsed.model_dump()
    logs.append("[STAGE 1: PARSE] Intent parsed via Groq AI Gateway.")

    # 2. Compliance Guardrails
    passed = True
    reason = "Transaction within safe limits."
    
    if intent.get("action_type") == "delete_purge":
        passed = False
        reason = "CRITICAL: 'delete_purge' actions are blocked by Enterprise Data Governance policies."
        logs.append(f"[STAGE 2: COMPLIANCE] ❌ REJECTED: {reason}")
    elif intent.get("amount", 0) > 5000:
        passed = False
        reason = "HIGH RISK: Transaction amount exceeds $5,000 auto-approval threshold."
        logs.append(f"[STAGE 2: COMPLIANCE] ⚠️ FLAGGED: {reason}")
    else:
        logs.append("[STAGE 2: COMPLIANCE] ✅ Cleared Enterprise Guardrails.")

    return {"parsed_intent": intent, "compliance_passed": passed, "compliance_reason": reason, "audit_log": logs}

def discover_resources_node(state: AgentState) -> Dict[str, Any]:
    """Stage 3: Resource Discovery (Software AG Predict)"""
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    discovered = {
        "Predict_Dictionary": "ADABAS_FILE_102_CUSTOMER",
        "CONNX_View": "VW_COMMERCIAL_ACCOUNTS",
        "Target_Zip": intent.get("zip_code_range", "N/A"),
        "Matched_Records": 3 # Hardcoded for demo diffs
    }
    logs.append("[STAGE 3: DISCOVERY] Metadata resolved via Software AG Predict Data Dictionary.")
    
    return {"discovered_resources": discovered, "audit_log": logs}

def simulate_and_diff_node(state: AgentState) -> Dict[str, Any]:
    """Stage 4: Adabas Simulation & Natural Code Gen"""
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    
    # Generate Natural 4GL Code snippet
    natural_code = f"""* GENERATED SOFTWARE AG NATURAL RPC
DEFINE DATA LOCAL
1 CUSTOMER-VIEW VIEW OF ADABAS_FILE_102
  2 ZIP-CODE
  2 CREDIT-BALANCE
END-DEFINE
FIND CUSTOMER-VIEW WITH ZIP-CODE = '{intent.get("zip_code_range")}'
  ADD {intent.get("amount")} TO CREDIT-BALANCE
  UPDATE
END-FIND
END TRANSACTION
END"""

    # Generate Data Diffs
    before_data = pd.DataFrame({
        "ISN (Adabas ID)": [1041, 1042, 1043],
        "Account": ["Corp_A", "Corp_B", "Corp_C"],
        "ZIP": [intent.get("zip_code_range")] * 3,
        "Credit_Balance": [0.0, 150.0, 50.0]
    })
    
    after_data = before_data.copy()
    if state.get("compliance_passed"):
        after_data["Credit_Balance"] = after_data["Credit_Balance"] + intent.get("amount", 0)

    sim_res = {
        "Status": "SIMULATION_COMPLETE",
        "Estimated_Cost": "0.03 SU (Service Units)"
    }
    logs.append("[STAGE 4: SIMULATION] Adabas impact calculated and Natural code generated.")

    return {
        "simulation_result": sim_res, 
        "natural_code": natural_code, 
        "before_df": before_data, 
        "after_df": after_data, 
        "audit_log": logs
    }

def commit_node(state: AgentState) -> Dict[str, Any]:
    """Stage 5: Commit Execution (Adabas REST)"""
    approved = state.get("human_approved", False)
    logs = state.get("audit_log", [])

    if not approved:
        logs.append("[STAGE 5: COMMIT] Human Operator REJECTED transaction.")
        return {"commit_status": "❌ ABORTED: Transaction rejected.", "audit_log": logs}

    if adabas_rest_url:
        try:
            payload = {"file": "ADABAS_FILE_102", "intent": state.get("parsed_intent")}
            response = requests.post(adabas_rest_url, json=payload, timeout=5)
            if response.status_code == 200:
                logs.append(f"[STAGE 5: COMMIT] Live execution via Software AG Adabas REST Server succeeded.")
                return {"commit_status": "✅ COMMITTED: Live Adabas REST HTTP transaction successful.", "audit_log": logs}
        except Exception as e:
            logs.append(f"[STAGE 5: COMMIT] REST error: {str(e)}. Falling back to Sandbox.")

    logs.append("[STAGE 5: COMMIT] Sandbox Adabas execution finalized.")
    return {"commit_status": "✅ COMMITTED: Sandbox Adabas transaction finalized.", "audit_log": logs}

# ==========================================
# 5. UI Rendering & Application Flow
# ==========================================
st.title("⚡ Adabas & Natural Agentic AI Gateway")

# Helper to render Architecture Diagram
def render_architecture():
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', size='8,4')
    graph.node('A', 'Groq LLM\n(AI Gateway)', shape='box', style='filled', fillcolor='#e1f5fe')
    graph.node('B', 'Software AG\nPredict', shape='cylinder', style='filled', fillcolor='#fff9c4')
    graph.node('C', 'Enterprise\nGuardrails', shape='diamond', style='filled', fillcolor='#ffe0b2')
    graph.node('D', 'HITL\nApproval', shape='box', style='filled', fillcolor='#f3e5f5')
    graph.node('E', 'Adabas REST\nServer', shape='cylinder', style='filled', fillcolor='#c8e6c9')
    
    graph.edge('A', 'C', label=' Intent')
    graph.edge('C', 'B', label=' Metadata')
    graph.edge('B', 'D', label=' Impact')
    graph.edge('D', 'E', label=' REST Payload')
    return graph

st.graphviz_chart(render_architecture())

# PHASE 1: Execution
if st.button("Simulate AI Workflow", type="primary"):
    with st.spinner("Processing through AI Gateway..."):
        initial_state: AgentState = {
            "raw_prompt": user_prompt, "parsed_intent": None, "compliance_passed": None,
            "compliance_reason": None, "discovered_resources": None, "simulation_result": None,
            "natural_code": None, "before_df": None, "after_df": None,
            "human_approved": None, "commit_status": None, "audit_log": []
        }
        
        s1 = parse_and_compliance_node(initial_state)
        s2 = discover_resources_node({**initial_state, **s1})
        s3 = simulate_and_diff_node({**initial_state, **s1, **s2})
        st.session_state["pending_txn"] = {**initial_state, **s1, **s2, **s3}

# PHASE 2: HITL Dashboard
if "pending_txn" in st.session_state:
    txn = st.session_state["pending_txn"]
    st.divider()
    
    # 🚨 Compliance Banner
    if txn["compliance_passed"]:
        st.success(f"✅ **Enterprise Guardrails Cleared:** {txn['compliance_reason']}")
    else:
        st.error(f"❌ **Policy Violation Detected:** {txn['compliance_reason']}")

    # 🗂️ Component Data Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Parsed Intent")
        st.caption("⚙️ **Component:** Groq Open-Weights LLM")
        st.json(txn["parsed_intent"])
        
        st.subheader("2. Generated Logic")
        st.caption("⚙️ **Component:** Software AG Natural 4GL")
        st.code(txn["natural_code"], language="text")

    with col2:
        st.subheader("3. Resource Discovery")
        st.caption("⚙️ **Component:** Software AG Predict Data Dictionary")
        st.json(txn["discovered_resources"])
        
        st.subheader("4. Target Table Diff")
        st.caption("⚙️ **Component:** Adabas Simulator Engine")
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.markdown("**Current Record State**")
            st.dataframe(txn["before_df"], hide_index=True, use_container_width=True)
        with diff_col2:
            st.markdown("**Predicted Post-Commit**")
            st.dataframe(txn["after_df"], hide_index=True, use_container_width=True)

    # 🛑 Execution & RBAC Controls
    st.divider()
    st.subheader("Action Approval")
    st.caption("⚙️ **Component:** API Gateway IAM")
    
    # Check permissions based on RBAC & Compliance
    can_approve = True
    if not txn["compliance_passed"]:
        st.warning("⚠️ Transaction violates governance policy. Approval disabled.")
        can_approve = False
    elif txn["parsed_intent"].get("amount", 0) > 1000 and user_role != "System Administrator":
        st.warning("⚠️ High transaction amount requires **System Administrator** role to approve.")
        can_approve = False

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✅ Approve & Commit to Adabas REST", use_container_width=True, disabled=not can_approve):
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