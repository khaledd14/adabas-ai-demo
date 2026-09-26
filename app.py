import os
import uuid
import random
import requests
import pandas as pd
import streamlit as st
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # newer langgraph renamed this
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver

# ==========================================
# STATUS LEGEND (kept honest, matches the architecture diagram in the deck)
# ------------------------------------------
# REAL      : Groq LLM intent parsing, LangGraph graph execution + checkpointing
#             (native .invoke()/interrupt()/Command(resume=...), not manual python if/else)
# MOCKED    : Predict lookup, CONNX SQL Gateway, EntireX RPC payload, Natural source text
#             (Software AG licensed products not reachable from this environment)
# CONDITIONAL: Adabas commit — LIVE if ADABAS_REST_URL points at a reachable endpoint
#             (e.g. a small REST bridge in front of Adabas Community Edition),
#             otherwise clearly labeled SIMULATED. Never silently faked as committed.
# ==========================================

# ==========================================
# 1. Page Setup & Configuration
# ==========================================
st.set_page_config(
    page_title="Software AG Agentic AI Gateway",
    page_icon="⚡",
    layout="wide"
)


def get_secret(key: str) -> str:
    val = os.environ.get(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return ""


groq_key = get_secret("GROQ_API_KEY")
adabas_rest_url = get_secret("ADABAS_REST_URL")

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
if adabas_rest_url:
    st.sidebar.success(f"Adabas commit: LIVE → {adabas_rest_url}")
else:
    st.sidebar.info("Adabas commit: SIMULATED (set ADABAS_REST_URL for a live write)")

st.sidebar.divider()
st.sidebar.title("🧪 Agentic Routing Scenarios")
st.sidebar.info("LangGraph's own conditional edges choose the CONNX or EntireX path — this build actually invokes the compiled graph, it no longer replays the logic in Python.")

preset_choice = st.sidebar.radio(
    "Load Test Prompt:",
    [
        "Read-Only Analytics (Routes to CONNX)",
        "Transactional Update (Routes to EntireX)",
        "Security Violation (Blocked)"
    ]
)

preset_prompts = {
    "Read-Only Analytics (Routes to CONNX)": "Show me the credit balances for commercial accounts in ZIP codes 75001-75010",
    "Transactional Update (Routes to EntireX)": "Apply a 15% emergency bill credit to all commercial accounts in ZIP codes 75001-75010 for October 2026",
    "Security Violation (Blocked)": "Purge all customer records from File 102",
}

user_prompt = st.text_input("Command Prompt:", value=preset_prompts[preset_choice])


# ==========================================
# 3. Pydantic Models & State Definition
# ==========================================
class ActionIntent(BaseModel):
    action_type: str = Field(description="Action: 'query', 'apply_credit', 'update_record', or 'delete_purge'")
    target_account_type: str = Field(description="Account category: 'commercial', 'individual', or 'all'")
    zip_code_range: str = Field(description="A single ZIP code or a range like '75001-75010'")
    credit_percent: float = Field(default=0, description="Percentage credit requested, 0 if this is not a percentage-based request")
    amount: float = Field(default=0, description="Flat dollar amount, 0 if this is a query or a percentage-based request")


class AgentState(TypedDict):
    raw_prompt: str
    parsed_intent: Optional[Dict[str, Any]]
    compliance_passed: Optional[bool]
    compliance_reason: Optional[str]
    predict_metadata: Optional[Dict[str, Any]]
    routing_decision: Optional[str]
    connx_sql: Optional[str]
    entirex_payload: Optional[Dict[str, Any]]
    natural_code: Optional[str]
    adabas_impact: Optional[Dict[str, Any]]
    before_df: Optional[List[Dict[str, Any]]]  # DataFrame stored as list of dicts for msgpack serialization
    after_df: Optional[List[Dict[str, Any]]]   # DataFrame stored as list of dicts for msgpack serialization
    human_approved: Optional[bool]
    commit_status: Optional[str]
    audit_log: List[str]


# ==========================================
# 4. Shared mock data helpers
#    (used by BOTH the read path and the write path's impact simulation,
#     so the two never disagree with each other)
# ==========================================
def _expand_zip_range(zip_range: str) -> List[str]:
    zip_range = (zip_range or "").strip()
    if "-" in zip_range:
        start, end = zip_range.split("-", 1)
        try:
            start_i, end_i = int(start), int(end)
            if 0 < (end_i - start_i) < 100:
                return [str(z) for z in range(start_i, end_i + 1)]
        except ValueError:
            pass
    return [zip_range] if zip_range else ["00000"]


def _mock_connx_accounts(zip_range: str, seed_salt: str = "") -> pd.DataFrame:
    """MOCKED — stands in for a real Adabas SQL Gateway (CONNX) query until that
    licensed product is reachable from this environment. Deterministic per
    zip_range so repeated queries in front of an audience return the same rows."""
    rng = random.Random(f"{zip_range}|{seed_salt}")
    zips = _expand_zip_range(zip_range)
    step = max(1, len(zips) // 6)
    sample_zips = zips[::step][:6] or [zip_range]
    rows = []
    for i, z in enumerate(sample_zips):
        rows.append({
            "ISN": 1041 + i,
            "Account": f"Corp_{chr(65 + i)}",
            "ZIP": z,
            "Credit_Balance": round(rng.uniform(50, 4000), 2),
        })
    return pd.DataFrame(rows)


# ==========================================
# 5. Agent Pipeline Nodes
# ==========================================
def parse_and_compliance_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("audit_log", [])

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_key)
    structured_llm = llm.with_structured_output(ActionIntent)
    parsed: ActionIntent = structured_llm.invoke(f"Parse this request: '{state['raw_prompt']}'")
    intent = parsed.model_dump()
    logs.append(f"[STAGE 1: PARSE] Intent parsed via Groq LLM (real call): {intent['action_type']}")

    passed = True
    reason = "Transaction within safe operational boundaries."

    if intent.get("action_type") == "delete_purge":
        passed = False
        reason = "CRITICAL: 'delete_purge' operations blocked by Enterprise Governance."
    elif intent.get("amount", 0) > 5000:
        passed = False
        reason = "HIGH RISK: Flat amount exceeds $5,000 threshold."
    elif intent.get("credit_percent", 0) > 50:
        passed = False
        reason = "HIGH RISK: Credit percentage exceeds 50% threshold."

    if passed:
        logs.append("[STAGE 2: GUARDRAILS] ✅ Cleared Enterprise Guardrails.")
    else:
        logs.append(f"[STAGE 2: GUARDRAILS] ❌ REJECTED: {reason}")

    return {"parsed_intent": intent, "compliance_passed": passed, "compliance_reason": reason, "audit_log": logs}


def blocked_node(state: AgentState) -> Dict[str, Any]:
    """Halts the run immediately — a rejected request never reaches EntireX or
    generates a Natural program, unlike a design that only disables a button
    at the very end."""
    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 3: HALT] Pipeline stopped before Predict/EntireX — {state.get('compliance_reason')}")
    return {"commit_status": "BLOCKED: request never reached EntireX", "audit_log": logs}


def predict_lookup_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("audit_log", [])
    predict_catalog = {
        "Predict_File_Number": "102",
        "Predict_Entity_Name": "CUSTOMER-MASTER",
        "Field_Mappings": {"AA": "CUSTOMER-ID", "AB": "ZIP-CODE", "AC": "CREDIT-BALANCE"},
    }
    logs.append("[STAGE 4: PREDICT] MOCKED data-dictionary lookup (static catalog, not a live Predict query).")
    return {"predict_metadata": predict_catalog, "audit_log": logs}


def route_intent(state: AgentState) -> str:
    if state["parsed_intent"]["action_type"] == "query":
        return "connx_read"
    return "entirex_prepare"


# --- BRANCH 1: CONNX (READ ONLY) ---
def connx_read_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    zip_range = intent.get("zip_code_range", "N/A")
    bounds = zip_range.split("-")

    connx_query = f"""SELECT ISN, ACCOUNT_NAME, ZIP_CODE, CREDIT_BALANCE
FROM CONNX_ADABAS.VW_COMMERCIAL_ACCOUNTS
WHERE ZIP_CODE BETWEEN '{bounds[0]}' AND '{bounds[-1]}';"""

    df = _mock_connx_accounts(zip_range, seed_salt="read")

    logs.append("[STAGE 5: CONNX] MOCKED read-only SQL Gateway query executed — no write possible on this path.")
    return {"routing_decision": "CONNX", "connx_sql": connx_query, "before_df": df.to_dict(orient="records"), "audit_log": logs}


# --- BRANCH 2: ENTIREX + NATURAL (TRANSACTIONAL WRITE, PREPARE ONLY) ---
def entirex_prepare_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("parsed_intent", {})
    logs = state.get("audit_log", [])
    zip_range = intent.get("zip_code_range", "N/A")
    bounds = zip_range.split("-")

    # Same mock CONNX helper the read path uses, so the write path's "before"
    # picture is never a second, disconnected set of hardcoded numbers.
    before_df = _mock_connx_accounts(zip_range, seed_salt="write")
    after_df = before_df.copy()

    pct = intent.get("credit_percent") or 0
    flat = intent.get("amount") or 0
    if pct:
        after_df["Credit_Balance"] = (after_df["Credit_Balance"] * (1 + pct / 100)).round(2)
    else:
        after_df["Credit_Balance"] = (after_df["Credit_Balance"] + flat).round(2)

    sample_delta = float((after_df["Credit_Balance"] - before_df["Credit_Balance"]).mean())
    zips = _expand_zip_range(zip_range)
    estimated_total_accounts = max(len(zips) * 350, len(before_df))
    estimated_total_cost = round(sample_delta * estimated_total_accounts, 2)

    payload = {
        "RpcServer": "SRV1",
        "Library": "FINANCE",
        "Program": "APPLY-CREDIT",
        "Parameters": {
            "p_zip_range": zip_range,
            "p_credit_percent": pct,
            "p_credit_amount": flat,
        },
    }

    natural_code = f"""* SOFTWARE AG NATURAL 4GL PROGRAM: APPLY-CREDIT (illustrative — not compiled or run)
DEFINE DATA PARAMETER
1 P-ZIP-LOW (A5)
1 P-ZIP-HIGH (A5)
1 P-CREDIT-PERCENT (N3.2)
LOCAL
1 CUST-VIEW VIEW OF CUSTOMER-BILLING-VIEW
  2 ZIP-CODE (A5)
  2 CREDIT-BALANCE (P10.2)
END-DEFINE
READ CUST-VIEW BY ZIP-CODE STARTING FROM P-ZIP-LOW
  IF ZIP-CODE > P-ZIP-HIGH
    ESCAPE BOTTOM
  END-IF
  COMPUTE CREDIT-BALANCE = CREDIT-BALANCE * (1 + P-CREDIT-PERCENT / 100)
  UPDATE
  WRITE LOG TO FILE 105 /* Mandatory Audit Trail
END-READ
END TRANSACTION
END"""

    impact = {
        "target_file": "FILE-102 & FILE-105",
        "sample_accounts_shown": int(len(before_df)),
        "estimated_total_accounts": int(estimated_total_accounts),
        "estimated_total_cost_usd": estimated_total_cost,
        "lock_type": "EXCLUSIVE WRITE",
    }

    logs.append("[STAGE 6: ENTIREX] MOCKED RPC payload + Natural source built (not executed yet).")
    logs.append("[STAGE 6: SIMULATE] Impact re-derived from the same CONNX mock as the read path.")

    return {
        "routing_decision": "ENTIREX",
        "entirex_payload": payload,
        "natural_code": natural_code,
        "before_df": before_df.to_dict(orient="records"),
        "after_df": after_df.to_dict(orient="records"),
        "adabas_impact": impact,
        "audit_log": logs,
    }


def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """REAL LangGraph human-in-the-loop gate. interrupt() pauses execution and
    checkpoints state; the graph resumes here (re-running this function from
    the top) once the caller sends Command(resume=True/False)."""
    payload = {
        "impact": state.get("adabas_impact", {}),
        "before": state.get("before_df") or [],  # Already list-of-dicts from the node
        "after": state.get("after_df") or [],    # Already list-of-dicts from the node
        "rpc_payload": state.get("entirex_payload", {}),
    }
    decision = interrupt(payload)

    logs = state.get("audit_log", [])
    logs.append(f"[STAGE 7: HITL] Operator {'APPROVED' if decision else 'REJECTED'} the transaction.")
    return {"human_approved": bool(decision), "audit_log": logs}


def commit_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("audit_log", [])

    if not state.get("human_approved"):
        logs.append("[STAGE 8: COMMIT] Transaction aborted — not approved.")
        return {"commit_status": "ABORTED: transaction rejected", "audit_log": logs}

    payload = state.get("entirex_payload", {})

    if adabas_rest_url:
        try:
            resp = requests.post(f"{adabas_rest_url.rstrip('/')}/apply-credit", json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            logs.append(f"[STAGE 8: COMMIT] LIVE write via {adabas_rest_url} → {data}")
            return {"commit_status": f"COMMITTED (live Adabas): {data}", "audit_log": logs}
        except requests.exceptions.RequestException as exc:
            logs.append(f"[STAGE 8: COMMIT] Live Adabas call failed ({exc}); falling back to simulation.")

    logs.append("[STAGE 8: COMMIT] SIMULATED — EntireX/Natural/Adabas were not actually called.")
    return {"commit_status": "SIMULATED COMMIT (no live Adabas endpoint reachable)", "audit_log": logs}


# ==========================================
# 6. LangGraph Agent Orchestration
#    — this graph is actually invoked below (Section 8), not hand-replayed.
# ==========================================
@st.cache_resource
def build_langgraph_pipeline():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_and_compliance", parse_and_compliance_node)
    workflow.add_node("blocked", blocked_node)
    workflow.add_node("predict_lookup", predict_lookup_node)
    workflow.add_node("connx_read", connx_read_node)
    workflow.add_node("entirex_prepare", entirex_prepare_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("commit", commit_node)

    workflow.set_entry_point("parse_and_compliance")

    workflow.add_conditional_edges(
        "parse_and_compliance",
        lambda s: "blocked" if not s.get("compliance_passed") else "predict_lookup",
        {"blocked": "blocked", "predict_lookup": "predict_lookup"},
    )
    workflow.add_edge("blocked", END)

    workflow.add_conditional_edges(
        "predict_lookup",
        route_intent,
        {"connx_read": "connx_read", "entirex_prepare": "entirex_prepare"},
    )
    workflow.add_edge("connx_read", END)

    workflow.add_edge("entirex_prepare", "human_approval")
    workflow.add_edge("human_approval", "commit")
    workflow.add_edge("commit", END)

    return workflow.compile(checkpointer=MemorySaver())


langgraph_agent = build_langgraph_pipeline()


# ==========================================
# 7. Architecture diagram
# ==========================================
def render_architecture():
    return """
    digraph {
        rankdir=LR;
        node [fontname="sans-serif", fontsize=9];

        A [label="Groq LLM Intent\\n(real)", shape=box, style=filled, fillcolor="#c8e6c9"];
        Z [label="Guardrails", shape=diamond, style=filled, fillcolor="#eeeeee"];
        BL [label="Blocked\\n(halts here)", shape=box, style=filled, fillcolor="#ffcdd2"];
        B [label="Predict Dictionary\\n(mocked)", shape=cylinder, style=filled, fillcolor="#fff9c4"];

        C [label="CONNX SQL\\n(mocked read)", shape=box, style=filled, fillcolor="#e1bee7"];
        D [label="EntireX + Natural\\n(mocked)", shape=box, style=filled, fillcolor="#ffe0b2"];

        G [label="HITL Approval\\n(real LangGraph interrupt)", shape=diamond, style=filled, fillcolor="#f3e5f5"];
        F [label="Adabas Engine\\nlive if ADABAS_REST_URL set,\\nelse simulated", shape=cylinder, style=filled, fillcolor="#c8e6c9"];

        A -> Z;
        Z -> BL [label=" fails"];
        Z -> B [label=" passes"];
        B -> C [label=" query"];
        B -> D [label=" write"];
        C -> F [label=" read-only", style=dashed];
        D -> G;
        G -> F [label=" approved"];
    }
    """


# ==========================================
# 8. Streamlit User Interface
# ==========================================
st.title("⚡ Dynamic Software AG Agentic AI Gateway")
st.caption(
    "Real: Groq intent parsing · LangGraph execution & checkpointing. "
    "Mocked: Predict, CONNX, EntireX, Natural. "
    "Adabas: live if ADABAS_REST_URL is reachable, otherwise a clearly-labeled simulation."
)
st.graphviz_chart(render_architecture())

run_col, reset_col = st.columns([3, 1])
with run_col:
    run_clicked = st.button("Simulate AI Workflow", type="primary", use_container_width=True)
with reset_col:
    if st.button("Reset Session", use_container_width=True):
        st.session_state.pop("thread_id", None)
        st.session_state.pop("graph_state", None)
        st.rerun()

if run_clicked:
    thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = thread_id
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "raw_prompt": user_prompt, "parsed_intent": None, "compliance_passed": None,
        "compliance_reason": None, "predict_metadata": None, "routing_decision": None,
        "connx_sql": None, "entirex_payload": None, "natural_code": None,
        "adabas_impact": None, "before_df": None, "after_df": None,
        "human_approved": None, "commit_status": None, "audit_log": [],
    }
    with st.spinner("Invoking the compiled LangGraph pipeline..."):
        result = langgraph_agent.invoke(initial_state, config=config)
    st.session_state["graph_state"] = result

state = st.session_state.get("graph_state")

if state:
    thread_config = {"configurable": {"thread_id": st.session_state.get("thread_id", "")}}
    interrupts = state.get("__interrupt__")

    # --- PENDING: graph is paused at the real LangGraph interrupt ---
    if interrupts:
        st.divider()
        st.warning("LangGraph has interrupted this run and checkpointed its state — waiting on a human decision.")

        impact = state.get("adabas_impact", {}) or {}
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. EntireX RPC translation")
            st.caption("⚙️ Component: EntireX API Gateway — MOCKED")
            st.json(state.get("entirex_payload", {}))

            st.subheader("3. Database impact (simulated)")
            st.caption("⚙️ Derived from the same CONNX mock as the read path")
            st.json(impact)
        with col2:
            st.subheader("2. Generated Natural program")
            st.caption("⚙️ Component: Software AG Natural — illustrative source, not compiled")
            st.code(state.get("natural_code", ""), language="text")

            st.subheader("4. Sample account diff")
            st.caption(f"Showing {impact.get('sample_accounts_shown', 0)} of an estimated {impact.get('estimated_total_accounts', 0):,} affected accounts")
            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                st.caption("Current State")
                before_data = state.get("before_df")
                if before_data:
                    st.dataframe(pd.DataFrame(before_data), hide_index=True)
            with diff_col2:
                st.caption("Predicted State")
                after_data = state.get("after_df")
                if after_data:
                    st.dataframe(pd.DataFrame(after_data), hide_index=True)

        st.divider()
        st.subheader("Human-in-the-Loop Gate")

        est_cost = impact.get("estimated_total_cost_usd", 0)
        can_approve = True
        if est_cost > 1000 and user_role != "System Administrator":
            st.warning(f"⚠️ Estimated impact (${est_cost:,.2f}) requires System Administrator role to approve.")
            can_approve = False

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Approve — resume LangGraph", use_container_width=True, disabled=not can_approve):
                with st.spinner("Resuming from checkpoint..."):
                    result = langgraph_agent.invoke(Command(resume=True), config=thread_config)
                st.session_state["graph_state"] = result
                st.rerun()
        with btn_col2:
            if st.button("❌ Reject Transaction", type="secondary", use_container_width=True):
                with st.spinner("Resuming from checkpoint..."):
                    result = langgraph_agent.invoke(Command(resume=False), config=thread_config)
                st.session_state["graph_state"] = result
                st.rerun()

    # --- READ PATH: ended at CONNX, no approval needed ---
    elif state.get("routing_decision") == "CONNX":
        st.divider()
        st.info("🔄 Routed to the read-only CONNX path — no write, no approval gate needed.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generated SQL Abstraction")
            st.caption("⚙️ Component: CONNX SQL Gateway — MOCKED")
            st.code(state.get("connx_sql", ""), language="sql")
        with col2:
            st.subheader("Adabas Query Results (simulated)")
            before_data = state.get("before_df")
            if before_data:
                st.dataframe(pd.DataFrame(before_data), hide_index=True, use_container_width=True)
            else:
                st.write("No data")

        st.markdown("### Agent Audit Trail")
        st.code("\n".join(state.get("audit_log", [])), language="text")

    # --- TERMINAL: blocked, aborted, or committed ---
    elif state.get("commit_status") is not None:
        st.divider()
        status = state["commit_status"]
        if status.startswith("COMMITTED"):
            st.success(status)
        elif status.startswith("BLOCKED"):
            st.error(f"{status} — {state.get('compliance_reason', '')}")
        elif status.startswith("SIMULATED"):
            st.warning(status)
        else:
            st.info(status)

        st.markdown("### Agent Audit Trail")
        st.code("\n".join(state.get("audit_log", [])), language="text")