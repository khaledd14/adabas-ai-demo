import streamlit as st
import time
import json
from typing import TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. MOCKS & SIMULATED ADABAS
# ==========================================

class PredictAPIMock:
    @staticmethod
    def get_field_mappings():
        time.sleep(0.3)
        return {
            "ddm_name": "COMMERCIAL-ACCOUNTS-DDM",
            "adabas_file_number": 12,
            "fields": {
                "account_type": {"fdt_code": "AT", "type": "A2"},
                "zip_code": {"fdt_code": "ZP", "type": "A5"},
                "balance": {"fdt_code": "BL", "type": "P9.2"},
                "credit_applied": {"fdt_code": "CR", "type": "P9.2"}
            }
        }

class CONNXSQLGatewayMock:
    @staticmethod
    def simulate_credit_update(zip_start, zip_end, credit_pct):
        time.sleep(0.5)
        records = [
            {"isn": 101, "acct_id": "C-75001-A", "current_bal": 1250.00, "credit_amount": 1250.00 * (credit_pct / 100)},
            {"isn": 104, "acct_id": "C-75003-B", "current_bal": 8400.50, "credit_amount": 8400.50 * (credit_pct / 100)},
            {"isn": 112, "acct_id": "C-75009-F", "current_bal": 3100.00, "credit_amount": 3100.00 * (credit_pct / 100)},
        ]
        return {
            "affected_isn_count": len(records),
            "matched_records": records,
            "total_credit_impact_usd": round(sum(r["credit_amount"] for r in records), 2)
        }

class AdabasSimulator:
    @staticmethod
    def update_records(file_number, records):
        time.sleep(0.4)
        return {
            "status": "COMMITTED_TO_ADABAS",
            "adabas_dbid": 12,
            "file_number": file_number,
            "updated_isns": [r["isn"] for r in records],
            "transaction_id": "TX-ADABAS-982341"
        }

# ==========================================
# 2. LANGGRAPH AGENT WORKFLOW
# ==========================================

class ParsedIntent(BaseModel):
    account_type: str = Field(description="Account type e.g., 'commercial'")
    zip_start: int = Field(description="Start ZIP")
    zip_end: int = Field(description="End ZIP")
    credit_percentage: float = Field(description="Credit percent e.g. 15.0")

class AgentState(TypedDict):
    raw_prompt: str
    parsed_intent: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    simulation_results: Optional[Dict[str, Any]]
    human_approved: Optional[bool]
    execution_result: Optional[Dict[str, Any]]
    audit_log: Optional[Dict[str, Any]]

def parse_intent_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ParsedIntent)
    res = structured_llm.invoke(f"Extract params: {state['raw_prompt']}")
    return {"parsed_intent": res.model_dump()}

def discover_metadata_node(state: AgentState):
    return {"metadata": PredictAPIMock.get_field_mappings()}

def simulate_impact_node(state: AgentState):
    intent = state["parsed_intent"]
    sim = CONNXSQLGatewayMock.simulate_credit_update(
        intent["zip_start"], intent["zip_end"], intent["credit_percentage"]
    )
    return {"simulation_results": sim}

def execute_commit_node(state: AgentState):
    if not state.get("human_approved"):
        return {"execution_result": {"status": "REJECTED_BY_USER"}}
    sim = state["simulation_results"]
    res = AdabasSimulator.update_records(12, sim["matched_records"])
    return {"execution_result": res}

def audit_log_node(state: AgentState):
    exec_res = state.get("execution_result", {})
    intent = state.get("parsed_intent", {})
    return {
        "audit_log": {
            "event": "ADABAS_CREDIT_UPDATE",
            "status": exec_res.get("status"),
            "transaction_id": exec_res.get("transaction_id"),
            "target_file": 12,
            "records_affected": len(exec_res.get("updated_isns", [])),
            "summary": f"Applied {intent.get('credit_percentage')}% to ZIPs {intent.get('zip_start')}-{intent.get('zip_end')}"
        }
    }

@st.cache_resource
def get_graph():
    builder = StateGraph(AgentState)
    builder.add_node("parse_intent", parse_intent_node)
    builder.add_node("discover_metadata", discover_metadata_node)
    builder.add_node("simulate_impact", simulate_impact_node)
    builder.add_node("execute_commit", execute_commit_node)
    builder.add_node("audit_log", audit_log_node)

    builder.add_edge(START, "parse_intent")
    builder.add_edge("parse_intent", "discover_metadata")
    builder.add_edge("discover_metadata", "simulate_impact")
    builder.add_edge("execute_commit", "audit_log")
    builder.add_edge("audit_log", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["execute_commit"])

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Adabas AI Demo", layout="wide")
st.title("Adabas & Natural Agentic AI Demo")

thread_config = {"configurable": {"thread_id": "mobile_demo_1"}}
graph = get_graph()

user_prompt = st.text_input(
    "Command Prompt:", 
    value="Apply 15% credit to commercial accounts ZIP 75001-75010"
)

if st.button("Start Workflow", type="primary"):
    graph.invoke({
        "raw_prompt": user_prompt,
        "parsed_intent": None,
        "metadata": None,
        "simulation_results": None,
        "human_approved": None,
        "execution_result": None,
        "audit_log": None
    }, thread_config)

current_state = graph.get_state(thread_config)

if current_state.values:
    state = current_state.values

    if state.get("parsed_intent"):
        with st.expander("1. Parsed Intent (GPT-4o)", expanded=True):
            st.json(state["parsed_intent"])

    if state.get("metadata"):
        with st.expander("2. Predict Metadata Catalog", expanded=False):
            st.json(state["metadata"])

    if state.get("simulation_results"):
        with st.expander("3. CONNX SQL Simulation Impact", expanded=True):
            st.json(state["simulation_results"])

    # Human-in-the-loop Gate
    if current_state.next == ("execute_commit",):
        st.error("⚠️ HUMAN GATE: Approve writing to Adabas DBID 12?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Approve & Commit"):
            graph.update_state(thread_config, {"human_approved": True}, as_node="simulate_impact")
            graph.invoke(None, thread_config)
            st.rerun()
        if col2.button("❌ Reject"):
            graph.update_state(thread_config, {"human_approved": False}, as_node="simulate_impact")
            graph.invoke(None, thread_config)
            st.rerun()

    if state.get("execution_result"):
        with st.expander("4. Adabas Execution Result", expanded=True):
            st.json(state["execution_result"])

    if state.get("audit_log"):
        with st.expander("5. System Audit Log", expanded=True):
            st.json(state["audit_log"])