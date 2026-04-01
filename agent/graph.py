"""
LangGraph Orchestration — MAPE-K Flow
Matches Agentic_Architecture.drawio exactly:

  Ingestion → Drift → Balance → Supervisor → Policy → Strategy
            → Training → Evaluation → Simulation → Deployment
            → Knowledge → END

Feedback loops:
  L1: Evaluation → Balance   (data correction & rebalancing)
  L2: Evaluation → Strategy  (strategy refinement)
  L3: Training  → Training   (iterative retraining self-loop)
  L4: Supervisor ↔ Policy    (governance loop)
  L5: Evaluation → KB        (simulated validation logging)

Knowledge Base is the central hub connected to all agents.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the agent package root is on sys.path so that
#   `from config import …` and `from agents.xxx import …`
# resolve correctly regardless of the working directory.
_AGENT_ROOT = Path(__file__).resolve().parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from langgraph.graph import END, START, StateGraph

from config import (
    DATASET_PATH,
    NUMERIC_FEATURE_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    MAX_L1_ITERATIONS,
    MAX_L2_ITERATIONS,
    MAX_L3_ITERATIONS,
    MAX_L4_ITERATIONS,
    MAX_L5_ITERATIONS,
    F1_THRESHOLD,
)
from state import PipelineState
from knowledge_base import KnowledgeBase
from agents.drift_agent import drift_agent
from agents.balance_agent import balance_agent
from agents.supervisor_agent import supervisor_agent
from agents.policy_agent import policy_agent
from agents.strategy_agent import strategy_agent
from agents.training_agent import training_agent
from agents.evaluation_agent import evaluation_agent
from agents.simulation_agent import simulation_agent
from agents.deployment_agent import deployment_agent
from agents.knowledge_agent import knowledge_agent


# ── Data Ingestion (first graph node) ──────────────────────────────

def ingest_node(state: dict) -> dict:
    """Load CSV, select features, split train/test, ingest scraped data, initialize KB."""
    csv_path = Path(state.get("csv_path") or DATASET_PATH)

    print(f"\n📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Filter to rows where the target exists and is 0 or 1
    df = df[df[TARGET_COL].isin([0, 1, -1])].copy()
    # Map -1 (non-fraud) to 0 for binary classification
    df[TARGET_COL] = df[TARGET_COL].map({1: 1, 0: 0, -1: 0})

    # Keep only numeric features that exist in the dataframe
    available_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    print(f"  Available features: {len(available_cols)}")

    # Fill NaN with 0 for numeric features
    df[available_cols] = df[available_cols].fillna(0).astype(float)

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    fraud_count = int(df[TARGET_COL].sum())
    print(f"  Fraud: {fraud_count}, Non-fraud: {len(df) - fraud_count}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Handle scraped data ingestion
    scraped_dir = _AGENT_ROOT / "output" / "scraped_data"
    scraped_df = None
    if scraped_dir.exists() and any(scraped_dir.iterdir()):
        print(f"  📥 Found scraped data in {scraped_dir}")
        dfs = []
        for file_path in scraped_dir.glob("*.csv"):
            try:
                sdf = pd.read_csv(file_path)
                sdf = sdf[sdf[TARGET_COL].isin([0, 1, -1])].copy()
                sdf[TARGET_COL] = sdf[TARGET_COL].map({1: 1, 0: 0, -1: 0})
                sdf[available_cols] = sdf[available_cols].fillna(0).astype(float)
                dfs.append(sdf)
                print(f"    - Loaded {file_path.name} ({len(sdf)} rows)")
            except Exception as e:
                print(f"    - Failed to load {file_path.name}: {e}")
        if dfs:
            scraped_df = pd.concat(dfs, ignore_index=True)
            print(f"  📊 Total scraped data available: {len(scraped_df)} rows")

    # Initialize Knowledge Base (preserve existing if looping)
    kb = state.get("knowledge_base")
    if kb is None:
        kb = KnowledgeBase()

    return {
        **state,
        "raw_df": df,
        "train_df": train_df,
        "test_df": test_df,
        "scraped_df": scraped_df,
        "feature_cols": available_cols,
        "target_col": TARGET_COL,

        "current_model": None,
        "knowledge_base": kb,
        "knowledge_log": [],
        # Initialize all feedback-loop counters
        "l1_count": 0,
        "l2_count": 0,
        "l3_count": 0,
        "l4_count": 0,
        "l5_count": 0,
        "kb_loop_count": 0,   # KB→Drift closed-loop re-cycle counter
    }


# ── Conditional Routing (Feedback Loops) ───────────────────────────

def route_after_supervisor(state: dict) -> str:
    """
    L4 feedback loop: Supervisor ↔ Policy (bidirectional).
    If the supervisor escalates and we haven't exceeded L4 iterations,
    route back to Balance Agent for re-balancing before Policy re-evaluates.
    Otherwise proceed normally to Policy.
    """
    decision = state.get("supervisor_decision", "proceed")
    l4_count = state.get("l4_count", 0)

    if decision == "escalate" and l4_count < MAX_L4_ITERATIONS:
        print(f"  🔄 L4 loop: Supervisor escalated → Balance (iteration {l4_count + 1})")
        return "balance_agent"  # re-balance before policy re-evaluates
    return "policy_agent"


def route_after_training(state: dict) -> str:
    """
    L3 feedback loop: Training → Training (self-loop).
    If training F1 is too low and we haven't hit the max, retrain.
    """
    metrics = state.get("training_metrics", {})
    train_f1 = metrics.get("train_f1", 0)
    l3_count = state.get("l3_count", 0)

    if train_f1 < F1_THRESHOLD and l3_count < MAX_L3_ITERATIONS:
        print(f"  🔄 L3 loop: Train F1={train_f1} < {F1_THRESHOLD} → Retrain (iteration {l3_count + 1})")
        return "training_agent"  # self-loop
    return "evaluation_agent"


def route_after_evaluation(state: dict) -> str:
    """
    L1: Evaluation → Balance (data correction & rebalancing)
    L2: Evaluation → Strategy (strategy refinement)
    Otherwise: proceed to Simulation.
    """
    needs_rebalance = state.get("needs_rebalance", False)
    needs_strategy = state.get("needs_strategy_refinement", False)
    l1_count = state.get("l1_count", 0)
    l2_count = state.get("l2_count", 0)

    # L1: poor data quality → loop back to Balance
    if needs_rebalance and l1_count < MAX_L1_ITERATIONS:
        print(f"  🔄 L1 loop: Evaluation → Balance (iteration {l1_count + 1})")
        return "balance_agent"

    # L2: strategy needs refinement → loop back to Strategy
    if needs_strategy and l2_count < MAX_L2_ITERATIONS:
        print(f"  🔄 L2 loop: Evaluation → Strategy (iteration {l2_count + 1})")
        return "strategy_agent"

    return "simulation_agent"


def route_after_policy(state: dict) -> str:
    """
    Route after Policy Agent decision.
    If policy says skip → go straight to knowledge/end.
    If rebalance → loop back to balance (L4 re-entry).
    Otherwise → proceed to Strategy.
    """
    if state.get("should_skip", False):
        print("  ⏭️  Policy: skip update → Knowledge Agent")
        return "knowledge_agent"
    if state.get("should_rebalance", False):
        l4_count = state.get("l4_count", 0)
        if l4_count < MAX_L4_ITERATIONS:
            print(f"  🔄 L4 loop: Policy → Balance (rebalance, iteration {l4_count + 1})")
            return "balance_agent"
    return "strategy_agent"


def route_after_knowledge(state: dict) -> str:
    """
    L5 closed loop: KB → Ingestion (Continuous Loop).
    If we have scraped data, loop back to ingestion to process it.
    If the knowledge base dictates a drift check, loop back to drift detection.
    """
    kb = state.get("knowledge_base")
    l5_count = state.get("l5_count", 0)
    
    if l5_count < MAX_L5_ITERATIONS:
        scraped_dir = _AGENT_ROOT / "output" / "scraped_data"
        has_scraped = scraped_dir.exists() and any(scraped_dir.glob("*.csv"))
        
        if has_scraped:
            print(f"  🔄 L5 Continuous Loop: Found scraped data, looping to Ingestion (cycle {l5_count + 1})")
            return "ingest_node"
            
        if kb and kb.should_retrigger_drift():
            print(f"  🔄 L5 KB loop: re-triggering Drift Detection (cycle {l5_count + 1})")
            return "drift_agent"
            
    print("  🛑 L5 loop complete or max iterations reached. Ending pipeline.")
    return END


# ── Build the LangGraph ───────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the MAPE-K LangGraph matching Agentic_Architecture.drawio:

    Ingestion → Drift Detection → Balance → Supervisor → Policy → Strategy
              → Training → Evaluation → Simulation → Deployment → Knowledge → END

    With feedback loops L1–L5 as conditional edges.
    """
    graph = StateGraph(PipelineState)

    # ── Add all agent nodes ────────────────────────────────────
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("drift_agent", drift_agent)
    graph.add_node("balance_agent", balance_agent)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("policy_agent", policy_agent)
    graph.add_node("strategy_agent", strategy_agent)
    graph.add_node("training_agent", training_agent)
    graph.add_node("evaluation_agent", evaluation_agent)
    graph.add_node("simulation_agent", simulation_agent)
    graph.add_node("deployment_agent", deployment_agent)
    graph.add_node("knowledge_agent", knowledge_agent)

    # ── Linear spine (matches diagram left-to-right/top-to-bottom) ──
    graph.add_edge(START, "ingest_node")
    graph.add_edge("ingest_node", "drift_agent")        # Ingestion → Drift
    graph.add_edge("drift_agent", "balance_agent")       # Drift → Balance
    graph.add_edge("balance_agent", "supervisor_agent")  # Balance → Supervisor

    # Supervisor → Policy | Balance (L4 governance loop — now a conditional edge)
    graph.add_conditional_edges(
        "supervisor_agent",
        route_after_supervisor,
        {
            "balance_agent": "balance_agent",  # L4: escalate → re-balance
            "policy_agent": "policy_agent",    # normal path
        },
    )

    # Policy → Strategy | Balance(L4) | Knowledge(skip)
    graph.add_conditional_edges(
        "policy_agent",
        route_after_policy,
        {
            "strategy_agent": "strategy_agent",
            "balance_agent": "balance_agent",
            "knowledge_agent": "knowledge_agent",
        },
    )

    # Strategy → Training
    graph.add_edge("strategy_agent", "training_agent")

    # Training → Evaluation | Training(L3 self-loop)
    graph.add_conditional_edges(
        "training_agent",
        route_after_training,
        {
            "evaluation_agent": "evaluation_agent",
            "training_agent": "training_agent",      # L3 self-loop
        },
    )

    # Evaluation → Simulation | Balance(L1) | Strategy(L2)
    graph.add_conditional_edges(
        "evaluation_agent",
        route_after_evaluation,
        {
            "simulation_agent": "simulation_agent",
            "balance_agent": "balance_agent",        # L1 feedback
            "strategy_agent": "strategy_agent",      # L2 feedback
        },
    )

    # Simulation → Deployment → Knowledge → END (or KB→Drift closed loop)
    graph.add_edge("simulation_agent", "deployment_agent")
    graph.add_edge("deployment_agent", "knowledge_agent")

    # ── KB→Drift/Ingestion closed loop (L5 / KB bidirectional) ──
    # If scraped data exists or KB requests, trigger a new cycle.
    graph.add_conditional_edges(
        "knowledge_agent",
        route_after_knowledge,
        {"drift_agent": "drift_agent", "ingest_node": "ingest_node", END: END},
    )

    return graph


def compile_and_run(csv_path: str | Path | None = None) -> dict:
    """Ingest data, build graph, compile, and run the full pipeline."""
    initial_state = {}
    if csv_path:
        initial_state["csv_path"] = str(csv_path)

    graph = build_graph()
    app = graph.compile()

    print("\n" + "🚀 " * 15)
    print("  STARTING MAPE-K FRAUD DETECTION PIPELINE")
    print("🚀 " * 15)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    if final_state.get("promoted"):
        print("  🏆 Model PROMOTED successfully.")
    else:
        print("  ℹ️  Model was NOT promoted.")

    return final_state
