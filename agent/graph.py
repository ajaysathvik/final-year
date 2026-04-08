"""
LangGraph Orchestration — MAPE-K Flow

Primary path:
  Ingestion → Drift → Balance → Supervisor → Policy → Strategy
            → Training → Strategy → Evaluation
            → Knowledge → Simulation → Deployment → END

Strategy is the central adaptation hub. It routes to training when a
candidate must be built or rebuilt, and to evaluation when a candidate or
current model is ready to be assessed.

Feedback loops:
  L1: Evaluation → Balance   (data correction & rebalancing)
  L2: Evaluation → Strategy  (strategy refinement)
  L3: Training  → Training   (iterative retraining self-loop)
  L4: Supervisor ↔ Policy    (governance loop)
  L5: Evaluation → KB        (evaluation handoff logging)

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
    TRAIN_BALANCED_FEATURE_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    MAX_L1_ITERATIONS,
    MAX_L2_ITERATIONS,
    MAX_L3_ITERATIONS,
    MAX_L4_ITERATIONS,
    F1_THRESHOLD,
    GENERATE_RANDOM_DRIFT_DATA,
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

def _prepare_selected_frame(
    df: pd.DataFrame,
    base_feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Keep configured feature columns, adding missing ones as empty values."""
    prepared = df.copy()
    for col in base_feature_cols:
        if col not in prepared.columns:
            prepared[col] = pd.NA
    return prepared[base_feature_cols + [target_col]].copy()


def _encode_feature_frames(
    frames: list[pd.DataFrame | None],
    base_feature_cols: list[str],
    target_col: str,
) -> tuple[list[pd.DataFrame | None], list[str]]:
    """Normalize configured features without one-hot expansion."""
    prepared_frames: list[pd.DataFrame | None] = []
    feature_frames: list[pd.DataFrame] = []

    for frame in frames:
        if frame is None:
            prepared_frames.append(None)
            continue
        prepared = _prepare_selected_frame(frame, base_feature_cols, target_col)
        prepared_frames.append(prepared)
        feature_frames.append(prepared[base_feature_cols].copy())

    if not feature_frames:
        return prepared_frames, []

    combined_features = pd.concat(feature_frames, ignore_index=True)
    normalized_features = pd.DataFrame(index=combined_features.index)

    for col in base_feature_cols:
        series = combined_features[col]

        if pd.api.types.is_bool_dtype(series):
            normalized_features[col] = series.fillna(False).astype(int)
            continue

        if pd.api.types.is_numeric_dtype(series):
            normalized_features[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)
            continue

        category_codes = series.astype("category").cat.codes.astype("int32")
        normalized_features[col] = category_codes

    encoded_feature_cols = list(normalized_features.columns)

    encoded_frames: list[pd.DataFrame | None] = []
    start = 0
    for prepared in prepared_frames:
        if prepared is None:
            encoded_frames.append(None)
            continue

        stop = start + len(prepared)
        encoded_part = normalized_features.iloc[start:stop].reset_index(drop=True).copy()
        encoded_part[target_col] = prepared[target_col].reset_index(drop=True).astype(int)
        encoded_frames.append(encoded_part)
        start = stop

    return encoded_frames, encoded_feature_cols

def ingest_node(state: dict) -> dict:
    """Load CSV, select features, split train/test, ingest scraped data, initialize KB."""

    # ── Optionally regenerate drift test dataset before loading anything ──
    if GENERATE_RANDOM_DRIFT_DATA:
        from generate_random_drift_data import generate_and_save
        generate_and_save()

    csv_path = Path(state.get("csv_path") or DATASET_PATH)

    print(f"\n📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Filter to rows where the target exists and is 0 or 1
    df = df[df[TARGET_COL].isin([0, 1, -1])].copy()
    # Map -1 (non-fraud) to 0 for binary classification
    df[TARGET_COL] = df[TARGET_COL].map({1: 1, 0: 0, -1: 0})

    available_base_cols = [c for c in TRAIN_BALANCED_FEATURE_COLS if c in df.columns]
    print(f"  Base features from train_balanced.csv: {len(available_base_cols)}")

    # Handle scraped data ingestion
    drift_file = _AGENT_ROOT / "drift_test_dataset.csv"
    scraped_df = None
    if drift_file.exists():
        print(f"  📥 Found scraped data in {drift_file}")
        try:
            sdf = pd.read_csv(drift_file)
            sdf = sdf[sdf[TARGET_COL].isin([0, 1, -1])].copy()
            sdf[TARGET_COL] = sdf[TARGET_COL].map({1: 1, 0: 0, -1: 0})
            scraped_df = sdf
            print(f"    - Loaded {drift_file.name} ({len(sdf)} rows)")
            print(f"  📊 Total scraped data available: {len(scraped_df)} rows")
        except Exception as e:
            print(f"    - Failed to load {drift_file.name}: {e}")

    encoded_frames, encoded_feature_cols = _encode_feature_frames(
        [df, scraped_df],
        base_feature_cols=available_base_cols,
        target_col=TARGET_COL,
    )
    df = encoded_frames[0]
    scraped_df = encoded_frames[1]

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    fraud_count = int(df[TARGET_COL].sum())
    print(f"  Runtime features used without one-hot expansion: {len(encoded_feature_cols)}")
    print(f"  Fraud: {fraud_count}, Non-fraud: {len(df) - fraud_count}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Initialize Knowledge Base
    kb = state.get("knowledge_base")
    if kb is None:
        kb = KnowledgeBase()

    return {
        **state,
        "raw_df": df,
        "train_df": train_df,
        "test_df": test_df,
        "scraped_df": scraped_df,
        "feature_cols": encoded_feature_cols,
        "target_col": TARGET_COL,

        "current_model": None,
        "candidate_needs_evaluation": False,
        "knowledge_base": kb,
        "knowledge_log": [],
        "knowledge_stage": "final",
        # Initialize all feedback-loop counters
        "l1_count": 0,
        "l2_count": 0,
        "l3_count": 0,
        "l4_count": 0,
        "l5_count": 0,
        "kb_loop_count": 0,
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
    Otherwise return to Strategy so Strategy remains the adaptation hub.
    """
    metrics = state.get("training_metrics", {})
    train_f1 = metrics.get("train_f1", 0)
    l3_count = state.get("l3_count", 0)

    if train_f1 < F1_THRESHOLD and l3_count < MAX_L3_ITERATIONS:
        print(f"  🔄 L3 loop: Train-set F1={train_f1} < {F1_THRESHOLD} → Retrain (iteration {l3_count + 1})")
        return "training_agent"  # self-loop
    return "strategy_agent"


def route_after_strategy(state: dict) -> str:
    """
    Strategy is the hub for adaptation decisions after policy, training,
    and evaluation feedback.
    """
    decision = state.get("strategy_decision", "training")

    if decision == "training":
        print("  ▶️  Strategy: route to Training")
        return "training_agent"
    if decision == "evaluation":
        print("  ▶️  Strategy: route to Evaluation")
        return "evaluation_agent"
    print(f"  ⚠️  Unknown strategy decision '{decision}', defaulting to Training")
    return "training_agent"


def route_after_evaluation(state: dict) -> str:
    """
    L1: Evaluation → Balance (data correction & rebalancing)
    L2: Evaluation → Strategy (strategy refinement)
    Otherwise: log the evaluation handoff in KB, then continue to Simulation.
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

    return "knowledge_agent"


def route_after_policy(state: dict) -> str:
    """
    Route after Policy Agent decision.
    If policy requests validation of the existing model, go to Evaluation.
    If policy says skip after validation → go straight to Knowledge.
    If rebalance → loop back to Balance (L4 re-entry).
    Otherwise → proceed to Strategy.
    """
    if state.get("should_validate_existing", False):
        print("  ✅ Policy: validate current model before deciding to skip")
        return "evaluation_agent"
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
    Route forward after KB logging.
    Post-evaluation KB logging continues to Simulation.
    Final KB logging ends the pipeline.
    """
    if state.get("knowledge_stage") == "post_evaluation_logged":
        print("  ▶️  Knowledge: continue to Simulation")
        return "simulation_agent"

    print("  🛑 Knowledge logging complete. Ending pipeline.")
    return END


# ── Build the LangGraph ───────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the MAPE-K LangGraph matching Agentic_Architecture.drawio:

    Ingestion → Drift Detection → Balance → Supervisor → Policy → Strategy
              → Training → Strategy → Evaluation
              → Knowledge → Simulation → Deployment → END

    With feedback loops L1–L4 and an L5 knowledge handoff after evaluation.
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

    # Policy → Strategy | Evaluation(validate current model) | Balance(L4) | Knowledge(skip)
    graph.add_conditional_edges(
        "policy_agent",
        route_after_policy,
        {
            "strategy_agent": "strategy_agent",
            "evaluation_agent": "evaluation_agent",
            "balance_agent": "balance_agent",
            "knowledge_agent": "knowledge_agent",
        },
    )

    # Strategy → Training | Evaluation
    graph.add_conditional_edges(
        "strategy_agent",
        route_after_strategy,
        {
            "training_agent": "training_agent",
            "evaluation_agent": "evaluation_agent",
        },
    )

    # Training → Strategy | Training(L3 self-loop)
    graph.add_conditional_edges(
        "training_agent",
        route_after_training,
        {
            "strategy_agent": "strategy_agent",
            "training_agent": "training_agent",      # L3 self-loop
        },
    )

    # Evaluation → Knowledge(L5) | Balance(L1) | Strategy(L2)
    graph.add_conditional_edges(
        "evaluation_agent",
        route_after_evaluation,
        {
            "knowledge_agent": "knowledge_agent",
            "balance_agent": "balance_agent",        # L1 feedback
            "strategy_agent": "strategy_agent",      # L2 feedback
        },
    )

    # Evaluation KB handoff → Simulation → Deployment → END
    graph.add_edge("simulation_agent", "deployment_agent")
    graph.add_edge("deployment_agent", END)

    graph.add_conditional_edges(
        "knowledge_agent",
        route_after_knowledge,
        {"simulation_agent": "simulation_agent", END: END},
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
