from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from config import (
    AUDIT_LOG_PATH,
    DEFAULT_RATIO_CANDIDATES,
    EXECUTION_TRACE_PATH,
    LABEL_NOISE_THRESHOLD,
    MAX_REVIEW_LOOPS,
    MIN_JS_DIVERGENCE_ACCEPT,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OUTPUT_DIR,
    RUN_REPORT_PATH,
    SLANG_DRIFT_THRESHOLD,
    TARGET_F1_THRESHOLD,
    TARGET_ROBUSTNESS_THRESHOLD,
)
from memory import AgentMemory
from tools import build_toolkit

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    ChatOllama = None
    HumanMessage = None
    SystemMessage = None
    StateGraph = None
    START = "START"
    END = "END"


class WorkflowState(TypedDict, total=False):
    iteration: int
    next_step: str
    ingestion_agent: dict[str, Any]
    balance_agent: dict[str, Any]
    training_agent: dict[str, Any]
    strategy_agent: dict[str, Any]
    evaluation_agent: dict[str, Any]
    decisions: list[dict[str, Any]]
    done: bool
    status: str


@dataclass
class AgentDecision:
    agent: str
    summary: str
    action: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL) -> None:
        self.model = model
        self.base_url = base_url
        self.client = None
        if ChatOllama is not None:
            self.client = ChatOllama(
                model=model,
                base_url=base_url,
                temperature=0.2,
                num_predict=800,
                reasoning=False,
            )

    def _coerce_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    chunks.append(str(text))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
        return str(content)

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        # Strip common wrappers returned by chat models.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        candidates = [cleaned]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _plain_text_decision(self, role: str, content: str, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        summary = cleaned[:280].strip() if cleaned else f"{role} assessment completed."
        if summary.endswith("```"):
            summary = summary[:-3].rstrip()
        return {
            "summary": summary or f"{role} assessment completed.",
            "action": "continue",
            "confidence": 0.6,
            "metadata": {"raw": content, "payload": payload, "response_format": "text"},
        }

    def decide(self, role: str, objective: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.client is None or HumanMessage is None or SystemMessage is None:
            return {
                "summary": f"{role} used rule-based fallback because LangChain/Ollama is unavailable.",
                "action": "continue",
                "confidence": 0.5,
                "metadata": payload,
            }

        prompt = (
            f"Role: {role}\n"
            f"Objective: {objective}\n"
            "You are one specialized agent in an agentic fraud-detection workflow.\n"
            "Return either a single JSON object with keys summary, action, confidence, metadata, "
            "or a short plain-text summary if JSON is inconvenient.\n"
            f"Payload:\n{json.dumps(payload, indent=2, default=str)}"
        )
        response = self.client.invoke(
            [
                SystemMessage(content="You reason briefly. JSON is preferred but plain text is allowed."),
                HumanMessage(content=prompt),
            ]
        )
        content = self._coerce_content_to_text(getattr(response, "content", ""))
        parsed = self._extract_json_object(content)
        if parsed is not None:
            return parsed
        return self._plain_text_decision(role, content, payload)


class FraudWorkflow:
    def __init__(self) -> None:
        self.memory = AgentMemory()
        self.tools = build_toolkit()
        self.decision_engine = DecisionEngine()
        self.run_started_at = datetime.now(timezone.utc).isoformat()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_LOG_PATH.write_text("", encoding="utf-8")
        EXECUTION_TRACE_PATH.write_text("", encoding="utf-8")

    def _trace_event(
        self,
        agent: str,
        event: str,
        *,
        iteration: int | None = None,
        step: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_started_at": self.run_started_at,
            "agent": agent,
            "event": event,
            "iteration": iteration,
            "step": step,
            "payload": payload or {},
        }
        with EXECUTION_TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _summarize_result(self, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"value": str(result)}
        summary_keys = (
            "returncode",
            "rows",
            "best_ratio",
            "best_f1",
            "robustness_score",
            "mean_jsd",
            "accepted",
            "new_posts_detected",
            "posts_after",
            "comments_after",
            "train_rows",
            "test_rows",
            "recommendation",
            "focus",
            "robustness_gain",
            "summary_exists",
            "robustness_curve_exists",
            "eval_returncode",
            "robustness_returncode",
        )
        summary = {key: result[key] for key in summary_keys if key in result}
        if "sync_result" in result and isinstance(result["sync_result"], dict):
            sync = result["sync_result"]
            summary["sync_result"] = {
                key: sync[key]
                for key in ("new_rows_added", "train_rows_added", "test_rows_added", "reason")
                if key in sync
            }
        return summary or {"keys": sorted(result.keys())}

    def _run_tool(self, agent: str, iteration: int, step: str, func, *args, **kwargs):
        print(f" -> {agent}.{step}: started")
        self._trace_event(agent, "step_started", iteration=iteration, step=step)
        result = func(*args, **kwargs)
        summary = self._summarize_result(result)
        print(f" -> {agent}.{step}: completed {json.dumps(summary, ensure_ascii=True)}")
        self._trace_event(agent, "step_completed", iteration=iteration, step=step, payload=summary)
        return result

    def _audit(self, decision: AgentDecision, iteration: int, next_step: str | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_started_at": self.run_started_at,
            "iteration": iteration,
            "agent": decision.agent,
            "action": decision.action,
            "confidence": round(decision.confidence, 4),
            "summary": decision.summary,
            "next_step": next_step,
        }
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _record(self, agent: str, decision: AgentDecision) -> None:
        self.memory.add_decision(agent, asdict(decision))

    def ingestion_agent(self, state: WorkflowState) -> WorkflowState:
        print("\n" + "="*60)
        print(" [ INGESTION AGENT ] Starting data ingestion...")
        print("="*60)
        iteration = int(state.get("iteration", 0)) + 1
        self._trace_event("ingestion_agent", "agent_started", iteration=iteration)
        scrape = self._run_tool("ingestion_agent", iteration, "reddit_scraper", self.tools["reddit_scraper"].run)
        label = self._run_tool("ingestion_agent", iteration, "reddit_labeller", self.tools["reddit_labeller"].run)
        post_process = self._run_tool("ingestion_agent", iteration, "post_processor", self.tools["post_processor"].run)
        append = self._run_tool("ingestion_agent", iteration, "dataset_append", self.tools["dataset_append"].run)
        profile = self._run_tool("ingestion_agent", iteration, "dataset_profiler", self.tools["dataset_profiler"].run)
        review = self._run_tool("ingestion_agent", iteration, "label_review", self.tools["label_review"].run)
        payload = {
            "scrape": scrape,
            "label": label,
            "post_process": post_process,
            "append": append,
            "profile": profile,
            "review": review,
        }
        llm = self.decision_engine.decide(
            "IngestionAgent",
            "Ingest Reddit data through scraping, labeling, post-processing, train/test append, then assess label quality and slang drift.",
            payload,
        )

        append_result = append.get("sync_result", {})
        new_rows_added = int(append_result.get("new_rows_added", 0))
        scrape_detected_new_posts = int(scrape.get("new_posts_detected", 0)) > 0
        labeling_failed = (
            scrape["returncode"] != 0
            or label["returncode"] != 0
            or post_process["returncode"] != 0
        )
        ingestion_gap = scrape_detected_new_posts and new_rows_added == 0
        quality_failed = (
            profile["label_noise_score"] > LABEL_NOISE_THRESHOLD
            or profile["slang_drift_score"] > SLANG_DRIFT_THRESHOLD
            or review["recommendation"] == "relabel"
        )
        needs_relabel = labeling_failed or ingestion_gap or quality_failed
        action = "retry_ingestion" if needs_relabel else "ready_for_balancing"
        decision = AgentDecision(
            agent="ingestion_agent",
            summary=llm.get("summary", "IngestionAgent assessment completed."),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        next_step = "ingestion_agent" if needs_relabel else "balance_agent"
        self.memory.add_snapshot({"stage": "ingestion_agent", **payload})
        self._record("ingestion_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "ingestion_agent": {
                "scrape": scrape,
                "label": label,
                "post_process": post_process,
                "append": append,
                "profile": profile,
                "review": review,
                "labeling_failed": labeling_failed,
                "ingestion_gap": ingestion_gap,
                "quality_failed": quality_failed,
                "needs_relabel": needs_relabel,
            },
            "next_step": next_step,
            "decisions": decisions,
        }
        print(f" ✅ IngestionAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "ingestion_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "needs_relabel": needs_relabel},
        )
        return res

    def balance_agent(self, state: WorkflowState) -> WorkflowState:
        print("\n" + "="*60)
        print(" [ BALANCE AGENT ] Balancing dataset...")
        print("="*60)
        iteration = int(state.get("iteration", 0)) + 1
        self._trace_event("balance_agent", "agent_started", iteration=iteration)
        ratio_search = self._run_tool(
            "balance_agent", iteration, "balance_search", self.tools["balance_search"].run, DEFAULT_RATIO_CANDIDATES
        )
        ctgan_result = self._run_tool("balance_agent", iteration, "ctgan_runner", self.tools["ctgan_runner"].run)
        synthetic_quality = {"mean_jsd": 1.0, "accepted": False}
        if ctgan_result.get("exists"):
            synthetic_quality = self._run_tool(
                "balance_agent",
                iteration,
                "synthetic_quality",
                self.tools["synthetic_quality"].run,
                Path(ctgan_result["output_path"]),
            )

        payload = {
            "ratio_search": ratio_search,
            "ctgan_result": ctgan_result,
            "synthetic_quality": synthetic_quality,
        }
        llm = self.decision_engine.decide(
            "BalanceAgent",
            "Pick a balancing ratio and reject low-fidelity CTGAN output if JSD is too high.",
            payload,
        )

        accept = ctgan_result.get("returncode") == 0 and synthetic_quality["mean_jsd"] <= MIN_JS_DIVERGENCE_ACCEPT
        action = "ready_for_strategy" if accept else "retry_balancing"
        decision = AgentDecision(
            agent="balance_agent",
            summary=llm.get("summary", "BalanceAgent completed balancing review."),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        next_step = "training_agent" if accept else "balance_agent"
        self.memory.add_snapshot({"stage": "balance_agent", **payload})
        self._record("balance_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "balance_agent": {
                "ratio_search": ratio_search,
                "ctgan_result": ctgan_result,
                "synthetic_quality": synthetic_quality,
                "accepted": accept,
            },
            "next_step": next_step,
            "decisions": decisions,
        }
        print(f" ✅ BalanceAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "balance_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "accepted": accept},
        )
        return res

    def training_agent(self, state: WorkflowState) -> WorkflowState:
        print("\n" + "="*60)
        print(" [ TRAINING AGENT ] Training base models...")
        print("="*60)
        iteration = int(state.get("iteration", 0)) + 1
        self._trace_event("training_agent", "agent_started", iteration=iteration)
        training = self._run_tool("training_agent", iteration, "classifier_training", self.tools["classifier_training"].run)
        payload = {"training": training}
        llm = self.decision_engine.decide(
            "TrainingAgent",
            "Evaluate base model training results. Proceed if F1 score is acceptable.",
            payload,
        )
        passed = training.get("returncode") == 0 and training["best_f1"] >= (TARGET_F1_THRESHOLD * 0.9)
        action = "ready_for_strategy" if passed else "retry_training"
        decision = AgentDecision(
            agent="training_agent",
            summary=llm.get("summary", "TrainingAgent completed base model training assessment."),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        next_step = "strategy_agent" if passed else "training_agent"
        self.memory.add_snapshot({"stage": "training_agent", **payload})
        self._record("training_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "training_agent": {
                "training": training,
                "passed": passed,
            },
            "next_step": next_step,
            "decisions": decisions,
        }
        print(f" ✅ TrainingAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "training_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "passed": passed},
        )
        return res

    def strategy_agent(self, state: WorkflowState) -> WorkflowState:
        print("\n" + "="*60)
        print(" [ STRATEGY AGENT ] Running adversarial training...")
        print("="*60)
        iteration = int(state.get("iteration", 0)) + 1
        self._trace_event("strategy_agent", "agent_started", iteration=iteration)
        balance_state = state.get("balance_agent", {})
        synthetic_quality = balance_state.get("synthetic_quality", {})
        focus = "generic"
        if synthetic_quality.get("mean_jsd", 1.0) > (MIN_JS_DIVERGENCE_ACCEPT * 0.8):
            focus = "distribution_shift_fraud"
        adversarial = self._run_tool(
            "strategy_agent", iteration, "adversarial_trainer", self.tools["adversarial_trainer"].run, focus=focus
        )
        payload = {
            "status": "executed",
            "focus": focus,
            "adversarial_training": adversarial,
        }
        llm = self.decision_engine.decide(
            "StrategyAgent",
            "Run adversarial training on the balanced model candidate, capture robustness gain, and decide whether the strategy stage is acceptable.",
            payload,
        )
        accepted = adversarial.get("accepted", False)
        action = "ready_for_evaluation" if accepted else "retry_strategy"
        decision = AgentDecision(
            agent="strategy_agent",
            summary=llm.get("summary", "StrategyAgent executed adversarial training and robustness checks."),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        next_step = "evaluation_agent" if accepted else "strategy_agent"
        self.memory.add_snapshot({"stage": "strategy_agent", **payload})
        self._record("strategy_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "strategy_agent": {
                "status": "executed",
                "accepted": accepted,
                "focus": focus,
                "adversarial_training": adversarial,
            },
            "next_step": next_step,
            "decisions": decisions,
        }
        print(f" ✅ StrategyAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "strategy_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "accepted": accepted, "focus": focus},
        )
        return res

    def evaluation_agent(self, state: WorkflowState) -> WorkflowState:
        print("\n" + "="*60)
        print(" [ EVALUATION AGENT ] Running final validation...")
        print("="*60)
        iteration = int(state.get("iteration", 0)) + 1
        self._trace_event("evaluation_agent", "agent_started", iteration=iteration)
        evaluation = self._run_tool("evaluation_agent", iteration, "evaluation_runner", self.tools["evaluation_runner"].run)
        payload = {"evaluation": evaluation}
        llm = self.decision_engine.decide(
            "EvaluationAgent",
            "Approve deployment only if F1 and robustness satisfy thresholds; otherwise issue correction orders.",
            payload,
        )
        passed = (
            evaluation["best_f1"] >= TARGET_F1_THRESHOLD
            and evaluation["robustness_score"] >= TARGET_ROBUSTNESS_THRESHOLD
            and evaluation["eval_returncode"] == 0
            and evaluation["robustness_returncode"] == 0
        )
        action = "deploy" if passed else "correction_order"
        correction_target = "complete"
        if not passed:
            correction_target = "balance_agent" if evaluation["best_f1"] < TARGET_F1_THRESHOLD else "strategy_agent"
        decision = AgentDecision(
            agent="evaluation_agent",
            summary=llm.get("summary", "EvaluationAgent finished validation."),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata={**payload, "correction_target": correction_target},
        )
        self.memory.add_snapshot({"stage": "evaluation_agent", **payload, "passed": passed})
        self._record("evaluation_agent", decision)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        next_step = "complete" if passed else correction_target
        self._audit(decision, iteration, next_step)
        res = {
            **state,
            "iteration": iteration,
            "evaluation_agent": {"evaluation": evaluation, "passed": passed, "correction_target": correction_target},
            "next_step": next_step,
            "done": passed,
            "decisions": decisions,
        }
        print(f" 🏁 EvaluationAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "evaluation_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "passed": passed, "correction_target": correction_target},
        )
        return res

    def route_after_evaluation_agent(self, state: WorkflowState) -> Literal["complete", "balance_agent", "strategy_agent"]:
        if state.get("done"):
            return "complete"
        target = state.get("evaluation_agent", {}).get("correction_target", "balance_agent")
        return "strategy_agent" if target == "strategy_agent" else "balance_agent"

    def route_after_ingestion_agent(self, state: WorkflowState) -> Literal["balance_agent", "ingestion_agent"]:
        accepted = not state.get("ingestion_agent", {}).get("needs_relabel", False)
        iteration = int(state.get("iteration", 0))
        if accepted or iteration >= MAX_REVIEW_LOOPS:
            return "balance_agent"
        return "ingestion_agent"

    def route_after_balance_agent(self, state: WorkflowState) -> Literal["training_agent", "balance_agent"]:
        accepted = state.get("balance_agent", {}).get("accepted", False)
        iteration = int(state.get("iteration", 0))
        if accepted or iteration >= MAX_REVIEW_LOOPS:
            return "training_agent"
        return "balance_agent"

    def route_after_training_agent(self, state: WorkflowState) -> Literal["strategy_agent", "training_agent"]:
        passed = state.get("training_agent", {}).get("passed", False)
        iteration = int(state.get("iteration", 0))
        if passed or iteration >= MAX_REVIEW_LOOPS:
            return "strategy_agent"
        return "training_agent"

    def route_after_strategy_agent(self, state: WorkflowState) -> Literal["evaluation_agent", "strategy_agent"]:
        accepted = state.get("strategy_agent", {}).get("accepted", False)
        iteration = int(state.get("iteration", 0))
        if accepted or iteration >= MAX_REVIEW_LOOPS:
            return "evaluation_agent"
        return "strategy_agent"

    def complete(self, state: WorkflowState) -> WorkflowState:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "completed" if state.get("done") else "stopped",
            "iteration": state.get("iteration", 0),
            "decisions": state.get("decisions", []),
            "ingestion_agent": state.get("ingestion_agent", {}),
            "balance_agent": state.get("balance_agent", {}),
            "training_agent": state.get("training_agent", {}),
            "strategy_agent": state.get("strategy_agent", {}),
            "evaluation_agent": state.get("evaluation_agent", {}),
        }
        RUN_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self.memory.record_run(report)
        self._trace_event("workflow", "run_completed", iteration=state.get("iteration", 0), payload=report)
        print("\n" + "="*60)
        print(f" [ WORKFLOW {report['status'].upper()} ] iteration: {report['iteration']}")
        print("="*60 + "\n")
        return {**state, "status": report["status"]}

    def build(self):
        if StateGraph is None:
            raise RuntimeError(
                "LangGraph is not installed. Install dependencies from agent-new/requirements.txt first."
            )
        graph = StateGraph(WorkflowState)
        graph.add_node("ingestion_agent", self.ingestion_agent)
        graph.add_node("balance_agent", self.balance_agent)
        graph.add_node("training_agent", self.training_agent)
        graph.add_node("strategy_agent", self.strategy_agent)
        graph.add_node("evaluation_agent", self.evaluation_agent)
        graph.add_node("complete", self.complete)

        graph.add_edge(START, "ingestion_agent")
        graph.add_conditional_edges("ingestion_agent", self.route_after_ingestion_agent)
        graph.add_conditional_edges("balance_agent", self.route_after_balance_agent)
        graph.add_conditional_edges("training_agent", self.route_after_training_agent)
        graph.add_conditional_edges("strategy_agent", self.route_after_strategy_agent)
        graph.add_conditional_edges("evaluation_agent", self.route_after_evaluation_agent)
        graph.add_edge("complete", END)
        return graph.compile()


def run_workflow() -> dict[str, Any]:
    workflow = FraudWorkflow()
    app = workflow.build()
    state: WorkflowState = {"iteration": 0, "decisions": [], "done": False, "status": "running"}
    return app.invoke(state)
