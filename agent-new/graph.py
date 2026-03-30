from __future__ import annotations

import json
import os
import re
import shutil
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
    STOP_AFTER_INGESTION_IF_NO_UPDATE,
    TARGET_F1_THRESHOLD,
    TARGET_NON_FRAUD_F1_THRESHOLD,
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

MANUAL_START_NODES = {
    "ingestion_agent",
    "balance_agent",
    "training_agent",
    "supervisor_agent",
    "policy_agent",
    "strategy_agent",
    "evaluation_agent",
    "simulation_agent",
}


class WorkflowState(TypedDict, total=False):
    iteration: int
    agent_attempts: dict[str, int]
    next_step: str
    ingestion_agent: dict[str, Any]
    balance_agent: dict[str, Any]
    training_agent: dict[str, Any]
    supervisor_agent: dict[str, Any]
    policy_agent: dict[str, Any]
    strategy_agent: dict[str, Any]
    evaluation_agent: dict[str, Any]
    simulation_agent: dict[str, Any]
    scores: dict[str, Any]
    proposal: dict[str, Any]
    approval: dict[str, Any]
    outcome: dict[str, Any]
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
        self._objective_cache: dict[tuple[str, str], str] = {}

    def _build_messages(self, role: str, objective: str, payload: dict[str, Any]) -> tuple[str, str]:
        system_prompt = "You reason briefly. JSON is preferred but plain text is allowed."
        user_prompt = (
            f"Role: {role}\n"
            f"Objective: {objective}\n"
            "You are one specialized agent in an agentic fraud-detection workflow.\n"
            "Return either a single JSON object with keys summary, action, confidence, metadata, "
            "or a short plain-text summary if JSON is inconvenient.\n"
            f"Payload:\n{json.dumps(payload, indent=2, default=str)}"
        )
        return system_prompt, user_prompt

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

    def generate_objective(self, role: str, default_objective: str) -> str:
        cache_key = (role, default_objective)
        if cache_key in self._objective_cache:
            return self._objective_cache[cache_key]

        if self.client is None or HumanMessage is None or SystemMessage is None:
            self._objective_cache[cache_key] = default_objective
            return default_objective

        system_prompt = "Rewrite the supplied objective as one concise operational sentence. Return plain text only."
        user_prompt = (
            f"Role: {role}\n"
            f"Base objective: {default_objective}\n"
            "Write a single sentence objective for runtime logging. Keep it under 24 words."
        )
        try:
            response = self.client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception:
            self._objective_cache[cache_key] = default_objective
            return default_objective

        content = self._coerce_content_to_text(getattr(response, "content", ""))
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            cleaned = default_objective
        self._objective_cache[cache_key] = cleaned
        return cleaned

    def decide(self, role: str, objective: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.client is None or HumanMessage is None or SystemMessage is None:
            return {
                "summary": f"{role} used rule-based fallback because LangChain/Ollama is unavailable.",
                "action": "continue",
                "confidence": 0.5,
                "metadata": {
                    "payload": payload,
                    "decision_debug": {
                        "model": self.model,
                        "base_url": self.base_url,
                        "system_prompt": None,
                        "user_prompt": None,
                        "raw_response": None,
                        "fallback_reason": "LangChain/Ollama unavailable",
                    },
                },
            }

        system_prompt, user_prompt = self._build_messages(role, objective, payload)
        try:
            response = self.client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:
            return {
                "summary": f"{role} used rule-based fallback because Ollama invocation failed: {exc}.",
                "action": "continue",
                "confidence": 0.5,
                "metadata": {
                    "payload": payload,
                    "decision_debug": {
                        "model": self.model,
                        "base_url": self.base_url,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "raw_response": None,
                        "fallback_reason": f"Ollama invocation failed: {exc}",
                    },
                },
            }
        content = self._coerce_content_to_text(getattr(response, "content", ""))
        parsed = self._extract_json_object(content)
        if parsed is not None:
            metadata = parsed.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}
            metadata["decision_debug"] = {
                "model": self.model,
                "base_url": self.base_url,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": content,
            }
            parsed["metadata"] = metadata
            return parsed
        decision = self._plain_text_decision(role, content, payload)
        metadata = decision.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        metadata["decision_debug"] = {
            "model": self.model,
            "base_url": self.base_url,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": content,
        }
        decision["metadata"] = metadata
        return decision


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

    AGENT_RUNTIME_INFO = {
        "ingestion_agent": {
            "label": "INGESTION AGENT",
            "status": "Starting data ingestion...",
        },
        "balance_agent": {
            "label": "BALANCE AGENT",
            "status": "Balancing dataset...",
        },
        "training_agent": {
            "label": "TRAINING AGENT",
            "status": "Training base models...",
        },
        "supervisor_agent": {
            "label": "SUPERVISOR AGENT",
            "status": "Routing decision...",
        },
        "policy_agent": {
            "label": "POLICY AGENT",
            "status": "Reviewing proposal...",
        },
        "strategy_agent": {
            "label": "STRATEGY AGENT",
            "status": "Running adversarial training...",
        },
        "evaluation_agent": {
            "label": "EVALUATION AGENT",
            "status": "Running final validation...",
        },
        "simulation_agent": {
            "label": "SIMULATION AGENT",
            "status": "Running offline release simulation...",
        },
    }

    TOOL_RUNTIME_INFO = {
        "reddit_scraper": "Scraper script",
        "reddit_labeller": "Labelling pipeline",
        "post_processor": "Post-processing pipeline",
        "dataset_append": "Dataset/train-test append",
        "dataset_profiler": "Dataset profiler",
        "label_review": "Label review sampler",
        "balance_search": "Balance ratio search",
        "ctgan_runner": "CTGAN generation runner",
        "synthetic_quality": "Synthetic quality scorer",
        "classifier_training": "Classifier training runner",
        "attack_surface": "Attack-surface analyzer",
        "adversarial_trainer": "Adversarial training runner",
        "evaluation_runner": "Evaluation runner",
    }

    def _normalize_action_value(self, value: Any) -> str:
        cleaned = str(value or "").strip().lower()
        cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned

    def _print_agent_header(self, agent: str, *, role: str | None = None, objective: str | None = None) -> None:
        info = self.AGENT_RUNTIME_INFO.get(agent, {})
        label = info.get("label", agent.replace("_", " ").upper())
        status = info.get("status", "Starting...")
        print("\n" + "=" * 60)
        print(f" [ {label} ] {status}")
        print("=" * 60)
        if role:
            print(f" Role: {role}")
        if objective:
            generated_objective = self.decision_engine.generate_objective(role or agent, objective)
            print(f" Objective: {generated_objective}")

    def _tool_display_name(self, step: str) -> str:
        return self.TOOL_RUNTIME_INFO.get(step, step)

    def _select_action(self, llm_action: Any, fallback_action: str, allowed_actions: set[str]) -> tuple[str, str]:
        normalized = self._normalize_action_value(llm_action)
        if normalized in allowed_actions:
            return normalized, "llm"
        return fallback_action, "rules"

    def _balance_summary(
        self,
        *,
        accepted: bool,
        ratio_search: dict[str, Any],
        synthetic_quality: dict[str, Any],
        ctgan_result: dict[str, Any],
    ) -> str:
        ratio = ratio_search.get("best_ratio")
        jsd = synthetic_quality.get("mean_jsd")
        threshold = MIN_JS_DIVERGENCE_ACCEPT
        required = int(ratio_search.get("required_non_fraud_rows", 0))
        if accepted:
            if required == 0:
                return f"BalanceAgent accepted the batch without CTGAN because no synthetic non-fraud rows were required. Best ratio={ratio}."
            return f"BalanceAgent accepted CTGAN output. Best ratio={ratio}, mean JSD={jsd}, threshold={threshold}."
        reason = "CTGAN output missing or failed."
        if ctgan_result.get("returncode") == 0 and not ctgan_result.get("fresh_output", False):
            reason = "CTGAN did not produce fresh output."
        elif jsd is not None:
            reason = f"mean JSD={jsd} exceeded threshold={threshold}."
        return f"BalanceAgent rejected balancing output. Best ratio={ratio}. {reason}"

    def _training_summary(self, *, passed: bool, training: dict[str, Any]) -> str:
        best_f1 = training.get("best_f1")
        threshold = round(TARGET_F1_THRESHOLD * 0.9, 4)
        status = "passed" if passed else "failed"
        return f"TrainingAgent {status} base-model validation. best_f1={best_f1}, required>={threshold}."

    def _supervisor_summary(self, *, lane: str, scores: dict[str, Any]) -> str:
        return (
            f"SupervisorAgent routed the workflow to {lane}. "
            f"base_model_f1={scores.get('base_model_f1')}, synthetic_jsd={scores.get('synthetic_jsd')}."
        )

    def _policy_summary(self, *, approved: bool, proposal: dict[str, Any]) -> str:
        status = "approved" if approved else "rejected"
        return (
            f"PolicyAgent {status} proposal. "
            f"action={proposal.get('action')}, mode={proposal.get('mode')}, "
            f"needs_adversarial_training={proposal.get('needs_adversarial_training')}."
        )

    def _strategy_summary(self, *, accepted: bool, focus: str, adversarial: dict[str, Any]) -> str:
        gain = adversarial.get("robustness_gain")
        clean_f1 = adversarial.get("adversarial_clean_f1")
        status = "accepted" if accepted else "rejected"
        return f"StrategyAgent {status} adversarial training. focus={focus}, robustness_gain={gain}, adversarial_clean_f1={clean_f1}."

    def _evaluation_summary(self, *, passed: bool, evaluation: dict[str, Any]) -> str:
        best_f1 = evaluation.get("best_f1")
        non_fraud_f1 = evaluation.get("non_fraud_f1")
        robustness = evaluation.get("robustness_score")
        status = "passed" if passed else "failed"
        return (
            f"EvaluationAgent {status} final validation. "
            f"best_f1={best_f1} vs threshold={TARGET_F1_THRESHOLD}, "
            f"non_fraud_f1={non_fraud_f1} vs threshold={TARGET_NON_FRAUD_F1_THRESHOLD}, "
            f"robustness_score={robustness} vs threshold={TARGET_ROBUSTNESS_THRESHOLD}."
        )

    def _simulation_summary(self, *, approved: bool, simulation: dict[str, Any]) -> str:
        status = "approved" if approved else "rejected"
        return (
            f"SimulationAgent {status} offline release simulation. "
            f"simulation_score={simulation.get('simulation_score')}, "
            f"guardrail_status={simulation.get('guardrail_status')}."
        )

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
            "non_fraud_f1",
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
                for key in (
                    "new_rows_added",
                    "new_fraud_rows_added",
                    "new_non_fraud_rows_added",
                    "train_rows_added",
                    "test_rows_added",
                    "bootstrap_used",
                    "meets_update_threshold",
                    "reason",
                )
                if key in sync
            }
        return summary or {"keys": sorted(result.keys())}

    def _run_tool(self, agent: str, iteration: int, step: str, func, *args, **kwargs):
        tool_name = self._tool_display_name(step)
        print(f" -> Tool: {step} ({tool_name})")
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

    def _register_attempt(self, state: WorkflowState, agent: str) -> tuple[int, dict[str, int]]:
        attempts = dict(state.get("agent_attempts", {}))
        attempt = int(attempts.get(agent, 0)) + 1
        attempts[agent] = attempt
        return attempt, attempts

    def _is_retry_exhausted(self, attempts: int) -> bool:
        return attempts >= MAX_REVIEW_LOOPS

    def ingestion_agent(self, state: WorkflowState) -> WorkflowState:
        role = "IngestionAgent"
        objective = "Ingest Reddit data through scraping, labeling, post-processing, train/test append, then assess label quality."
        self._print_agent_header("ingestion_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "ingestion_agent")
        stub_downstream = str(os.getenv("AGENT_STUB_DOWNSTREAM", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if stub_downstream:
            payload = {
                "scrape": {"returncode": 0, "new_posts_detected": 0, "stubbed": True},
                "label": {"returncode": 0, "rows": 0, "stubbed": True},
                "post_process": {"returncode": 0, "rows": 0, "stubbed": True},
                "append": {
                    "sync_result": {
                        "new_rows_added": 2,
                        "new_fraud_rows_added": 2,
                        "new_non_fraud_rows_added": 0,
                        "train_rows_added": 2,
                        "test_rows_added": 0,
                        "bootstrap_used": True,
                        "meets_update_threshold": True,
                        "stubbed": True,
                    }
                },
                "profile": {"label_noise_score": 0.0, "stubbed": True},
                "review": {"recommendation": "keep", "stubbed": True},
            }
            decision = AgentDecision(
                agent="ingestion_agent",
                summary="IngestionAgent stubbed to hand off directly to downstream agents.",
                action="ready_for_balancing",
                confidence=1.0,
                metadata=payload,
            )
            next_step = "balance_agent"
            self.memory.add_snapshot({"stage": "ingestion_agent", **payload})
            self._record("ingestion_agent", decision)
            self._audit(decision, iteration, next_step)
            decisions = list(state.get("decisions", []))
            decisions.append(asdict(decision))
            res = {
                **state,
                "iteration": iteration,
                "agent_attempts": agent_attempts,
                "ingestion_agent": {
                    **payload,
                    "new_rows_added": 2,
                    "new_fraud_rows_added": 2,
                    "new_non_fraud_rows_added": 0,
                    "bootstrap_used": True,
                    "should_update_model": True,
                    "skip_model_update": False,
                    "labeling_failed": False,
                    "ingestion_gap": False,
                    "quality_failed": False,
                    "needs_relabel": False,
                    "attempt": attempt,
                    "retries_exhausted": False,
                    "stubbed": True,
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
                payload={"next_step": next_step, "stubbed": True},
            )
            return res
        skip_labeling = str(os.getenv("AGENT_SKIP_LABELING", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self._trace_event("ingestion_agent", "agent_started", iteration=iteration)
        previous_ingestion = state.get("ingestion_agent", {})
        previous_scrape = previous_ingestion.get("scrape", {})
        retry_without_rescrape = (
            attempt > 1
            and isinstance(previous_scrape, dict)
            and int(previous_scrape.get("returncode", 1)) == 0
            and int(previous_scrape.get("new_posts_detected", 0)) > 0
        )
        if retry_without_rescrape:
            scrape = {
                **previous_scrape,
                "reused_from_previous_attempt": True,
                "reused_attempt": attempt - 1,
                "stdout_tail": "Reused previous scrape output on ingestion retry.",
            }
            print(" -> ingestion_agent.reddit_scraper: skipped (reusing previous scrape output)")
            self._trace_event(
                "ingestion_agent",
                "step_completed",
                iteration=iteration,
                step="reddit_scraper",
                payload=self._summarize_result(scrape),
            )
        else:
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
        llm = self.decision_engine.decide(role, objective, payload)

        append_result = append.get("sync_result", {})
        new_rows_added = int(append_result.get("new_rows_added", 0))
        new_fraud_rows_added = int(append_result.get("new_fraud_rows_added", 0))
        new_non_fraud_rows_added = int(append_result.get("new_non_fraud_rows_added", 0))
        bootstrap_used = bool(append_result.get("bootstrap_used", False))
        meets_update_threshold = bool(append_result.get("meets_update_threshold", False))
        scrape_detected_new_posts = int(scrape.get("new_posts_detected", 0)) > 0
        labeling_failed = (
            scrape["returncode"] != 0
            or label["returncode"] != 0
            or post_process["returncode"] != 0
        )
        labeler_had_posts = int(label.get("new_posts_to_label", 0)) > 0
        labeler_produced_nothing = labeler_had_posts and int(label.get("new_labels_created", 0)) == 0
        ingestion_gap = scrape_detected_new_posts and new_rows_added == 0 and not labeler_produced_nothing
        if skip_labeling:
            labeling_failed = scrape["returncode"] != 0
            ingestion_gap = False
        quality_failed = (
            profile["label_noise_score"] > LABEL_NOISE_THRESHOLD
            or review["recommendation"] == "relabel"
        )
        needs_relabel = labeling_failed or ingestion_gap or quality_failed
        should_update_model = bootstrap_used or meets_update_threshold or skip_labeling
        skip_model_update = (
            STOP_AFTER_INGESTION_IF_NO_UPDATE
            and not needs_relabel
            and not should_update_model
        )
        retries_exhausted = needs_relabel and self._is_retry_exhausted(attempt)
        fallback_action = "retry_ingestion" if needs_relabel else (
            "ready_for_balancing" if (should_update_model or not STOP_AFTER_INGESTION_IF_NO_UPDATE) else "skip_model_update"
        )
        action, action_source = self._select_action(
            llm.get("action"),
            fallback_action,
            {"retry_ingestion", "ready_for_balancing", "skip_model_update"},
        )
        decision = AgentDecision(
            agent="ingestion_agent",
            summary=llm.get("summary", "IngestionAgent assessment completed."),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        if retries_exhausted and action == "retry_ingestion":
            next_step = "complete"
        elif action == "retry_ingestion":
            next_step = "ingestion_agent"
        elif action == "ready_for_balancing":
            next_step = "balance_agent"
        else:
            next_step = "complete"
        self.memory.add_snapshot({"stage": "ingestion_agent", **payload})
        self._record("ingestion_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "ingestion_agent": {
                "scrape": scrape,
                "label": label,
                "post_process": post_process,
                "append": append,
                "profile": profile,
                "review": review,
                "new_rows_added": new_rows_added,
                "new_fraud_rows_added": new_fraud_rows_added,
                "new_non_fraud_rows_added": new_non_fraud_rows_added,
                "bootstrap_used": bootstrap_used,
                "should_update_model": should_update_model,
                "skip_model_update": skip_model_update,
                "labeling_failed": labeling_failed,
                "ingestion_gap": ingestion_gap,
                "quality_failed": quality_failed,
                "needs_relabel": needs_relabel,
                "retry_without_rescrape": retry_without_rescrape,
                "action_source": action_source,
                "attempt": attempt,
                "retries_exhausted": retries_exhausted,
            },
            "next_step": next_step,
            "done": action == "skip_model_update",
            "decisions": decisions,
        }
        print(f" ✅ IngestionAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "ingestion_agent",
            "agent_completed",
            iteration=iteration,
            payload={
                "next_step": next_step,
                "needs_relabel": needs_relabel,
                "new_fraud_rows_added": new_fraud_rows_added,
                "bootstrap_used": bootstrap_used,
                "skip_model_update": skip_model_update,
                "retry_without_rescrape": retry_without_rescrape,
                "action_source": action_source,
                "attempt": attempt,
                "retries_exhausted": retries_exhausted,
            },
        )
        return res

    def balance_agent(self, state: WorkflowState) -> WorkflowState:
        role = "BalanceAgent"
        objective = "Pick a balancing ratio and reject low-fidelity CTGAN output if JSD is too high."
        self._print_agent_header("balance_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "balance_agent")
        self._trace_event("balance_agent", "agent_started", iteration=iteration)
        ingestion_state = state.get("ingestion_agent", {})
        batch_fraud_count = int(ingestion_state.get("new_fraud_rows_added", 0))
        batch_non_fraud_count = int(ingestion_state.get("new_non_fraud_rows_added", 0))
        ratio_search = self._run_tool(
            "balance_agent",
            iteration,
            "balance_search",
            self.tools["balance_search"].run,
            DEFAULT_RATIO_CANDIDATES,
        )
        ctgan_result = self._run_tool(
            "balance_agent",
            iteration,
            "ctgan_runner",
            self.tools["ctgan_runner"].run,
            ratio_search.get("best_ratio"),
            fraud_count=int(ratio_search.get("fraud_count", 0)),
            non_fraud_count=int(ratio_search.get("non_fraud_count", 0)),
            required_non_fraud_rows=int(ratio_search.get("required_non_fraud_rows", 0)),
        )
        synthetic_quality = {"mean_jsd": 1.0, "accepted": False}
        if ctgan_result.get("fresh_output"):
            synthetic_quality = self._run_tool(
                "balance_agent",
                iteration,
                "synthetic_quality",
                self.tools["synthetic_quality"].run,
                Path(ctgan_result["output_path"]),
            )

        payload = {
            "batch_fraud_count": batch_fraud_count,
            "batch_non_fraud_count": batch_non_fraud_count,
            "ratio_search": ratio_search,
            "ctgan_result": ctgan_result,
            "synthetic_quality": synthetic_quality,
        }
        llm = self.decision_engine.decide(role, objective, payload)

        accept = ratio_search.get("required_non_fraud_rows", 0) == 0 or (
            ctgan_result.get("returncode") == 0
            and ctgan_result.get("fresh_output", False)
            and synthetic_quality["mean_jsd"] <= MIN_JS_DIVERGENCE_ACCEPT
        )
        retries_exhausted = (not accept) and self._is_retry_exhausted(attempt)
        fallback_action = "ready_for_strategy" if accept else "retry_balancing"
        action, action_source = self._select_action(
            llm.get("action"),
            fallback_action,
            {"ready_for_strategy", "retry_balancing"},
        )
        decision = AgentDecision(
            agent="balance_agent",
            summary=self._balance_summary(
                accepted=accept,
                ratio_search=ratio_search,
                synthetic_quality=synthetic_quality,
                ctgan_result=ctgan_result,
            ),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        if retries_exhausted and action == "retry_balancing":
            next_step = "complete"
        elif action == "retry_balancing":
            next_step = "balance_agent"
        else:
            next_step = "training_agent"
        self.memory.add_snapshot({"stage": "balance_agent", **payload})
        self._record("balance_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "balance_agent": {
                "ratio_search": ratio_search,
                "ctgan_result": ctgan_result,
                "synthetic_quality": synthetic_quality,
                "accepted": accept,
                "action_source": action_source,
                "attempt": attempt,
                "retries_exhausted": retries_exhausted,
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
            payload={"next_step": next_step, "accepted": accept, "attempt": attempt, "retries_exhausted": retries_exhausted},
        )
        return res

    def training_agent(self, state: WorkflowState) -> WorkflowState:
        role = "TrainingAgent"
        objective = "Evaluate base model training results. Proceed if F1 score is acceptable."
        self._print_agent_header("training_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "training_agent")
        self._trace_event("training_agent", "agent_started", iteration=iteration)
        training = self._run_tool("training_agent", iteration, "classifier_training", self.tools["classifier_training"].run)
        payload = {"training": training}
        llm = self.decision_engine.decide(role, objective, payload)
        passed = training.get("returncode") == 0 and training["best_f1"] >= (TARGET_F1_THRESHOLD * 0.9)
        retries_exhausted = (not passed) and self._is_retry_exhausted(attempt)
        fallback_action = "ready_for_strategy" if passed else "retry_training"
        action, action_source = self._select_action(
            llm.get("action"),
            fallback_action,
            {"ready_for_strategy", "retry_training"},
        )
        decision = AgentDecision(
            agent="training_agent",
            summary=self._training_summary(passed=passed, training=training),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        if retries_exhausted and action == "retry_training":
            next_step = "complete"
        elif action == "retry_training":
            next_step = "training_agent"
        else:
            next_step = "supervisor_agent"
        self.memory.add_snapshot({"stage": "training_agent", **payload})
        self._record("training_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "training_agent": {
                "training": training,
                "passed": passed,
                "action_source": action_source,
                "attempt": attempt,
                "retries_exhausted": retries_exhausted,
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
            payload={"next_step": next_step, "passed": passed, "attempt": attempt, "retries_exhausted": retries_exhausted},
        )
        return res

    def supervisor_agent(self, state: WorkflowState) -> WorkflowState:
        role = "SupervisorAgent"
        objective = "Route the workflow based on model quality, dataset health, and synthetic-data fidelity. Send abnormal data back for correction before policy review."
        self._print_agent_header("supervisor_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "supervisor_agent")
        self._trace_event("supervisor_agent", "agent_started", iteration=iteration)
        training = state.get("training_agent", {}).get("training", {})
        balance = state.get("balance_agent", {})
        scores = dict(state.get("scores", {}))
        scores.update(
            {
                "base_model_f1": round(float(training.get("best_f1", 0.0)), 4),
                "synthetic_jsd": round(float(balance.get("synthetic_quality", {}).get("mean_jsd", 1.0)), 4),
            }
        )
        lane = "policy_agent"
        if scores["synthetic_jsd"] > MIN_JS_DIVERGENCE_ACCEPT:
            lane = "balance_agent"
        payload = {
            "scores": scores,
            "training_passed": state.get("training_agent", {}).get("passed", False),
            "balance_agent": balance,
        }
        llm = self.decision_engine.decide(role, objective, payload)
        action, action_source = self._select_action(
            llm.get("action"),
            lane,
            {"policy_agent", "ingestion_agent", "balance_agent"},
        )
        decision = AgentDecision(
            agent="supervisor_agent",
            summary=self._supervisor_summary(lane=action, scores=scores),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        self.memory.add_snapshot({"stage": "supervisor_agent", **payload})
        self._record("supervisor_agent", decision)
        self._audit(decision, iteration, action)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "supervisor_agent": {
                "lane": action,
                "scores": scores,
                "action_source": action_source,
                "attempt": attempt,
            },
            "scores": scores,
            "next_step": action,
            "decisions": decisions,
        }
        print(f" ✅ SupervisorAgent: {decision.summary}")
        print(f" -> Next step: {action}")
        self._trace_event(
            "supervisor_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": action, "lane": action, "attempt": attempt},
        )
        return res

    def policy_agent(self, state: WorkflowState) -> WorkflowState:
        role = "PolicyAgent"
        objective = "Review the attack surface, create a policy proposal for the next workflow step, and approve it when the proposal is internally consistent."
        self._print_agent_header("policy_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "policy_agent")
        self._trace_event("policy_agent", "agent_started", iteration=iteration)
        scores = dict(state.get("scores", {}))
        attack_surface = self._run_tool("policy_agent", iteration, "attack_surface", self.tools["attack_surface"].run)
        escalation_score = round(
            (0.6 * float(scores.get("synthetic_jsd", 0.0)))
            + (0.4 * max(0.0, TARGET_F1_THRESHOLD - float(scores.get("base_model_f1", 0.0)))),
            4,
        )
        recommended_action = "harden_with_adversarial_training" if escalation_score >= 0.08 else "standard_policy_review"
        needs_adversarial_training = escalation_score >= 0.08
        proposal = {
            "action": "advance_pipeline",
            "mode": "hardened" if needs_adversarial_training else "standard",
            "needs_adversarial_training": needs_adversarial_training,
            "reason": recommended_action,
        }
        approval = {
            "approved": True,
            "owner": "policy_agent",
            "review_mode": proposal["mode"],
        }
        payload = {
            "scores": scores,
            "attack_surface": attack_surface,
            "escalation_score": escalation_score,
            "recommended_action": recommended_action,
            "proposal": proposal,
            "approval": approval,
        }
        llm = self.decision_engine.decide(role, objective, payload)
        action, action_source = self._select_action(
            llm.get("action"),
            "strategy_agent",
            {"strategy_agent", "complete"},
        )
        decision = AgentDecision(
            agent="policy_agent",
            summary=self._policy_summary(approved=True, proposal=proposal),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        self.memory.add_snapshot({"stage": "policy_agent", **payload})
        self._record("policy_agent", decision)
        self._audit(decision, iteration, action)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "policy_agent": {
                "attack_surface": attack_surface,
                "escalation_score": escalation_score,
                "recommended_action": recommended_action,
                "proposal": proposal,
                "approval": approval,
                "action_source": action_source,
                "attempt": attempt,
            },
            "proposal": proposal,
            "approval": approval,
            "next_step": action,
            "decisions": decisions,
        }
        print(f" ✅ PolicyAgent: {decision.summary}")
        print(f" -> Next step: {action}")
        self._trace_event(
            "policy_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": action, "approved": True, "escalation_score": escalation_score, "attempt": attempt},
        )
        return res

    def strategy_agent(self, state: WorkflowState) -> WorkflowState:
        role = "StrategyAgent"
        objective = "Run adversarial training on the balanced model candidate, capture robustness gain, and decide whether the strategy stage is acceptable."
        self._print_agent_header("strategy_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "strategy_agent")
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
        llm = self.decision_engine.decide(role, objective, payload)
        accepted = adversarial.get("accepted", False)
        scores = dict(state.get("scores", {}))
        scores["robustness_gain"] = round(float(adversarial.get("robustness_gain", 0.0)), 4)
        retries_exhausted = (not accepted) and self._is_retry_exhausted(attempt)
        fallback_action = "ready_for_evaluation" if accepted else "retry_strategy"
        action, action_source = self._select_action(
            llm.get("action"),
            fallback_action,
            {"ready_for_evaluation", "retry_strategy"},
        )
        decision = AgentDecision(
            agent="strategy_agent",
            summary=self._strategy_summary(accepted=accepted, focus=focus, adversarial=adversarial),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        if retries_exhausted and action == "retry_strategy":
            next_step = "complete"
        elif action == "retry_strategy":
            next_step = "strategy_agent"
        else:
            next_step = "evaluation_agent"
        self.memory.add_snapshot({"stage": "strategy_agent", **payload})
        self._record("strategy_agent", decision)
        self._audit(decision, iteration, next_step)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "strategy_agent": {
                "status": "executed",
                "accepted": accepted,
                "focus": focus,
                "adversarial_training": adversarial,
                "action_source": action_source,
                "attempt": attempt,
                "retries_exhausted": retries_exhausted,
            },
            "scores": scores,
            "next_step": next_step,
            "decisions": decisions,
        }
        print(f" ✅ StrategyAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "strategy_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "accepted": accepted, "focus": focus, "attempt": attempt, "retries_exhausted": retries_exhausted},
        )
        return res

    def evaluation_agent(self, state: WorkflowState) -> WorkflowState:
        role = "EvaluationAgent"
        objective = "Approve deployment only if F1 and robustness satisfy thresholds; otherwise issue correction orders."
        self._print_agent_header("evaluation_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "evaluation_agent")
        self._trace_event("evaluation_agent", "agent_started", iteration=iteration)
        evaluation = self._run_tool("evaluation_agent", iteration, "evaluation_runner", self.tools["evaluation_runner"].run)
        payload = {"evaluation": evaluation}
        llm = self.decision_engine.decide(role, objective, payload)
        non_fraud_f1 = evaluation.get("non_fraud_f1", 0.0)
        passed = (
            evaluation["best_f1"] >= TARGET_F1_THRESHOLD
            and float(non_fraud_f1) >= TARGET_NON_FRAUD_F1_THRESHOLD
            and evaluation["robustness_score"] >= TARGET_ROBUSTNESS_THRESHOLD
            and evaluation["eval_returncode"] == 0
            and evaluation["robustness_returncode"] == 0
        )
        correction_target = "complete"
        retries_exhausted = (not passed) and self._is_retry_exhausted(attempt)
        if not passed:
            correction_target = "balance_agent" if evaluation["best_f1"] < TARGET_F1_THRESHOLD else "strategy_agent"
        fallback_action = "deploy" if passed else correction_target
        action, action_source = self._select_action(
            llm.get("action"),
            fallback_action,
            {"deploy", "balance_agent", "strategy_agent", "correction_order"},
        )
        if action in {"balance_agent", "strategy_agent"}:
            correction_target = action
        elif action == "correction_order":
            action = correction_target
        llm_metadata = llm.get("metadata") if isinstance(llm.get("metadata"), dict) else {}
        self._write_stage_output(
            "evaluation",
            "decision_model_log.json",
            {
                "role": "EvaluationAgent",
                "objective": objective,
                "payload": payload,
                "model_input": llm_metadata.get("decision_debug", {}),
                "model_output": {
                    "summary": llm.get("summary"),
                    "action": llm.get("action"),
                    "confidence": llm.get("confidence"),
                    "metadata": llm_metadata,
                },
                "resolved_transition": {
                    "action_after_normalization": action,
                    "correction_target": correction_target,
                },
            },
        )
        decision = AgentDecision(
            agent="evaluation_agent",
            summary=self._evaluation_summary(passed=passed, evaluation=evaluation),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata={**payload, "correction_target": correction_target},
        )
        self.memory.add_snapshot({"stage": "evaluation_agent", **payload, "passed": passed})
        self._record("evaluation_agent", decision)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        outcome = {
            "deployment_ready": action == "deploy",
            "correction_target": correction_target,
            "evaluation_passed": passed,
        }
        next_step = "complete" if retries_exhausted else ("simulation_agent" if action == "deploy" else correction_target)
        self._audit(decision, iteration, next_step)
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "evaluation_agent": {
                "evaluation": evaluation,
                "passed": passed,
                "correction_target": correction_target,
                "action_source": action_source,
                "attempt": attempt,
                "retries_exhausted": retries_exhausted,
            },
            "outcome": outcome,
            "next_step": next_step,
            "done": False,
            "decisions": decisions,
        }
        print(f" 🏁 EvaluationAgent: {decision.summary}")
        print(f" -> Next step: {next_step}")
        self._trace_event(
            "evaluation_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": next_step, "passed": passed, "correction_target": correction_target, "attempt": attempt, "retries_exhausted": retries_exhausted},
        )
        return res

    def simulation_agent(self, state: WorkflowState) -> WorkflowState:
        role = "SimulationAgent"
        objective = "Simulate the release decision offline and approve completion only when all guardrails still pass."
        self._print_agent_header("simulation_agent", role=role, objective=objective)
        iteration = int(state.get("iteration", 0)) + 1
        attempt, agent_attempts = self._register_attempt(state, "simulation_agent")
        self._trace_event("simulation_agent", "agent_started", iteration=iteration)
        evaluation = state.get("evaluation_agent", {}).get("evaluation", {})
        proposal = state.get("proposal", {})
        simulation_score = round(
            min(
                float(evaluation.get("best_f1", 0.0)),
                float(evaluation.get("robustness_score", 0.0)),
            ),
            4,
        )
        approved = (
            state.get("evaluation_agent", {}).get("passed", False)
            and simulation_score >= min(TARGET_F1_THRESHOLD, TARGET_ROBUSTNESS_THRESHOLD)
        )
        simulation = {
            "proposal_mode": proposal.get("mode", "standard"),
            "simulation_score": simulation_score,
            "guardrail_status": "pass" if approved else "fail",
        }
        payload = {"simulation": simulation, "evaluation": evaluation, "proposal": proposal}
        llm = self.decision_engine.decide(role, objective, payload)
        fallback_action = "complete" if approved else state.get("evaluation_agent", {}).get("correction_target", "strategy_agent")
        action, action_source = self._select_action(
            llm.get("action"),
            fallback_action,
            {"complete", "balance_agent", "strategy_agent"},
        )
        outcome = {
            "deployment_ready": approved,
            "simulation_score": simulation_score,
            "final_action": action,
        }
        decision = AgentDecision(
            agent="simulation_agent",
            summary=self._simulation_summary(approved=approved, simulation=simulation),
            action=action,
            confidence=float(llm.get("confidence", 0.7)),
            metadata=payload,
        )
        self.memory.add_snapshot({"stage": "simulation_agent", **payload})
        self._record("simulation_agent", decision)
        self._audit(decision, iteration, action)
        decisions = list(state.get("decisions", []))
        decisions.append(asdict(decision))
        res = {
            **state,
            "iteration": iteration,
            "agent_attempts": agent_attempts,
            "simulation_agent": {
                "simulation": simulation,
                "approved": approved,
                "action_source": action_source,
                "attempt": attempt,
            },
            "approval": {
                **state.get("approval", {}),
                "approved": approved,
                "owner": "simulation_agent",
            },
            "outcome": outcome,
            "next_step": action,
            "done": action == "complete" and approved,
            "decisions": decisions,
        }
        print(f" ✅ SimulationAgent: {decision.summary}")
        print(f" -> Next step: {action}")
        self._trace_event(
            "simulation_agent",
            "agent_completed",
            iteration=iteration,
            payload={"next_step": action, "approved": approved, "attempt": attempt},
        )
        return res

    def route_after_evaluation_agent(self, state: WorkflowState) -> Literal["complete", "balance_agent", "strategy_agent", "simulation_agent"]:
        next_step = state.get("next_step")
        if next_step in {"complete", "balance_agent", "strategy_agent", "simulation_agent"}:
            return next_step
        if state.get("done"):
            return "complete"
        if state.get("evaluation_agent", {}).get("retries_exhausted", False):
            return "complete"
        target = state.get("evaluation_agent", {}).get("correction_target", "balance_agent")
        return "strategy_agent" if target == "strategy_agent" else "balance_agent"

    def route_after_ingestion_agent(self, state: WorkflowState) -> Literal["balance_agent", "ingestion_agent", "complete"]:
        next_step = state.get("next_step")
        if next_step in {"balance_agent", "ingestion_agent", "complete"}:
            return next_step
        if state.get("ingestion_agent", {}).get("skip_model_update", False):
            return "complete"
        accepted = not state.get("ingestion_agent", {}).get("needs_relabel", False)
        if accepted:
            return "balance_agent"
        if state.get("ingestion_agent", {}).get("retries_exhausted", False):
            return "complete"
        return "ingestion_agent"

    def route_after_balance_agent(self, state: WorkflowState) -> Literal["training_agent", "balance_agent", "complete"]:
        next_step = state.get("next_step")
        if next_step in {"training_agent", "balance_agent", "complete"}:
            return next_step
        accepted = state.get("balance_agent", {}).get("accepted", False)
        if accepted:
            return "training_agent"
        if state.get("balance_agent", {}).get("retries_exhausted", False):
            return "complete"
        return "balance_agent"

    def route_after_training_agent(self, state: WorkflowState) -> Literal["supervisor_agent", "training_agent", "complete"]:
        next_step = state.get("next_step")
        if next_step in {"supervisor_agent", "training_agent", "complete"}:
            return next_step
        passed = state.get("training_agent", {}).get("passed", False)
        if passed:
            return "supervisor_agent"
        if state.get("training_agent", {}).get("retries_exhausted", False):
            return "complete"
        return "training_agent"

    def route_after_supervisor_agent(self, state: WorkflowState) -> Literal["policy_agent", "ingestion_agent", "balance_agent"]:
        next_step = state.get("next_step")
        if next_step in {"policy_agent", "ingestion_agent", "balance_agent"}:
            return next_step
        return "policy_agent"

    def route_after_policy_agent(self, state: WorkflowState) -> Literal["strategy_agent", "complete"]:
        next_step = state.get("next_step")
        if next_step in {"strategy_agent", "complete"}:
            return next_step
        return "strategy_agent"

    def route_after_strategy_agent(self, state: WorkflowState) -> Literal["evaluation_agent", "strategy_agent", "complete"]:
        next_step = state.get("next_step")
        if next_step in {"evaluation_agent", "strategy_agent", "complete"}:
            return next_step
        accepted = state.get("strategy_agent", {}).get("accepted", False)
        if accepted:
            return "evaluation_agent"
        if state.get("strategy_agent", {}).get("retries_exhausted", False):
            return "complete"
        return "strategy_agent"

    def route_after_simulation_agent(self, state: WorkflowState) -> Literal["complete", "balance_agent", "strategy_agent"]:
        next_step = state.get("next_step")
        if next_step in {"complete", "balance_agent", "strategy_agent"}:
            return next_step
        return "complete" if state.get("simulation_agent", {}).get("approved", False) else "strategy_agent"

    def _write_stage_output(self, subdir: str, filename: str, data: Any) -> Path:
        """Write a JSON file into output/<subdir>/<filename>."""
        stage_dir = OUTPUT_DIR / subdir
        stage_dir.mkdir(parents=True, exist_ok=True)
        out_path = stage_dir / filename
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
        return out_path

    def _copy_adversarial_charts(self) -> list[str]:
        """Copy adversarial result charts into output/adversarial/charts/."""
        from config import ADVERSARIAL_RESULTS_DIR
        charts_dir = OUTPUT_DIR / "adversarial" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        if ADVERSARIAL_RESULTS_DIR.exists():
            for src in sorted(ADVERSARIAL_RESULTS_DIR.glob("*.png")):
                dst = charts_dir / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst))
        return copied

    def _generate_run_summary(self, report: dict[str, Any]) -> str:
        """Generate a human-readable Markdown summary of the pipeline run."""
        lines: list[str] = []
        status = report.get("status", "unknown").upper()
        iteration = report.get("iteration", 0)
        lines.append(f"# Agentic Fraud Pipeline — Run Summary")
        lines.append(f"")
        lines.append(f"**Status:** {status}  ")
        lines.append(f"**Total iterations:** {iteration}  ")
        lines.append(f"**Generated at:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
        lines.append("")

        # --- Scraping ---
        ing = report.get("ingestion_agent", {})
        scrape = ing.get("scrape", {})
        lines.append("## 1. Scraping (Ingestion Agent)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Posts before | {scrape.get('posts_before', 'N/A')} |")
        lines.append(f"| Posts after | {scrape.get('posts_after', 'N/A')} |")
        lines.append(f"| New posts detected | {scrape.get('new_posts_detected', 0)} |")
        lines.append(f"| Comments after | {scrape.get('comments_after', 'N/A')} |")
        lines.append(f"| Smoke test | {scrape.get('smoke_test', 'N/A')} |")
        lines.append(f"| Return code | {scrape.get('returncode', 'N/A')} |")
        lines.append("")

        # --- Labeling ---
        label = ing.get("label", {})
        post_process = ing.get("post_process", {})
        lines.append("## 2. Labeling & Post-Processing")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Labeling skipped | {label.get('skipped', False)} |")
        lines.append(f"| New posts to label | {label.get('new_posts_to_label', 0)} |")
        lines.append(f"| New labels created | {label.get('new_labels_created', 0)} |")
        lines.append(f"| Post-processing skipped | {post_process.get('skipped', False)} |")
        lines.append(f"| Processed rows | {post_process.get('rows', 'N/A')} |")
        lines.append(f"| Label return code | {label.get('returncode', 'N/A')} |")
        lines.append("")

        # --- Dataset profile ---
        profile = ing.get("profile", {})
        review = ing.get("review", {})
        lines.append("### Dataset Profile")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Train rows | {profile.get('train_rows', 'N/A')} |")
        lines.append(f"| Test rows | {profile.get('test_rows', 'N/A')} |")
        lines.append(f"| Label noise score | {profile.get('label_noise_score', 'N/A')} |")
        lines.append(f"| Label review recommendation | {review.get('recommendation', 'N/A')} |")
        lines.append(f"| Ambiguous ratio | {review.get('ambiguous_ratio', 'N/A')} |")
        lines.append("")

        # --- Balance / CTGAN ---
        bal = report.get("balance_agent", {})
        sq = bal.get("synthetic_quality", {})
        lines.append("## 3. Balancing (CTGAN)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Best ratio | {bal.get('ratio_search', {}).get('best_ratio', 'N/A')} |")
        lines.append(f"| CTGAN return code | {bal.get('ctgan_result', {}).get('returncode', 'N/A')} |")
        lines.append(f"| Synthetic rows | {sq.get('synthetic_rows', 'N/A')} |")
        lines.append(f"| Mean JSD | {sq.get('mean_jsd', 'N/A')} |")
        lines.append(f"| Quality accepted | {sq.get('accepted', 'N/A')} |")
        lines.append(f"| Balance accepted | {bal.get('accepted', 'N/A')} |")
        lines.append("")

        # --- Training ---
        tr = report.get("training_agent", {})
        tr_data = tr.get("training", {})
        lines.append("## 4. Classifier Training")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Return code | {tr_data.get('returncode', 'N/A')} |")
        lines.append(f"| Best F1 | {tr_data.get('best_f1', 'N/A')} |")
        lines.append(f"| Passed | {tr.get('passed', 'N/A')} |")
        lines.append("")

        supervisor = report.get("supervisor_agent", {})
        policy = report.get("policy_agent", {})
        lines.append("## 5. Supervisor / Policy")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Route lane | {supervisor.get('lane', 'N/A')} |")
        lines.append(f"| Escalation score | {policy.get('escalation_score', 'N/A')} |")
        lines.append(f"| Recommended action | {policy.get('recommended_action', 'N/A')} |")
        lines.append(f"| Policy mode | {policy.get('proposal', {}).get('mode', 'N/A')} |")
        lines.append(f"| Policy approved | {policy.get('approval', {}).get('approved', 'N/A')} |")
        lines.append("")

        # --- Adversarial Training ---
        strat = report.get("strategy_agent", {})
        adv = strat.get("adversarial_training", {})
        lines.append("## 6. Adversarial Training (Strategy Agent)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Focus | {adv.get('focus', 'N/A')} |")
        lines.append(f"| Recommended focus | {adv.get('recommended_focus', 'N/A')} |")
        lines.append(f"| Training return code | {adv.get('training_returncode', 'N/A')} |")
        lines.append(f"| Robustness return code | {adv.get('robustness_returncode', 'N/A')} |")
        lines.append(f"| Baseline FGSM F1 | {adv.get('baseline_attack_f1', 'N/A')} |")
        lines.append(f"| Adversarial FGSM F1 | {adv.get('adversarial_attack_f1', 'N/A')} |")
        lines.append(f"| Adversarial Clean F1 | {adv.get('adversarial_clean_f1', 'N/A')} |")
        lines.append(f"| Robustness gain (Δ) | {adv.get('robustness_gain', 'N/A')} |")
        lines.append(f"| Accepted | {strat.get('accepted', 'N/A')} |")
        lines.append("")

        # Attack surface
        attack = adv.get("attack_surface", {})
        if attack:
            lines.append("### Attack Surface")
            lines.append("")
            channels = attack.get("channel_counts", {})
            if channels:
                lines.append("| Fraud Channel | Count |")
                lines.append("|---------------|-------|")
                for ch, cnt in channels.items():
                    lines.append(f"| {ch} | {cnt} |")
                lines.append("")

        # --- Evaluation ---
        ev = report.get("evaluation_agent", {})
        ev_data = ev.get("evaluation", {})
        lines.append("## 7. Final Evaluation")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Eval return code | {ev_data.get('eval_returncode', 'N/A')} |")
        lines.append(f"| Robustness return code | {ev_data.get('robustness_returncode', 'N/A')} |")
        lines.append(f"| Best F1 | {ev_data.get('best_f1', 'N/A')} |")
        lines.append(f"| Robustness score | {ev_data.get('robustness_score', 'N/A')} |")
        lines.append(f"| Passed | {ev.get('passed', 'N/A')} |")
        lines.append(f"| Correction target | {ev.get('correction_target', 'N/A')} |")
        lines.append("")

        sim = report.get("simulation_agent", {})
        sim_data = sim.get("simulation", {})
        lines.append("## 8. Simulation Gate")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Proposal mode | {sim_data.get('proposal_mode', 'N/A')} |")
        lines.append(f"| Simulation score | {sim_data.get('simulation_score', 'N/A')} |")
        lines.append(f"| Guardrail status | {sim_data.get('guardrail_status', 'N/A')} |")
        lines.append(f"| Approved | {sim.get('approved', 'N/A')} |")
        lines.append("")

        # --- Agent decisions timeline ---
        decisions = report.get("decisions", [])
        if decisions:
            lines.append("## Agent Decisions Timeline")
            lines.append("")
            lines.append("| # | Agent | Action | Confidence | Summary |")
            lines.append("|---|-------|--------|------------|---------|")
            for i, d in enumerate(decisions, 1):
                summary = (d.get("summary", "") or "")[:120].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {i} | {d.get('agent', '')} | {d.get('action', '')} | {d.get('confidence', '')} | {summary} |")
            lines.append("")

        # --- Charts ---
        charts_dir = OUTPUT_DIR / "adversarial" / "charts"
        if charts_dir.exists():
            chart_files = sorted(charts_dir.glob("*.png"))
            if chart_files:
                lines.append("## Adversarial Training Charts")
                lines.append("")
                lines.append("Charts copied to `output/adversarial/charts/`:")
                lines.append("")
                for cf in chart_files:
                    lines.append(f"- `{cf.name}`")
                lines.append("")

        return "\n".join(lines)

    def complete(self, state: WorkflowState) -> WorkflowState:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "completed" if state.get("done") else "stopped",
            "iteration": state.get("iteration", 0),
            "agent_attempts": state.get("agent_attempts", {}),
            "decisions": state.get("decisions", []),
            "ingestion_agent": state.get("ingestion_agent", {}),
            "balance_agent": state.get("balance_agent", {}),
            "training_agent": state.get("training_agent", {}),
            "supervisor_agent": state.get("supervisor_agent", {}),
            "policy_agent": state.get("policy_agent", {}),
            "strategy_agent": state.get("strategy_agent", {}),
            "evaluation_agent": state.get("evaluation_agent", {}),
            "simulation_agent": state.get("simulation_agent", {}),
            "scores": state.get("scores", {}),
            "proposal": state.get("proposal", {}),
            "approval": state.get("approval", {}),
            "outcome": state.get("outcome", {}),
        }
        RUN_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        # ── Per-stage outputs ──────────────────────────────────────────
        ing = state.get("ingestion_agent", {})
        self._write_stage_output("scraping", "scraping_result.json", ing.get("scrape", {}))
        self._write_stage_output("labeling", "labeling_result.json", ing.get("label", {}))
        self._write_stage_output("labeling", "post_processing_result.json", ing.get("post_process", {}))
        self._write_stage_output("labeling", "dataset_append_result.json", ing.get("append", {}))
        self._write_stage_output("labeling", "dataset_profile.json", ing.get("profile", {}))
        self._write_stage_output("labeling", "label_review.json", ing.get("review", {}))

        bal = state.get("balance_agent", {})
        self._write_stage_output("training", "balance_search.json", bal.get("ratio_search", {}))
        self._write_stage_output("training", "ctgan_result.json", bal.get("ctgan_result", {}))
        self._write_stage_output("training", "synthetic_quality.json", bal.get("synthetic_quality", {}))

        tr = state.get("training_agent", {})
        self._write_stage_output("training", "classifier_training.json", tr.get("training", {}))
        self._write_stage_output("training", "supervisor_decision.json", state.get("supervisor_agent", {}))
        self._write_stage_output("training", "policy_decision.json", state.get("policy_agent", {}))

        strat = state.get("strategy_agent", {})
        self._write_stage_output("adversarial", "adversarial_training.json", strat.get("adversarial_training", {}))

        ev = state.get("evaluation_agent", {})
        self._write_stage_output("evaluation", "evaluation_result.json", ev.get("evaluation", {}))
        self._write_stage_output("evaluation", "simulation_result.json", state.get("simulation_agent", {}))

        # ── Copy adversarial charts ────────────────────────────────────
        chart_paths = self._copy_adversarial_charts()

        # ── Run summary (human-readable) ──────────────────────────────
        summary_md = self._generate_run_summary(report)
        summary_path = OUTPUT_DIR / "run_summary.md"
        summary_path.write_text(summary_md, encoding="utf-8")

        self.memory.record_run(report)
        self._trace_event("workflow", "run_completed", iteration=state.get("iteration", 0), payload=report)
        print("\n" + "="*60)
        print(f" [ WORKFLOW {report['status'].upper()} ] iteration: {report['iteration']}")
        print(f" 📂 Outputs written to: {OUTPUT_DIR}")
        print(f" 📝 Summary: {summary_path}")
        if chart_paths:
            print(f" 📊 Charts copied: {len(chart_paths)} files")
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
        graph.add_node("supervisor_agent", self.supervisor_agent)
        graph.add_node("policy_agent", self.policy_agent)
        graph.add_node("strategy_agent", self.strategy_agent)
        graph.add_node("evaluation_agent", self.evaluation_agent)
        graph.add_node("simulation_agent", self.simulation_agent)
        graph.add_node("complete", self.complete)

        graph.add_edge(START, "ingestion_agent")
        graph.add_conditional_edges("ingestion_agent", self.route_after_ingestion_agent)
        graph.add_conditional_edges("balance_agent", self.route_after_balance_agent)
        graph.add_conditional_edges("training_agent", self.route_after_training_agent)
        graph.add_conditional_edges("supervisor_agent", self.route_after_supervisor_agent)
        graph.add_conditional_edges("policy_agent", self.route_after_policy_agent)
        graph.add_conditional_edges("strategy_agent", self.route_after_strategy_agent)
        graph.add_conditional_edges("evaluation_agent", self.route_after_evaluation_agent)
        graph.add_conditional_edges("simulation_agent", self.route_after_simulation_agent)
        graph.add_edge("complete", END)
        return graph.compile()


def _base_state() -> WorkflowState:
    return {"iteration": 0, "agent_attempts": {}, "decisions": [], "done": False, "status": "running"}


def _deep_merge_state(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_state(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_stub_state_from_file(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stub state file must contain a JSON object: {path}")
    return payload


def _manual_run(workflow: FraudWorkflow, start_at: str, state: WorkflowState) -> dict[str, Any]:
    step_handlers = {
        "ingestion_agent": workflow.ingestion_agent,
        "balance_agent": workflow.balance_agent,
        "training_agent": workflow.training_agent,
        "supervisor_agent": workflow.supervisor_agent,
        "policy_agent": workflow.policy_agent,
        "strategy_agent": workflow.strategy_agent,
        "evaluation_agent": workflow.evaluation_agent,
        "simulation_agent": workflow.simulation_agent,
    }
    route_handlers = {
        "ingestion_agent": workflow.route_after_ingestion_agent,
        "balance_agent": workflow.route_after_balance_agent,
        "training_agent": workflow.route_after_training_agent,
        "supervisor_agent": workflow.route_after_supervisor_agent,
        "policy_agent": workflow.route_after_policy_agent,
        "strategy_agent": workflow.route_after_strategy_agent,
        "evaluation_agent": workflow.route_after_evaluation_agent,
        "simulation_agent": workflow.route_after_simulation_agent,
    }

    current = start_at
    while current != "complete":
        state = step_handlers[current](state)
        current = route_handlers[current](state)
    return workflow.complete(state)


def run_workflow() -> dict[str, Any]:
    workflow = FraudWorkflow()
    start_at = os.getenv("AGENT_START_AT", "ingestion_agent").strip() or "ingestion_agent"
    if start_at not in MANUAL_START_NODES:
        raise ValueError(f"Unsupported AGENT_START_AT={start_at!r}. Expected one of: {sorted(MANUAL_START_NODES)}")

    stub_state = _load_stub_state_from_file(os.getenv("AGENT_STATE_FILE"))
    state: WorkflowState = _deep_merge_state(_base_state(), stub_state)

    if start_at == "ingestion_agent":
        app = workflow.build()
        return app.invoke(state)
    return _manual_run(workflow, start_at, state)
