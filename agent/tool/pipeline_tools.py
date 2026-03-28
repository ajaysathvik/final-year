from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


@dataclass
class ToolSpec:
    name: str
    description: str
    inputs: list[str]
    outputs: list[str]

    @property
    def signature(self) -> str:
        payload = "|".join([self.name, self.description, ",".join(self.inputs), ",".join(self.outputs)])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "signature": self.signature,
        }


class FraudIntelTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="latest_fraud_intel",
            description="Pulls recent fraud and scam attack signals from current news/RSS search feeds.",
            inputs=["search queries", "max_items"],
            outputs=["dated fraud signal list with titles and source links"],
        )
        self.queries = [
            "financial fraud scam phishing bank impersonation",
            "gift card scam fraud alert",
            "crypto investment scam fraud alert",
        ]

    def _rss_url(self, query: str) -> str:
        return (
            "https://news.google.com/rss/search?q="
            f"{quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )

    def _parse_feed(self, query: str) -> list[dict[str, Any]]:
        request = Request(
            self._rss_url(query),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=20) as response:
            content = response.read()

        root = ET.fromstring(content)
        signals = []
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            if not title or not link:
                continue
            published_at = pub_date
            if pub_date:
                try:
                    published_at = parsedate_to_datetime(pub_date).isoformat()
                except (TypeError, ValueError, IndexError):
                    published_at = pub_date
            signals.append(
                {
                    "query": query,
                    "title": title,
                    "link": link,
                    "published_at": published_at,
                }
            )
        return signals

    def run(self, max_items: int = 8) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        seen_links: set[str] = set()
        for query in self.queries:
            for signal in self._parse_feed(query):
                if signal["link"] in seen_links:
                    continue
                seen_links.add(signal["link"])
                signals.append(signal)

        signals.sort(key=lambda item: item.get("published_at", ""), reverse=True)
        return signals[:max_items]


class PythonScriptTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="python_pipeline_runner",
            description="Runs one Python stage in the Fin-Fraud AI pipeline and captures stdout/stderr.",
            inputs=["script_path"],
            outputs=["returncode", "stdout", "stderr"],
        )

    def run(self, script_path: Path, cwd: Path, stream: bool = True) -> dict[str, Any]:
        if not stream:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
            return {
                "script": str(script_path),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout_chunks: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stdout_chunks.append(line)
        process.wait()
        return {
            "script": str(script_path),
            "returncode": process.returncode,
            "stdout": "".join(stdout_chunks),
            "stderr": "",
        }


class OllamaAdvisorTool:
    def __init__(self, model: str = "qwen3.5:0.8b") -> None:
        self.model = model
        self.spec = ToolSpec(
            name="ollama_classifier_advisor",
            description="Uses a local Ollama model to assess new fraud patterns and suggest classifier updates.",
            inputs=["dataset snapshot", "latest fraud signals"],
            outputs=["llm assessment with retraining and feature recommendations"],
        )

    def run(self, dataset_snapshot: dict[str, Any], fraud_signals: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage
            chat = ChatOllama(
                model=self.model,
                temperature=0.3,
                num_predict=1024,
            )
        except ImportError as exc:
            return {
                "model": self.model,
                "status": "unavailable",
                "error": f"langchain_ollama or langchain_core missing: {exc}",
                "assessment": "",
            }

        signals_block = fraud_signals[:5]
        prompt = (
            "You are the decision-maker for a fraud-detection training pipeline.\n"
            "Given the current dataset summary, recent Reddit-ingestion results, previous selected model, "
            "and recent fraud/cybercrime signals, decide the training plan yourself.\n\n"
            f"Dataset snapshot:\n{dataset_snapshot}\n\n"
            f"Recent fraud signals:\n{signals_block}\n\n"
            "Output format:\n"
            "Thinking...\n"
            "Thinking Process:\n"
            "1. Short analysis of the dataset changes and new fraud signals.\n"
            "2. Short decision notes on retraining and augmentation choice.\n"
            "...done thinking.\n\n"
            "Final Answer:\n"
            "Return one valid JSON object only after the 'Final Answer:' label. "
            "Use these keys exactly:\n"
            "{\n"
            '  "risk_level": "LOW|MEDIUM|HIGH",\n'
            '  "retrain": true,\n'
            '  "stop_early": false,\n'
            '  "augmentation_choice": "none|ctgan|adv_ctgan|full",\n'
            '  "evaluation_dataset_filter": ["Original", "CTGAN", "Adv-CTGAN"],\n'
            '  "reason": "short reason",\n'
            '  "confidence": 0.0,\n'
            '  "new_attack_types": [],\n'
            '  "feature_updates": [],\n'
            '  "classifier_updates": [],\n'
            '  "data_collection_updates": [],\n'
            '  "summary": {}\n'
            "}\n"
            "Rules:\n"
            "- If nothing materially new was ingested, you may set stop_early=true and augmentation_choice='none'.\n"
            "- If only one augmentation path should run, choose 'ctgan' or 'adv_ctgan'.\n"
            "- If both should run, choose 'full'.\n"
            "- evaluation_dataset_filter must match the augmentation choice.\n"
            "- Keep the thinking section concise.\n"
            "- Do not use markdown fences.\n"
            "- The JSON object must be the last thing in the response."
        )

        try:
            response = chat.invoke([HumanMessage(content=prompt)])
            content = response.content
        except Exception as exc:
            return {
                "model": self.model,
                "status": "error",
                "error": str(exc),
                "assessment": "",
            }

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        content = str(content).strip()
        if not content:
            return {
                "model": self.model,
                "status": "empty",
                "error": "Ollama returned an empty response.",
                "assessment": "",
                "visible_thinking": "",
                "final_json_text": "",
                "parsed_assessment": None,
            }

        parsed_assessment, json_text = self._extract_json_payload(content)
        visible_thinking = content
        if json_text:
            visible_thinking = content[: content.rfind(json_text)].rstrip()
        status = "ok" if parsed_assessment is not None else "malformed"
        return {
            "model": self.model,
            "status": status,
            "assessment": content,
            "visible_thinking": visible_thinking,
            "final_json_text": json_text,
            "parsed_assessment": parsed_assessment,
        }

    def _extract_json_payload(self, content: str) -> tuple[dict[str, Any] | None, str]:
        candidate_texts: list[str] = []
        stripped = content.strip()
        if stripped:
            candidate_texts.append(stripped)

        final_answer_index = content.rfind("Final Answer:")
        if final_answer_index != -1:
            candidate_texts.append(content[final_answer_index + len("Final Answer:"):].strip())

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate_texts.append(content[start:end + 1].strip())

        seen: set[str] = set()
        for candidate in candidate_texts:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                return json.loads(candidate), candidate
            except Exception:
                continue
        return None, ""
