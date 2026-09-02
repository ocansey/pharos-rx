"""Triage node — the first gate."""

from __future__ import annotations

from typing import Any

from pharos.agent.state import PharosState, trace_event
from pharos.safety.triage import SafetyTriage, Verdict


def make_triage_node(triage: SafetyTriage, llm: Any, use_model: bool = True):
    """Build the triage node.

    The rule layer runs unconditionally. The model layer runs only on queries the
    rules cleared, and only to *escalate* — so a model failure degrades to the
    rule verdict rather than opening the gate. Any exception from the model is
    swallowed for the same reason: an unreachable API must not turn into an
    unguarded answer.
    """

    def node(state: PharosState) -> dict:
        query = state["query"]
        result = triage.classify_rules(query)

        if use_model and result.verdict is Verdict.INFORMATIONAL:
            from pharos.llm.prompts import TRIAGE_PROMPT

            try:
                response = llm.invoke(TRIAGE_PROMPT.format_messages(query=query))
                model_verdict = SafetyTriage.parse_model_verdict(str(response.content))
                result = SafetyTriage.merge(result, model_verdict)
            except Exception as exc:
                result.rationale += f" (model layer unavailable: {type(exc).__name__})"

        return {
            "triage_verdict": result.verdict.value,
            "triage_rationale": result.rationale,
            "triage_source": result.source,
            "triage_matched": result.matched,
            "trace": [
                trace_event(
                    "triage",
                    verdict=result.verdict.value,
                    source=result.source,
                    matched=result.matched,
                )
            ],
        }

    return node
