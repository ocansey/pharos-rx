"""Synthesis and repair nodes."""

from __future__ import annotations

from typing import Any

from pharos.agent.state import PharosState, trace_event
from pharos.config import PharosConfig
from pharos.llm.prompts import REPAIR_PROMPT, SYNTHESIZE_PROMPT


def make_synthesize_node(llm: Any, cfg: PharosConfig):
    """Build the synthesis node."""

    def node(state: PharosState) -> dict:
        messages = SYNTHESIZE_PROMPT.format_messages(
            query=state["query"],
            statistics=state.get("statistics_text", ""),
            panel=state.get("panel_text", ""),
            n_units=state.get("n_units", 0),
        )
        response = llm.invoke(messages)
        draft = str(response.content).strip()
        return {
            "draft": draft,
            "trace": [
                trace_event("synthesize", chars=len(draft), round=state.get("repair_rounds", 0))
            ],
        }

    return node


def make_repair_node(llm: Any, cfg: PharosConfig):
    """Build the repair node.

    Repair is *subtractive*. The prompt asks for the failed claims to be dropped,
    not rescued, and the loop is bounded at ``agent.max_repair_rounds``. An
    unbounded loop against a model that keeps reasserting the same ungrounded
    claim is not self-correction, it is an expensive way to arrive at the same
    answer; §5.5 of the results shows the second round buys nothing.
    """

    def node(state: PharosState) -> dict:
        failed = [c for c in state.get("claims", []) if c["verdict"] == "UNSUPPORTED"]
        failed_text = "\n".join(f"- {c['text']}  ({c['reason']})" for c in failed) or "(none)"

        messages = REPAIR_PROMPT.format_messages(
            query=state["query"],
            statistics=state.get("statistics_text", ""),
            panel=state.get("panel_text", ""),
            draft=state.get("draft", ""),
            failed=failed_text,
        )
        response = llm.invoke(messages)
        draft = str(response.content).strip()
        rounds = state.get("repair_rounds", 0) + 1
        return {
            "draft": draft,
            "repair_rounds": rounds,
            "trace": [trace_event("repair", round=rounds, dropped_targets=len(failed))],
        }

    return node
