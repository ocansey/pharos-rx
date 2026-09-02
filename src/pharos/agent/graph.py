"""The PHAROS graph.

                          ┌──────────┐
                          │  triage  │
                          └────┬─────┘
              refuse ◄─────────┤ (crisis · personal advice · out of scope)
                               │ informational
                          ┌────▼─────┐
                          │ retrieve │
                          └────┬─────┘
             abstain ◄─────────┤ (panel below the sufficiency floor)
                               │
                          ┌────▼─────┐
                          │ compute  │   deterministic statistics
                          └────┬─────┘
                          ┌────▼──────┐
                          │ synthesize│
                          └────┬──────┘
                          ┌────▼─────┐
                          │  verify  │   structural + entailment
                          └────┬─────┘
                    repair ◄───┤ (unsupported claims, budget remaining)
                       │       │
                       └───────┤
                          ┌────▼─────┐
                          │ finalize │
                          └──────────┘

The shape is the argument. Refusal comes before retrieval, so a question this
system should not answer never touches the corpus. Statistics come before
generation, so the model receives numbers rather than being asked for them.
Verification comes after generation and can send the answer back, so grounding is
enforced rather than requested. And every edge that could loop is bounded.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from pharos.agent.nodes.compute import make_compute_node
from pharos.agent.nodes.finalize import (
    make_abstain_node,
    make_finalize_node,
    make_refuse_node,
)
from pharos.agent.nodes.retrieve import make_retrieve_node
from pharos.agent.nodes.synthesize import make_repair_node, make_synthesize_node
from pharos.agent.nodes.triage import make_triage_node
from pharos.agent.nodes.verify import make_verify_node
from pharos.agent.state import PharosState, new_state
from pharos.config import PharosConfig
from pharos.data.cohort import CohortStatistics
from pharos.llm.factory import build_chat_model, provider_is_live
from pharos.retrieval.retriever import PharosRetriever
from pharos.safety.triage import SafetyTriage, Verdict


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #
def route_after_triage(state: PharosState) -> str:
    return "retrieve" if state.get("triage_verdict") == Verdict.INFORMATIONAL.value else "refuse"


def route_after_retrieve(state: PharosState) -> str:
    return "compute" if state.get("retrieval_sufficient") else "abstain"


def make_route_after_verify(cfg: PharosConfig):
    def route(state: PharosState) -> str:
        unsupported = state.get("unsupported_count", 0)
        rounds = state.get("repair_rounds", 0)
        if unsupported > 0 and rounds < cfg.agent.max_repair_rounds:
            return "repair"
        return "finalize"

    return route


# --------------------------------------------------------------------------- #
def build_graph(
    retriever: PharosRetriever,
    stats: CohortStatistics,
    cfg: PharosConfig,
    llm: Any | None = None,
):
    """Compile the graph.

    The model layers of triage and verification are enabled only when a live
    provider is configured. Running them against the deterministic mock would
    produce verdicts that look like judgements but are pattern matches — and
    would put those pretend judgements into the evaluation numbers.
    """
    llm = llm or build_chat_model(cfg.llm)
    live = provider_is_live(cfg.llm)
    vocabulary = set(retriever.index.by_drug) | set(retriever.index.by_condition)
    triage = SafetyTriage(cfg.safety, vocabulary=vocabulary)

    graph = StateGraph(PharosState)
    graph.add_node("triage", make_triage_node(triage, llm, use_model=live))
    graph.add_node("retrieve", make_retrieve_node(retriever, cfg))
    graph.add_node("compute", make_compute_node(stats, cfg))
    graph.add_node("synthesize", make_synthesize_node(llm, cfg))
    graph.add_node("verify", make_verify_node(llm, cfg, use_model=live))
    graph.add_node("repair", make_repair_node(llm, cfg))
    graph.add_node("finalize", make_finalize_node(cfg))
    graph.add_node("refuse", make_refuse_node())
    graph.add_node("abstain", make_abstain_node(cfg))

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage", route_after_triage, {"retrieve": "retrieve", "refuse": "refuse"}
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"compute": "compute", "abstain": "abstain"}
    )
    graph.add_edge("compute", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify", make_route_after_verify(cfg), {"repair": "repair", "finalize": "finalize"}
    )
    graph.add_edge("repair", "verify")
    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)
    graph.add_edge("abstain", END)

    return graph.compile()


# --------------------------------------------------------------------------- #
class PharosAgent:
    """Convenience wrapper: build once, ask many times."""

    def __init__(
        self,
        retriever: PharosRetriever,
        stats: CohortStatistics,
        cfg: PharosConfig,
        llm: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.stats = stats
        self.llm = llm or build_chat_model(cfg.llm)
        self.graph = build_graph(retriever, stats, cfg, self.llm)

    def ask(
        self, query: str, drug_name: str | None = None, condition: str | None = None
    ) -> PharosState:
        """Answer one question. Returns the full final state, not just the text.

        Returning the state rather than a string is the point: the answer is one
        field among the triage verdict, the panel, the statistics, the claim
        verdicts and the trace. Anyone auditing the answer needs the rest.
        """
        return self.graph.invoke(new_state(query, drug_name, condition))

    def answer(self, query: str, **kwargs: Any) -> str:
        return self.ask(query, **kwargs).get("answer", "")
