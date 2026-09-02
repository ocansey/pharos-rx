"""Graph state.

A single ``TypedDict`` threaded through every node. Two rules keep it honest:

* Nodes return only the keys they changed. LangGraph merges shallowly, so a node
  that returns the whole state will clobber a sibling's work the moment anything
  runs in parallel.
* Nothing is deleted. The final state is the audit trail — triage verdict,
  retrieval diagnostics, the statistics that were computed, the draft before
  verification, every claim verdict, and what the repair loop changed. An answer
  you cannot reconstruct the derivation of is an answer you cannot defend.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class Claim(TypedDict):
    """One atomic assertion extracted from the draft."""

    index: int
    text: str
    citations: list[str]
    verdict: Literal["SUPPORTED", "UNSUPPORTED", "UNCHECKED"]
    reason: str


class PharosState(TypedDict, total=False):
    """State for the PHAROS graph."""

    # --- input ---------------------------------------------------------- #
    query: str
    drug_name: str | None
    condition: str | None
    messages: Annotated[list, add_messages]

    # --- triage --------------------------------------------------------- #
    triage_verdict: str
    triage_rationale: str
    triage_source: str
    triage_matched: list[str]

    # --- planning ------------------------------------------------------- #
    plan_drug: str | None
    plan_condition: str | None
    plan_aspects: list[str]

    # --- retrieval ------------------------------------------------------ #
    panel_text: str
    panel_unit_ids: list[str]
    n_units: int
    n_candidates: int
    valence_skew: float
    rating_error: float
    cohort_distribution: dict[str, float]
    cohort_mean_rating: float | None
    allocation: dict[str, Any]
    retrieval_sufficient: bool

    # --- computed statistics -------------------------------------------- #
    statistics_text: str
    statistic_ids: list[str]
    statistics_detail: list[dict[str, Any]]

    # --- generation ----------------------------------------------------- #
    draft: str
    claims: list[Claim]
    repair_rounds: int
    unsupported_count: int

    # --- output --------------------------------------------------------- #
    answer: str
    abstained: bool
    abstain_reason: str
    #: Appended to, never replaced. LangGraph's default merge is last-write-wins,
    #: which would leave the final state holding only the last node's record and
    #: silently destroy the audit trail this system's credibility rests on.
    trace: Annotated[list[dict[str, Any]], operator.add]


def new_state(
    query: str, drug_name: str | None = None, condition: str | None = None
) -> PharosState:
    return PharosState(
        query=query,
        drug_name=drug_name,
        condition=condition,
        messages=[],
        repair_rounds=0,
        abstained=False,
        trace=[],
    )


def trace_event(node: str, **fields: Any) -> dict[str, Any]:
    """One structured trace record. Appended, never replaced."""
    return {"node": node, **fields}
