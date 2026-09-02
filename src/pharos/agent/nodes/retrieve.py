"""Retrieval node — builds the evidence panel."""

from __future__ import annotations

from pharos.agent.state import PharosState, trace_event
from pharos.config import PharosConfig
from pharos.retrieval.retriever import PharosRetriever


def make_retrieve_node(retriever: PharosRetriever, cfg: PharosConfig):
    """Build the retrieval node.

    Sufficiency is decided here rather than at generation time, and it is decided
    on the *panel*, not on scores. A panel of two units can produce a fluent
    answer; it cannot produce a defensible one, and a system that only notices
    that after generating has already lost — the fluent draft will be hard to
    throw away.
    """

    def node(state: PharosState) -> dict:
        result = retriever.retrieve(
            state["query"],
            drug_name=state.get("drug_name"),
            condition=state.get("condition"),
        )

        # A panel with no identified cohort is not sufficient, however many units
        # it contains. Every statistic this system computes is cohort-relative,
        # so an answer assembled from units belonging to no particular drug or
        # indication would be anecdote with citations attached -- exactly the
        # failure mode the architecture exists to prevent.
        has_cohort = bool(result.plan.drug_name or result.plan.condition)
        sufficient = has_cohort and result.is_sufficient(cfg.retrieval.abstain_below_units)

        return {
            "plan_drug": result.plan.drug_name,
            "plan_condition": result.plan.condition,
            "plan_aspects": [a.value for a in result.plan.aspects],
            "panel_text": retriever.format_panel(result),
            "panel_unit_ids": [ru.unit.unit_id for ru in result.units],
            "n_units": len(result.units),
            "n_candidates": result.n_candidates,
            "valence_skew": result.valence_skew,
            "rating_error": result.rating_error,
            "cohort_distribution": {s.value: v for s, v in result.cohort_distribution.items()},
            "cohort_mean_rating": result.cohort_mean_rating,
            "allocation": result.allocation,
            "retrieval_sufficient": sufficient,
            "trace": [
                trace_event(
                    "retrieve",
                    plan=result.plan.describe(),
                    candidates=result.n_candidates,
                    panel=len(result.units),
                    stratified=result.stratified,
                    valence_skew=round(result.valence_skew, 4),
                    rating_error=round(result.rating_error, 3),
                    filtered_to_empty=result.filtered_to_empty,
                    has_cohort=has_cohort,
                )
            ],
        }

    return node
