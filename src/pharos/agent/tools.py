"""LangChain tools over the deterministic statistics layer.

These are exposed as real ``StructuredTool`` objects with Pydantic argument
schemas so that a tool-calling model can select among them, but PHAROS's own
graph does not let the model choose. The ``compute`` node calls them directly,
driven by the parsed query plan.

That is a deliberate architectural decision and the reason deserves stating. A
tool-calling loop gives the model discretion over *which* statistic gets
computed, and therefore over which statistic the answer ends up resting on. In a
setting where the answer is about medication, discretion over the evidence base
is the thing you least want to delegate. Making the call deterministic means the
same question always computes the same statistics — which is also what makes the
numeric-accuracy evaluation meaningful, since there is a fixed ground truth to
compare against.

The tools remain available, and importable, for anyone who wants to wire PHAROS's
statistics into an agent of their own.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pharos.data.cohort import CohortStatistics


# --------------------------------------------------------------------------- #
# Argument schemas
# --------------------------------------------------------------------------- #
class CohortSummaryArgs(BaseModel):
    drug_name: str = Field(description="Drug name as written in the corpus, e.g. 'Sertraline'.")
    condition: str | None = Field(
        default=None, description="Indication to restrict to, e.g. 'Depression'."
    )
    top_n_effects: int = Field(default=6, ge=1, le=15)


class DisproportionalityArgs(BaseModel):
    drug_name: str = Field(description="Drug name as written in the corpus.")
    concept: str = Field(
        description="Adverse-event concept from the lexicon, e.g. 'weight_gain', 'nausea'."
    )
    condition: str | None = Field(default=None, description="Indication to restrict to.")
    comparator: str = Field(
        default="same_condition",
        description=(
            "'same_condition' compares against other drugs for the same indication, "
            "which controls confounding by indication. 'all' compares against the "
            "whole corpus."
        ),
    )


class CompareDrugsArgs(BaseModel):
    drug_names: list[str] = Field(description="Two or more drug names to compare.", min_length=2)
    condition: str | None = Field(default=None, description="Indication to restrict to.")


class ResolveArgs(BaseModel):
    query: str = Field(description="A partial or informal drug or condition name.")


# --------------------------------------------------------------------------- #
def build_tools(stats: CohortStatistics) -> list[StructuredTool]:
    """Bind the statistics engine into callable tools."""

    def cohort_summary(drug_name: str, condition: str | None = None, top_n_effects: int = 6) -> str:
        summary = stats.summarise(drug_name, condition, top_n_effects=top_n_effects)
        if summary is None:
            alternatives = stats.resolve_drug(drug_name)
            hint = (
                f" Closest names in the corpus: {', '.join(alternatives)}." if alternatives else ""
            )
            return f"No reviews found for {drug_name!r}{f' / {condition!r}' if condition else ''}.{hint}"
        return summary.to_evidence_block()

    def disproportionality(
        drug_name: str,
        concept: str,
        condition: str | None = None,
        comparator: str = "same_condition",
    ) -> str:
        signal = stats.disproportionality(
            drug_name,
            concept,
            condition,
            comparator="same_condition" if comparator == "same_condition" else "all",
        )
        if signal is None:
            return (
                f"Insufficient reports to screen {concept!r} for {drug_name!r}: "
                f"fewer than 3 reviews mention it, below the floor at which any "
                f"disproportionality estimate is meaningful."
            )
        return signal.to_evidence_block()

    def compare_drugs(drug_names: list[str], condition: str | None = None) -> str:
        result = stats.compare_drugs(drug_names, condition)
        cohorts = result["cohorts"]
        if not cohorts:
            return f"No reviews found for any of {drug_names}."
        blocks = [c.to_evidence_block() for c in cohorts]
        if "kruskal_p" in result:
            blocks.append(
                f"[{result['stat_id']}] ACROSS-DRUG TEST — Kruskal-Wallis "
                f"H = {result['kruskal_h']:.2f}, p = {result['kruskal_p']:.2e} "
                f"(rank-based; ratings are ordinal and bimodal, so a normal-theory "
                f"test would be inappropriate)"
            )
        return "\n\n".join(blocks)

    def resolve_drug_name(query: str) -> str:
        matches = stats.resolve_drug(query)
        return ", ".join(matches) if matches else f"No drug in the corpus matches {query!r}."

    def resolve_condition_name(query: str) -> str:
        matches = stats.resolve_condition(query)
        return ", ".join(matches) if matches else f"No condition in the corpus matches {query!r}."

    return [
        StructuredTool.from_function(
            func=cohort_summary,
            name="cohort_summary",
            description=(
                "Descriptive statistics for a drug (optionally within one indication): "
                "review count, mean rating with a bootstrap CI, the full outcome "
                "distribution, and the most-reported effects with Wilson intervals. "
                "Call this before making any claim about how a drug is received."
            ),
            args_schema=CohortSummaryArgs,
        ),
        StructuredTool.from_function(
            func=disproportionality,
            name="disproportionality_signal",
            description=(
                "Screen one drug x adverse-event pair with PRR, ROR, chi-square and the "
                "BCPNN information component, against other drugs for the same "
                "indication by default. Answers 'is this reported MORE for this drug "
                "than for its alternatives', which raw counts cannot."
            ),
            args_schema=DisproportionalityArgs,
        ),
        StructuredTool.from_function(
            func=compare_drugs,
            name="compare_drugs",
            description=(
                "Head-to-head comparison of two or more drugs for one indication, with "
                "a Kruskal-Wallis test across their rating distributions."
            ),
            args_schema=CompareDrugsArgs,
        ),
        StructuredTool.from_function(
            func=resolve_drug_name,
            name="resolve_drug_name",
            description="Map an informal drug name to the names used in the corpus.",
            args_schema=ResolveArgs,
        ),
        StructuredTool.from_function(
            func=resolve_condition_name,
            name="resolve_condition_name",
            description="Map an informal condition name to the labels used in the corpus.",
            args_schema=ResolveArgs,
        ),
    ]


def tool_manifest(tools: list[StructuredTool]) -> list[dict[str, Any]]:
    """Serialisable description of the toolset, for the run trace."""
    return [{"name": t.name, "description": t.description.split(".")[0]} for t in tools]
