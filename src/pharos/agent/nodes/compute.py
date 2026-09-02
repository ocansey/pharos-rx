"""Compute node — deterministic statistics, selected without model discretion.

Which statistics get computed is a function of the parsed plan and the aspects
of the retrieved panel, not of a model's choice. Three things follow, and all
three are the point:

* The same question always produces the same statistics, so the numeric-accuracy
  evaluation has a fixed ground truth to score against.
* The model cannot decline to compute an inconvenient statistic, or compute a
  flattering one instead.
* A statistic that would be unstable is never produced at all, rather than being
  produced and hedged. Below the support floor, the honest output is silence.
"""

from __future__ import annotations

from pharos.agent.state import PharosState, trace_event
from pharos.config import PharosConfig
from pharos.data.cohort import CohortStatistics
from pharos.data.schema import Aspect

#: Adverse-event concepts screened for disproportionality when the question is
#: about side effects. Capped so a single answer cannot carry a screen of every
#: concept in the lexicon — that would be a multiple-comparisons problem dressed
#: up as thoroughness.
MAX_SCREENED_CONCEPTS = 4


def make_compute_node(stats: CohortStatistics, cfg: PharosConfig):
    def node(state: PharosState) -> dict:
        blocks: list[str] = []
        detail: list[dict] = []
        drug = state.get("plan_drug")
        condition = state.get("plan_condition")

        if not drug or not cfg.agent.enable_cohort_stats:
            return {
                "statistics_text": "(no cohort identified in the question; no statistics computed)",
                "statistic_ids": [],
                "statistics_detail": [],
                "trace": [trace_event("compute", computed=0, reason="no drug in plan")],
            }

        # --- descriptive ------------------------------------------------ #
        summary = stats.summarise(drug, condition)
        if summary is None and condition:
            # The drug is in the corpus but not for this indication. Widen once,
            # and say so in the trace rather than silently answering about a
            # different population than the one asked about.
            summary = stats.summarise(drug, None)
            if summary is not None:
                condition = None

        if summary is not None:
            blocks.append(summary.to_evidence_block())
            detail.append(
                {
                    "stat_id": summary.stat_id,
                    "kind": "cohort_summary",
                    "drug": summary.drug_name,
                    "condition": summary.condition,
                    "n_reviews": summary.n_reviews,
                    "mean_rating": round(summary.mean_rating, 3),
                    "rating_ci": [round(v, 3) for v in summary.rating_ci],
                    "stratum_distribution": summary.stratum_distribution,
                    "low_support": summary.low_support,
                }
            )

        # --- disproportionality ---------------------------------------- #
        asks_adverse = Aspect.ADVERSE_EFFECT.value in (state.get("plan_aspects") or [])
        if cfg.agent.enable_disproportionality and summary is not None and asks_adverse:
            for concept, _count, _prop, _ci in summary.top_adverse_concepts[:MAX_SCREENED_CONCEPTS]:
                signal = stats.disproportionality(drug, concept, condition)
                if signal is None:
                    continue
                blocks.append(signal.to_evidence_block())
                detail.append(
                    {
                        "stat_id": signal.stat_id,
                        "kind": "disproportionality",
                        "drug": signal.drug_name,
                        "concept": signal.concept,
                        "prr": round(signal.prr, 3),
                        "prr_ci": [round(v, 3) for v in signal.prr_ci],
                        "chi2": round(signal.chi2, 3),
                        "ic025": round(signal.information_component - 0.0, 3),
                        "flagged": signal.flagged,
                    }
                )

        text = (
            "\n\n".join(blocks) if blocks else "(no statistics could be computed for this cohort)"
        )
        return {
            "statistics_text": text,
            "statistic_ids": [d["stat_id"] for d in detail],
            "statistics_detail": detail,
            "trace": [
                trace_event(
                    "compute",
                    computed=len(detail),
                    kinds=sorted({d["kind"] for d in detail}),
                    drug=drug,
                    condition=condition,
                )
            ],
        }

    return node
