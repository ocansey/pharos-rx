"""Terminal nodes: refusal, abstention, and finalisation."""

from __future__ import annotations

from pharos.agent.state import PharosState, trace_event
from pharos.config import PharosConfig
from pharos.llm.prompts import (
    INSUFFICIENT_EVIDENCE_TEMPLATE,
    REFUSAL_CRISIS,
    REFUSAL_OUT_OF_SCOPE,
    REFUSAL_PERSONAL,
    STANDING_DISCLAIMER,
)
from pharos.safety.triage import Verdict


def make_refuse_node():
    """Emit the refusal matching the triage verdict."""

    responses = {
        Verdict.PERSONAL_MEDICAL_ADVICE.value: REFUSAL_PERSONAL,
        Verdict.CRISIS.value: REFUSAL_CRISIS,
        Verdict.OUT_OF_SCOPE.value: REFUSAL_OUT_OF_SCOPE,
    }

    def node(state: PharosState) -> dict:
        verdict = state.get("triage_verdict", Verdict.OUT_OF_SCOPE.value)
        return {
            "answer": responses.get(verdict, REFUSAL_OUT_OF_SCOPE),
            "abstained": True,
            "abstain_reason": f"triage:{verdict}",
            "trace": [trace_event("refuse", verdict=verdict)],
        }

    return node


def make_abstain_node(cfg: PharosConfig):
    """Emit an insufficient-evidence answer, naming what is actually missing.

    Saying *why* the evidence is thin is not politeness. "No reviews for this drug
    at all" and "reviews exist but not for this indication" send the user to
    different next steps, and a generic "I don't know" sends them nowhere.
    """

    def node(state: PharosState) -> dict:
        n_units = state.get("n_units", 0)
        drug = state.get("plan_drug")
        condition = state.get("plan_condition")

        if not drug:
            detail = (
                "I couldn't identify a specific medication in the question, and this "
                "corpus is organised by drug. Naming one would let me answer."
            )
        elif n_units == 0:
            detail = (
                f"The corpus holds no reviews for {drug}"
                f"{f' used for {condition}' if condition else ''}."
            )
        else:
            detail = (
                f"Only {n_units} evidence unit"
                f"{'s' if n_units != 1 else ''} matched, below the "
                f"{cfg.retrieval.abstain_below_units}-unit floor this system uses. "
                f"Anything built on that would be an anecdote presented as a pattern."
            )

        return {
            "answer": INSUFFICIENT_EVIDENCE_TEMPLATE.format(detail=detail),
            "abstained": True,
            "abstain_reason": f"insufficient_evidence:{n_units}_units",
            "trace": [trace_event("abstain", n_units=n_units, drug=drug, condition=condition)],
        }

    return node


def make_finalize_node(cfg: PharosConfig):
    """Assemble the answer: strip what failed verification, append provenance."""

    def node(state: PharosState) -> dict:
        claims = state.get("claims", [])
        draft = state.get("draft", "")

        if cfg.agent.strip_unsupported and claims:
            kept = [c["text"] for c in claims if c["verdict"] == "SUPPORTED"]
            dropped = [c for c in claims if c["verdict"] == "UNSUPPORTED"]
            body = " ".join(kept) if kept else ""
        else:
            body = draft
            dropped = []

        if not body.strip():
            return {
                "answer": INSUFFICIENT_EVIDENCE_TEMPLATE.format(
                    detail=(
                        "Every claim in the draft failed verification against the "
                        "retrieved evidence, so there is nothing left that I can stand behind."
                    )
                ),
                "abstained": True,
                "abstain_reason": "all_claims_unsupported",
                "trace": [trace_event("finalize", kept=0, dropped=len(dropped))],
            }

        sections = [body.strip()]

        if dropped:
            sections.append(
                f"_{len(dropped)} statement"
                f"{'s' if len(dropped) != 1 else ''} removed by the verifier for lacking "
                f"grounding in the retrieved evidence._"
            )

        n_units = state.get("n_units", 0)
        n_stats = len(state.get("statistic_ids", []))
        skew = state.get("valence_skew", 0.0)
        sections.append(
            f"_Evidence: {n_units} patient-review excerpt"
            f"{'s' if n_units != 1 else ''}, balanced across outcome strata "
            f"(divergence from the cohort's true rating distribution: {skew:.3f}), "
            f"and {n_stats} statistic{'s' if n_stats != 1 else ''} computed over the "
            f"full corpus._"
        )

        if cfg.safety.always_disclaim:
            sections.append(f"_{STANDING_DISCLAIMER}_")

        return {
            "answer": "\n\n".join(sections),
            "abstained": False,
            "trace": [
                trace_event(
                    "finalize",
                    kept=len(claims) - len(dropped),
                    dropped=len(dropped),
                    answer_chars=len("\n\n".join(sections)),
                )
            ],
        }

    return node
