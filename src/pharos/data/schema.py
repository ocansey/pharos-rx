"""Canonical data contracts.

Two record types carry the whole system.

``Review`` is one patient narrative as it appears in the source corpus, after
cleaning. ``EvidenceUnit`` is the atom of retrieval: a single clause-level span
of a review, carrying enough of its parent's metadata to be stratified, filtered,
and cited without a join.

The denormalisation is deliberate. Retrieval touches evidence units hundreds of
times per query; the quota allocator needs ``rating`` and ``stratum`` on the unit
itself, and the citation formatter needs ``drug_name`` there too. Paying a few
bytes per unit buys an allocator that never leaves the array.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Aspect(str, Enum):
    """What a span of patient narrative is *about*.

    These are the facets a reader of drug reviews actually wants separated. A
    query about side effects should not be answered with efficacy testimony, and
    a question about how long a drug takes to work should not be answered with
    complaints about the copay.
    """

    EFFICACY = "efficacy"
    ADVERSE_EFFECT = "adverse_effect"
    ONSET_DURATION = "onset_duration"
    DOSING = "dosing"
    ACCESS_COST = "access_cost"
    DISCONTINUATION = "discontinuation"
    COMPARISON = "comparison"
    CONTEXT = "context"

    @classmethod
    def clinical(cls) -> tuple[Aspect, ...]:
        """Aspects that carry clinical signal, as opposed to framing."""
        return (
            cls.EFFICACY,
            cls.ADVERSE_EFFECT,
            cls.ONSET_DURATION,
            cls.DOSING,
            cls.DISCONTINUATION,
        )


class Stratum(str, Enum):
    """Outcome strata used by the quota allocator.

    The source ratings run 1–10 and are sharply bimodal: 32% of the corpus sits
    at 9–10 and 18% at 1–2. Collapsing to three ordered strata gives the
    allocator enough resolution to preserve dissent without shattering small
    cohorts into empty cells.
    """

    NEGATIVE = "negative"  # ratings 1–4
    MIXED = "mixed"  # ratings 5–6
    POSITIVE = "positive"  # ratings 7–10

    @classmethod
    def from_rating(cls, rating: float) -> Stratum:
        if rating <= 4:
            return cls.NEGATIVE
        if rating <= 6:
            return cls.MIXED
        return cls.POSITIVE

    @classmethod
    def ordered(cls) -> tuple[Stratum, ...]:
        return (cls.NEGATIVE, cls.MIXED, cls.POSITIVE)


class Review(BaseModel):
    """One cleaned patient review."""

    review_id: int
    drug_name: str
    condition: str | None
    text: str
    rating: float = Field(ge=1.0, le=10.0)
    review_date: date
    useful_count: int = Field(ge=0)
    split: str = "train"

    #: Set when cleaning repaired a truncated or mojibake condition label.
    condition_repaired: bool = False
    #: Set when de-identification rewrote part of the text.
    deidentified: bool = False

    @property
    def stratum(self) -> Stratum:
        return Stratum.from_rating(self.rating)

    @field_validator("drug_name", "text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("field must not be empty")
        return v


class EvidenceUnit(BaseModel):
    """A citable clause-level span of a review.

    ``unit_id`` is the token that appears in a generated answer. It is stable
    across rebuilds for a fixed config, which is what lets a saved transcript be
    re-audited months later.
    """

    unit_id: str
    review_id: int
    ordinal: int  # position of this unit within its parent review

    text: str
    aspects: list[Aspect] = Field(default_factory=list)

    # Denormalised parent metadata — see the module docstring.
    drug_name: str
    condition: str | None
    rating: float
    stratum: Stratum
    review_date: date
    useful_count: int

    #: Adverse-event lexicon terms detected in this span, after negation and
    #: hedging are accounted for.
    adverse_terms: list[str] = Field(default_factory=list)
    #: True when the span is inside the scope of a negation cue ("no nausea").
    negated: bool = False
    #: True when the span is hedged ("might be the cause", "I think").
    hedged: bool = False

    @property
    def cohort_key(self) -> tuple[str, str | None]:
        return (self.drug_name, self.condition)

    def citation(self) -> str:
        return f"[{self.unit_id}]"

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata for a LangChain ``Document``."""
        return {
            "unit_id": self.unit_id,
            "review_id": self.review_id,
            "drug_name": self.drug_name,
            "condition": self.condition,
            "rating": self.rating,
            "stratum": self.stratum.value,
            "aspects": [a.value for a in self.aspects],
            "adverse_terms": self.adverse_terms,
            "negated": self.negated,
            "hedged": self.hedged,
            "review_date": self.review_date.isoformat(),
            "useful_count": self.useful_count,
        }


class RetrievedUnit(BaseModel):
    """An evidence unit with the scores that put it in the panel.

    Keeping the component scores alongside the fused one is not bookkeeping for
    its own sake: the ablation table in ``docs/RESULTS.md`` is computed by
    replaying these fields, and a reviewer can see exactly which arm surfaced a
    given citation.
    """

    unit: EvidenceUnit
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    #: Which stratum quota this unit was admitted under, when stratifying.
    admitted_under: Stratum | None = None

    model_config = {"arbitrary_types_allowed": True}
