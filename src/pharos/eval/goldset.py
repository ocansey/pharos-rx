"""Gold-set construction for retrieval evaluation.

The hard problem in evaluating retrieval over an unlabelled corpus is that there
are no relevance judgements, and the two usual substitutes both fail here.

*Hand annotation* does not scale past a few dozen queries, and a few dozen
queries cannot separate configurations whose nDCG differs by 0.02.

*LLM-as-judge* scales, but it makes the evaluation a measurement of the judge.
The judge's biases become the leaderboard, the numbers move when the model moves,
and — decisively for a project whose whole thesis is that ungrounded model output
should not be trusted — it would be incoherent to argue that case and then
validate the argument with ungrounded model output.

So relevance is **derived from structured metadata that already exists**. Every
evidence unit carries its drug, its indication, its aspect labels and its outcome
stratum. A query is generated *from* a (drug, condition, aspect) triple, and the
relevant set is defined as the units whose metadata matches that triple. The
judgements are then exactly as reliable as the metadata — which is auditable,
deterministic, and free.

The limitation is real and is stated rather than hidden: this measures whether
retrieval finds units matching the query's structured intent, not whether a human
would find them useful. It is a *necessary* condition for good retrieval, not a
sufficient one. Aspect labels come from the weak labeller in
:mod:`pharos.nlp.aspects`, so aspect-conditioned relevance inherits its error
rate; the agreement study in ``docs/RESULTS.md`` bounds that.

Graded relevance, on a 0-3 scale:

    3   drug + condition + aspect all match
    2   drug + condition match, aspect does not
    1   drug matches, condition does not
    0   everything else
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from pharos.data.schema import Aspect, EvidenceUnit, Stratum

#: Natural-language templates per aspect. Several per aspect, so the query set
#: does not become a test of one phrasing.
_TEMPLATES: dict[Aspect, list[str]] = {
    Aspect.ADVERSE_EFFECT: [
        "What side effects do people report on {drug} for {condition}?",
        "Are there bad reactions to {drug} when used for {condition}?",
        "What problems do reviewers describe with {drug} for {condition}?",
    ],
    Aspect.EFFICACY: [
        "Does {drug} work for {condition} according to reviewers?",
        "How effective is {drug} for {condition} in patient reviews?",
        "Do people say {drug} helped their {condition}?",
    ],
    Aspect.ONSET_DURATION: [
        "How long does {drug} take to work for {condition}?",
        "When do reviewers say {drug} started working for {condition}?",
        "How quickly does {drug} act on {condition}?",
    ],
    Aspect.DOSING: [
        "What doses of {drug} do reviewers mention for {condition}?",
        "How do people describe dosing {drug} for {condition}?",
        "What dosage schedules come up for {drug} in {condition}?",
    ],
    Aspect.DISCONTINUATION: [
        "What do reviewers say about stopping {drug} for {condition}?",
        "How do people describe coming off {drug} taken for {condition}?",
        "Is withdrawal discussed for {drug} in {condition} reviews?",
    ],
    Aspect.ACCESS_COST: [
        "What do reviewers say about the cost of {drug} for {condition}?",
        "Do people mention insurance or price for {drug} in {condition}?",
    ],
}


@dataclass
class GoldQuery:
    """One evaluation query with derived graded relevance judgements."""

    query_id: str
    text: str
    drug_name: str
    condition: str
    aspect: Aspect
    relevance: dict[str, int] = field(default_factory=dict)  # unit_id -> grade
    cohort_size: int = 0
    cohort_mean_rating: float = 0.0
    cohort_distribution: dict[str, float] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> set[str]:
        return {uid for uid, grade in self.relevance.items() if grade > 0}

    @property
    def highly_relevant_ids(self) -> set[str]:
        return {uid for uid, grade in self.relevance.items() if grade >= 3}

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "drug_name": self.drug_name,
            "condition": self.condition,
            "aspect": self.aspect.value,
            "n_relevant": len(self.relevant_ids),
            "n_highly_relevant": len(self.highly_relevant_ids),
            "cohort_size": self.cohort_size,
            "cohort_mean_rating": round(self.cohort_mean_rating, 3),
        }


def build_goldset(
    units: list[EvidenceUnit],
    n_queries: int = 300,
    seed: int = 7,
    min_cohort_units: int = 30,
    min_relevant: int = 6,
) -> list[GoldQuery]:
    """Generate queries from cohorts that can actually support an evaluation.

    A cohort with three matching units cannot distinguish a good retriever from a
    lucky one: Recall@10 is 1.0 for anything that works at all, and the metric
    carries no information. Worse for this project specifically, a cohort smaller
    than the panel is retrieved *exhaustively*, so its panel is perfectly
    representative no matter what the retriever does -- and every fidelity metric
    reads 0 for reasons that have nothing to do with the mechanism being tested.
    The defaults require a cohort at least 2.5x the panel size.

    Sampling is stratified over aspects so that no single facet - and in this
    corpus that would be adverse effects, which are everywhere - dominates the
    query set and turns the benchmark into a side-effect benchmark.
    """
    rng = random.Random(seed)

    by_cohort: dict[tuple[str, str], list[EvidenceUnit]] = {}
    for unit in units:
        if not unit.condition:
            continue
        by_cohort.setdefault((unit.drug_name, unit.condition), []).append(unit)

    eligible = {k: v for k, v in by_cohort.items() if len(v) >= min_cohort_units}
    if not eligible:
        return []

    by_drug: dict[str, list[EvidenceUnit]] = {}
    for unit in units:
        by_drug.setdefault(unit.drug_name, []).append(unit)

    aspects = list(_TEMPLATES.keys())
    candidates: list[tuple[tuple[str, str], Aspect]] = []
    for cohort_key, cohort_units in eligible.items():
        present = {a for u in cohort_units for a in u.aspects}
        for aspect in aspects:
            if aspect in present:
                candidates.append((cohort_key, aspect))
    if not candidates:
        return []

    # Round-robin over aspects for balance, shuffling within each aspect so the
    # selection is not ordered by cohort size.
    grouped: dict[Aspect, list[tuple[str, str]]] = {a: [] for a in aspects}
    for cohort_key, aspect in candidates:
        grouped[aspect].append(cohort_key)
    for aspect in aspects:
        rng.shuffle(grouped[aspect])

    selected: list[tuple[tuple[str, str], Aspect]] = []
    cursor = dict.fromkeys(aspects, 0)
    budget = n_queries * 3
    while len(selected) < budget and any(cursor[a] < len(grouped[a]) for a in aspects):
        for aspect in aspects:
            if cursor[aspect] < len(grouped[aspect]):
                selected.append((grouped[aspect][cursor[aspect]], aspect))
                cursor[aspect] += 1
            if len(selected) >= budget:
                break

    queries: list[GoldQuery] = []
    for (drug, condition), aspect in selected:
        if len(queries) >= n_queries:
            break
        template = rng.choice(_TEMPLATES[aspect])
        text = template.format(drug=drug, condition=condition)

        relevance: dict[str, int] = {}
        for unit in by_drug.get(drug, []):
            same_condition = unit.condition == condition
            has_aspect = aspect in unit.aspects
            if same_condition and has_aspect:
                relevance[unit.unit_id] = 3
            elif same_condition:
                relevance[unit.unit_id] = 2
            else:
                relevance[unit.unit_id] = 1

        if sum(1 for g in relevance.values() if g == 3) < min_relevant:
            continue

        cohort_units = eligible[(drug, condition)]
        review_ratings: dict[int, float] = {u.review_id: u.rating for u in cohort_units}
        ratings = list(review_ratings.values())

        counts: dict[str, float] = {}
        for rating in ratings:
            key = Stratum.from_rating(rating).value
            counts[key] = counts.get(key, 0.0) + 1
        distribution = {k: v / len(ratings) for k, v in counts.items()}

        queries.append(
            GoldQuery(
                query_id=f"Q{len(queries):04d}",
                text=text,
                drug_name=drug,
                condition=condition,
                aspect=aspect,
                relevance=relevance,
                cohort_size=len(review_ratings),
                cohort_mean_rating=sum(ratings) / len(ratings),
                cohort_distribution=distribution,
            )
        )

    return queries
