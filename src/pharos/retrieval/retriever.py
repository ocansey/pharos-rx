"""The retriever: query understanding, hybrid search, stratification, packing.

Pipeline for one query:

    parse ──► filter ──► dense ──┐
                        lexical ─┴─► RRF ──► MMR ──► stratify ──► panel

Each stage is switchable from config, and the ablation study in
``docs/RESULTS.md`` is produced by toggling exactly these switches, so the table
measures the code that ships rather than a reimplementation of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pharos.config import PharosConfig
from pharos.data.cohort import CohortStatistics
from pharos.data.schema import Aspect, RetrievedUnit, Stratum
from pharos.index.store import CorpusIndex
from pharos.nlp.aspects import infer_query_aspects
from pharos.retrieval.fusion import maximal_marginal_relevance, reciprocal_rank_fusion
from pharos.retrieval.stratified import (
    StratifiedSampler,
    cohort_rating_error,
    valence_skew_divergence,
)


@dataclass
class QueryPlan:
    """A parsed query: what the user asked, and what to constrain the search to."""

    raw: str
    drug_name: str | None = None
    condition: str | None = None
    aspects: list[Aspect] = field(default_factory=list)
    resolved_drug_candidates: list[str] = field(default_factory=list)
    resolved_condition_candidates: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if self.drug_name:
            bits.append(f"drug={self.drug_name}")
        if self.condition:
            bits.append(f"condition={self.condition}")
        if self.aspects:
            bits.append("aspects=" + ",".join(a.value for a in self.aspects))
        return "; ".join(bits) or "unconstrained"


@dataclass
class RetrievalResult:
    """A panel, plus every diagnostic needed to evaluate or audit it."""

    plan: QueryPlan
    units: list[RetrievedUnit]
    cohort_distribution: dict[Stratum, float]
    cohort_mean_rating: float | None
    valence_skew: float
    rating_error: float
    n_candidates: int
    stratified: bool
    allocation: dict[str, Any] = field(default_factory=dict)
    filtered_to_empty: bool = False

    @property
    def panel_ratings(self) -> list[float]:
        return [ru.unit.rating for ru in self.units]

    def is_sufficient(self, min_units: int) -> bool:
        return len(self.units) >= min_units


class PharosRetriever:
    """Hybrid, metadata-aware, stratification-capable retriever."""

    def __init__(
        self,
        index: CorpusIndex,
        stats: CohortStatistics,
        cfg: PharosConfig,
    ) -> None:
        self.index = index
        self.stats = stats
        self.cfg = cfg
        self.sampler = StratifiedSampler(
            min_slots_per_stratum=cfg.retrieval.min_slots_per_stratum,
            max_units_per_review=cfg.retrieval.max_units_per_review,
        )
        self._drug_vocab = self._build_vocab(self.index.by_drug.keys(), min_len=3)
        self._condition_vocab = self._build_vocab(self.index.by_condition.keys(), min_len=4)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_vocab(keys, min_len: int) -> list[str]:
        """Vocabulary sorted longest-first, so 'wellbutrin xl' beats 'wellbutrin'."""
        return sorted((k for k in keys if len(k) >= min_len), key=len, reverse=True)

    def parse(
        self,
        query: str,
        drug_name: str | None = None,
        condition: str | None = None,
        aspects: list[Aspect] | None = None,
    ) -> QueryPlan:
        """Extract entity constraints and aspect intent from a question.

        Entity extraction is vocabulary matching against the corpus's own drug
        and condition names, longest first, on word boundaries. An LLM-based
        extractor would be more flexible and would also make the retrieval
        evaluation depend on a model version — the retrieval numbers would then
        move when the model moved, for reasons unrelated to retrieval. Keeping
        the parser deterministic is what lets the ablation table isolate the
        thing it claims to isolate.
        """
        lowered = query.casefold()
        plan = QueryPlan(raw=query, drug_name=drug_name, condition=condition)

        if plan.drug_name is None:
            for name in self._drug_vocab:
                if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered):
                    matches = self.stats.resolve_drug(name)
                    plan.drug_name = matches[0] if matches else name
                    plan.resolved_drug_candidates = matches
                    break

        if plan.condition is None:
            for name in self._condition_vocab:
                if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered):
                    matches = self.stats.resolve_condition(name)
                    plan.condition = matches[0] if matches else name
                    plan.resolved_condition_candidates = matches
                    break

        plan.aspects = aspects if aspects is not None else infer_query_aspects(query)
        return plan

    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        drug_name: str | None = None,
        condition: str | None = None,
        aspects: list[Aspect] | None = None,
        stratify: bool | None = None,
        mode: str | None = None,
    ) -> RetrievalResult:
        """Retrieve an evidence panel for one query."""
        rcfg = self.cfg.retrieval
        k = k or rcfg.final_k
        stratify = rcfg.stratify if stratify is None else stratify
        mode = mode or rcfg.mode

        plan = self.parse(query, drug_name, condition, aspects)

        # --- candidate restriction ------------------------------------- #
        # Aspect is a soft preference, not a hard filter: a question about side
        # effects is still well served by an efficacy unit that mentions one.
        # Drug and condition are hard, because answering about the wrong
        # medicine is not a ranking error, it is a wrong answer.
        candidates = self.index.candidate_ids(drug_name=plan.drug_name, condition=plan.condition)
        filtered_to_empty = candidates is not None and candidates.size == 0

        if filtered_to_empty and plan.condition:
            # Back off the indication before backing off the drug: reviews of
            # the right drug for a neighbouring indication are far more useful
            # than reviews of a different drug for the right one.
            candidates = self.index.candidate_ids(drug_name=plan.drug_name)
            filtered_to_empty = candidates is not None and candidates.size == 0

        # --- arms ------------------------------------------------------ #
        # Encoded once and reused by MMR below. Encoding is the single most
        # expensive operation in a query when a neural encoder is configured,
        # and doing it twice doubles latency for nothing.
        query_vector: np.ndarray | None = None
        arms: dict[str, list[tuple[int, float]]] = {}
        if mode in ("dense", "hybrid"):
            query_vector = self.index.encoder.encode([query])[0]
            arms["dense"] = self.index.dense_search(query_vector, rcfg.candidate_k, candidates)
        if mode in ("lexical", "hybrid"):
            arms["lexical"] = self.index.lexical_search(query, rcfg.candidate_k, candidates)

        weights = {
            "lexical": rcfg.lexical_weight,
            "dense": 1.0 - rcfg.lexical_weight,
        }
        fused = reciprocal_rank_fusion(arms, weights=weights, k=rcfg.rrf_k)
        n_candidates = len(fused)

        if not fused:
            return self._empty_result(plan, filtered_to_empty)

        # --- aspect preference ----------------------------------------- #
        if plan.aspects:
            wanted = {a.value for a in plan.aspects}
            fused = sorted(
                fused,
                key=lambda t: (
                    -(1.15 if wanted & {a.value for a in self.index.units[t[0]].aspects} else 1.0)
                    * t[1]
                ),
            )

        ranked = [(doc_id, score) for doc_id, score, _prov in fused]
        provenance = {doc_id: prov for doc_id, _score, prov in fused}

        # --- redundancy control ---------------------------------------- #
        # MMR runs over a pool wider than k so that stratification still has
        # candidates in every stratum to choose from afterwards.
        if rcfg.rerank == "mmr" and mode != "lexical" and query_vector is not None:
            pool = ranked[: max(k * 6, 48)]
            pool_ids = [doc_id for doc_id, _ in pool]
            order = maximal_marginal_relevance(
                query_vector,
                self.index.vectors[pool_ids],
                pool_ids,
                k=len(pool_ids),
                lambda_mult=rcfg.mmr_lambda,
            )
            score_map = dict(pool)
            reordered = [(i, score_map[i]) for i in order]
            ranked = reordered + ranked[len(pool) :]

        # --- reference distribution ------------------------------------ #
        cohort_dist = self.stats.stratum_distribution(plan.drug_name, plan.condition)
        cohort_frame = self.stats.cohort_frame(plan.drug_name, plan.condition)
        cohort_mean = float(cohort_frame["rating"].mean()) if len(cohort_frame) else None

        if rcfg.strata_source == "uniform":
            cohort_dist = {s: 1.0 / len(Stratum.ordered()) for s in Stratum.ordered()}

        # --- panel selection ------------------------------------------- #
        allocation_info: dict[str, Any] = {}
        if stratify:
            selected, allocation = self.sampler.select(ranked, self.index.units, cohort_dist, k)
            allocation_info = {
                "quotas": {s.value: n for s, n in allocation.quotas.items()},
                "target": {s.value: round(v, 4) for s, v in allocation.target.items()},
                "achieved": {s.value: round(v, 4) for s, v in allocation.achieved.items()},
                "reallocated": {s.value: n for s, n in allocation.reallocated.items()},
                "shortfall": round(allocation.shortfall(), 4),
            }
        else:
            selected = [(pos, score, self.index.units[pos].stratum) for pos, score in ranked[:k]]

        units = []
        for pos, score, admitted in selected:
            prov = provenance.get(pos, {})
            units.append(
                RetrievedUnit(
                    unit=self.index.units[pos],
                    score=float(score),
                    dense_rank=prov.get("dense"),
                    lexical_rank=prov.get("lexical"),
                    rrf_score=float(score),
                    admitted_under=admitted if stratify else None,
                )
            )

        ratings = [ru.unit.rating for ru in units]
        return RetrievalResult(
            plan=plan,
            units=units,
            cohort_distribution=cohort_dist,
            cohort_mean_rating=cohort_mean,
            valence_skew=valence_skew_divergence(ratings, cohort_dist),
            rating_error=cohort_rating_error(ratings, cohort_mean) if cohort_mean else 0.0,
            n_candidates=n_candidates,
            stratified=stratify,
            allocation=allocation_info,
            filtered_to_empty=filtered_to_empty,
        )

    # ------------------------------------------------------------------ #
    def _empty_result(self, plan: QueryPlan, filtered_to_empty: bool) -> RetrievalResult:
        return RetrievalResult(
            plan=plan,
            units=[],
            cohort_distribution={s: 0.0 for s in Stratum.ordered()},
            cohort_mean_rating=None,
            valence_skew=0.0,
            rating_error=0.0,
            n_candidates=0,
            stratified=False,
            filtered_to_empty=filtered_to_empty,
        )

    # ------------------------------------------------------------------ #
    def format_panel(self, result: RetrievalResult, max_chars: int = 6000) -> str:
        """Render a panel as citable context for the generator.

        Each unit is prefixed with its identifier and annotated with the rating
        and date. Showing the rating is not decoration: it is what lets the
        generator write "reviewers who rated it 2/10 describe…" instead of
        flattening a bimodal population into a single false consensus.
        """
        lines: list[str] = []
        budget = max_chars
        for ru in result.units:
            u = ru.unit
            flags = []
            if u.negated:
                flags.append("negated-mention")
            if u.hedged:
                flags.append("hedged")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            block = (
                f"[{u.unit_id}] {u.drug_name} — {u.condition or 'unspecified'} — "
                f"rated {u.rating:.0f}/10 — {u.review_date}{flag_str}\n"
                f'  "{u.text}"'
            )
            if len(block) > budget:
                break
            lines.append(block)
            budget -= len(block) + 1
        return "\n".join(lines)
