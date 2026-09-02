"""Stratified Evidence Sampling — the project's core contribution.

These are the tests that matter most. The allocator's job is to make an evidence
panel representative of its cohort; if it silently stops doing that, every claim
this project makes stops being true, and no other test in the suite would notice.
"""

from __future__ import annotations

from datetime import date

import pytest

from pharos.data.schema import Aspect, EvidenceUnit, Stratum
from pharos.retrieval.stratified import (
    StratifiedSampler,
    cohort_rating_error,
    largest_remainder_apportionment,
    valence_skew_divergence,
)


def make_unit(unit_id: str, rating: float, review_id: int = 1) -> EvidenceUnit:
    return EvidenceUnit(
        unit_id=unit_id,
        review_id=review_id,
        ordinal=0,
        text=f"unit {unit_id} with enough text to be a plausible evidence span",
        aspects=[Aspect.EFFICACY],
        drug_name="Alphamed",
        condition="Chronic Pain",
        rating=rating,
        stratum=Stratum.from_rating(rating),
        review_date=date(2015, 6, 1),
        useful_count=3,
    )


class TestApportionment:
    def test_allocation_sums_exactly_to_the_budget(self):
        target = {Stratum.NEGATIVE: 0.34, Stratum.MIXED: 0.33, Stratum.POSITIVE: 0.33}
        for k in range(1, 40):
            assert sum(largest_remainder_apportionment(target, k).values()) == k

    def test_allocation_tracks_the_target_distribution(self):
        target = {Stratum.NEGATIVE: 0.5, Stratum.MIXED: 0.2, Stratum.POSITIVE: 0.3}
        alloc = largest_remainder_apportionment(target, 10, min_per_nonzero=0)
        assert alloc[Stratum.NEGATIVE] == 5
        assert alloc[Stratum.MIXED] == 2
        assert alloc[Stratum.POSITIVE] == 3

    def test_minority_stratum_is_never_rounded_out(self):
        # 2% of the cohort had a bad time. They must still be in the panel.
        target = {Stratum.NEGATIVE: 0.02, Stratum.MIXED: 0.03, Stratum.POSITIVE: 0.95}
        alloc = largest_remainder_apportionment(target, 12, min_per_nonzero=1)
        assert alloc[Stratum.NEGATIVE] >= 1
        assert alloc[Stratum.MIXED] >= 1
        assert sum(alloc.values()) == 12

    def test_empty_strata_receive_nothing(self):
        target = {Stratum.NEGATIVE: 0.0, Stratum.MIXED: 0.0, Stratum.POSITIVE: 1.0}
        alloc = largest_remainder_apportionment(target, 8)
        assert alloc[Stratum.NEGATIVE] == 0
        assert alloc[Stratum.POSITIVE] == 8

    def test_budget_smaller_than_the_stratum_count(self):
        target = {Stratum.NEGATIVE: 0.2, Stratum.MIXED: 0.3, Stratum.POSITIVE: 0.5}
        alloc = largest_remainder_apportionment(target, 2, min_per_nonzero=1)
        assert sum(alloc.values()) == 2
        # The two largest strata get the seats.
        assert alloc[Stratum.POSITIVE] == 1
        assert alloc[Stratum.MIXED] == 1

    def test_zero_budget(self):
        target = {Stratum.NEGATIVE: 0.5, Stratum.POSITIVE: 0.5}
        assert sum(largest_remainder_apportionment(target, 0).values()) == 0

    def test_is_deterministic(self):
        target = {Stratum.NEGATIVE: 0.371, Stratum.MIXED: 0.144, Stratum.POSITIVE: 0.485}
        first = largest_remainder_apportionment(target, 13)
        for _ in range(20):
            assert largest_remainder_apportionment(target, 13) == first


class TestSampler:
    @pytest.fixture
    def skewed_candidates(self):
        """A ranked list dominated by positive units, as similarity search gives."""
        units = [make_unit(f"EU-{i}", 10.0, review_id=i) for i in range(20)]
        units += [make_unit(f"EU-{100 + i}", 2.0, review_id=100 + i) for i in range(10)]
        units += [make_unit(f"EU-{200 + i}", 5.0, review_id=200 + i) for i in range(5)]
        # Positives rank first — the bias the sampler exists to correct.
        ranked = [(i, 1.0 - i * 0.01) for i in range(len(units))]
        return ranked, units

    def test_panel_matches_the_cohort_distribution(self, skewed_candidates):
        ranked, units = skewed_candidates
        target = {Stratum.NEGATIVE: 0.4, Stratum.MIXED: 0.2, Stratum.POSITIVE: 0.4}
        selected, allocation = StratifiedSampler().select(ranked, units, target, k=10)

        assert len(selected) == 10
        counts = {s: 0 for s in Stratum.ordered()}
        for pos, _score, _admitted in selected:
            counts[units[pos].stratum] += 1
        assert counts[Stratum.NEGATIVE] == 4
        assert counts[Stratum.MIXED] == 2
        assert counts[Stratum.POSITIVE] == 4
        assert allocation.shortfall() == pytest.approx(0.0, abs=1e-9)

    def test_unstratified_top_k_would_be_all_positive(self, skewed_candidates):
        # The baseline this project measures against, made explicit.
        ranked, units = skewed_candidates
        top_k = [units[pos].stratum for pos, _ in ranked[:10]]
        assert set(top_k) == {Stratum.POSITIVE}

    def test_reallocates_when_a_stratum_is_empty(self):
        units = [make_unit(f"EU-{i}", 10.0, review_id=i) for i in range(12)]
        ranked = [(i, 1.0) for i in range(12)]
        target = {Stratum.NEGATIVE: 0.5, Stratum.MIXED: 0.0, Stratum.POSITIVE: 0.5}
        selected, allocation = StratifiedSampler().select(ranked, units, target, k=6)
        # No negative candidates exist, so the panel is still full from positives.
        assert len(selected) == 6
        assert allocation.reallocated

    def test_per_review_cap_prevents_one_reviewer_dominating(self):
        # Ten units, all from the same review.
        units = [make_unit(f"EU-{i}", 9.0, review_id=7) for i in range(10)]
        units += [make_unit(f"EU-{100 + i}", 9.0, review_id=100 + i) for i in range(10)]
        ranked = [(i, 1.0 - i * 0.01) for i in range(len(units))]
        target = {Stratum.POSITIVE: 1.0}
        selected, _ = StratifiedSampler(max_units_per_review=2).select(ranked, units, target, k=8)
        from_review_7 = sum(1 for pos, _, _ in selected if units[pos].review_id == 7)
        assert from_review_7 <= 2

    def test_returns_a_full_panel_even_on_a_thin_cohort(self):
        units = [make_unit(f"EU-{i}", 8.0, review_id=1) for i in range(4)]
        ranked = [(i, 1.0) for i in range(4)]
        selected, _ = StratifiedSampler(max_units_per_review=2).select(
            ranked, units, {Stratum.POSITIVE: 1.0}, k=4
        )
        # The per-review cap is relaxed last rather than returning a short panel.
        assert len(selected) == 4

    def test_no_unit_is_selected_twice(self):
        units = [make_unit(f"EU-{i}", float(1 + i % 10), review_id=i) for i in range(30)]
        ranked = [(i, 1.0 - i * 0.01) for i in range(30)]
        target = {Stratum.NEGATIVE: 0.33, Stratum.MIXED: 0.33, Stratum.POSITIVE: 0.34}
        selected, _ = StratifiedSampler().select(ranked, units, target, k=12)
        positions = [pos for pos, _, _ in selected]
        assert len(positions) == len(set(positions))

    def test_empty_candidates(self):
        selected, _ = StratifiedSampler().select([], [], {Stratum.POSITIVE: 1.0}, k=5)
        assert selected == []


class TestValenceSkew:
    def test_perfect_match_is_zero(self):
        # 5 negative, 5 positive against a 50/50 cohort.
        ratings = [1.0] * 5 + [9.0] * 5
        assert valence_skew_divergence(
            ratings, {Stratum.NEGATIVE: 0.5, Stratum.MIXED: 0.0, Stratum.POSITIVE: 0.5}
        ) == pytest.approx(0.0, abs=1e-9)

    def test_complete_mismatch_approaches_one(self):
        skew = valence_skew_divergence(
            [10.0] * 10, {Stratum.NEGATIVE: 1.0, Stratum.MIXED: 0.0, Stratum.POSITIVE: 0.0}
        )
        assert skew == pytest.approx(1.0, abs=1e-6)

    def test_is_finite_when_a_stratum_is_missing(self):
        # The case that makes KL divergence useless: it would return infinity
        # for exactly the panels that are most badly skewed.
        skew = valence_skew_divergence(
            [10.0] * 12, {Stratum.NEGATIVE: 0.4, Stratum.MIXED: 0.1, Stratum.POSITIVE: 0.5}
        )
        assert 0.0 < skew < 1.0

    def test_is_bounded(self):
        for ratings in ([1.0], [10.0] * 50, [1.0, 5.0, 10.0]):
            for dist in (
                {Stratum.NEGATIVE: 1.0},
                {Stratum.POSITIVE: 1.0},
                {Stratum.NEGATIVE: 0.3, Stratum.MIXED: 0.3, Stratum.POSITIVE: 0.4},
            ):
                assert 0.0 <= valence_skew_divergence(ratings, dist) <= 1.0

    def test_empty_panel(self):
        assert valence_skew_divergence([], {Stratum.POSITIVE: 1.0}) == 0.0

    def test_rating_error_is_absolute_difference_of_means(self):
        assert cohort_rating_error([8.0, 10.0], 6.0) == pytest.approx(3.0)
        assert cohort_rating_error([], 6.0) == 0.0
