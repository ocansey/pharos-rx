"""Cohort statistics and disproportionality estimators.

Numeric correctness here is load-bearing: every quantity in a PHAROS answer comes
from this module, and a wrong interval would be reported to the reader with full
confidence and a citation.
"""

from __future__ import annotations

import math

import pytest

from pharos.data.cohort import bootstrap_mean_ci, wilson_interval
from pharos.data.schema import Stratum


class TestWilsonInterval:
    def test_matches_published_values(self):
        # Wilson 95% for 2/8; the textbook value, and the case where Wald fails.
        lo, hi = wilson_interval(2, 8)
        assert lo == pytest.approx(0.0710, abs=1e-3)
        assert hi == pytest.approx(0.5901, abs=1e-3)

    def test_stays_inside_the_unit_interval_where_wald_does_not(self):
        for successes, n in ((0, 5), (5, 5), (1, 3), (0, 1), (100, 100)):
            lo, hi = wilson_interval(successes, n)
            assert 0.0 <= lo <= hi <= 1.0

    def test_contains_the_point_estimate(self):
        for successes, n in ((3, 10), (17, 40), (1, 100)):
            lo, hi = wilson_interval(successes, n)
            assert lo <= successes / n <= hi

    def test_narrows_as_n_grows(self):
        widths = [
            (lambda t: t[1] - t[0])(wilson_interval(int(0.3 * n), n))
            for n in (10, 100, 1000, 10000)
        ]
        assert widths == sorted(widths, reverse=True)

    def test_zero_denominator(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)


class TestBootstrapCI:
    def test_brackets_the_sample_mean(self):
        import numpy as np

        values = np.array([1.0, 1.0, 2.0, 9.0, 10.0, 10.0, 10.0])
        lo, hi = bootstrap_mean_ci(values, n_boot=500, seed=1)
        assert lo <= values.mean() <= hi

    def test_is_deterministic_under_a_fixed_seed(self):
        import numpy as np

        values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert bootstrap_mean_ci(values, seed=42) == bootstrap_mean_ci(values, seed=42)

    def test_degenerate_inputs(self):
        import numpy as np

        assert bootstrap_mean_ci(np.array([5.0])) == (5.0, 5.0)
        assert all(math.isnan(v) for v in bootstrap_mean_ci(np.array([])))


class TestCohortSummary:
    def test_summarises_a_known_cohort(self, stats):
        summary = stats.summarise("Alphamed", "Chronic Pain")
        assert summary is not None
        assert summary.n_reviews == 52
        assert 1.0 <= summary.mean_rating <= 10.0
        assert summary.rating_ci[0] <= summary.mean_rating <= summary.rating_ci[1]
        assert sum(summary.rating_histogram.values()) == summary.n_reviews
        assert sum(summary.stratum_distribution.values()) == pytest.approx(1.0)

    def test_missing_cohort_returns_none(self, stats):
        assert stats.summarise("Nonexistium", "Chronic Pain") is None

    def test_evidence_block_carries_the_low_support_warning(self, stats):
        summary = stats.summarise("Alphamed", "Chronic Pain")
        block = summary.to_evidence_block()
        assert summary.stat_id in block
        assert "reviews analysed" in block
        assert "LOW SUPPORT" not in block  # 52 reviews clears the floor

    def test_stat_ids_are_unique(self, stats):
        ids = {
            stats.summarise(drug, cond).stat_id
            for drug, cond in (
                ("Alphamed", "Chronic Pain"),
                ("Betacine", "Chronic Pain"),
                ("Alphamed", "Insomnia"),
            )
        }
        assert len(ids) == 3

    def test_counts_a_concept_once_per_review(self, stats):
        # A review mentioning nausea in three units is one nausea report.
        summary = stats.summarise("Betacine", "Chronic Pain")
        for _concept, count, _prop, _ci in summary.top_adverse_concepts:
            assert count <= summary.n_reviews


class TestStratumDistribution:
    def test_sums_to_one(self, stats):
        dist = stats.stratum_distribution("Alphamed", "Chronic Pain")
        assert sum(dist.values()) == pytest.approx(1.0)
        assert set(dist) == set(Stratum.ordered())

    def test_reflects_the_designed_skew(self, stats):
        # Alphamed is well received, Betacine is not.
        alpha = stats.stratum_distribution("Alphamed", "Chronic Pain")
        beta = stats.stratum_distribution("Betacine", "Chronic Pain")
        assert alpha[Stratum.POSITIVE] > beta[Stratum.POSITIVE]
        assert beta[Stratum.NEGATIVE] > alpha[Stratum.NEGATIVE]

    def test_unknown_cohort_falls_back_to_uniform(self, stats):
        dist = stats.stratum_distribution("Nonexistium", None)
        assert all(v == pytest.approx(1 / 3) for v in dist.values())

    def test_is_cached(self, stats):
        first = stats.stratum_distribution("Alphamed", "Chronic Pain")
        assert stats.stratum_distribution("Alphamed", "Chronic Pain") is first


class TestDisproportionality:
    def test_computes_a_signal_for_a_real_pair(self, stats):
        summary = stats.summarise("Betacine", "Chronic Pain")
        concept = summary.top_adverse_concepts[0][0]
        signal = stats.disproportionality("Betacine", concept, "Chronic Pain")
        if signal is None:
            pytest.skip("synthetic cohort did not clear the a>=3 floor")
        assert signal.prr > 0
        assert signal.prr_ci[0] <= signal.prr <= signal.prr_ci[1]
        assert signal.ror_ci[0] <= signal.ror <= signal.ror_ci[1]
        assert 0.0 <= signal.p_value <= 1.0
        assert signal.ic025 <= signal.information_component

    def test_refuses_below_the_support_floor(self, stats):
        assert stats.disproportionality("Alphamed", "seizure", "Chronic Pain") is None

    def test_evidence_block_carries_the_provenance_caveat(self, stats):
        summary = stats.summarise("Betacine", "Chronic Pain")
        concept = summary.top_adverse_concepts[0][0]
        signal = stats.disproportionality("Betacine", concept, "Chronic Pain")
        if signal is None:
            pytest.skip("synthetic cohort did not clear the a>=3 floor")
        block = signal.to_evidence_block()
        assert "not spontaneous adverse-event reports" in block
        assert "Not a regulatory signal" in block

    def test_unknown_drug(self, stats):
        assert stats.disproportionality("Nonexistium", "nausea") is None


class TestResolution:
    def test_exact_match_wins(self, stats):
        assert stats.resolve_drug("alphamed") == ["Alphamed"]

    def test_prefix_before_containment(self, stats):
        assert "Alphamed" in stats.resolve_drug("alpha")

    def test_unknown_returns_empty(self, stats):
        assert stats.resolve_drug("zzzzz") == []

    def test_condition_resolution(self, stats):
        assert "Chronic Pain" in stats.resolve_condition("chronic pain")


class TestCompareDrugs:
    def test_runs_a_rank_based_test(self, stats):
        result = stats.compare_drugs(["Alphamed", "Betacine"], "Chronic Pain")
        assert len(result["cohorts"]) == 2
        assert "kruskal_p" in result
        # The two cohorts were constructed to differ; the test should see it.
        assert result["kruskal_p"] < 0.05

    def test_skips_missing_drugs(self, stats):
        result = stats.compare_drugs(["Alphamed", "Nonexistium"], "Chronic Pain")
        assert len(result["cohorts"]) == 1


class TestCohortIndexKeys:
    """Regression tests for the composite (drug, condition) index key.

    An earlier implementation joined the two fields with a NUL byte and grouped
    on the resulting string. pandas < 3 silently drops NUL from
    ``Series + str + Series``, so every composite lookup missed and every cohort
    came back empty — passing on pandas 3, failing on pandas 2, with no error
    anywhere. These tests pin the behaviour that fix depends on.
    """

    def test_composite_lookup_returns_the_right_rows(self, stats):
        frame = stats.cohort_frame("Alphamed", "Chronic Pain")
        assert len(frame) == 52
        assert set(frame["drug_name"]) == {"Alphamed"}
        assert set(frame["condition"]) == {"Chronic Pain"}

    def test_same_drug_different_conditions_do_not_merge(self, stats):
        pain = stats.cohort_frame("Alphamed", "Chronic Pain")
        insomnia = stats.cohort_frame("Alphamed", "Insomnia")
        assert len(pain) and len(insomnia)
        assert set(pain["review_id"]).isdisjoint(insomnia["review_id"])
        assert len(pain) + len(insomnia) == len(stats.cohort_frame("Alphamed"))

    def test_lookup_is_case_insensitive(self, stats):
        expected = len(stats.cohort_frame("Alphamed", "Chronic Pain"))
        for drug, cond in (
            ("alphamed", "chronic pain"),
            ("ALPHAMED", "CHRONIC PAIN"),
            ("AlPhAmEd", "ChRoNiC pAiN"),
        ):
            assert len(stats.cohort_frame(drug, cond)) == expected

    def test_unknown_cohort_returns_empty_not_everything(self, stats):
        # The dangerous failure: a missed lookup falling back to the whole
        # corpus would silently answer about the wrong population.
        assert len(stats.cohort_frame("Alphamed", "Nonexistent Condition")) == 0
        assert len(stats.cohort_frame("Nonexistium", "Chronic Pain")) == 0

    def test_composite_keys_cannot_collide_across_a_boundary(self):
        """A delimited string key can collide; a tuple key cannot.

        ("alpha", "medchronic pain") and ("alphamed", "chronic pain") concatenate
        to the same string once a separator is lost or appears in the data.
        """
        from pharos.data.cohort import CohortStatistics

        groups = CohortStatistics._positional_groups(
            [("alpha", "medchronic pain"), ("alphamed", "chronic pain")]
        )
        assert len(groups) == 2
        assert groups[("alpha", "medchronic pain")].tolist() == [0]
        assert groups[("alphamed", "chronic pain")].tolist() == [1]

    def test_positional_groups_preserves_order_and_covers_every_row(self):
        from pharos.data.cohort import CohortStatistics

        keys = ["b", "a", "b", "c", "a", "b"]
        groups = CohortStatistics._positional_groups(keys)
        assert groups["b"].tolist() == [0, 2, 5]
        assert groups["a"].tolist() == [1, 4]
        assert sum(len(v) for v in groups.values()) == len(keys)
