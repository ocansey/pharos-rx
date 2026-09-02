"""Retrieval: BM25, fusion, metadata filtering, and the end-to-end retriever."""

from __future__ import annotations

import numpy as np
import pytest

from pharos.data.schema import Aspect, Stratum
from pharos.index.bm25 import BM25Index, tokenize
from pharos.retrieval.fusion import maximal_marginal_relevance, reciprocal_rank_fusion


class TestTokenizer:
    def test_lowercases_and_splits_on_non_alphanumerics(self):
        assert tokenize("Nausea, dizziness; 20mg!") == ["nausea", "dizziness", "20mg"]

    def test_keeps_clinically_meaningful_negation_words(self):
        # "no" and "not" invert a clinical claim and must survive the stoplist.
        tokens = tokenize("no nausea and not dizzy")
        assert "no" in tokens
        assert "not" in tokens

    def test_removes_ordinary_stopwords(self):
        assert "the" not in tokenize("the medication worked")


class TestBM25:
    @pytest.fixture(scope="class")
    def bm25(self):
        docs = [
            "severe nausea and vomiting after taking this medication",
            "no nausea at all and my headaches resolved completely",
            "terrible headaches every single morning for a month",
            "the weight gain was significant and unwelcome",
        ]
        return BM25Index().fit(docs), docs

    def test_ranks_the_on_topic_document_first(self, bm25):
        index, _ = bm25
        results = index.top_k("nausea vomiting", k=3)
        assert results[0][0] == 0

    def test_returns_only_scoring_documents(self, bm25):
        index, _ = bm25
        assert index.top_k("cardiomyopathy", k=5) == []

    def test_candidate_restriction_is_exact(self, bm25):
        # The filter is applied during scoring, so the top-k is the true top-k
        # of the filtered set rather than a post-filtered top-k that came back short.
        index, _ = bm25
        results = index.top_k("nausea", k=5, candidates=np.array([1, 2, 3]))
        assert all(doc_id in {1, 2, 3} for doc_id, _ in results)
        assert results[0][0] == 1

    def test_empty_candidate_set_returns_nothing(self, bm25):
        index, _ = bm25
        assert index.top_k("nausea", k=5, candidates=np.array([], dtype=np.int64)) == []

    def test_scores_are_non_negative(self, bm25):
        index, _ = bm25
        _ids, scores = index.score("nausea headaches weight")
        assert (scores >= 0).all()

    def test_idf_penalises_ubiquitous_terms(self, bm25):
        index, _ = bm25
        # "nausea" appears in 2/4 docs, "cardiomyopathy" in none.
        assert index.idf["nausea"] < index.idf["vomiting"]

    def test_empty_index(self):
        assert BM25Index().fit([]).top_k("anything", k=5) == []


class TestReciprocalRankFusion:
    def test_rewards_agreement_between_arms(self):
        # Doc 2 is found by both arms; doc 1 is ranked first by one arm alone.
        # Corroboration across arms should outweigh a single arm's confidence.
        fused = reciprocal_rank_fusion(
            {"dense": [(1, 0.9), (2, 0.8)], "lexical": [(2, 5.0), (3, 4.0)]}
        )
        ranked = [doc_id for doc_id, _score, _prov in fused]
        assert ranked[0] == 2

    def test_head_ties_break_toward_the_better_single_rank(self):
        # With the standard damping constant of 60 the discount curve is nearly
        # linear over the head, so two documents with mirrored ranks are almost
        # tied and the one holding rank 1 wins by a hair. Documented because it
        # is easy to assume RRF strongly rewards symmetry here, and it does not.
        fused = reciprocal_rank_fusion(
            {"dense": [(1, 0.9), (2, 0.8), (3, 0.7)], "lexical": [(3, 5.0), (2, 4.0), (1, 3.0)]}
        )
        scores = {doc: score for doc, score, _ in fused}
        assert scores[1] == pytest.approx(scores[3], rel=1e-9)
        assert scores[1] > scores[2]
        assert (scores[1] - scores[2]) < 1e-4

    def test_is_invariant_to_score_scale(self):
        # The reason RRF is used rather than a weighted sum: BM25 scores are
        # unbounded, cosine similarities are not.
        a = reciprocal_rank_fusion({"x": [(1, 0.9), (2, 0.1)]})
        b = reciprocal_rank_fusion({"x": [(1, 9000.0), (2, 1.0)]})
        assert [d for d, _, _ in a] == [d for d, _, _ in b]

    def test_weights_shift_the_ordering(self):
        arms = {"dense": [(1, 1.0)], "lexical": [(2, 1.0)]}
        dense_heavy = reciprocal_rank_fusion(arms, weights={"dense": 0.9, "lexical": 0.1})
        lexical_heavy = reciprocal_rank_fusion(arms, weights={"dense": 0.1, "lexical": 0.9})
        assert dense_heavy[0][0] == 1
        assert lexical_heavy[0][0] == 2

    def test_provenance_records_each_arm_rank(self):
        fused = reciprocal_rank_fusion({"dense": [(7, 1.0)], "lexical": [(7, 1.0), (8, 0.5)]})
        provenance = {doc: prov for doc, _score, prov in fused}
        assert provenance[7] == {"dense": 1, "lexical": 1}
        assert provenance[8] == {"lexical": 2}

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []


class TestMMR:
    def test_prefers_diversity_over_a_near_duplicate(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.6, 0.8]], dtype=np.float32)
        # Doc 1 is nearly identical to doc 0; doc 2 adds coverage.
        selected = maximal_marginal_relevance(query, vectors, [0, 1, 2], k=2, lambda_mult=0.3)
        assert selected == [0, 2]

    def test_lambda_one_is_pure_relevance(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.6, 0.8]], dtype=np.float32)
        selected = maximal_marginal_relevance(query, vectors, [0, 1, 2], k=2, lambda_mult=1.0)
        assert selected == [0, 1]

    def test_handles_empty_and_oversized_k(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        assert maximal_marginal_relevance(query, np.zeros((0, 2), np.float32), [], k=3) == []
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        assert maximal_marginal_relevance(query, vectors, [5], k=10) == [5]


class TestQueryParsing:
    def test_extracts_drug_and_condition_from_free_text(self, retriever):
        plan = retriever.parse("What side effects do people get from Alphamed for chronic pain?")
        assert plan.drug_name == "Alphamed"
        assert plan.condition == "Chronic Pain"
        assert Aspect.ADVERSE_EFFECT in plan.aspects

    def test_infers_onset_intent(self, retriever):
        plan = retriever.parse("How long does Alphamed take to work?")
        assert Aspect.ONSET_DURATION in plan.aspects

    def test_explicit_arguments_override_extraction(self, retriever):
        plan = retriever.parse("something vague", drug_name="Betacine", condition="Chronic Pain")
        assert plan.drug_name == "Betacine"

    def test_unknown_entities_leave_the_plan_open(self, retriever):
        plan = retriever.parse("Tell me about Nonexistium")
        assert plan.drug_name is None


class TestRetriever:
    def test_returns_a_full_panel(self, retriever, test_config):
        result = retriever.retrieve("Does Alphamed work for chronic pain?")
        assert len(result.units) == test_config.retrieval.final_k

    def test_respects_the_drug_filter(self, retriever):
        result = retriever.retrieve("side effects", drug_name="Betacine")
        assert {ru.unit.drug_name for ru in result.units} == {"Betacine"}

    def test_stratification_reduces_valence_skew(self, retriever):
        query = "Does Alphamed work for chronic pain?"
        stratified = retriever.retrieve(query, stratify=True)
        plain = retriever.retrieve(query, stratify=False)
        assert stratified.valence_skew <= plain.valence_skew

    def test_stratified_panel_covers_more_strata(self, retriever):
        query = "Does Alphamed work for chronic pain?"
        stratified = {ru.unit.stratum for ru in retriever.retrieve(query, stratify=True).units}
        plain = {ru.unit.stratum for ru in retriever.retrieve(query, stratify=False).units}
        assert len(stratified) >= len(plain)

    def test_records_the_allocation_for_audit(self, retriever):
        result = retriever.retrieve("Does Alphamed work for chronic pain?", stratify=True)
        assert set(result.allocation) >= {"quotas", "target", "achieved"}

    def test_lexical_and_dense_modes_both_work(self, retriever):
        for mode in ("dense", "lexical", "hybrid"):
            result = retriever.retrieve("nausea and dizziness", mode=mode)
            assert result.units

    def test_unknown_drug_yields_an_empty_or_unfiltered_panel(self, retriever):
        result = retriever.retrieve("Tell me about Nonexistium please")
        assert result.plan.drug_name is None

    def test_panel_formatting_carries_identifiers_and_ratings(self, retriever):
        result = retriever.retrieve("Does Alphamed work for chronic pain?")
        panel = retriever.format_panel(result)
        for ru in result.units[:3]:
            assert ru.unit.unit_id in panel
        assert "/10" in panel

    def test_panel_respects_the_character_budget(self, retriever):
        result = retriever.retrieve("Does Alphamed work for chronic pain?")
        assert len(retriever.format_panel(result, max_chars=300)) <= 400


class TestIndexPostings:
    def test_metadata_filter_is_an_intersection(self, index):
        ids = index.candidate_ids(drug_name="Alphamed", condition="Insomnia")
        assert ids is not None and ids.size > 0
        for pos in ids:
            assert index.units[pos].drug_name == "Alphamed"
            assert index.units[pos].condition == "Insomnia"

    def test_impossible_combination_returns_empty_not_none(self, index):
        # Empty means "constraints matched nothing" and is grounds for
        # abstention; None means "no constraint". Conflating them is a bug.
        ids = index.candidate_ids(drug_name="Alphamed", condition="High Blood Pressure")
        assert ids is not None
        assert ids.size == 0

    def test_no_constraints_returns_none(self, index):
        assert index.candidate_ids() is None

    def test_stratum_postings_are_consistent(self, index):
        for stratum in Stratum.ordered():
            positions = index.by_stratum.get(stratum.value)
            if positions is None:
                continue
            assert all(index.units[p].stratum is stratum for p in positions)

    def test_round_trips_through_disk(self, index, tmp_path):
        index.save(tmp_path / "idx")
        from pharos.index.store import CorpusIndex

        reloaded = CorpusIndex.load(tmp_path / "idx")
        assert len(reloaded) == len(index)
        assert reloaded.build_info["encoder"] == index.build_info["encoder"]
        assert np.allclose(reloaded.vectors, index.vectors)

    def test_missing_index_raises_an_actionable_error(self, tmp_path):
        from pharos.index.store import CorpusIndex

        with pytest.raises(FileNotFoundError, match="build-index"):
            CorpusIndex.load(tmp_path / "nothing")
