"""Segmentation, aspect labelling, and the clinical lexicon."""

from __future__ import annotations

import pytest

from pharos.data.schema import Aspect, Stratum
from pharos.nlp.aspects import infer_query_aspects, label_aspects, score_aspects
from pharos.nlp.lexicon import (
    CONCEPT_TO_SOC,
    TERM_TO_CONCEPT,
    extract_adverse_concepts,
    find_adverse_terms,
    is_hedged,
    is_negated,
)
from pharos.nlp.segment import segment_review


class TestSegmentation:
    def test_splits_on_sentence_boundaries(self):
        segments = segment_review(
            "It worked well for my anxiety within a week of starting the medication. "
            "The nausea in the mornings was hard to deal with at first.",
            min_chars=20,
        )
        assert len(segments) == 2

    def test_splits_on_a_contrast_connective(self):
        # The single most important boundary in this corpus: "but" is where the
        # efficacy claim ends and the adverse-effect claim begins.
        segments = segment_review(
            "Worked great for my anxiety within about a week of starting it "
            "but the weight gain was absolutely awful and I had to stop",
            min_chars=20,
        )
        assert len(segments) == 2
        assert "but" in segments[1].text.lower()

    def test_keeps_the_connective_with_the_clause_it_governs(self):
        segments = segment_review(
            "This medication helped my chronic pain considerably over two months "
            "however the drowsiness made driving genuinely unsafe for me",
            min_chars=20,
        )
        assert segments[1].text.lower().startswith("however")

    def test_merges_fragments_below_the_minimum(self):
        segments = segment_review(
            "Great. This medication really helped my chronic back pain a lot.", min_chars=40
        )
        assert all(len(s.text) >= 20 for s in segments)

    def test_splits_oversized_units(self):
        long_text = "word " * 400
        segments = segment_review(long_text, min_chars=20, max_chars=200)
        assert all(len(s.text) <= 200 for s in segments)

    def test_does_not_split_on_a_decimal_dose(self):
        segments = segment_review(
            "I take 2.5 mg every morning and it has worked well", min_chars=10
        )
        assert len(segments) == 1

    def test_offsets_are_within_the_source(self):
        text = "It worked well for me. The side effects were tolerable overall."
        for segment in segment_review(text, min_chars=10):
            assert 0 <= segment.start <= segment.end <= len(text)

    def test_ordinals_are_contiguous(self):
        segments = segment_review(
            "First claim about efficacy here. Second claim about side effects here. "
            "Third claim about the cost.",
            min_chars=10,
        )
        assert [s.ordinal for s in segments] == list(range(len(segments)))

    def test_no_review_disappears(self):
        assert len(segment_review("Short.", min_chars=100)) == 1

    def test_empty_input(self):
        assert segment_review("") == []
        assert segment_review("   ") == []


class TestLexicon:
    def test_maps_lay_terms_to_clinical_concepts(self):
        # The point of a lay lexicon: patients write "knocked me out".
        assert TERM_TO_CONCEPT["knocked me out"] == "somnolence"
        assert TERM_TO_CONCEPT["room spinning"] == "dizziness"
        assert TERM_TO_CONCEPT["brain zaps"] == "withdrawal"

    def test_every_concept_has_a_system_organ_class(self):
        for concept in set(TERM_TO_CONCEPT.values()):
            assert concept in CONCEPT_TO_SOC

    def test_prefers_the_longest_match(self):
        found = {surface for surface, _c, _s, _e in find_adverse_terms("I had muscle cramps daily")}
        assert "muscle cramps" in found
        assert "cramps" not in found

    def test_matches_on_word_boundaries(self):
        assert not find_adverse_terms("the therapist was helpful")


class TestNegation:
    def test_detects_a_negated_mention(self):
        text = "I had no nausea at all"
        matches = find_adverse_terms(text)
        assert matches
        assert is_negated(text, matches[0][2])

    def test_negation_does_not_leak_past_a_contrast(self):
        # "no nausea but terrible headaches" -- the headache is affirmed.
        text = "I had no nausea but terrible headaches every day"
        concepts, saw_negation = extract_adverse_concepts(text)
        assert "headache" in concepts
        assert "nausea" not in concepts
        assert saw_negation

    def test_negation_does_not_leak_past_a_sentence_boundary(self):
        text = "There was no dizziness. The nausea was constant."
        concepts, _ = extract_adverse_concepts(text)
        assert "nausea" in concepts

    def test_affirmed_mentions_survive(self):
        concepts, saw_negation = extract_adverse_concepts("Terrible nausea and dizziness")
        assert set(concepts) >= {"nausea", "dizziness"}
        assert not saw_negation


class TestHedging:
    @pytest.mark.parametrize(
        "text",
        [
            "I think it might be the pill",
            "possibly related to the drug",
            "not sure if it caused it",
        ],
    )
    def test_detects_hedges(self, text):
        assert is_hedged(text)

    def test_plain_assertion_is_not_hedged(self):
        assert not is_hedged("The drug gave me severe nausea")


class TestAspectLabelling:
    @pytest.mark.parametrize(
        ("text", "aspect"),
        [
            ("This medication worked wonderfully for my condition", Aspect.EFFICACY),
            ("Terrible side effects, constant nausea and dizziness", Aspect.ADVERSE_EFFECT),
            ("It took about three weeks to start working properly", Aspect.ONSET_DURATION),
            ("I take 20 mg twice a day with food", Aspect.DOSING),
            ("The copay was $120 a month and insurance would not cover it", Aspect.ACCESS_COST),
            ("I stopped taking it and had terrible withdrawal", Aspect.DISCONTINUATION),
            ("Much better than the sertraline I tried before", Aspect.COMPARISON),
        ],
    )
    def test_assigns_the_right_primary_aspect(self, text, aspect):
        assert aspect in label_aspects(text)

    def test_assigns_multiple_aspects_to_a_multi_facet_span(self):
        aspects = label_aspects("It worked within three days but gave me an awful rash")
        assert Aspect.ADVERSE_EFFECT in aspects
        assert len(aspects) >= 2

    def test_uncued_text_is_labelled_context(self):
        assert label_aspects("I am a 45 year old woman from Ohio") == [Aspect.CONTEXT]

    def test_caps_the_number_of_aspects(self):
        text = (
            "It worked within three days at 20 mg twice a day but the nausea was awful "
            "and the copay was $200 so I stopped taking it, much worse than my last drug"
        )
        assert len(label_aspects(text)) <= 3

    def test_the_lexicon_reinforces_the_adverse_aspect(self):
        scores = score_aspects("constant nausea, dizziness and hair loss")
        assert scores[Aspect.ADVERSE_EFFECT] > 1.0

    def test_negated_symptoms_do_not_create_an_adverse_label(self):
        aspects = label_aspects("I had no nausea and no dizziness at all from this")
        assert Aspect.ADVERSE_EFFECT not in aspects


class TestQueryAspectInference:
    @pytest.mark.parametrize(
        ("query", "aspect"),
        [
            ("What side effects does it cause?", Aspect.ADVERSE_EFFECT),
            ("Does it actually work?", Aspect.EFFICACY),
            ("How long does it take to work?", Aspect.ONSET_DURATION),
            ("What dose do people take?", Aspect.DOSING),
            ("Is it expensive?", Aspect.ACCESS_COST),
            ("What happens when you stop taking it?", Aspect.DISCONTINUATION),
        ],
    )
    def test_infers_intent(self, query, aspect):
        assert aspect in infer_query_aspects(query)

    def test_no_signal_returns_empty_rather_than_guessing(self):
        assert infer_query_aspects("Tell me about this medication") == []


class TestStrata:
    @pytest.mark.parametrize(
        ("rating", "stratum"),
        [
            (1.0, Stratum.NEGATIVE),
            (4.0, Stratum.NEGATIVE),
            (5.0, Stratum.MIXED),
            (6.0, Stratum.MIXED),
            (7.0, Stratum.POSITIVE),
            (10.0, Stratum.POSITIVE),
        ],
    )
    def test_rating_boundaries(self, rating, stratum):
        assert Stratum.from_rating(rating) is stratum

    def test_ordering_is_stable(self):
        assert Stratum.ordered() == (Stratum.NEGATIVE, Stratum.MIXED, Stratum.POSITIVE)
