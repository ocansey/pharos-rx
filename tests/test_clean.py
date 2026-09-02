"""Cleaning, label repair, de-identification, and de-duplication."""

from __future__ import annotations

import pandas as pd
import pytest

from pharos.data.clean import (
    ConditionRepairer,
    deidentify,
    find_near_duplicates,
    normalise_text,
    shingle_hashes,
    strip_export_quotes,
    unescape_fully,
)


class TestUnescaping:
    def test_decodes_numeric_entities(self):
        text, changed = unescape_fully("can&#039;t sleep")
        assert text == "can't sleep"
        assert changed

    def test_decodes_double_escaping(self):
        # Not present in the shipped corpus, but a re-export would introduce it
        # and a single-pass unescape would leave `039` for the tokenizer.
        text, changed = unescape_fully("can&amp;#039;t")
        assert text == "can't"
        assert changed

    def test_is_a_no_op_on_clean_text(self):
        text, changed = unescape_fully("no entities here")
        assert text == "no entities here"
        assert not changed

    def test_terminates_on_pathological_input(self):
        # Must not spin: a bounded number of rounds, always.
        text, _ = unescape_fully("&amp;" * 200, max_rounds=3)
        assert isinstance(text, str)


class TestQuoteStripping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('"It worked."', "It worked."),
            ('""It worked.""', "It worked."),
            ("It worked.", "It worked."),
            ('"He said "hi" to me"', 'He said "hi" to me'),
        ],
    )
    def test_strips_only_wrapping_quotes(self, raw, expected):
        assert strip_export_quotes(raw)[0] == expected


class TestConditionRepair:
    @pytest.fixture(scope="class")
    def repairer(self):
        return ConditionRepairer()

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Bipolar Disorde", "Bipolar Disorder"),
            ("Breast Cance", "Breast Cancer"),
            ("Stomach Ulce", "Stomach Ulcer"),
            ("Overactive Bladde", "Overactive Bladder"),
            ("Typhoid Feve", "Typhoid Fever"),
            ("Tinea Versicol", "Tinea Versicolor"),
        ],
    )
    def test_repairs_trailing_deletion(self, repairer, raw, expected):
        repaired, status = repairer.repair(raw)
        assert repaired == expected
        assert "trailing" in status

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ibromyalgia", "Fibromyalgia"),
            ("mance Anxiety", "Performance Anxiety"),
            ("mis", "Dermatitis Herpetiformis"),
            ("me", "Glioblastoma Multiforme"),
            ("t Care", "Foot Care"),
        ],
    )
    def test_repairs_leading_deletion(self, repairer, raw, expected):
        repaired, status = repairer.repair(raw)
        assert repaired == expected
        assert "leading" in status

    def test_repairs_both_deletions_together(self, repairer):
        repaired, status = repairer.repair("zen Shoulde")
        assert repaired == "Frozen Shoulder"
        assert status == "leading+trailing"

    def test_footer_artifact_becomes_null(self, repairer):
        repaired, status = repairer.repair("3</span> users found this comment helpful.")
        assert repaired is None
        assert status == "artifact"

    def test_whitelisted_eponym_is_left_alone(self, repairer):
        # Starts lowercase, which the detector would otherwise treat as a
        # leading deletion. It is simply how the condition is spelled.
        repaired, status = repairer.repair("von Willebrand's Disease")
        assert repaired == "von Willebrand's Disease"
        assert status == "ok"

    def test_leaked_drug_name_is_nulled_not_guessed(self, repairer):
        repaired, status = repairer.repair("min / sitagliptin)")
        assert repaired is None
        assert status == "unrepairable"

    def test_clean_labels_pass_through_untouched(self, repairer):
        for label in ("Depression", "Birth Control", "ADHD", "Diabetes, Type 2"):
            assert repairer.repair(label) == (label, "ok")

    def test_missing_label(self, repairer):
        assert repairer.repair(None)[1] == "missing"
        assert repairer.repair("   ")[1] == "missing"

    def test_audit_reports_every_label(self, repairer):
        labels = pd.Series(["Bipolar Disorde", "Depression", "ibromyalgia", None])
        audit = repairer.audit(labels)
        assert len(audit) == 4
        assert set(audit["status"]) == {"trailing", "ok", "leading", "missing"}


class TestDeidentification:
    @pytest.mark.parametrize(
        ("raw", "tag"),
        [
            ("email me at jane.doe@example.com ok", "EMAIL"),
            ("call 555-123-4567 anytime", "PHONE"),
            ("see https://example.com/x for more", "URL"),
            ("my ssn is 123-45-6789", "SSN"),
            ("order 1234567890123 shipped", "IDNUM"),
        ],
    )
    def test_replaces_identifiers(self, raw, tag):
        out, found = deidentify(raw)
        assert f"[{tag}]" in out
        assert found[tag] >= 1

    def test_leaves_clinical_numbers_alone(self):
        # 20 mg must survive: it is the clinical content, not an identifier.
        out, found = deidentify("I take 20 mg twice a day and it costs $45")
        assert "20 mg" in out
        assert not found


class TestNormalisation:
    def test_collapses_whitespace_and_runaway_punctuation(self):
        assert normalise_text("so   bad!!!!!  really") == "so bad!! really"

    def test_normalises_smart_quotes(self):
        assert normalise_text("it’s “fine”") == 'it\'s "fine"'


class TestNearDuplicates:
    def test_finds_near_duplicates_and_keeps_the_first(self):
        texts = [
            "This medication worked well for my chronic back pain and I sleep better now",
            "This medication worked well for my chronic back pain and I sleep better now!",
            "Completely different review about an entirely unrelated skin condition here",
        ]
        drop = find_near_duplicates(texts, threshold=0.9)
        assert drop == {1}

    def test_respects_blocking(self):
        # Identical text under different drugs is two independent reports.
        texts = ["Worked well for my chronic pain and I can sleep at night now"] * 2
        assert find_near_duplicates(texts, threshold=0.9, blocks=["A", "B"]) == set()
        assert find_near_duplicates(texts, threshold=0.9, blocks=["A", "A"]) == {1}

    def test_is_deterministic_across_calls(self):
        texts = [f"review number {i} about medication effects and outcomes" for i in range(40)]
        assert find_near_duplicates(texts, seed=3) == find_near_duplicates(texts, seed=3)

    def test_shingle_hashes_are_process_stable(self):
        # A salted hash here would make the corpus differ between runs.
        assert shingle_hashes("stable input") == shingle_hashes("stable input")

    def test_handles_empty_input(self):
        assert find_near_duplicates([]) == set()
