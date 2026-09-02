"""The LangGraph agent: routing, verification, repair, and abstention."""

from __future__ import annotations

import pytest

from pharos.agent.nodes.verify import (
    decompose_claims,
    strip_quotations,
    structural_check,
)
from pharos.agent.state import Claim, new_state
from pharos.config import LLMConfig
from pharos.llm.factory import DeterministicMockChatModel, build_chat_model


class TestClaimDecomposition:
    def test_splits_sentences_and_attaches_citations(self):
        draft = (
            "Reviewers report nausea [EU-1-0]. The cohort mean is 6.2/10 [STAT-0001]. "
            "Some describe relief [EU-2-1]."
        )
        claims = decompose_claims(draft)
        assert len(claims) == 3
        assert claims[0]["citations"] == ["EU-1-0"]
        assert claims[1]["citations"] == ["STAT-0001"]

    def test_splits_on_bullets(self):
        claims = decompose_claims(
            "- First point about the drug [EU-1-0]\n- Second point about it [EU-2-0]"
        )
        assert len(claims) == 2

    def test_ignores_headers_and_fragments(self):
        claims = decompose_claims("What reviewers report:\nOK.\nA real claim here [EU-1-0].")
        assert len(claims) == 1

    def test_indexes_are_one_based_and_contiguous(self):
        claims = decompose_claims("Claim one here [EU-1-0]. Claim two here [EU-2-0].")
        assert [c["index"] for c in claims] == [1, 2]

    def test_empty_draft(self):
        assert decompose_claims("") == []


class TestQuotationStripping:
    def test_removes_quoted_spans(self):
        assert "several" not in strip_quotations(
            'One reviewer writes "I had several bad days" [EU-1-0].'
        )

    def test_leaves_unquoted_text(self):
        assert "Several" in strip_quotations("Several reviewers report nausea [EU-1-0].")


class TestStructuralChecks:
    def make(self, text: str, citations: list[str]) -> Claim:
        return Claim(index=1, text=text, citations=citations, verdict="UNCHECKED", reason="")

    def test_accepts_a_cited_qualitative_claim(self):
        ok, _ = structural_check(
            self.make("Reviewers describe nausea [EU-1-0].", ["EU-1-0"]), {"EU-1-0"}, False
        )
        assert ok

    def test_rejects_an_uncited_claim(self):
        ok, reason = structural_check(
            self.make("Reviewers describe nausea.", []), {"EU-1-0"}, False
        )
        assert not ok and reason == "no citation"

    def test_rejects_a_fabricated_identifier(self):
        # The most dangerous failure: it looks exactly like a real citation.
        ok, reason = structural_check(
            self.make("Reviewers describe nausea [EU-9-9].", ["EU-9-9"]), {"EU-1-0"}, False
        )
        assert not ok and "not in the retrieved evidence" in reason

    @pytest.mark.parametrize(
        "quantifier", ["most", "many", "the majority", "commonly", "rarely", "almost all"]
    )
    def test_rejects_vague_quantifiers_without_a_statistic(self, quantifier):
        ok, reason = structural_check(
            self.make(f"{quantifier} reviewers report nausea [EU-1-0].", ["EU-1-0"]),
            {"EU-1-0"},
            False,
        )
        assert not ok and "quantity claim" in reason

    def test_accepts_a_quantifier_backed_by_a_statistic(self):
        ok, _ = structural_check(
            self.make("Most reviewers report nausea [STAT-0001].", ["STAT-0001"]),
            {"STAT-0001"},
            True,
        )
        assert ok

    def test_does_not_penalise_a_quantifier_inside_a_quotation(self):
        ok, _ = structural_check(
            self.make('One reviewer writes "many bad nights" [EU-1-0].', ["EU-1-0"]),
            {"EU-1-0"},
            False,
        )
        assert ok

    def test_rejects_an_ungrounded_percentage(self):
        ok, reason = structural_check(
            self.make("About 40% of reviewers report nausea [EU-1-0].", ["EU-1-0"]),
            {"EU-1-0"},
            False,
        )
        assert not ok and "numeric claim" in reason

    @pytest.mark.parametrize(
        "advice",
        [
            "You should stop taking it [EU-1-0].",
            "I recommend switching to another drug [EU-1-0].",
            "Consider taking it with food [EU-1-0].",
        ],
    )
    def test_rejects_advice_however_well_cited(self, advice):
        ok, reason = structural_check(self.make(advice, ["EU-1-0"]), {"EU-1-0"}, True)
        assert not ok and "advice" in reason


class TestMockModel:
    def test_is_deterministic(self):
        model = DeterministicMockChatModel()
        prompt = 'TASK: SYNTHESIZE\n[EU-1-0] Alphamed\n  "it worked well for me"'
        assert model._respond(prompt) == model._respond(prompt)

    def test_routes_by_task_marker(self):
        model = DeterministicMockChatModel()
        assert "CLASSIFICATION" in model._respond("TASK: TRIAGE\nShould I stop my meds?")
        assert "CLAIM" in model._respond("TASK: VERIFY\n[EU-1-0]\nCLAIM 1: x [EU-1-0]")

    def test_cites_only_identifiers_present_in_the_prompt(self):
        response = DeterministicMockChatModel()._respond(
            'TASK: SYNTHESIZE\n[EU-42-1] Alphamed — Chronic Pain — rated 9/10\n  "it helped a lot"'
        )
        assert "[EU-42-1]" in response

    def test_missing_key_falls_back_to_the_mock_with_a_warning(self, monkeypatch):
        # Silent degradation to a different model is how an evaluation ends up
        # reporting numbers from a system nobody meant to run.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.warns(RuntimeWarning, match="ANTHROPIC_API_KEY"):
            model = build_chat_model(LLMConfig(provider="anthropic"))
        assert isinstance(model, DeterministicMockChatModel)


class TestGraphRouting:
    def test_informational_question_reaches_an_answer(self, agent):
        state = agent.ask("What side effects do reviewers report on Alphamed for chronic pain?")
        assert state["triage_verdict"] == "INFORMATIONAL"
        assert state["answer"]
        assert not state["abstained"]

    def test_personal_advice_never_reaches_retrieval(self, agent):
        state = agent.ask("Should I stop taking my Alphamed?")
        assert state["abstained"]
        nodes = [event["node"] for event in state["trace"]]
        assert "retrieve" not in nodes

    def test_crisis_is_routed_to_support_resources(self, agent):
        state = agent.ask("I want to die and I have my Alphamed here")
        assert state["triage_verdict"] == "CRISIS"
        assert "988" in state["answer"]

    def test_trace_accumulates_across_nodes(self, agent):
        # Regression: LangGraph's default merge is last-write-wins, which would
        # leave only the final node's record and destroy the audit trail.
        state = agent.ask("Does Alphamed work for chronic pain?")
        nodes = [event["node"] for event in state["trace"]]
        assert nodes[0] == "triage"
        assert "retrieve" in nodes and "compute" in nodes and "finalize" in nodes
        assert len(nodes) >= 5

    def test_answer_carries_the_standing_disclaimer(self, agent):
        state = agent.ask("Does Alphamed work for chronic pain?")
        assert "not medical advice" in state["answer"]

    def test_answer_reports_its_evidence_base(self, agent):
        state = agent.ask("Does Alphamed work for chronic pain?")
        assert "patient-review excerpt" in state["answer"]

    def test_statistics_are_computed_before_generation(self, agent):
        state = agent.ask("What side effects do reviewers report on Betacine for chronic pain?")
        assert state["statistic_ids"]
        trace = {event["node"]: event for event in state["trace"]}
        assert trace["compute"]["computed"] >= 1

    def test_every_surviving_claim_is_cited(self, agent):
        state = agent.ask("Does Alphamed work for chronic pain?")
        for claim in state["claims"]:
            if claim["verdict"] == "SUPPORTED":
                assert claim["citations"]

    def test_repair_loop_is_bounded(self, agent, test_config):
        state = agent.ask("What side effects do reviewers report on Betacine for chronic pain?")
        assert state.get("repair_rounds", 0) <= test_config.agent.max_repair_rounds

    def test_unknown_drug_abstains_rather_than_inventing(self, agent):
        state = agent.ask("What side effects does Nonexistium medication cause in patients?")
        assert state["abstained"] or "enough evidence" in state["answer"]


class TestState:
    def test_new_state_initialises_the_accumulators(self):
        state = new_state("a question")
        assert state["query"] == "a question"
        assert state["trace"] == []
        assert state["repair_rounds"] == 0
