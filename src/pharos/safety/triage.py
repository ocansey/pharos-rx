"""Safety triage.

Two layers, in this order.

**A deterministic rule layer** runs first and can refuse on its own. It exists
because the hazards here are recognisable from surface form, and because a
refusal that depends on a language-model call is a refusal that fails open when
the API is down, the key is missing, or the response is malformed. On a system
that fields questions about medication, failing open is not an acceptable
default.

**A model layer** runs second, on whatever survives, and catches the cases the
rules cannot see — indirection, framing, an individualised request phrased in the
third person. It can escalate a rule verdict but never soften one.

The design choice worth stating plainly: this system refuses an entire *class* of
question it is technically capable of answering. Asked "should I stop my
sertraline", it could assemble a fluent, well-cited, extremely persuasive answer
from hundreds of reviews. That capability is exactly the danger. The reviews
describe other people, the asker's situation is not in them, and fluency would
read as authority. The refusal is a design position, not a limitation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from pharos.config import SafetyConfig


class Verdict(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    PERSONAL_MEDICAL_ADVICE = "PERSONAL_MEDICAL_ADVICE"
    CRISIS = "CRISIS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class TriageResult:
    verdict: Verdict
    rationale: str
    source: str  # "rules" | "model" | "default"
    matched: list[str]

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.INFORMATIONAL


# --------------------------------------------------------------------------- #
# Crisis — checked first, and never overridden by anything downstream.
# --------------------------------------------------------------------------- #
_CRISIS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "suicidal_ideation",
        re.compile(
            r"\b(kill myself|killing myself|end my life|ending my life|want to die|"
            r"wanna die|suicidal|suicide|not want to (?:be here|live)|better off dead)\b",
            re.I,
        ),
    ),
    ("self_harm", re.compile(r"\b(self[- ]harm|hurt myself|cutting myself|harm myself)\b", re.I)),
    (
        "overdose",
        re.compile(
            r"\b(overdose|od'?ed|took (?:too many|the whole bottle|all my)|"
            r"how many .{0,30}(?:to die|would kill)|lethal dose|fatal dose)\b",
            re.I,
        ),
    ),
    (
        "acute_emergency",
        re.compile(
            r"\b(can'?t breathe|cannot breathe|chest pain right now|throat (?:is )?clos\w+|"
            r"anaphyla\w+|unconscious|unresponsive|seizing)\b",
            re.I,
        ),
    ),
)

# --------------------------------------------------------------------------- #
# Individualised clinical requests.
# --------------------------------------------------------------------------- #
_PERSONAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Deliberative modals in any person. Third-person framing ("asking for a
    # friend: should she stop?") is the same request wearing a coat, and an
    # earlier first-person-only version let every one of them through.
    (
        "should_i",
        re.compile(
            r"\b(should (?:i|we|he|she|they|my \w+|her \w+|his \w+)|shall i|"
            r"do i need to|ought i to|would you (?:recommend|suggest)|"
            r"asking for a friend|is it (?:ok|okay|safe|alright|fine) (?:for me )?(?:to|if))\b",
            re.I,
        ),
    ),
    # A possessive within a short window of a regimen noun. The window matters:
    # "my 10mg dose of escitalopram" puts two tokens between the two cues, and
    # an adjacency-only pattern misses it.
    (
        "my_regimen",
        re.compile(
            r"\b(?:my|our|his|her|their|i(?:'m| am))\b[^.?!]{0,24}?\b"
            r"(?:doctor|dr\.?|prescription|prescribed|dose|dosage|meds?|medication|"
            r"pills?|tablets?|regimen|treatment|currently (?:on|taking)|taking)\b",
            re.I,
        ),
    ),
    (
        "dose_for_me",
        re.compile(
            r"\b(how (?:much|many) should (?:i|we|he|she|they)|what dose should|"
            r"can i (?:take|have|use)|how do i (?:taper|stop|come off|wean)|"
            r"(?:safe|maximum|max|right|correct) dose .{0,30}\bfor me\b|"
            r"\d+\s?(?:mg|mcg|ml|g)\b[^.?!]{0,40}\b(?:too much|too high|too low|safe|enough)\b)",
            re.I,
        ),
    ),
    # "for me", "for myself", "in my case" - an explicit request to individualise,
    # whatever the surrounding phrasing.
    (
        "about_me",
        re.compile(
            r"\b(for me\b|for myself\b|in my case\b|my situation|i should\b|"
            r"i'?m on\b|i am on\b|i started\b|i've been (?:taking|on)\b)",
            re.I,
        ),
    ),
    (
        "interaction_personal",
        re.compile(
            r"\b(can i (?:take|combine|mix)|is it safe to (?:take|combine|mix)|"
            r"will .{0,25}interact with (?:my|his|her|their))\b",
            re.I,
        ),
    ),
    (
        "diagnosis",
        re.compile(
            r"\b(do i have|what'?s wrong with me|"
            r"is (?:this|that|it|my)\b[^.?!]{0,32}?\b(?:normal|serious|dangerous|"
            r"a side ?effect|side ?effect|an? allergic reaction|something worse)|"
            r"diagnose|am i (?:having|experiencing|allergic)|"
            r"what does (?:my|this) symptom mean)",
            re.I,
        ),
    ),
    (
        "switch_stop",
        re.compile(
            r"\b(should (?:i|we|he|she|they) (?:stop|switch|quit|change|skip)|"
            r"can i (?:stop|skip|double)|is it (?:ok|okay|safe|fine) (?:to|if i) (?:stop|skip)|"
            r"how do i get off)\b",
            re.I,
        ),
    ),
)

# --------------------------------------------------------------------------- #
# Aggregate phrasing that clears an otherwise personal-looking question.
# "My doctor mentioned X — what do reviewers say about it?" is informational.
# --------------------------------------------------------------------------- #
_AGGREGATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(what (?:do|did) (?:people|patients|reviewers|users)|"
        r"how (?:do|did) (?:people|patients|reviewers|users))\b",
        re.I,
    ),
    re.compile(
        r"\b(most (?:commonly )?reported|commonly reported|in the reviews|"
        r"according to reviews|reviewers report|patients report)\b",
        re.I,
    ),
    re.compile(r"\b(what side effects (?:are|do people|do patients|get) )", re.I),
    re.compile(r"\b(compare|comparison|versus|vs\.?) \b", re.I),
    re.compile(r"\b(how (?:is|are) .{2,40} rated|ratings? for|reviews? (?:of|for))\b", re.I),
)

# Stems rather than whole words, and no trailing word boundary: "side effects",
# "medications", "reviewers" and "prescribed" must all match. An earlier version
# anchored on \b and classified "What side effects do people report on
# sertraline?" as out of scope, because the boundary after "effect" fell inside
# "effects" -- a reminder that a boundary assertion is not a free safety net.
_IN_SCOPE = re.compile(
    r"\b(drugs?|medicat\w*|medicin\w*|pills?|tablets?|capsules?|dos\w*|"
    r"side[- ]?effect\w*|adverse|review\w*|patients?|treat\w*|prescri\w*|"
    r"\d+\s?(?:mg|mcg|ml)|symptom\w*|therap\w*|withdrawal|taper\w*|"
    r"generic|brand|pharmac\w*|antidepress\w*|antibiotic\w*|inject\w*)",
    re.I,
)


class SafetyTriage:
    """The rule layer, plus an optional model escalation."""

    def __init__(self, cfg: SafetyConfig, vocabulary: set[str] | None = None) -> None:
        self.cfg = cfg
        # Naming a drug the corpus knows about is itself evidence that a question
        # is in scope. Without this, "Does Lipitor work for high cholesterol?"
        # is classified out of scope, because it contains no generic medical
        # keyword at all -- only a proper noun the keyword list cannot anticipate.
        self.vocabulary = {v.casefold() for v in (vocabulary or set()) if len(v) >= 4}

    def _mentions_known_entity(self, text: str) -> bool:
        if not self.vocabulary:
            return False
        lowered = text.casefold()
        return any(
            re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered) for name in self.vocabulary
        )

    # ------------------------------------------------------------------ #
    def classify_rules(self, query: str) -> TriageResult:
        """Deterministic classification. Never raises, never calls out."""
        if not self.cfg.enabled:
            return TriageResult(Verdict.INFORMATIONAL, "safety disabled", "default", [])

        text = query.strip()

        if self.cfg.crisis_routing:
            crisis_hits = [name for name, p in _CRISIS_PATTERNS if p.search(text)]
            if crisis_hits:
                return TriageResult(
                    Verdict.CRISIS,
                    "crisis language detected; routed to support resources",
                    "rules",
                    crisis_hits,
                )

        personal_hits = [name for name, p in _PERSONAL_PATTERNS if p.search(text)]
        aggregate = any(p.search(text) for p in _AGGREGATE_PATTERNS)

        if personal_hits and self.cfg.refuse_personal_medical_advice:
            # A single weak cue ("my doctor") inside an otherwise clearly
            # aggregate question is context, not a request for personal advice.
            # Two or more cues, or any strong cue, is not rescued by phrasing.
            strong = {
                "should_i",
                "dose_for_me",
                "switch_stop",
                "diagnosis",
                "interaction_personal",
                "about_me",
            }
            if aggregate and len(personal_hits) == 1 and not (set(personal_hits) & strong):
                pass
            else:
                return TriageResult(
                    Verdict.PERSONAL_MEDICAL_ADVICE,
                    "request is individualised and would require the asker's clinical history",
                    "rules",
                    personal_hits,
                )

        if not _IN_SCOPE.search(text) and not self._mentions_known_entity(text):
            return TriageResult(
                Verdict.OUT_OF_SCOPE, "no medication-related content detected", "rules", []
            )

        return TriageResult(Verdict.INFORMATIONAL, "aggregate question about reviews", "rules", [])

    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_model_verdict(response: str) -> Verdict | None:
        match = re.search(r"CLASSIFICATION:\s*([A-Z_]+)", response)
        if not match:
            return None
        try:
            return Verdict(match.group(1).strip())
        except ValueError:
            return None

    @staticmethod
    def merge(rule_result: TriageResult, model_verdict: Verdict | None) -> TriageResult:
        """Combine the two layers. The model may escalate; it may not soften.

        Precedence is CRISIS > PERSONAL_MEDICAL_ADVICE > OUT_OF_SCOPE >
        INFORMATIONAL. A model that returns INFORMATIONAL for something the rules
        flagged is overruled — the asymmetry is deliberate, because the cost of
        the two errors is not symmetric.
        """
        if model_verdict is None or rule_result.verdict is Verdict.CRISIS:
            return rule_result

        severity = {
            Verdict.INFORMATIONAL: 0,
            Verdict.OUT_OF_SCOPE: 1,
            Verdict.PERSONAL_MEDICAL_ADVICE: 2,
            Verdict.CRISIS: 3,
        }
        if severity[model_verdict] > severity[rule_result.verdict]:
            return TriageResult(
                model_verdict,
                "escalated by model classifier",
                "model",
                rule_result.matched,
            )
        return rule_result
