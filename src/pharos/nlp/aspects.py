"""Aspect labelling for evidence units.

Each unit is tagged with the facets it speaks to — efficacy, adverse effect,
onset, dosing, cost, discontinuation, comparison — so that retrieval can be
conditioned on what the user actually asked about. A question about how long a
drug takes to work should not be answered with a paragraph about the copay.

The labeller is a **deterministic weak supervisor**: high-precision surface cues
scored per aspect, with the adverse-effect aspect additionally driven by the
clinical lexicon and suppressed under negation. This choice is deliberate.

* It is *auditable*. Every label traces to a cue you can read, which matters
  when the labels go on to shape a pharmacovigilance statistic.
* It is *free*. Labelling 70k units with an LLM would cost real money and
  introduce a dependency on a model version, defeating reproducibility.
* It is *honest about its ceiling*. `docs/RESULTS.md` reports agreement against
  a 300-unit hand-annotated sample rather than asserting the labels are correct.

Where a neural labeller would win, the interface allows one: any callable with
the signature of :func:`label_aspects` can be substituted at index time.
"""

from __future__ import annotations

import re
from typing import Final

from pharos.data.schema import Aspect
from pharos.nlp.lexicon import extract_adverse_concepts

#: Cue patterns per aspect. Weights encode precision, not frequency: a phrase
#: that is almost always diagnostic scores higher than one that merely correlates.
_CUES: Final[dict[Aspect, list[tuple[str, float]]]] = {
    Aspect.EFFICACY: [
        (r"\b(work(s|ed|ing)?|effective|helps?|helped|helping)\b", 1.0),
        (r"\b(life ?saver|miracle|game changer|godsend)\b", 1.6),
        (
            r"\b(no (?:relief|help|difference|change)|did ?n[o']t (?:work|help)|useless|waste of money)\b",
            1.6,
        ),
        (r"\b(symptom-?free|cleared (?:it )?up|under control|in remission|cured)\b", 1.4),
        (r"\b(pain (?:is )?gone|much better|so much better|improved|improvement)\b", 1.2),
        (r"\b(still (?:in pain|struggling|suffering)|no better|got worse|worsened)\b", 1.2),
    ],
    Aspect.ADVERSE_EFFECT: [
        (r"\bside ?effects?\b", 1.8),
        (r"\b(adverse|reaction|intolerab\w+|unbearable)\b", 1.2),
        (r"\b(made me (?:feel )?(?:sick|awful|terrible|horrible))\b", 1.4),
        (r"\b(nightmare|hell|worst (?:drug|experience|thing))\b", 0.8),
    ],
    Aspect.ONSET_DURATION: [
        # Spelled-out quantities matter here: "it took about three weeks" is the
        # canonical way patients express onset, and a digits-only pattern misses
        # the majority of onset statements in the corpus.
        (
            r"\b(?:after|within|took|takes|for|in)\s+(?:about|around|roughly|nearly)?\s*"
            r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several)\s*"
            r"(?:of\s+)?(?:min(?:ute)?s?|hours?|hrs?|days?|weeks?|months?|years?)\b",
            1.6,
        ),
        (r"\b(first (?:day|week|month)|second (?:day|week|month)|third (?:day|week|month))\b", 1.3),
        (
            r"\b(right away|immediately|instantly|kicked in|kicks in|started working|wore off|wears off)\b",
            1.5,
        ),
        (r"\b(long ?term|short ?term|over time|eventually|gradually|so far)\b", 0.9),
    ],
    Aspect.DOSING: [
        (r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|ml|g|iu|units?)\b", 1.8),
        (
            r"\b(dose|dosage|dosing|titrat\w+|taper\w+|increased? (?:to|my)|decreased? (?:to|my)|upped|lowered)\b",
            1.3,
        ),
        (
            r"\b(twice a day|once a day|every (?:day|morning|night|\d+ hours)|bid|tid|prn|as needed|at bedtime|with food|empty stomach)\b",
            1.4,
        ),
        (
            r"\b(half a? (?:pill|tablet)|split the (?:pill|tablet)|extended release|er\b|xr\b|generic|brand)\b",
            0.9,
        ),
    ],
    Aspect.ACCESS_COST: [
        (r"[$£€]\s?\d", 1.8),
        (
            r"\b(cost|costs|expensive|cheap|price|pricing|copay|co-?pay|insurance|coverage|covered|pharmacy|out of pocket|afford|goodrx|coupon)\b",
            1.4,
        ),
        (r"\b(prior authorization|formulary|manufacturer (?:coupon|program))\b", 1.6),
    ],
    Aspect.DISCONTINUATION: [
        (
            r"\b(stopp?ed? (?:taking|it)|quit|came off|coming off|discontinu\w+|weaning|weaned|taper(?:ing|ed)? off)\b",
            1.7,
        ),
        (r"\b(withdrawal|brain zaps|rebound|switched (?:to|off)|had to stop)\b", 1.6),
        (r"\b(back on it|restarted|went back)\b", 1.0),
    ],
    Aspect.COMPARISON: [
        (
            r"\b(compared to|versus|vs\.?|better than|worse than|instead of|switched from|tried .* before|unlike)\b",
            1.5,
        ),
        (r"\b(the only (?:thing|one) that)\b", 1.1),
    ],
}

_COMPILED: Final[dict[Aspect, list[tuple[re.Pattern[str], float]]]] = {
    aspect: [(re.compile(p, re.IGNORECASE), w) for p, w in pairs] for aspect, pairs in _CUES.items()
}

#: A unit must clear this to receive an aspect at all.
DEFAULT_THRESHOLD: Final[float] = 1.0
#: An aspect within this factor of the top scorer is also assigned, so a unit
#: that genuinely speaks to two facets ("worked in 3 days but gave me a rash")
#: is retrievable under both.
SECONDARY_RATIO: Final[float] = 0.6
#: Weight contributed by each distinct lexicon-confirmed adverse concept.
LEXICON_WEIGHT: Final[float] = 1.5


def score_aspects(text: str) -> dict[Aspect, float]:
    """Raw per-aspect scores. Exposed for inspection and for the agreement study."""
    scores: dict[Aspect, float] = {}
    for aspect, patterns in _COMPILED.items():
        total = 0.0
        for pattern, weight in patterns:
            if pattern.search(text):
                total += weight
        if total:
            scores[aspect] = total

    concepts, _ = extract_adverse_concepts(text)
    if concepts:
        scores[Aspect.ADVERSE_EFFECT] = scores.get(
            Aspect.ADVERSE_EFFECT, 0.0
        ) + LEXICON_WEIGHT * min(len(concepts), 3)
    return scores


def label_aspects(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    secondary_ratio: float = SECONDARY_RATIO,
    max_aspects: int = 3,
) -> list[Aspect]:
    """Assign aspects to one evidence unit.

    Returns the highest-scoring aspect plus any aspect within
    ``secondary_ratio`` of it, capped at ``max_aspects``. A unit with no cue
    above ``threshold`` is labelled :attr:`Aspect.CONTEXT` — narrative framing
    rather than a clinical claim — which the retriever de-prioritises but does
    not discard, because context is what makes a citation readable.
    """
    scores = score_aspects(text)
    if not scores:
        return [Aspect.CONTEXT]

    top = max(scores.values())
    if top < threshold:
        return [Aspect.CONTEXT]

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].value))
    keep = [a for a, s in ranked if s >= max(threshold, top * secondary_ratio)]
    return keep[:max_aspects] or [Aspect.CONTEXT]


#: Query-side cues, used to infer which aspects a *question* is asking about so
#: the retriever can weight the panel accordingly.
_QUERY_CUES: Final[dict[Aspect, re.Pattern[str]]] = {
    Aspect.ADVERSE_EFFECT: re.compile(
        r"\b(side ?effects?|adverse|risks?|dangers?|safe|harm|reaction|tolerat)", re.I
    ),
    Aspect.EFFICACY: re.compile(
        r"\b(work|effective|efficacy|help|success|worth it|any good|results?)", re.I
    ),
    Aspect.ONSET_DURATION: re.compile(
        r"\b(how long|how soon|when (?:will|does)|onset|kick in|takes? to work|duration)", re.I
    ),
    Aspect.DOSING: re.compile(r"\b(dose|dosage|mg|how much|how many|titrat|taper|schedule)", re.I),
    Aspect.ACCESS_COST: re.compile(r"\b(cost|price|expensive|insurance|copay|afford|cover)", re.I),
    Aspect.DISCONTINUATION: re.compile(
        r"\b(stop|quit|come off|coming off|withdrawal|discontinu|wean)", re.I
    ),
    Aspect.COMPARISON: re.compile(
        r"\b(compare|versus|vs\.?|better than|instead of|or\b.*\?)", re.I
    ),
}


def infer_query_aspects(query: str) -> list[Aspect]:
    """Infer the aspects a natural-language question is asking about.

    An empty result means "no strong signal", and the retriever then treats all
    clinical aspects as equally admissible rather than guessing.
    """
    hits = [aspect for aspect, pattern in _QUERY_CUES.items() if pattern.search(query)]
    return hits
