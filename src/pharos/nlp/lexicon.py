"""Clinical lexicons for lay patient narrative.

The vocabulary here is deliberately *lay*, not clinical. Patients write "knocked
me out", not "somnolence"; "the room was spinning", not "vertigo". A lexicon
lifted from MedDRA or SNOMED matches almost nothing in this corpus. Every term
below was selected by frequency analysis over the 215,063-review corpus and then
mapped to its clinical concept, so the surface forms match what people write
while the concepts aggregate the way a pharmacovigilance analyst needs.

Three families of cue are modelled:

* **Adverse-event terms**, grouped by system-organ class.
* **Negation cues**, because "no nausea at all" and "nausea" are opposite
  reports and a bag-of-words retriever cannot tell them apart.
* **Hedge and attribution cues**, because "I think it might be the pill" is a
  weaker report than "the pill did this", and a disproportionality statistic
  that counts them equally is overstating its evidence.
"""

from __future__ import annotations

import re
from typing import Final

# --------------------------------------------------------------------------- #
# Adverse events, by system-organ class, in the words patients actually use
# --------------------------------------------------------------------------- #
ADVERSE_EVENT_LEXICON: Final[dict[str, dict[str, list[str]]]] = {
    "nervous_system": {
        "headache": ["headache", "head ache", "migraine", "head pain", "pounding head"],
        "dizziness": [
            "dizzy",
            "dizziness",
            "lightheaded",
            "light headed",
            "vertigo",
            "room spinning",
            "woozy",
        ],
        "somnolence": [
            "drowsy",
            "drowsiness",
            "sleepy",
            "knocked me out",
            "zombie",
            "groggy",
            "sedated",
            "sluggish",
        ],
        "insomnia": [
            "insomnia",
            "cant sleep",
            "can't sleep",
            "couldn't sleep",
            "trouble sleeping",
            "kept me awake",
            "wide awake",
        ],
        "tremor": ["tremor", "shaky", "shakes", "trembling", "twitching", "jitters", "jittery"],
        "cognitive_impairment": [
            "brain fog",
            "foggy",
            "memory loss",
            "forgetful",
            "cant focus",
            "can't concentrate",
            "spacey",
            "word finding",
        ],
        "seizure": ["seizure", "convulsion", "fit"],
        "paresthesia": ["tingling", "numbness", "pins and needles", "burning sensation"],
    },
    "psychiatric": {
        "anxiety": ["anxious", "anxiety", "panic attack", "panicky", "on edge", "restless"],
        "depression_mood": [
            "depressed",
            "depression",
            "hopeless",
            "crying spells",
            "numb",
            "flat",
            "emotionless",
            "no motivation",
        ],
        "irritability": ["irritable", "angry", "rage", "short fuse", "agitated", "aggressive"],
        "suicidal_ideation": [
            "suicidal",
            "suicidal thoughts",
            "wanted to die",
            "self harm",
            "end my life",
        ],
        "libido_change": [
            "libido",
            "sex drive",
            "no interest in sex",
            "cant orgasm",
            "can't orgasm",
            "anorgasmia",
        ],
        "vivid_dreams": ["vivid dreams", "nightmares", "weird dreams", "night terrors"],
    },
    "gastrointestinal": {
        "nausea": ["nausea", "nauseous", "nauseated", "queasy", "sick to my stomach"],
        "vomiting": ["vomit", "vomiting", "threw up", "throwing up", "puking"],
        "diarrhea": ["diarrhea", "diarrhoea", "loose stools", "the runs", "watery stool"],
        "constipation": ["constipation", "constipated", "cant go", "backed up"],
        "abdominal_pain": [
            "stomach pain",
            "stomach ache",
            "cramps",
            "cramping",
            "abdominal pain",
            "gut pain",
            "bloating",
            "bloated",
        ],
        "dyspepsia": ["heartburn", "acid reflux", "indigestion", "gerd"],
        "appetite_change": [
            "no appetite",
            "loss of appetite",
            "not hungry",
            "appetite suppressed",
            "always hungry",
            "increased appetite",
            "cravings",
        ],
        "dry_mouth": ["dry mouth", "cotton mouth", "parched"],
    },
    "cardiovascular": {
        "palpitations": [
            "heart racing",
            "palpitations",
            "racing heart",
            "heart pounding",
            "rapid heartbeat",
            "tachycardia",
        ],
        "hypotension": [
            "low blood pressure",
            "blood pressure dropped",
            "fainted",
            "passed out",
            "syncope",
        ],
        "hypertension": ["blood pressure went up", "high blood pressure", "bp spiked"],
        "chest_pain": ["chest pain", "chest tightness", "chest pressure"],
        "edema": [
            "swelling",
            "swollen ankles",
            "swollen feet",
            "edema",
            "water retention",
            "puffy",
        ],
    },
    "dermatologic": {
        "rash": ["rash", "hives", "welts", "breakout", "skin reaction", "itchy skin"],
        "pruritus": ["itching", "itchy", "itch"],
        "alopecia": ["hair loss", "losing hair", "hair falling out", "thinning hair", "bald"],
        "dry_skin": ["dry skin", "peeling", "flaking", "chapped lips", "cracked lips"],
        "photosensitivity": ["sun sensitivity", "sunburn easily", "photosensitive"],
        "sweating": ["night sweats", "sweating", "hot flashes", "sweaty"],
    },
    "musculoskeletal": {
        "myalgia": ["muscle pain", "muscle aches", "sore muscles", "body aches", "achy"],
        "arthralgia": ["joint pain", "joint aches", "stiff joints"],
        "weakness": [
            "weakness",
            "weak",
            "no strength",
            "fatigue",
            "exhausted",
            "tired all the time",
            "lethargic",
        ],
        "cramps": ["leg cramps", "muscle cramps", "charlie horse"],
    },
    "metabolic": {
        "weight_gain": [
            "weight gain",
            "gained weight",
            "put on weight",
            "gaining weight",
            "packed on",
        ],
        "weight_loss": ["weight loss", "lost weight", "losing weight", "dropped pounds"],
        "hyperglycemia": ["blood sugar", "sugar spiked", "glucose high"],
    },
    "genitourinary": {
        "urinary_change": [
            "frequent urination",
            "peeing a lot",
            "urinary retention",
            "cant pee",
            "burning when i pee",
        ],
        "menstrual_change": [
            "spotting",
            "breakthrough bleeding",
            "heavy period",
            "no period",
            "irregular period",
            "missed period",
            "cramping",
        ],
        "yeast_infection": ["yeast infection", "thrush"],
    },
    "systemic": {
        "allergic_reaction": [
            "allergic reaction",
            "anaphylaxis",
            "throat closed",
            "swelling of face",
            "trouble breathing",
        ],
        "flu_like": ["flu like symptoms", "chills", "fever", "achy and feverish"],
        "withdrawal": [
            "withdrawal",
            "brain zaps",
            "rebound",
            "coming off",
            "tapering",
            "discontinuation",
        ],
    },
}

#: Flat surface-form -> concept map, built once at import.
TERM_TO_CONCEPT: Final[dict[str, str]] = {
    surface: concept
    for soc in ADVERSE_EVENT_LEXICON.values()
    for concept, surfaces in soc.items()
    for surface in surfaces
}

#: Concept -> system-organ class.
CONCEPT_TO_SOC: Final[dict[str, str]] = {
    concept: soc_name for soc_name, soc in ADVERSE_EVENT_LEXICON.items() for concept in soc
}

# Longest-first so that "muscle cramps" wins over "cramps".
_SORTED_TERMS: Final[list[str]] = sorted(TERM_TO_CONCEPT, key=len, reverse=True)

# A trailing plural is absorbed rather than blocking the match. Without the
# `(?:e?s)?` group, "headaches" -- by far the more common surface form in this
# corpus than the singular -- matches nothing at all, because the closing
# boundary assertion falls between "headache" and its own plural "s". The
# captured group excludes the suffix, so the concept lookup still resolves.
_ADVERSE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(t) for t in _SORTED_TERMS) + r")(?:e?s)?(?!\w)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Negation and hedging
# --------------------------------------------------------------------------- #
NEGATION_CUES: Final[tuple[str, ...]] = (
    "no",
    "not",
    "none",
    "never",
    "without",
    "denies",
    "denied",
    "zero",
    "free of",
    "free from",
    "absent",
    "lack of",
    "didn't have",
    "did not have",
    "haven't had",
    "have not had",
    "hasn't",
    "no more",
    "went away",
    "gone away",
    "resolved",
    "cleared up",
    "stopped having",
    "nothing",
    "neither",
    "nor",
    "no longer",
    "wasn't",
    "isn't",
    "aren't",
)
_NEGATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(c) for c in NEGATION_CUES) + r")(?![\w])", re.IGNORECASE
)

#: Tokens that terminate a negation's scope. Without these, "no nausea but
#: terrible headaches" would mark the headache as negated too.
NEGATION_TERMINATORS: Final[tuple[str, ...]] = (
    "but",
    "however",
    "although",
    "though",
    "except",
    "yet",
    "still",
    ";",
    ".",
    "!",
    "?",
)

HEDGE_CUES: Final[tuple[str, ...]] = (
    "maybe",
    "might",
    "may be",
    "possibly",
    "perhaps",
    "i think",
    "i believe",
    "not sure",
    "unsure",
    "could be",
    "probably",
    "seems",
    "seemed",
    "appears",
    "hard to say",
    "who knows",
    "i guess",
    "supposedly",
    "allegedly",
)
_HEDGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(c) for c in HEDGE_CUES) + r")(?![\w])", re.IGNORECASE
)

#: Scope window, in characters, searched backwards from a matched term.
NEGATION_WINDOW: Final[int] = 48


def find_adverse_terms(text: str) -> list[tuple[str, str, int, int]]:
    """Return ``(surface, concept, start, end)`` for every adverse-event match."""
    out = []
    for m in _ADVERSE_RE.finditer(text):
        surface = m.group(1).lower()
        concept = TERM_TO_CONCEPT.get(surface)
        if concept:
            out.append((surface, concept, m.start(1), m.end(1)))
    return out


def is_negated(text: str, start: int, window: int = NEGATION_WINDOW) -> bool:
    """Whether the span beginning at ``start`` falls inside a negation scope.

    A bounded left-window scan with explicit scope terminators. It is not a
    dependency parse, and it does not need to be: on lay narrative the
    constructions are short and coordination is marked by exactly the tokens in
    ``NEGATION_TERMINATORS``. The evaluation in docs/RESULTS.md reports its
    agreement with hand annotation on a 200-span sample.
    """
    left = text[max(0, start - window) : start]
    # Truncate the window at the nearest scope terminator, so a negation cannot
    # leak across a clause boundary.
    cut = 0
    lowered = left.lower()
    for term in NEGATION_TERMINATORS:
        idx = lowered.rfind(term)
        if idx > cut:
            cut = idx + len(term)
    return bool(_NEGATION_RE.search(left[cut:]))


def is_hedged(text: str) -> bool:
    """Whether the span carries epistemic hedging or a tentative attribution."""
    return bool(_HEDGE_RE.search(text))


def extract_adverse_concepts(text: str) -> tuple[list[str], bool]:
    """Return ``(affirmed_concepts, any_negated)`` for a span.

    Negated mentions are excluded from the returned concepts but reported via
    the flag, because "no weight gain at all" is genuinely informative — it just
    must not be counted as a weight-gain report.
    """
    affirmed: list[str] = []
    saw_negation = False
    for _surface, concept, start, _end in find_adverse_terms(text):
        if is_negated(text, start):
            saw_negation = True
            continue
        if concept not in affirmed:
            affirmed.append(concept)
    return affirmed, saw_negation
