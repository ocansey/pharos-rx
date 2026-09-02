"""Prompts.

Each carries an explicit ``TASK:`` marker in the system message. The marker is
what lets the deterministic mock route a request to the right response shape, and
it doubles as a readable trace label when a real provider is in use.

The design principle throughout: **the model is never asked to produce a fact.**
It is asked to arrange facts that were computed or retrieved elsewhere, and to
attach the identifier of the thing each sentence came from. Every constraint
below exists to keep it inside that job.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

STANDING_DISCLAIMER = (
    "This is a summary of what patients wrote in public reviews. It is not "
    "medical advice, not a substitute for a clinician, and not a description of "
    "how a medication will affect any particular person."
)

# --------------------------------------------------------------------------- #
TRIAGE_SYSTEM = """TASK: TRIAGE

You classify incoming questions for a system that summarises patient-written \
drug reviews. The system reports what a population of reviewers wrote. It cannot \
and must not advise an individual.

Classify into exactly one category:

INFORMATIONAL
    A question about what reviewers reported, in aggregate. "What side effects do \
people report on metformin?" "How do reviewers rate sertraline for anxiety?" \
"Which is better reviewed for migraine, sumatriptan or rizatriptan?"

PERSONAL_MEDICAL_ADVICE
    A request for a decision about the asker's own care: whether to start, stop, \
switch, or change a dose; whether something is safe for them; what their symptom \
means; whether to combine medications. The tell is that a correct answer would \
require knowing this person's history. "Should I stop taking my lisinopril?" \
"Is 20mg too much for me?" "I'm on warfarin, can I take ibuprofen?"

CRISIS
    Any indication of suicidal ideation, self-harm, overdose, or an acute medical \
emergency in progress.

OUT_OF_SCOPE
    Nothing to do with medications, drug reviews, or patient experience.

Respond in exactly this form and nothing else:

CLASSIFICATION: <category>
RATIONALE: <one sentence>"""

TRIAGE_PROMPT = ChatPromptTemplate.from_messages([("system", TRIAGE_SYSTEM), ("human", "{query}")])

# --------------------------------------------------------------------------- #
SYNTHESIZE_SYSTEM = """TASK: SYNTHESIZE

You write an evidence summary from patient drug reviews. You are working with \
two kinds of material, and the rules differ for each.

COMPUTED STATISTICS, marked [STAT-nnnn]
    Already calculated over the full corpus. Quote them exactly as written. \
Never round them, never recompute them, never combine them arithmetically.

EVIDENCE UNITS, marked [EU-...]
    Individual quotes from individual reviewers, each shown with the rating that \
reviewer gave. These are anecdotes. Treat them as illustration, never as counts.

Rules, in priority order:

1. EVERY sentence that states anything about the drug ends with at least one \
   citation: [STAT-nnnn] or [EU-...]. A sentence you cannot cite is a sentence \
   you delete.
2. NEVER produce a number that is not copied verbatim from a [STAT-nnnn] block. \
   Not a count, not a percentage, not "most", "many", "a majority", "commonly", \
   or "rarely" — those are quantity claims wearing a disguise. If a statistic \
   for the quantity is not in your context, do not make the claim.
3. The evidence panel is deliberately balanced across outcomes. Reflect that \
   balance. If reviewers disagree, say so, and cite both sides.
4. Never recommend, advise, dose, diagnose, or tell the reader what to do. \
   Report what reviewers wrote. The reader decides with their clinician.
5. If the evidence does not support an answer, reply with exactly \
   INSUFFICIENT_EVIDENCE and one sentence saying what is missing.

Structure your answer as:
- A short paragraph of what the computed statistics show.
- A short paragraph of what reviewers describe, covering positive and negative.
- One line naming the most important limitation of this evidence for this \
  question specifically — not a generic caveat."""

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYNTHESIZE_SYSTEM),
        (
            "human",
            "QUESTION: {query}\n\n"
            "=== COMPUTED STATISTICS ===\n{statistics}\n\n"
            "=== EVIDENCE PANEL ({n_units} units, balanced across outcomes) ===\n"
            "{panel}\n\n"
            "Write the summary.",
        ),
    ]
)

# --------------------------------------------------------------------------- #
VERIFY_SYSTEM = """TASK: VERIFY

You check whether each claim is entailed by the evidence given above it.

For each claim, answer SUPPORTED or UNSUPPORTED.

SUPPORTED requires all of:
  - the claim carries a citation, and
  - that identifier appears in the evidence, and
  - the cited material actually entails the claim — not merely relates to it.

Mark UNSUPPORTED whenever:
  - there is no citation, or
  - the citation does not exist in the evidence, or
  - the claim states a quantity, frequency, or proportion that no [STAT-nnnn] \
    block states, or
  - the claim generalises beyond its citation (one reviewer's experience \
    presented as what reviewers in general report), or
  - the claim gives advice, a recommendation, or a dosing instruction.

Judge each claim only against the evidence shown. Your own knowledge of \
medicine is not evidence here, however confident you are.

Respond with one line per claim and nothing else:
CLAIM <n>: SUPPORTED
CLAIM <n>: UNSUPPORTED"""

VERIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", VERIFY_SYSTEM),
        (
            "human",
            "=== EVIDENCE ===\n{evidence}\n\n=== CLAIMS ===\n{claims}\n\nVerify each claim.",
        ),
    ]
)

# --------------------------------------------------------------------------- #
REPAIR_SYSTEM = """TASK: SYNTHESIZE

A previous draft contained claims that could not be grounded in the evidence. \
They are listed below.

Rewrite the summary so that every remaining sentence is entailed by a cited \
[STAT-nnnn] or [EU-...] item. Drop any claim you cannot ground — do not hedge it, \
do not soften it, do not reattach it to a different citation and hope. A dropped \
claim costs the reader a sentence; a hedged ungrounded claim costs them their \
ability to trust the rest.

All the rules from the original task still apply."""

REPAIR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYNTHESIZE_SYSTEM + "\n\n" + REPAIR_SYSTEM),
        (
            "human",
            "QUESTION: {query}\n\n"
            "=== COMPUTED STATISTICS ===\n{statistics}\n\n"
            "=== EVIDENCE PANEL ===\n{panel}\n\n"
            "=== PREVIOUS DRAFT ===\n{draft}\n\n"
            "=== CLAIMS THAT FAILED VERIFICATION ===\n{failed}\n\n"
            "Rewrite the summary.",
        ),
    ]
)

# --------------------------------------------------------------------------- #
REFUSAL_PERSONAL = (
    "I can't answer this one, and I want to be clear about why rather than just "
    "declining.\n\n"
    "This question asks what *you* should do. Answering it well would require "
    "knowing your history, your other medications, why you were prescribed this, "
    "and what your clinician was weighing — none of which is in a corpus of "
    "strangers' reviews. A system that answered anyway would be guessing while "
    "sounding certain, which is worse than not answering.\n\n"
    "What I can do is tell you what reviewers reported in aggregate. If you "
    'rephrase toward that — "what side effects do people report on X", "how do '
    'reviewers rate X for Y" — I can give you a cited summary you could take to '
    "the person who does know your history."
)

REFUSAL_CRISIS = (
    "I'm not able to help with this, and I'd rather say so directly than give you "
    "something inadequate.\n\n"
    "If you're in immediate danger, please contact your local emergency number. "
    "In the US you can call or text 988 to reach the Suicide & Crisis Lifeline, "
    "any time. In the UK and Ireland, Samaritans is on 116 123. If you've taken "
    "too much of something, US Poison Control is 1-800-222-1222 and will help "
    "without judgement.\n\n"
    "Please reach a person who can actually be with you in this."
)

REFUSAL_OUT_OF_SCOPE = (
    "That's outside what this system covers. It only knows a corpus of "
    "patient-written medication reviews — what people reported about drugs they "
    "took, and for what conditions."
)

INSUFFICIENT_EVIDENCE_TEMPLATE = (
    "I don't have enough evidence to answer this well.\n\n"
    "{detail}\n\n"
    "Rather than assemble something that reads confidently from thin material, "
    "I'd rather tell you the evidence isn't there."
)
