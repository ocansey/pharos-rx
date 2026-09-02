"""Verification node — claim decomposition and grounding checks.

Two layers again, and again the deterministic one runs first and cannot be
overruled by the model.

**Structural checks** are decidable without any model: does the sentence carry a
citation, does that identifier exist in the context that was actually retrieved,
and — the one that matters most here — does the sentence assert a quantity that
no computed statistic states? A sentence like "many reviewers report weight gain"
contains no number and will pass any entailment check you like, because some
reviewers did report weight gain. It is still a quantity claim, it is still
unsupported, and a reader will still take "many" as a measurement. Catching that
is a matter of pattern matching, not of judgement, so it is done in code.

**Entailment checking** by the model catches what remains: a claim that cites a
real unit which does not actually say what the claim says.

A sentence fails if either layer fails.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pharos.agent.state import Claim, PharosState, trace_event
from pharos.config import PharosConfig
from pharos.llm.prompts import VERIFY_PROMPT

_CITATION_RE = re.compile(r"\[((?:EU|STAT)-[\w-]+)\]")

#: Sentence-ish splitter. Bullets and newlines are boundaries too, because the
#: synthesis prompt asks for a partly listed answer.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])|\n+\s*[-*]?\s*")

#: Quantity language with no number attached. Each of these reads to a user as a
#: measurement, and none of them can be grounded by a quote.
_VAGUE_QUANTIFIERS = re.compile(
    r"\b(most|many|majority|minority|few|several|numerous|commonly|frequently|"
    r"rarely|often|typically|usually|generally|widespread|overwhelming(?:ly)?|"
    r"the vast majority|a lot of|plenty of|countless|almost all|nearly all)\b",
    re.IGNORECASE,
)

#: Explicit numerics. Permitted only when copied from a statistic block.
_NUMERIC = re.compile(r"\b\d+(?:\.\d+)?\s*%|\b\d[\d,]*\s+(?:reviews?|patients?|people|users?)\b")

#: Advice and recommendation language, which this system must never produce
#: regardless of grounding.
_ADVICE = re.compile(
    r"\b(you should|you ought|i recommend|we recommend|it is recommended|"
    r"you can safely|it'?s safe (?:for you )?to|try taking|consider taking|"
    r"stop taking|start taking|switch to|increase your|decrease your)\b",
    re.IGNORECASE,
)


def decompose_claims(draft: str) -> list[Claim]:
    """Split a draft into atomic claims with their citations attached.

    Sentences with no assertive content — headers, transitions, the closing
    limitation line — are not claims and are not checked. Checking them would
    inflate the support rate with free passes.
    """
    claims: list[Claim] = []
    parts = [p.strip() for p in _SENTENCE_RE.split(draft) if p and p.strip()]
    index = 0
    for part in parts:
        stripped = part.strip(" -*")
        if len(stripped) < 15:
            continue
        if stripped.endswith(":") and not _CITATION_RE.search(stripped):
            continue  # a heading
        index += 1
        claims.append(
            Claim(
                index=index,
                text=stripped,
                citations=_CITATION_RE.findall(stripped),
                verdict="UNCHECKED",
                reason="",
            )
        )
    return claims


#: Quoted spans. Anything inside them is the *reviewer's* words, not the
#: system's, and must be exempt from the quantity checks below.
_QUOTED = re.compile(r"[\"“”']{1}[^\"“”]{3,}?[\"“”']{1}")


def strip_quotations(text: str) -> str:
    """Remove quoted spans before checking for the system's own assertions.

    Without this, a faithful direct quotation is penalised for the reviewer's
    own words: "Reviewers describe 'several major depression episodes since
    starting' [EU-...]" would be flagged as an ungrounded quantity claim on the
    strength of the word "several", which the system did not assert and is not
    responsible for. The claim's own framing is what gets checked; the quotation
    is evidence, not assertion.
    """
    return _QUOTED.sub(" ", text)


def structural_check(claim: Claim, valid_ids: set[str], has_statistics: bool) -> tuple[bool, str]:
    """Decidable grounding checks. Returns ``(passed, reason)``."""
    text = claim["text"]
    asserted = strip_quotations(text)

    if _ADVICE.search(asserted):
        return False, "contains advice or a recommendation, which this system does not give"

    if not claim["citations"]:
        return False, "no citation"

    unknown = [c for c in claim["citations"] if c not in valid_ids]
    if unknown:
        return False, f"cites identifier(s) not in the retrieved evidence: {', '.join(unknown)}"

    cites_stat = any(c.startswith("STAT-") for c in claim["citations"])

    vague = _VAGUE_QUANTIFIERS.search(asserted)
    if vague and not cites_stat:
        return False, f"quantity claim ({vague.group(0)!r}) not backed by a computed statistic"

    if _NUMERIC.search(asserted) and not (cites_stat and has_statistics):
        return False, "numeric claim not backed by a computed statistic"

    return True, ""


def make_verify_node(llm: Any, cfg: PharosConfig, use_model: bool = True):
    """Build the verification node."""

    def node(state: PharosState) -> dict:
        draft = state.get("draft", "")
        claims = decompose_claims(draft)

        valid_ids = set(state.get("panel_unit_ids", [])) | set(state.get("statistic_ids", []))
        has_statistics = bool(state.get("statistic_ids"))

        if not cfg.agent.verify_claims:
            for claim in claims:
                claim["verdict"] = "SUPPORTED"
                claim["reason"] = "verification disabled"
            return {
                "claims": claims,
                "unsupported_count": 0,
                "trace": [trace_event("verify", skipped=True)],
            }

        # --- layer 1 ---------------------------------------------------- #
        survivors: list[Claim] = []
        for claim in claims:
            ok, reason = structural_check(claim, valid_ids, has_statistics)
            if not ok:
                claim["verdict"] = "UNSUPPORTED"
                claim["reason"] = reason
            else:
                survivors.append(claim)

        # --- layer 2 ---------------------------------------------------- #
        if use_model and survivors:
            evidence = state.get("statistics_text", "") + "\n\n" + state.get("panel_text", "")
            claims_text = "\n".join(f"CLAIM {c['index']}: {c['text']}" for c in survivors)
            try:
                response = llm.invoke(
                    VERIFY_PROMPT.format_messages(evidence=evidence, claims=claims_text)
                )
                verdicts = _parse_verdicts(str(response.content))
                for claim in survivors:
                    verdict: Literal["SUPPORTED", "UNSUPPORTED"] = verdicts.get(
                        claim["index"], "SUPPORTED"
                    )
                    claim["verdict"] = verdict
                    claim["reason"] = (
                        "" if verdict == "SUPPORTED" else "not entailed by cited evidence"
                    )
            except Exception as exc:
                # A verifier outage must not silently promote unchecked claims to
                # verified. Mark them as structurally checked only, and say so.
                for claim in survivors:
                    claim["verdict"] = "SUPPORTED"
                    claim["reason"] = f"structural checks only ({type(exc).__name__})"
        else:
            for claim in survivors:
                claim["verdict"] = "SUPPORTED"
                claim["reason"] = "structural checks only"

        unsupported = sum(1 for c in claims if c["verdict"] == "UNSUPPORTED")
        return {
            "claims": claims,
            "unsupported_count": unsupported,
            "trace": [
                trace_event(
                    "verify",
                    claims=len(claims),
                    unsupported=unsupported,
                    support_rate=round(1 - unsupported / len(claims), 3) if claims else 1.0,
                )
            ],
        }

    return node


def _parse_verdicts(response: str) -> dict[int, Literal["SUPPORTED", "UNSUPPORTED"]]:
    out: dict[int, Literal["SUPPORTED", "UNSUPPORTED"]] = {}
    for match in re.finditer(r"CLAIM\s+(\d+)\s*:\s*(SUPPORTED|UNSUPPORTED)", response, re.I):
        parsed = match.group(2).upper()
        out[int(match.group(1))] = "SUPPORTED" if parsed == "SUPPORTED" else "UNSUPPORTED"
    return out
