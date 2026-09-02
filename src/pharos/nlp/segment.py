"""Evidence-unit segmentation.

Chunking is usually treated as a plumbing decision — pick 512 tokens, pick a
100-token overlap, move on. On this corpus that is wrong, and measurably so.

A drug review is not prose with a topic; it is a *sequence of distinct claims*
compressed into four sentences: "Worked great for my anxiety within a week, but
the weight gain was awful and I had to come off it." Indexed whole, that review
is retrieved for "does it work?" and for "does it cause weight gain?" with the
same vector, and whichever question you asked, the generator sees both answers
mixed together. Split at clause boundaries, and the efficacy claim and the
adverse-effect claim become separately retrievable, separately citable, and
separately countable.

Segmentation is therefore contrastive: we split on sentence boundaries *and* on
discourse connectives that mark a change of polarity or topic ("but", "however",
"although", "on the other hand"). Fragments shorter than the configured minimum
are merged back into their neighbour, so the split never manufactures a citation
too small to stand alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentence boundary: terminal punctuation followed by whitespace and a capital
# or digit. The negative lookbehind protects common abbreviations and the
# decimal points in dosages ("2.5 mg"), which are frequent here.
_ABBREV = r"(?<!\bmg)(?<!\bmcg)(?<!\bml)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bvs)(?<!\bapprox)(?<!\bi\.e)(?<!\be\.g)"
_SENTENCE_BOUNDARY = re.compile(rf"{_ABBREV}(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9])")

# Discourse connectives that reliably mark a polarity or topic shift in lay
# review prose. Splitting *before* the connective keeps it with the clause it
# governs, which matters because "but" is the single strongest cue in the corpus
# that an adverse effect is about to be described.
_CONTRAST_CONNECTIVES = (
    "but",
    "however",
    "although",
    "though",
    "whereas",
    "on the other hand",
    "that said",
    "downside",
    "only problem",
    "only issue",
    "the one thing",
    "unfortunately",
    "sadly",
    "except",
)
_CONTRAST_BOUNDARY = re.compile(
    r"(?<=[\w,;)])\s+(?=(?:" + "|".join(re.escape(c) for c in _CONTRAST_CONNECTIVES) + r")\b)",
    re.IGNORECASE,
)

# A trailing "and then ..." enumeration is also a claim boundary when it is long
# enough to stand alone; short "and" coordination is left intact.
_ENUM_BOUNDARY = re.compile(r"(?<=[\w)])\s*(?:;|\s+and then\s+|\s+also\s+)(?=[a-z])", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    """A candidate evidence unit with its offsets in the parent review."""

    text: str
    start: int
    end: int
    ordinal: int


def _split_keep_offsets(text: str, pattern: re.Pattern[str]) -> list[tuple[str, int, int]]:
    """Split on ``pattern`` while tracking character offsets into ``text``."""
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    for m in pattern.finditer(text):
        if m.start() > cursor:
            pieces.append((text[cursor : m.start()], cursor, m.start()))
        cursor = m.end()
    if cursor < len(text):
        pieces.append((text[cursor:], cursor, len(text)))
    return pieces or [(text, 0, len(text))]


def _merge_short(pieces: list[tuple[str, int, int]], min_chars: int) -> list[tuple[str, int, int]]:
    """Fold undersized fragments into their neighbour.

    Forward-merge first (a fragment usually continues into what follows); fall
    back to a backward merge for a trailing fragment. The result is that no
    emitted unit is shorter than ``min_chars`` unless the entire review is.
    """
    if not pieces:
        return []
    out: list[list] = []
    for text, start, end in pieces:
        stripped = text.strip()
        if not stripped:
            continue
        if out and len(stripped) < min_chars:
            out[-1][0] = (out[-1][0] + " " + stripped).strip()
            out[-1][2] = end
        else:
            out.append([stripped, start, end])
    # A leading fragment can still be short if it was the first piece.
    if len(out) > 1 and len(out[0][0]) < min_chars:
        out[1][0] = (out[0][0] + " " + out[1][0]).strip()
        out[1][1] = out[0][1]
        out.pop(0)
    return [(t, s, e) for t, s, e in out]


def _split_oversized(
    pieces: list[tuple[str, int, int]], max_chars: int
) -> list[tuple[str, int, int]]:
    """Break any remaining over-long piece at the nearest whitespace.

    Reviews run to 10,787 characters at the tail. An unbounded unit would
    dominate a lexical index by sheer length and would blow the generator's
    context budget with a single citation.
    """
    out: list[tuple[str, int, int]] = []
    for text, start, end in pieces:
        while len(text) > max_chars:
            cut = text.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append((text[:cut].strip(), start, start + cut))
            text = text[cut:].lstrip()
            start = start + cut
        if text.strip():
            out.append((text.strip(), start, end))
    return out


def segment_review(text: str, min_chars: int = 25, max_chars: int = 420) -> list[Segment]:
    """Split one review into evidence units.

    Args:
        text: A cleaned review body.
        min_chars: Fragments below this are merged into a neighbour.
        max_chars: Units above this are split at whitespace.

    Returns:
        Ordered segments with offsets into ``text``. Always at least one segment
        for non-empty input, so no review silently disappears from the index.
    """
    text = text.strip()
    if not text:
        return []

    pieces = _split_keep_offsets(text, _SENTENCE_BOUNDARY)

    expanded: list[tuple[str, int, int]] = []
    for piece, start, _end in pieces:
        for sub, s_off, _e_off in _split_keep_offsets(piece, _CONTRAST_BOUNDARY):
            for sub2, s2, e2 in _split_keep_offsets(sub, _ENUM_BOUNDARY):
                expanded.append((sub2, start + s_off + s2, start + s_off + e2))

    merged = _merge_short(expanded, min_chars)
    final = _split_oversized(merged, max_chars)
    return [Segment(t, s, e, i) for i, (t, s, e) in enumerate(final)]
