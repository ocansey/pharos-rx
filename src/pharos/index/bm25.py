"""Okapi BM25 over a sparse inverted index.

Written out rather than imported from ``rank_bm25`` for two reasons that matter
to this project specifically.

First, ``rank_bm25`` scores every document in the collection on every query — a
dense pass over 70,000 units per query, which makes the 300-query evaluation
sweep take minutes instead of seconds. The implementation here walks only the
postings lists of the query terms, which is what an inverted index is *for*.

Second, the retriever needs metadata-constrained lexical search: "BM25 over units
belonging to this drug, for this indication". A candidate-restricted scoring pass
is trivial against an inverted index and impossible against a library that owns
its own corpus array. Filtering *after* scoring — the usual workaround — silently
returns fewer than k results, and that failure is invisible in an evaluation that
only measures what came back.

Scores use the standard Robertson/Sparck-Jones IDF with the ``+1`` guard, so a
term appearing in more than half the collection cannot contribute negatively.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Removing these costs nothing on recall and materially shortens the postings
#: lists that dominate query latency. Deliberately *not* a full English stoplist:
#: "no", "not" and "off" carry clinical meaning here and are kept.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenisation with a clinically-aware stoplist."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


@dataclass
class BM25Index:
    """An inverted index with BM25 scoring.

    Attributes:
        k1: Term-frequency saturation. 1.4 is slightly above the usual 1.2
            because review text repeats symptom words within a single unit.
        b: Length normalisation. 0.72 rather than 0.75 because evidence units are
            already length-bounded by segmentation, so less correction is needed.
    """

    k1: float = 1.4
    b: float = 0.72

    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    doc_len: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    avg_doc_len: float = 0.0
    n_docs: int = 0
    idf: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def fit(self, documents: list[str]) -> BM25Index:
        """Build the inverted index."""
        self.n_docs = len(documents)
        lengths = np.zeros(self.n_docs, dtype=np.int32)
        postings: dict[str, list[tuple[int, int]]] = {}

        for doc_id, text in enumerate(documents):
            tokens = tokenize(text)
            lengths[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_id, tf))

        self.postings = postings
        self.doc_len = lengths
        self.avg_doc_len = float(lengths.mean()) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }
        return self

    # ------------------------------------------------------------------ #
    def score(
        self, query: str, candidates: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score documents against a query.

        Args:
            query: Free text.
            candidates: Optional document ids to restrict scoring to. Passing a
                candidate set is how metadata-filtered lexical search stays
                exact — the filter is applied *during* scoring, so the top-k is
                the true top-k of the filtered set.

        Returns:
            ``(doc_ids, scores)`` sorted by descending score, containing only
            documents with a non-zero score.
        """
        terms = tokenize(query)
        if not terms or self.n_docs == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        allowed: set[int] | None = set(candidates.tolist()) if candidates is not None else None
        accum: dict[int, float] = {}

        for term, q_tf in Counter(terms).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            # A query term repeated by the user is weighted, but sub-linearly —
            # "pain pain pain" is emphasis, not three independent requirements.
            q_weight = 1.0 + math.log(q_tf)
            for doc_id, tf in plist:
                if allowed is not None and doc_id not in allowed:
                    continue
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_doc_len or 1.0))
                accum[doc_id] = (
                    accum.get(doc_id, 0.0) + q_weight * idf * (tf * (self.k1 + 1)) / denom
                )

        if not accum:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        doc_ids = np.fromiter(accum.keys(), dtype=np.int64, count=len(accum))
        scores = np.fromiter(accum.values(), dtype=np.float32, count=len(accum))
        order = np.argsort(-scores, kind="stable")
        return doc_ids[order], scores[order]

    # ------------------------------------------------------------------ #
    def top_k(
        self, query: str, k: int, candidates: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        doc_ids, scores = self.score(query, candidates)
        return list(zip(doc_ids[:k].tolist(), scores[:k].tolist(), strict=False))

    def vocabulary_size(self) -> int:
        return len(self.postings)
