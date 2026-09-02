"""Rank fusion and redundancy control.

Reciprocal Rank Fusion is used rather than score interpolation, and the reason is
concrete here: BM25 scores are unbounded sums over query terms while cosine
similarities live in [-1, 1]. Any weighted sum of the two is really a weighted
sum of one arm and the *noise* of the other, and the weight that works for a
three-word query is wrong for a twenty-word one. RRF discards magnitudes and
combines ranks, so it is invariant to both scales — which is what lets one
``lexical_weight`` setting hold across the whole query distribution.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[int, float]]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse ranked lists by weighted reciprocal rank.

    Args:
        ranked_lists: Arm name -> ``[(doc_id, score), ...]`` in rank order.
        weights: Arm name -> weight. Missing arms default to 1.0.
        k: Damping constant. The standard 60 flattens the head enough that a
            document ranked 1 by one arm and 30 by the other beats a document
            ranked 5 by both — which is the behaviour you want when the two arms
            disagree because they are seeing different things, and lexical and
            dense retrieval over clinical narrative disagree constantly.

    Returns:
        ``[(doc_id, fused_score, {arm: rank})]``, descending. The per-arm ranks
        travel with the result so that the provenance of every citation survives
        into the answer transcript.
    """
    weights = weights or {}
    fused: dict[int, float] = defaultdict(float)
    provenance: dict[int, dict[str, int]] = defaultdict(dict)

    for arm, results in ranked_lists.items():
        w = weights.get(arm, 1.0)
        for rank, (doc_id, _score) in enumerate(results, start=1):
            fused[doc_id] += w / (k + rank)
            provenance[doc_id][arm] = rank

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc_id, score, provenance[doc_id]) for doc_id, score in ordered]


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_ids: list[int],
    k: int,
    lambda_mult: float = 0.65,
) -> list[int]:
    """Greedy MMR re-ranking for redundancy control.

    This corpus is unusually redundant: hundreds of reviews of the same
    contraceptive say almost the same sentence about the same side effect. A
    plain top-k returns twelve paraphrases of one claim and the generator, seeing
    twelve of them, reports it as overwhelming consensus. MMR trades a little
    relevance for coverage, which here is a trade for *accuracy*, not just for
    variety.

    Args:
        lambda_mult: 1.0 is pure relevance, 0.0 pure diversity. 0.65 was chosen
            by sweep on the development split; see docs/RESULTS.md §5.3.
    """
    if candidate_vectors.shape[0] == 0 or k <= 0:
        return []
    k = min(k, candidate_vectors.shape[0])

    relevance = candidate_vectors @ query_vector
    selected: list[int] = [int(np.argmax(relevance))]
    remaining = set(range(candidate_vectors.shape[0])) - set(selected)

    while len(selected) < k and remaining:
        rem = np.fromiter(remaining, dtype=np.int64, count=len(remaining))
        sim_to_selected = candidate_vectors[rem] @ candidate_vectors[selected].T
        redundancy = sim_to_selected.max(axis=1)
        mmr = lambda_mult * relevance[rem] - (1 - lambda_mult) * redundancy
        chosen = int(rem[int(np.argmax(mmr))])
        selected.append(chosen)
        remaining.discard(chosen)

    return [candidate_ids[i] for i in selected]
