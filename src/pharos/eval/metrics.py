"""Metrics.

Three families, and the second is the one this project exists to introduce.

**Retrieval quality** — Recall@k, nDCG@k, MRR, Precision@k. Standard, and here to
establish that the system retrieves competently before any claim is made about
what it retrieves.

**Distributional fidelity** — Valence Skew Divergence, Cohort Rating Error, and
Stratum Coverage. These ask a question standard IR metrics cannot: not "did you
find relevant passages" but "is the *set* you found representative of the
population it is summarising". A retriever can score a perfect nDCG and still
hand the generator an evidence panel that misrepresents the cohort, because
relevance and representativeness are different properties and only one of them
is conventionally measured.

**Answer grounding** — Claim Support Rate, Citation Validity, Numeric Grounding
Rate, and Abstention Correctness.

Every aggregate carries a bootstrap confidence interval. A results table of bare
point estimates invites the reader to over-read a 0.01 difference, and on 300
queries a 0.01 difference is noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Retrieval quality
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / min(k, len(retrieved))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevance: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance.

    Uses the linear gain ``rel`` rather than the exponential ``2^rel - 1``. With a
    0-3 scale the exponential form makes a grade-3 unit worth seven times a
    grade-1 one, which would let a configuration that finds a single perfect unit
    outscore one that finds a well-rounded panel. Given that panel composition is
    exactly what this project is about, that weighting would bias the metric
    against the hypothesis it is meant to test.
    """
    gains = [relevance.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Distributional fidelity
# --------------------------------------------------------------------------- #
def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, safe against zero cells."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def stratum_coverage(panel_strata: Sequence[str], cohort_strata: dict[str, float]) -> float:
    """Fraction of the cohort's non-empty strata that the panel represents.

    A blunter companion to VSD, and the one that catches the failure people
    actually care about. VSD 0.31 is abstract; "the panel contained no negative
    reviews at all, and 24 % of the cohort rated it 1-4" is not.
    """
    present_in_cohort = {s for s, share in cohort_strata.items() if share > 0}
    if not present_in_cohort:
        return 1.0
    return len(set(panel_strata) & present_in_cohort) / len(present_in_cohort)


# --------------------------------------------------------------------------- #
# Answer grounding
# --------------------------------------------------------------------------- #
@dataclass
class GroundingScores:
    claim_support_rate: float
    citation_validity: float
    numeric_grounding_rate: float
    n_claims: int
    n_citations: int
    n_numeric_claims: int


def score_grounding(claims: list[dict], valid_ids: set[str]) -> GroundingScores:
    """Score a verified answer's grounding.

    ``citation_validity`` is the share of emitted citation identifiers that
    actually exist in the retrieved evidence. It is the cheapest possible check
    and it catches the most damaging failure mode there is: a fabricated
    identifier, which looks exactly like a real one to a reader and is
    unfalsifiable without the index in hand.
    """
    import re

    if not claims:
        return GroundingScores(1.0, 1.0, 1.0, 0, 0, 0)

    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    all_citations = [cid for c in claims for cid in c.get("citations", [])]
    valid_citations = [cid for cid in all_citations if cid in valid_ids]

    numeric_re = re.compile(r"\d")
    numeric_claims = [c for c in claims if numeric_re.search(c.get("text", ""))]
    numeric_grounded = [
        c for c in numeric_claims if any(cid.startswith("STAT-") for cid in c.get("citations", []))
    ]

    return GroundingScores(
        claim_support_rate=supported / len(claims),
        citation_validity=(len(valid_citations) / len(all_citations)) if all_citations else 0.0,
        numeric_grounding_rate=(
            len(numeric_grounded) / len(numeric_claims) if numeric_claims else 1.0
        ),
        n_claims=len(claims),
        n_citations=len(all_citations),
        n_numeric_claims=len(numeric_claims),
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` with a percentile bootstrap interval."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(arr.mean()), float(lo), float(hi))


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on ``mean(a) - mean(b)``.

    Paired, because both configurations are evaluated on the same queries and the
    query-to-query variance dwarfs the effect being measured. An unpaired
    interval on this data would be wide enough to swallow every result in the
    table and would say nothing.

    Returns ``(delta, lo, hi, p_two_sided)``, where the p-value is the bootstrap
    proportion of resamples whose sign disagrees with the observed delta,
    doubled.
    """
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    if arr_a.size != arr_b.size or arr_a.size == 0:
        raise ValueError("paired bootstrap requires equal-length, non-empty samples")

    diff = arr_a - arr_b
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    resampled = diff[idx].mean(axis=1)
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if observed >= 0:
        p = 2.0 * float((resampled <= 0).mean())
    else:
        p = 2.0 * float((resampled >= 0).mean())
    return (observed, float(lo), float(hi), min(1.0, p))


def format_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if any(math.isnan(v) for v in (mean, lo, hi)):
        return "n/a"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
