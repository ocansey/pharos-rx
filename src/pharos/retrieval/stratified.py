"""Stratified Evidence Sampling — the core contribution.

The problem
-----------
Retrieval-augmented generation over an opinion corpus inherits a bias that has
nothing to do with retrieval quality. Similarity search returns the passages most
similar to the query; in a corpus of patient narratives, the passages most
similar to "does sertraline help with anxiety" are the ones that talk about
sertraline and anxiety *most emphatically*. Emphasis correlates with extremity,
extremity correlates with outcome, and so the panel of evidence handed to the
generator is systematically skewed relative to the population it is supposed to
summarise. The generator then reports the skew faithfully. Every citation is
real, every sentence is grounded, and the answer is still wrong about the world.

We call this **retrieval-induced valence skew**, and it is measurable: take the
rating distribution of the retrieved panel and compare it against the rating
distribution of the cohort the query is about. On this corpus, conventional
top-k retrieval produces panels whose outcome distribution diverges sharply from
the cohort's — the measurement is in ``docs/RESULTS.md`` §5.1.

The mechanism
-------------
Treat retrieval as *sampling from a population*, not as ranked lookup.

1. Compute the cohort's true outcome distribution over the whole corpus — not
   over the retrieved candidates, which would already carry the bias.
2. Allocate the ``k`` panel slots across outcome strata to match it, using
   largest-remainder apportionment so the allocation is exact and stable rather
   than drifting with floating-point rounding.
3. Guarantee every non-empty stratum at least ``min_slots_per_stratum`` slots, so
   a dissenting minority is never rounded out of existence.
4. Fill each stratum's quota with its own highest-ranked candidates, capping the
   units any single review may contribute so one voluble narrator cannot
   colonise the panel.
5. Reallocate the quota of any stratum that cannot be filled to the strata that
   can, so the panel is always exactly ``k`` when ``k`` units exist.

What this is and is not
-----------------------
This is not a debiasing claim about the *corpus*. The underlying reviews are
self-selected and no sampling scheme fixes that; a drug whose reviewers are
disproportionately unhappy will still look that way, correctly. What the
allocator removes is the *additional* distortion that retrieval imposes on top —
the gap between "what the corpus says" and "what the retriever showed the model".
That gap is entirely an artifact of the system, and so it is the system's to
close.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pharos.data.schema import EvidenceUnit, Stratum


@dataclass
class Allocation:
    """A quota allocation, with enough detail to audit after the fact."""

    quotas: dict[Stratum, int]
    target: dict[Stratum, float]
    achieved: dict[Stratum, float] = field(default_factory=dict)
    reallocated: dict[Stratum, int] = field(default_factory=dict)
    total: int = 0

    def shortfall(self) -> float:
        """Total variation distance between target and achieved shares."""
        if not self.achieved:
            return 0.0
        return 0.5 * sum(
            abs(self.target.get(s, 0.0) - self.achieved.get(s, 0.0)) for s in Stratum.ordered()
        )


def largest_remainder_apportionment(
    proportions: dict[Stratum, float], total: int, min_per_nonzero: int = 1
) -> dict[Stratum, int]:
    """Apportion ``total`` slots to strata in proportion to ``proportions``.

    Largest-remainder (Hamilton) apportionment rather than naive rounding.
    Rounding each share independently does not sum to ``total`` — with three
    strata at 0.34/0.33/0.33 and k=12 it gives 4/4/4 by luck and 5/4/4 or 4/4/3
    depending on floating-point noise, which makes panel composition
    irreproducible across machines. Hamilton's method allocates the integer
    floors, then hands the remaining seats to the largest fractional remainders,
    and always sums exactly.

    The floor guarantee is applied first, then the remaining seats are
    apportioned. If the floor alone exceeds the budget, seats are taken back from
    the largest strata, which are the ones that can spare them.
    """
    strata = [s for s in Stratum.ordered()]
    nonzero = [s for s in strata if proportions.get(s, 0.0) > 0]
    if total <= 0 or not nonzero:
        return {s: 0 for s in strata}

    floor_alloc = {s: (min_per_nonzero if s in nonzero else 0) for s in strata}
    reserved = sum(floor_alloc.values())

    if reserved >= total:
        # Not enough seats to honour the floor everywhere. Give seats to the
        # strata with the most cohort mass, largest first.
        alloc = {s: 0 for s in strata}
        for s in sorted(nonzero, key=lambda x: -proportions.get(x, 0.0))[:total]:
            alloc[s] = 1
        return alloc

    remaining = total - reserved
    mass = sum(proportions.get(s, 0.0) for s in nonzero)
    exact = {s: remaining * proportions.get(s, 0.0) / mass for s in nonzero}
    alloc = dict(floor_alloc)
    for s in nonzero:
        alloc[s] += int(exact[s])

    seats_left = total - sum(alloc.values())
    if seats_left > 0:
        remainders = sorted(nonzero, key=lambda s: (-(exact[s] - int(exact[s])), s.value))
        for s in remainders[:seats_left]:
            alloc[s] += 1
    return alloc


class StratifiedSampler:
    """Fills a panel to match a reference outcome distribution."""

    def __init__(
        self,
        min_slots_per_stratum: int = 1,
        max_units_per_review: int = 2,
    ) -> None:
        self.min_slots_per_stratum = min_slots_per_stratum
        self.max_units_per_review = max_units_per_review

    # ------------------------------------------------------------------ #
    def allocate(self, target_distribution: dict[Stratum, float], k: int) -> Allocation:
        quotas = largest_remainder_apportionment(
            target_distribution, k, min_per_nonzero=self.min_slots_per_stratum
        )
        return Allocation(quotas=quotas, target=dict(target_distribution), total=k)

    # ------------------------------------------------------------------ #
    def select(
        self,
        ranked: list[tuple[int, float]],
        units: list[EvidenceUnit],
        target_distribution: dict[Stratum, float],
        k: int,
    ) -> tuple[list[tuple[int, float, Stratum]], Allocation]:
        """Choose ``k`` units from a ranked candidate list to fill the quotas.

        Args:
            ranked: ``[(unit_position, fused_score)]`` in descending rank order.
            units: The index's unit list, positionally aligned with ``ranked``'s
                first element.
            target_distribution: The cohort's true outcome distribution.
            k: Panel size.

        Returns:
            ``(selected, allocation)`` where ``selected`` is
            ``[(unit_position, score, admitted_under_stratum)]`` and
            ``allocation`` records what was asked for and what was achieved.
        """
        allocation = self.allocate(target_distribution, k)
        quotas = dict(allocation.quotas)

        by_stratum: dict[Stratum, list[tuple[int, float]]] = defaultdict(list)
        for pos, score in ranked:
            by_stratum[units[pos].stratum].append((pos, score))

        selected: list[tuple[int, float, Stratum]] = []
        per_review: dict[int, int] = defaultdict(int)

        def take(stratum: Stratum, budget: int) -> int:
            """Fill up to ``budget`` slots from one stratum; return slots used."""
            used = 0
            for pos, score in by_stratum[stratum]:
                if used >= budget:
                    break
                review_id = units[pos].review_id
                if per_review[review_id] >= self.max_units_per_review:
                    continue
                if any(p == pos for p, _, _ in selected):
                    continue
                selected.append((pos, score, stratum))
                per_review[review_id] += 1
                used += 1
            return used

        # Pass 1 — honour each stratum's quota.
        unfilled = 0
        for stratum in Stratum.ordered():
            budget = quotas.get(stratum, 0)
            if budget <= 0:
                continue
            used = take(stratum, budget)
            unfilled += budget - used

        # Pass 2 — a stratum with no candidates gives its seats to those that
        # have them, so a thin cohort still yields a full panel.
        if unfilled > 0:
            order = sorted(Stratum.ordered(), key=lambda s: -target_distribution.get(s, 0.0))
            for stratum in order:
                if unfilled <= 0:
                    break
                used = take(stratum, unfilled)
                allocation.reallocated[stratum] = allocation.reallocated.get(stratum, 0) + used
                unfilled -= used

        # Pass 3 — the per-review cap can still leave the panel short on a very
        # small cohort. Relax it last rather than return fewer than k.
        if len(selected) < k:
            chosen = {p for p, _, _ in selected}
            for pos, score in ranked:
                if len(selected) >= k:
                    break
                if pos in chosen:
                    continue
                selected.append((pos, score, units[pos].stratum))
                chosen.add(pos)

        selected.sort(key=lambda t: -t[1])

        if selected:
            counts: dict[Stratum, int] = defaultdict(int)
            for pos, _, _ in selected:
                counts[units[pos].stratum] += 1
            allocation.achieved = {s: counts.get(s, 0) / len(selected) for s in Stratum.ordered()}
        return selected, allocation


# --------------------------------------------------------------------------- #
# Skew measurement
# --------------------------------------------------------------------------- #
def valence_skew_divergence(
    panel_ratings: list[float], cohort_distribution: dict[Stratum, float]
) -> float:
    """Jensen–Shannon divergence between a panel and its cohort, in bits.

    The headline metric for the contribution. Jensen–Shannon rather than
    Kullback–Leibler because it is symmetric, bounded in [0, 1] with base-2 logs,
    and — decisively — finite when a stratum is empty in one distribution but not
    the other, which is precisely the failure case being measured. KL would
    return infinity for the very panels that are most badly skewed, making the
    metric useless exactly where it matters.

    0.0 means the panel's outcome distribution matches the cohort's. 1.0 means
    they share no support at all.
    """
    if not panel_ratings:
        return 0.0

    counts = dict.fromkeys(Stratum.ordered(), 0)
    for rating in panel_ratings:
        counts[Stratum.from_rating(float(rating))] += 1
    n = len(panel_ratings)

    p = np.array([counts[s] / n for s in Stratum.ordered()], dtype=float)
    q = np.array([cohort_distribution.get(s, 0.0) for s in Stratum.ordered()], dtype=float)
    q_sum = q.sum()
    if q_sum <= 0:
        return 0.0
    q = q / q_sum

    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def cohort_rating_error(panel_ratings: list[float], cohort_mean: float) -> float:
    """Absolute error between the panel's mean rating and the cohort's.

    A second, more interpretable view of the same distortion, on the 1–10 scale
    a reader already understands. VSD says the shape is wrong; this says by how
    many stars a naive reader of the panel would be misled.
    """
    if not panel_ratings:
        return 0.0
    return abs(float(np.mean(panel_ratings)) - cohort_mean)
