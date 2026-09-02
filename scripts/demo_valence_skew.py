#!/usr/bin/env python3
"""Demonstrate retrieval-induced valence skew, and its correction.

Runs the same query through a conventional dense top-k retriever and through
PHAROS, and prints both panels beside the cohort's true outcome distribution.

This is the single clearest way to see what the project is about, and it is the
figure worth putting in a talk or a post.

    python scripts/demo_valence_skew.py
    python scripts/demo_valence_skew.py --drug Sertraline --condition Depression
"""

from __future__ import annotations

import argparse
import sys

from pharos.config import load_config
from pharos.data.cohort import CohortStatistics
from pharos.index.build import load_corpus, load_index
from pharos.retrieval.retriever import PharosRetriever


def pick_cohort(stats: CohortStatistics, min_reviews: int = 40) -> tuple[str, str]:
    """Choose a cohort large enough for the effect to be visible.

    Preference goes to a cohort with genuine disagreement among reviewers: the
    distortion is invisible where everyone agrees, because any panel drawn from a
    unanimous cohort is representative by default.
    """
    grouped = stats.reviews.groupby(["drug_name", "condition"])["rating"]
    counts, means = grouped.count(), grouped.mean()
    eligible = counts[counts >= min_reviews].index
    if len(eligible) == 0:
        raise SystemExit(f"no cohort has {min_reviews}+ reviews in this build")
    # Closest to a 5/10 mean = maximum disagreement.
    best = min(eligible, key=lambda k: abs(means[k] - 5.5))
    return str(best[0]), str(best[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug")
    parser.add_argument("--condition")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        reviews, units, _ = load_corpus(cfg)
        index = load_index(cfg)
    except FileNotFoundError as exc:
        print(f"{exc}\n\nRun `make build` first.", file=sys.stderr)
        return 1

    stats = CohortStatistics(reviews, units, seed=cfg.data.seed)
    drug, condition = (
        (args.drug, args.condition)
        if args.drug and args.condition
        else pick_cohort(stats)
    )

    pharos = PharosRetriever(index, stats, cfg)
    naive = PharosRetriever(
        index,
        stats,
        load_config(retrieval={"stratify": False, "rerank": "none", "mode": "dense"}),
    )

    query = f"What side effects do people report on {drug} for {condition}?"
    a = naive.retrieve(query, k=args.k, drug_name=drug, condition=condition)
    b = pharos.retrieve(query, k=args.k, drug_name=drug, condition=condition)

    cohort = stats.cohort_frame(drug, condition)
    truth = {s.value: v for s, v in b.cohort_distribution.items()}

    print(f"\nQUESTION: {query}\n")
    print(f"THE COHORT ({len(cohort)} reviews) — ground truth")
    print("   " + "   ".join(f"{k} {v * 100:5.1f}%" for k, v in truth.items()))
    print(f"   mean rating {b.cohort_mean_rating:.2f}/10\n")

    for label, result in (("CONVENTIONAL top-k", a), ("STRATIFIED (PHAROS)", b)):
        ratings = sorted(int(u.unit.rating) for u in result.units)
        mean = sum(ratings) / len(ratings)
        print(label)
        print(f"   panel ratings: {ratings}")
        print(
            f"   panel mean {mean:.2f}/10"
            f"     VSD {result.valence_skew:.4f}"
            f"     off by {result.rating_error:.2f} stars"
        )

    improvement = (1 - b.valence_skew / a.valence_skew) * 100 if a.valence_skew else 0.0
    print(f"\n   → stratification reduces valence skew by {improvement:.0f}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
