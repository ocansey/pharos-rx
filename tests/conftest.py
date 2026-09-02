"""Shared fixtures.

The suite runs entirely on a synthetic corpus built in memory. No network, no
model download, no dependency on the real 215,063-review dataset having been
downloaded — which is what lets the whole thing run in CI on a fresh checkout in
seconds.

The synthetic corpus is not arbitrary. It reproduces the structural properties
the system's mechanisms are designed around: a bimodal rating distribution, two
drugs sharing an indication, unequal cohort sizes, and text carrying the exact
defects the cleaner is supposed to repair.
"""

from __future__ import annotations

import random
from datetime import date

import pandas as pd
import pytest

from pharos.config import load_config
from pharos.data.cohort import CohortStatistics
from pharos.index.build import build_index
from pharos.retrieval.retriever import PharosRetriever

# Sentence pools, split by aspect and valence, so the synthetic corpus exercises
# the aspect labeller and the stratifier rather than just the plumbing.
_POSITIVE = [
    "This medication worked wonderfully for me and my symptoms cleared up within two weeks.",
    "It has been a life saver, my pain is gone and I finally sleep through the night.",
    "After three days on this drug I noticed a real improvement and I have had no side effects.",
    "Very effective for my condition, the relief started right away and has held up.",
]
_NEGATIVE = [
    "Terrible side effects, I had constant nausea and dizziness and had to stop taking it.",
    "It did not work at all for me and gave me headaches and severe stomach cramps.",
    "The weight gain was awful, I gained twenty pounds and my hair started falling out.",
    "I had to come off it after two weeks because of the insomnia and the brain fog.",
]
_MIXED = [
    "It helped somewhat but the drowsiness was hard to live with during the day.",
    "Works okay for the pain, however the cost is high and my insurance would not cover it.",
    "Mixed experience, some improvement in symptoms but I had mild nausea most mornings.",
]


def _text_for(rating: float, rng: random.Random) -> str:
    if rating >= 7:
        pool = _POSITIVE
    elif rating <= 4:
        pool = _NEGATIVE
    else:
        pool = _MIXED
    return " ".join(rng.sample(pool, k=2))


@pytest.fixture(scope="session")
def synthetic_reviews() -> pd.DataFrame:
    """A small corpus with deliberately unequal, bimodal cohorts."""
    rng = random.Random(11)
    rows = []
    review_id = 1

    # Alphamed is well received; Betacine is poorly received. Both treat the same
    # indication, which is what makes the same-condition comparator meaningful.
    spec = [
        ("Alphamed", "Chronic Pain", [10] * 22 + [9] * 14 + [6] * 4 + [2] * 8 + [1] * 4),
        ("Betacine", "Chronic Pain", [10] * 6 + [8] * 5 + [5] * 5 + [2] * 16 + [1] * 12),
        ("Alphamed", "Insomnia", [9] * 12 + [7] * 6 + [3] * 6 + [1] * 4),
        ("Gammatab", "Insomnia", [10] * 9 + [8] * 8 + [5] * 3 + [2] * 6),
        ("Deltapril", "High Blood Pressure", [9] * 15 + [7] * 7 + [4] * 5 + [1] * 5),
    ]
    for drug, condition, ratings in spec:
        for rating in ratings:
            rows.append(
                {
                    "review_id": review_id,
                    "drug_name": drug,
                    "condition": condition,
                    "text": _text_for(float(rating), rng),
                    "rating": float(rating),
                    "review_date": date(2015, 1 + (review_id % 12), 1 + (review_id % 28)),
                    "useful_count": rng.randint(0, 90),
                    "split": "train",
                    "condition_repaired": False,
                    "deidentified": False,
                    "label_status": "ok",
                }
            )
            review_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def test_config():
    """Config tuned for a small corpus and a fast test run."""
    return load_config(
        data={"sample_size": None, "dedup_enabled": False, "min_condition_support": 1},
        index={"encoder": "hashing", "embedding_dim": 128, "tfidf_min_df": 1},
        retrieval={"candidate_k": 60, "final_k": 9},
        llm={"provider": "mock"},
        eval={"n_queries": 20, "bootstrap_n": 50},
    )


@pytest.fixture(scope="session")
def synthetic_units(synthetic_reviews, test_config) -> pd.DataFrame:
    from pharos.data.schema import Stratum
    from pharos.nlp.aspects import label_aspects
    from pharos.nlp.lexicon import extract_adverse_concepts, is_hedged
    from pharos.nlp.segment import segment_review

    rows = []
    for row in synthetic_reviews.itertuples(index=False):
        for seg in segment_review(
            row.text,
            min_chars=test_config.data.min_unit_chars,
            max_chars=test_config.data.max_unit_chars,
        ):
            concepts, negated = extract_adverse_concepts(seg.text)
            rows.append(
                {
                    "unit_id": f"EU-{row.review_id}-{seg.ordinal}",
                    "review_id": int(row.review_id),
                    "ordinal": seg.ordinal,
                    "text": seg.text,
                    "aspects": [a.value for a in label_aspects(seg.text)],
                    "drug_name": row.drug_name,
                    "condition": row.condition,
                    "rating": float(row.rating),
                    "stratum": Stratum.from_rating(float(row.rating)).value,
                    "review_date": row.review_date,
                    "useful_count": int(row.useful_count),
                    "adverse_terms": concepts,
                    "negated": negated,
                    "hedged": is_hedged(seg.text),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def stats(synthetic_reviews, synthetic_units) -> CohortStatistics:
    return CohortStatistics(synthetic_reviews, synthetic_units, min_support=5, seed=0)


@pytest.fixture(scope="session")
def index(synthetic_units, test_config):
    return build_index(synthetic_units, test_config)


@pytest.fixture(scope="session")
def retriever(index, stats, test_config) -> PharosRetriever:
    return PharosRetriever(index, stats, test_config)


@pytest.fixture(scope="session")
def agent(retriever, stats, test_config):
    from pharos.agent.graph import PharosAgent

    return PharosAgent(retriever, stats, test_config)
