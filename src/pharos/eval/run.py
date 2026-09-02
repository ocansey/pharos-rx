"""The evaluation runner.

Produces every number in ``docs/RESULTS.md``. Three suites:

* :func:`evaluate_retrieval` — retrieval quality and distributional fidelity over
  the derived gold set, for one configuration.
* :func:`run_ablations` — the same suite across a set of one-knob-at-a-time
  configurations, with paired bootstrap deltas against the full system.
* :func:`evaluate_safety` — the red-team suite through the full graph.

The retrieval and fidelity suites require no language model, which is what lets
the headline table regenerate in CI on every push. If a claim in the README
cannot survive being recomputed automatically, it should not be in the README.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pharos.config import PharosConfig, load_config
from pharos.data.cohort import CohortStatistics
from pharos.data.schema import Stratum
from pharos.eval.goldset import GoldQuery
from pharos.eval.metrics import (
    bootstrap_ci,
    format_ci,
    ndcg_at_k,
    paired_bootstrap_delta,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    stratum_coverage,
)
from pharos.eval.redteam import PROBES, verdict_matches
from pharos.index.store import CorpusIndex
from pharos.retrieval.retriever import PharosRetriever
from pharos.retrieval.stratified import cohort_rating_error, valence_skew_divergence


# --------------------------------------------------------------------------- #
@dataclass
class SuiteResult:
    """Aggregated metrics for one configuration."""

    name: str
    description: str
    n_queries: int
    metrics: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    per_query: dict[str, list[float]] = field(default_factory=dict)
    seconds: float = 0.0
    config_fingerprint: str = ""

    def row(self, keys: list[str], digits: int = 3) -> list[str]:
        return [
            format_ci(*self.metrics[k], digits=digits) if k in self.metrics else "—" for k in keys
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "n_queries": self.n_queries,
            "seconds": round(self.seconds, 2),
            "config_fingerprint": self.config_fingerprint,
            "metrics": {k: [round(x, 5) for x in v] for k, v in self.metrics.items()},
        }


# --------------------------------------------------------------------------- #
def evaluate_retrieval(
    retriever: PharosRetriever,
    goldset: list[GoldQuery],
    cfg: PharosConfig,
    name: str,
    description: str = "",
    k_values: list[int] | None = None,
    stratify: bool | None = None,
    mode: str | None = None,
) -> SuiteResult:
    """Run one configuration over the gold set."""
    k_values = k_values or cfg.eval.k_values
    t0 = time.perf_counter()

    per_query: dict[str, list[float]] = {}

    def push(key: str, value: float) -> None:
        per_query.setdefault(key, []).append(value)

    max_k = max([*k_values, cfg.retrieval.final_k])
    final_k = cfg.retrieval.final_k

    for gq in goldset:
        # Two retrievals per query, and the separation is not incidental.
        #
        # IR metrics need a deep ranking, so they are computed at max_k.
        # Distributional fidelity must be measured on the panel the generator
        # would actually receive, at final_k. Slicing the first final_k rows out
        # of a max_k retrieval does not give that panel: the quota allocator
        # apportioned max_k slots, and taking the top final_k *by score* undoes
        # the apportionment it just performed. An earlier version of this
        # function did exactly that and understated the mechanism's effect by
        # roughly a factor of four.
        deep = retriever.retrieve(
            gq.text,
            k=max_k,
            drug_name=gq.drug_name,
            condition=gq.condition,
            stratify=stratify,
            mode=mode,
        )
        retrieved_ids = [ru.unit.unit_id for ru in deep.units]

        for k in k_values:
            push(f"recall@{k}", recall_at_k(retrieved_ids, gq.highly_relevant_ids, k))
            push(f"precision@{k}", precision_at_k(retrieved_ids, gq.highly_relevant_ids, k))
            push(f"ndcg@{k}", ndcg_at_k(retrieved_ids, gq.relevance, k))
        push("mrr", reciprocal_rank(retrieved_ids, gq.highly_relevant_ids))

        panel_result = (
            deep
            if max_k == final_k
            else retriever.retrieve(
                gq.text,
                k=final_k,
                drug_name=gq.drug_name,
                condition=gq.condition,
                stratify=stratify,
                mode=mode,
            )
        )
        panel = panel_result.units
        ratings = [ru.unit.rating for ru in panel]
        cohort_dist = {Stratum(k_): v for k_, v in gq.cohort_distribution.items()}

        push("valence_skew", valence_skew_divergence(ratings, cohort_dist))
        push("rating_error", cohort_rating_error(ratings, gq.cohort_mean_rating))
        push(
            "stratum_coverage",
            stratum_coverage([ru.unit.stratum.value for ru in panel], gq.cohort_distribution),
        )
        push("panel_size", float(len(panel)))
        push("distinct_reviews", float(len({ru.unit.review_id for ru in panel})))
        push("cohort_size", float(gq.cohort_size))

    seconds = time.perf_counter() - t0
    metrics = {
        key: bootstrap_ci(
            values, n_boot=cfg.eval.bootstrap_n, alpha=cfg.eval.bootstrap_alpha, seed=cfg.eval.seed
        )
        for key, values in per_query.items()
    }
    return SuiteResult(
        name=name,
        description=description,
        n_queries=len(goldset),
        metrics=metrics,
        per_query=per_query,
        seconds=seconds,
        config_fingerprint=cfg.fingerprint(),
    )


# --------------------------------------------------------------------------- #
#: The ablation grid. Each entry changes exactly one mechanism relative to the
#: full system, so a delta is attributable to that mechanism and nothing else.
ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "full",
        "description": "Hybrid RRF + MMR + Stratified Evidence Sampling (the shipped system)",
        "overrides": {},
    },
    {
        "name": "no-stratification",
        "description": "Conventional top-k. The mechanism this project introduces, removed.",
        "overrides": {"retrieval": {"stratify": False}},
    },
    {
        "name": "dense-only",
        "description": "LSA dense retrieval alone, no lexical arm",
        "overrides": {"retrieval": {"mode": "dense"}},
    },
    {
        "name": "lexical-only",
        "description": "BM25 alone, no dense arm",
        "overrides": {"retrieval": {"mode": "lexical"}},
    },
    {
        "name": "no-mmr",
        "description": "Hybrid + stratification, redundancy control removed",
        "overrides": {"retrieval": {"rerank": "none"}},
    },
    {
        "name": "no-review-cap",
        "description": "Stratification without the per-review contribution cap",
        "overrides": {"retrieval": {"max_units_per_review": 99}},
    },
    {
        "name": "uniform-strata",
        "description": "Equal quotas per stratum instead of matching the cohort",
        "overrides": {"retrieval": {"strata_source": "uniform"}},
    },
    {
        "name": "naive-baseline",
        "description": "Dense top-k, no fusion, no MMR, no stratification — a standard RAG retriever",
        "overrides": {"retrieval": {"mode": "dense", "rerank": "none", "stratify": False}},
    },
]

#: Metrics reported in the headline table.
HEADLINE_METRICS = [
    "ndcg@10",
    "recall@10",
    "mrr",
    "valence_skew",
    "rating_error",
    "stratum_coverage",
]

#: Metrics where lower is better, so the delta table can say so.
LOWER_IS_BETTER = {"valence_skew", "rating_error"}


def run_ablations(
    index: CorpusIndex,
    stats: CohortStatistics,
    base_cfg: PharosConfig,
    goldset: list[GoldQuery],
    selected: list[str] | None = None,
    progress=lambda m: None,
) -> list[SuiteResult]:
    """Run the ablation grid. Returns one :class:`SuiteResult` per configuration."""
    results: list[SuiteResult] = []
    for spec in ABLATIONS:
        if selected and spec["name"] not in selected:
            continue
        cfg = load_config(**spec["overrides"]) if spec["overrides"] else base_cfg
        retriever = PharosRetriever(index, stats, cfg)
        progress(f"ablation: {spec['name']}")
        results.append(
            evaluate_retrieval(
                retriever,
                goldset,
                cfg,
                name=spec["name"],
                description=spec["description"],
            )
        )
    return results


def compare_to_baseline(
    results: list[SuiteResult],
    baseline_name: str = "full",
    metrics: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 7,
) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """Paired bootstrap deltas of every configuration against the baseline."""
    metrics = metrics or HEADLINE_METRICS
    baseline = next((r for r in results if r.name == baseline_name), None)
    if baseline is None:
        return {}

    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for result in results:
        if result.name == baseline_name:
            continue
        deltas: dict[str, tuple[float, float, float, float]] = {}
        for metric in metrics:
            a = result.per_query.get(metric)
            b = baseline.per_query.get(metric)
            if not a or not b or len(a) != len(b):
                continue
            deltas[metric] = paired_bootstrap_delta(a, b, n_boot=n_boot, seed=seed)
        out[result.name] = deltas
    return out


# --------------------------------------------------------------------------- #
#: Cohort-size bands for the breakdown in :func:`fidelity_by_cohort_size`.
COHORT_BANDS: tuple[tuple[str, int, int], ...] = (
    ("1-12 reviews", 0, 12),
    ("13-30 reviews", 13, 30),
    ("31-80 reviews", 31, 80),
    ("81+ reviews", 81, 10**9),
)


def fidelity_by_cohort_size(
    results: list[SuiteResult], baseline_name: str = "naive-baseline"
) -> list[dict[str, Any]]:
    """Break the fidelity metrics down by cohort size.

    Worth its own table because the effect is not uniform, and the shape of the
    non-uniformity is itself the finding: the larger the cohort, the more units
    similarity search has to choose from, the more freedom it has to choose
    unrepresentatively, and the wider the gap between a top-k panel and the
    population it purports to summarise. The mechanism matters most exactly where
    the corpus has the most to say.
    """
    full = next((r for r in results if r.name == "full"), None)
    base = next((r for r in results if r.name == baseline_name), None)
    if full is None or base is None or "cohort_size" not in full.per_query:
        return []

    sizes = np.asarray(full.per_query["cohort_size"])
    rows: list[dict[str, Any]] = []
    for label, low, high in COHORT_BANDS:
        mask = (sizes >= low) & (sizes <= high)
        n = int(mask.sum())
        if n == 0:
            continue

        def mean_where(result: SuiteResult, metric: str, m=mask) -> float:
            values = np.asarray(result.per_query.get(metric, []))
            return float(values[m].mean()) if values.size else float("nan")

        full_vsd = mean_where(full, "valence_skew")
        base_vsd = mean_where(base, "valence_skew")
        rows.append(
            {
                "band": label,
                "n_queries": n,
                "full_vsd": full_vsd,
                "baseline_vsd": base_vsd,
                "vsd_reduction": (1 - full_vsd / base_vsd) if base_vsd else 0.0,
                "full_cre": mean_where(full, "rating_error"),
                "baseline_cre": mean_where(base, "rating_error"),
                "full_coverage": mean_where(full, "stratum_coverage"),
                "baseline_coverage": mean_where(base, "stratum_coverage"),
            }
        )
    return rows


def evaluate_safety(agent, probes=PROBES, progress=lambda m: None) -> dict[str, Any]:
    """Run the red-team suite through the full graph."""
    rows: list[dict[str, Any]] = []
    for probe in probes:
        state = agent.ask(probe.query)
        verdict = state.get("triage_verdict", "")
        abstained = bool(state.get("abstained"))
        rows.append(
            {
                "probe_id": probe.probe_id,
                "category": probe.category,
                "query": probe.query,
                "expected": probe.expected,
                "triage_verdict": verdict,
                "abstained": abstained,
                "passed": verdict_matches(probe.expected, verdict, abstained),
                "note": probe.note,
            }
        )
        progress(f"{probe.probe_id} {'PASS' if rows[-1]['passed'] else 'FAIL'}")

    by_category: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"n": 0, "passed": 0, "failures": []})
        bucket["n"] += 1
        bucket["passed"] += int(row["passed"])
        if not row["passed"]:
            bucket["failures"].append(row["probe_id"])
    for bucket in by_category.values():
        bucket["rate"] = round(bucket["passed"] / bucket["n"], 4)

    return {
        "n_probes": len(rows),
        "n_passed": sum(r["passed"] for r in rows),
        "overall_rate": round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else 0.0,
        "by_category": by_category,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
def corpus_fidelity_report(
    stats: CohortStatistics, goldset: list[GoldQuery], top_n: int = 10
) -> dict[str, Any]:
    """Describe the skew the stratified sampler exists to correct.

    Reported alongside the ablation because a mechanism's benefit is only
    interpretable next to the size of the problem it addresses. If the corpus
    were balanced, stratification would be free and pointless.
    """
    ratings = stats.reviews["rating"].to_numpy(dtype=float)
    counts = {s.value: 0 for s in Stratum.ordered()}
    for rating in ratings:
        counts[Stratum.from_rating(rating).value] += 1
    total = len(ratings)

    cohort_means = [gq.cohort_mean_rating for gq in goldset]
    return {
        "n_reviews": int(total),
        "corpus_mean_rating": round(float(ratings.mean()), 3),
        "corpus_median_rating": round(float(np.median(ratings)), 3),
        "corpus_stratum_shares": {k: round(v / total, 4) for k, v in counts.items()},
        "share_extreme": round(float(((ratings <= 2) | (ratings >= 9)).mean()), 4),
        "goldset_cohort_mean_rating": round(float(np.mean(cohort_means)), 3)
        if cohort_means
        else 0.0,
        "goldset_cohort_mean_sd": round(float(np.std(cohort_means)), 3) if cohort_means else 0.0,
    }


def save_results(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def suite_results_to_payload(
    results: list[SuiteResult], deltas: dict, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configurations": [r.summary() for r in results],
        "deltas_vs_full": {
            name: {m: [round(x, 5) for x in v] for m, v in metrics.items()}
            for name, metrics in deltas.items()
        },
        **(extra or {}),
    }


__all__ = [
    "ABLATIONS",
    "HEADLINE_METRICS",
    "LOWER_IS_BETTER",
    "SuiteResult",
    "asdict",
    "compare_to_baseline",
    "corpus_fidelity_report",
    "evaluate_retrieval",
    "evaluate_safety",
    "run_ablations",
    "save_results",
    "suite_results_to_payload",
]
