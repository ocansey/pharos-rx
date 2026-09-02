"""Cohort statistics and disproportionality analysis.

This module is the answer to the single largest failure mode of RAG over review
corpora: **numeric hallucination**. Ask a conventional system "how many people
report weight gain on this drug?" and it will read a dozen retrieved passages
and produce a number — one that is not computed from anything, cannot be
checked, and is wrong in a way no citation will reveal, because each individual
citation is real.

PHAROS never lets the language model produce a number. Every quantity in an
answer is computed here, over the whole corpus, and handed to the generator as a
pre-formatted statistic with an identifier the verifier can check. The language
model's job is reduced to what it is actually good at: putting the computed
facts into readable prose.

The estimators are the standard pharmacovigilance toolkit, chosen because they
are the ones a regulator would recognise:

* **Wilson score interval** for proportions — not the Wald interval, which is
  badly behaved exactly where this corpus lives (small cohorts, proportions near
  0 or 1, where Wald produces intervals extending past 0 or 1).
* **Proportional Reporting Ratio** with the Gaussian interval on the log scale,
  the MHRA's screening statistic.
* **Reporting Odds Ratio**, the EMA/Netherlands Pharmacovigilance Centre variant,
  which is the estimator that remains valid under case-control sampling.
* **Information Component** from the Bayesian Confidence Propagation Neural
  Network used by the WHO Uppsala Monitoring Centre — a shrinkage estimator that
  does not scream at a 2-of-2 cell the way PRR does.

A caveat is carried in the code, not just in the paper: these statistics are
computed over *self-selected patient reviews*, not spontaneous adverse-event
reports. They measure what people write, and people write when a drug is
remarkable. Every emitted statistic carries that framing, and
:meth:`CohortStatistics.disproportionality` refuses to report at all below a
support floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

from pharos.data.schema import Stratum


# --------------------------------------------------------------------------- #
# Interval estimators
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval throughout. With 2 events in 8 reviews —
    an entirely ordinary cell in this corpus — Wald gives (-0.05, 0.55), which
    includes impossible values and understates uncertainty; Wilson gives
    (0.07, 0.59), which is both admissible and honest.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = float(sps.norm.ppf(1 - alpha / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean.

    Used for mean ratings, whose distribution is sharply bimodal — 32% of the
    corpus sits at 9–10 and 18% at 1–2. A normal-theory interval on a mean drawn
    from that shape is not defensible; resampling makes no distributional claim.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class CohortSummary:
    """Descriptive statistics for one (drug, condition) cohort."""

    drug_name: str
    condition: str | None
    n_reviews: int
    n_units: int
    mean_rating: float
    rating_ci: tuple[float, float]
    median_rating: float
    rating_histogram: dict[int, int]
    stratum_distribution: dict[str, float]
    date_range: tuple[str, str]
    mean_useful_count: float
    top_adverse_concepts: list[tuple[str, int, float, tuple[float, float]]] = field(
        default_factory=list
    )
    low_support: bool = False
    stat_id: str = ""

    def to_evidence_block(self) -> str:
        """Render as a citable block for the generator's context window.

        Formatted as text rather than JSON on purpose: the generator's task is
        to quote these numbers verbatim, and a flat labelled layout produces
        markedly fewer transcription errors than nested JSON in our testing.
        """
        cond = self.condition or "any indication"
        lo, hi = self.rating_ci
        lines = [
            f"[{self.stat_id}] COHORT STATISTICS — {self.drug_name} for {cond}",
            f"  reviews analysed: {self.n_reviews:,} ({self.n_units:,} evidence units)",
            f"  mean rating: {self.mean_rating:.2f}/10  (95% CI {lo:.2f}–{hi:.2f})"
            f"   median: {self.median_rating:.1f}",
            "  outcome distribution: "
            + ", ".join(f"{k} {v * 100:.1f}%" for k, v in self.stratum_distribution.items()),
            f"  review dates: {self.date_range[0]} to {self.date_range[1]}",
        ]
        if self.top_adverse_concepts:
            lines.append("  most-reported effects (share of reviews, 95% CI):")
            for concept, count, prop, (clo, chi) in self.top_adverse_concepts:
                lines.append(
                    f"    - {concept.replace('_', ' ')}: {count}/{self.n_reviews} "
                    f"= {prop * 100:.1f}% ({clo * 100:.1f}–{chi * 100:.1f}%)"
                )
        if self.low_support:
            lines.append("  ** LOW SUPPORT: too few reviews for a stable estimate **")
        return "\n".join(lines)


@dataclass
class DisproportionalitySignal:
    """One drug × adverse-event disproportionality result."""

    drug_name: str
    concept: str
    a: int  # drug and event
    b: int  # drug, not event
    c: int  # event, other drugs
    d: int  # neither
    prr: float
    prr_ci: tuple[float, float]
    ror: float
    ror_ci: tuple[float, float]
    chi2: float
    p_value: float
    information_component: float
    ic025: float
    flagged: bool
    stat_id: str = ""

    def to_evidence_block(self) -> str:
        lines = [
            f"[{self.stat_id}] DISPROPORTIONALITY — {self.concept.replace('_', ' ')} "
            f"in {self.drug_name} reviews",
            f"  contingency: a={self.a} b={self.b} c={self.c} d={self.d}",
            f"  PRR = {self.prr:.2f} (95% CI {self.prr_ci[0]:.2f}–{self.prr_ci[1]:.2f})",
            f"  ROR = {self.ror:.2f} (95% CI {self.ror_ci[0]:.2f}–{self.ror_ci[1]:.2f})",
            f"  chi-square = {self.chi2:.2f}, p = {self.p_value:.2e}",
            f"  IC = {self.information_component:.2f}, IC025 = {self.ic025:.2f}",
            f"  signal criterion (a>=3, PRR>=2, chi2>=4): {'MET' if self.flagged else 'not met'}",
            "  NOTE: computed over self-selected patient reviews, not spontaneous "
            "adverse-event reports. Not a regulatory signal.",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class CohortStatistics:
    """Deterministic statistics over the cleaned corpus.

    Holds two frames: ``reviews`` (one row per review) and ``units`` (one row per
    evidence unit, with an exploded ``adverse_terms`` column). All statistics
    are computed over ``reviews`` — the review, not the unit, is the reporting
    entity, and counting units would let one voluble reviewer count five times.
    """

    def __init__(
        self,
        reviews: pd.DataFrame,
        units: pd.DataFrame,
        min_support: int = 10,
        seed: int = 0,
    ) -> None:
        self.reviews = reviews
        self.units = units
        self.min_support = min_support
        self.seed = seed
        self._stat_counter = 0
        self._review_concepts = self._build_review_concepts()

        # Cohort lookup is on the hot path: the retriever asks for a reference
        # distribution on every query, and a `str.casefold()` comparison across
        # the whole review table each time turns a millisecond of retrieval into
        # a hundred. Casefolded keys are computed once and grouped into positional
        # indices; lookup is then a dict hit and an `iloc`.
        drug_key = reviews["drug_name"].astype(str).str.casefold()
        condition_key = reviews["condition"].fillna("").astype(str).str.casefold()
        pair_key = drug_key + "\x00" + condition_key

        self._drug_index = self._positional_groups(drug_key)
        self._pair_index = self._positional_groups(pair_key)
        self._distribution_cache: dict[tuple[str | None, str | None], dict[Stratum, float]] = {}

    # ---------------------------------------------------------------- #
    @staticmethod
    def _positional_groups(key: pd.Series) -> dict[str, np.ndarray]:
        """Map each distinct key to the *positional* row indices carrying it.

        Positional rather than label-based, so the lookup survives a frame whose
        index is not a clean range — which is what happens the moment someone
        passes a filtered slice of the corpus.
        """
        positions = np.arange(len(key), dtype=np.int64)
        out: dict[str, np.ndarray] = {}
        order = np.argsort(key.to_numpy().astype(str), kind="stable")
        sorted_keys = key.to_numpy().astype(str)[order]
        sorted_positions = positions[order]
        if len(sorted_keys) == 0:
            return out
        boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(sorted_keys)]])
        for start, end in zip(starts, ends, strict=True):
            out[sorted_keys[start]] = sorted_positions[start:end]
        return out

    def _next_stat_id(self, prefix: str) -> str:
        self._stat_counter += 1
        return f"{prefix}-{self._stat_counter:04d}"

    def _build_review_concepts(self) -> pd.DataFrame:
        """Collapse unit-level adverse concepts to a review-level binary table.

        A review that mentions nausea in three separate units still contributes
        one nausea report. Failing to collapse here is the most common way an
        analysis of this corpus inflates its own denominators.
        """
        if self.units.empty or "adverse_terms" not in self.units:
            return pd.DataFrame(columns=["review_id", "concept"])
        exploded = self.units[["review_id", "adverse_terms"]].explode("adverse_terms")
        exploded = exploded.dropna(subset=["adverse_terms"])
        exploded = exploded.rename(columns={"adverse_terms": "concept"})
        return exploded.drop_duplicates().reset_index(drop=True)

    # ---------------------------------------------------------------- #
    def cohort_frame(
        self, drug_name: str | None = None, condition: str | None = None
    ) -> pd.DataFrame:
        """Rows matching the requested cohort. Matching is case-insensitive."""
        if not drug_name and not condition:
            return self.reviews

        if drug_name and condition:
            key = f"{drug_name.casefold()}\x00{condition.casefold()}"
            positions = self._pair_index.get(key)
            if positions is not None:
                return self.reviews.iloc[positions]
            return self.reviews.iloc[[]]

        if drug_name:
            positions = self._drug_index.get(drug_name.casefold())
            if positions is not None:
                return self.reviews.iloc[positions]
            return self.reviews.iloc[[]]

        assert condition is not None  # narrowed by the early return above
        mask = self.reviews["condition"].fillna("").str.casefold() == condition.casefold()
        return self.reviews[mask]

    def stratum_distribution(
        self, drug_name: str | None = None, condition: str | None = None
    ) -> dict[Stratum, float]:
        """The cohort's true outcome distribution.

        This is the reference distribution the Stratified Evidence Sampling
        allocator matches. Computing it over the *whole* cohort rather than over
        the retrieved candidates is the entire point: a distribution estimated
        from what similarity search returned would already carry the bias the
        allocator exists to remove.

        An empty cohort returns a uniform distribution. That is the right
        default: with no information about the population, the allocator should
        reserve slots for every outcome rather than silently concentrate on
        whichever one similarity search happened to favour.
        """
        cache_key = (drug_name, condition)
        cached = self._distribution_cache.get(cache_key)
        if cached is not None:
            return cached

        df = self.cohort_frame(drug_name, condition)
        if df.empty:
            result = {s: 1.0 / len(Stratum.ordered()) for s in Stratum.ordered()}
        else:
            ratings = df["rating"].to_numpy(dtype=float)
            counts = dict.fromkeys(Stratum.ordered(), 0)
            for rating in ratings:
                counts[Stratum.from_rating(rating)] += 1
            total = len(ratings)
            result = {s: c / total for s, c in counts.items()}

        self._distribution_cache[cache_key] = result
        return result

    # ---------------------------------------------------------------- #
    def summarise(
        self,
        drug_name: str,
        condition: str | None = None,
        top_n_effects: int = 6,
        alpha: float = 0.05,
    ) -> CohortSummary | None:
        """Descriptive statistics for one cohort, or ``None`` if it is empty."""
        df = self.cohort_frame(drug_name, condition)
        if df.empty:
            return None

        ratings = df["rating"].to_numpy(dtype=float)
        hist = {int(r): int((ratings == r).sum()) for r in range(1, 11)}
        strata = self.stratum_distribution(drug_name, condition)

        review_ids = set(df["review_id"])
        n_units = int(self.units["review_id"].isin(review_ids).sum()) if not self.units.empty else 0

        concepts = self._review_concepts[self._review_concepts["review_id"].isin(review_ids)]
        counts = concepts["concept"].value_counts()
        n = len(df)
        top: list[tuple[str, int, float, tuple[float, float]]] = []
        for concept, count in counts.head(top_n_effects).items():
            top.append((str(concept), int(count), count / n, wilson_interval(int(count), n, alpha)))

        dates = pd.to_datetime(df["review_date"])
        return CohortSummary(
            drug_name=drug_name,
            condition=condition,
            n_reviews=n,
            n_units=n_units,
            mean_rating=float(ratings.mean()),
            rating_ci=bootstrap_mean_ci(ratings, alpha=alpha, seed=self.seed),
            median_rating=float(np.median(ratings)),
            rating_histogram=hist,
            stratum_distribution={s.value: v for s, v in strata.items()},
            date_range=(str(dates.min().date()), str(dates.max().date())),
            mean_useful_count=float(df["useful_count"].mean()),
            top_adverse_concepts=top,
            low_support=n < self.min_support,
            stat_id=self._next_stat_id("STAT"),
        )

    # ---------------------------------------------------------------- #
    def disproportionality(
        self,
        drug_name: str,
        concept: str,
        condition: str | None = None,
        comparator: Literal["all", "same_condition"] = "same_condition",
        alpha: float = 0.05,
    ) -> DisproportionalitySignal | None:
        """Screen one drug × event pair against a comparator population.

        ``comparator="same_condition"`` restricts the background to other drugs
        used for the same indication. This is the confounding-by-indication
        control: comparing sedation reports for a sleep aid against the whole
        corpus would flag every sleep aid ever written about, because the
        indication, not the drug, is doing the work.

        Returns ``None`` when the exposed cell has fewer than three reports —
        below that, every disproportionality estimator is noise, and emitting one
        with a wide interval invites a reader to over-read it.
        """
        exposed = self.cohort_frame(drug_name, condition)
        if exposed.empty:
            return None

        if comparator == "same_condition" and condition:
            background = self.reviews[
                (self.reviews["condition"].fillna("").str.casefold() == condition.casefold())
                & (self.reviews["drug_name"].str.casefold() != drug_name.casefold())
            ]
        else:
            background = self.reviews[
                self.reviews["drug_name"].str.casefold() != drug_name.casefold()
            ]
        if background.empty:
            return None

        with_concept = set(
            self._review_concepts.loc[self._review_concepts["concept"] == concept, "review_id"]
        )
        exposed_ids, background_ids = set(exposed["review_id"]), set(background["review_id"])

        a = len(exposed_ids & with_concept)
        b = len(exposed_ids) - a
        c = len(background_ids & with_concept)
        d = len(background_ids) - c

        if a < 3 or (a + b) == 0 or (c + d) == 0:
            return None

        # Haldane–Anscombe continuity correction: 0.5 added to every cell so a
        # zero background cell yields a finite, if wide, interval instead of an
        # infinity that would propagate into the answer.
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5

        prr = (aa / (aa + bb)) / (cc / (cc + dd))
        se_log_prr = math.sqrt(1 / aa - 1 / (aa + bb) + 1 / cc - 1 / (cc + dd))
        z = float(sps.norm.ppf(1 - alpha / 2))
        prr_ci = (prr * math.exp(-z * se_log_prr), prr * math.exp(z * se_log_prr))

        ror = (aa / bb) / (cc / dd)
        se_log_ror = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        ror_ci = (ror * math.exp(-z * se_log_ror), ror * math.exp(z * se_log_ror))

        table = np.array([[a, b], [c, d]])
        try:
            chi2, p_value = sps.chi2_contingency(table, correction=True)[:2]
        except ValueError:
            chi2, p_value = 0.0, 1.0

        # BCPNN Information Component, WHO-UMC shrinkage form.
        n_total = a + b + c + d
        expected = max(((a + b) * (a + c)) / n_total, 1e-9)
        ic = math.log2((a + 0.5) / (expected + 0.5))
        var_ic = (1 / math.log(2) ** 2) * (
            (n_total - a + 0.5) / ((a + 1) * (1 + n_total + 1))
            + (n_total - (a + b) + 0.5) / ((a + b + 1) * (1 + n_total + 1))
            + (n_total - (a + c) + 0.5) / ((a + c + 1) * (1 + n_total + 1))
        )
        ic025 = ic - 2 * math.sqrt(max(var_ic, 0.0))

        return DisproportionalitySignal(
            drug_name=drug_name,
            concept=concept,
            a=a,
            b=b,
            c=c,
            d=d,
            prr=prr,
            prr_ci=prr_ci,
            ror=ror,
            ror_ci=ror_ci,
            chi2=float(chi2),
            p_value=float(p_value),
            information_component=ic,
            ic025=ic025,
            # Evans' criteria, the conventional screening threshold.
            flagged=bool(a >= 3 and prr >= 2.0 and chi2 >= 4.0),
            stat_id=self._next_stat_id("STAT"),
        )

    # ---------------------------------------------------------------- #
    def compare_drugs(
        self, drug_names: list[str], condition: str | None = None, alpha: float = 0.05
    ) -> dict[str, Any]:
        """Head-to-head comparison across drugs for one indication.

        Includes a Kruskal–Wallis test across the rating distributions. Ratings
        are ordinal and bimodal, so a one-way ANOVA would be assuming a shape the
        data plainly does not have.
        """
        rows, samples = [], []
        for name in drug_names:
            summary = self.summarise(name, condition, alpha=alpha)
            if summary is None:
                continue
            rows.append(summary)
            samples.append(self.cohort_frame(name, condition)["rating"].to_numpy(dtype=float))

        result: dict[str, Any] = {
            "condition": condition,
            "cohorts": rows,
            "stat_id": self._next_stat_id("STAT"),
        }
        if len(samples) >= 2 and all(len(s) >= 5 for s in samples):
            h, p = sps.kruskal(*samples)
            result["kruskal_h"] = float(h)
            result["kruskal_p"] = float(p)
        return result

    # ---------------------------------------------------------------- #
    def resolve_drug(self, query: str, limit: int = 5) -> list[str]:
        """Fuzzy drug-name resolution against the corpus vocabulary.

        Users write "wellbutrin"; the corpus holds "Wellbutrin XL", "Wellbutrin
        SR" and "Bupropion". Exact match, then prefix, then containment — no
        edit distance, because a typo-tolerant match on drug names is a way to
        silently answer about the wrong medicine.
        """
        q = query.casefold().strip()
        names = self.reviews["drug_name"].dropna().unique().tolist()
        exact = [n for n in names if n.casefold() == q]
        if exact:
            return exact[:limit]
        prefix = sorted(n for n in names if n.casefold().startswith(q))
        contains = sorted(n for n in names if q in n.casefold() and n not in prefix)
        return (prefix + contains)[:limit]

    def resolve_condition(self, query: str, limit: int = 5) -> list[str]:
        """Fuzzy condition resolution against the repaired label vocabulary."""
        q = query.casefold().strip()
        names = self.reviews["condition"].dropna().unique().tolist()
        exact = [n for n in names if n.casefold() == q]
        if exact:
            return exact[:limit]
        contains = sorted(n for n in names if q in n.casefold() or n.casefold() in q)
        return contains[:limit]
