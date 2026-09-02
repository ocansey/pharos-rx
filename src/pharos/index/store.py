"""The corpus index: units, vectors, lexical index, and metadata postings.

One object owns everything retrieval needs and knows how to persist itself. The
piece worth pointing at is :attr:`CorpusIndex.by_drug` and its siblings —
pre-built posting lists from metadata values to unit positions.

The reason is a correctness one, not a performance one. Metadata filtering in
most vector stores is applied *after* the approximate nearest-neighbour search,
so a filter for a drug with 40 reviews in a 70,000-unit index frequently returns
nothing at all: none of the 200 nearest neighbours happened to be that drug. The
system then answers from an empty panel, or silently widens, and the user cannot
tell which. Here the filter produces a candidate id array *before* scoring, both
arms score only within it, and a small cohort is retrieved exhaustively and
exactly.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pharos.data.schema import Aspect, EvidenceUnit, Stratum
from pharos.index.bm25 import BM25Index
from pharos.index.embeddings import Encoder


@dataclass
class CorpusIndex:
    """An immutable, queryable index over evidence units."""

    units: list[EvidenceUnit]
    vectors: np.ndarray
    bm25: BM25Index
    encoder: Encoder

    #: Metadata posting lists, casefolded keys -> unit positions.
    by_drug: dict[str, np.ndarray] = field(default_factory=dict)
    by_condition: dict[str, np.ndarray] = field(default_factory=dict)
    by_aspect: dict[str, np.ndarray] = field(default_factory=dict)
    by_stratum: dict[str, np.ndarray] = field(default_factory=dict)
    by_review: dict[int, np.ndarray] = field(default_factory=dict)

    build_info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.units)

    @classmethod
    def build_postings(cls, units: list[EvidenceUnit]) -> dict[str, dict[Any, np.ndarray]]:
        drug: dict[str, list[int]] = {}
        condition: dict[str, list[int]] = {}
        aspect: dict[str, list[int]] = {}
        stratum: dict[str, list[int]] = {}
        review: dict[int, list[int]] = {}

        for pos, unit in enumerate(units):
            drug.setdefault(unit.drug_name.casefold(), []).append(pos)
            if unit.condition:
                condition.setdefault(unit.condition.casefold(), []).append(pos)
            for asp in unit.aspects:
                aspect.setdefault(asp.value, []).append(pos)
            stratum.setdefault(unit.stratum.value, []).append(pos)
            review.setdefault(unit.review_id, []).append(pos)

        def freeze(d: dict) -> dict:
            return {k: np.asarray(v, dtype=np.int64) for k, v in d.items()}

        return {
            "by_drug": freeze(drug),
            "by_condition": freeze(condition),
            "by_aspect": freeze(aspect),
            "by_stratum": freeze(stratum),
            "by_review": freeze(review),
        }

    # ------------------------------------------------------------------ #
    def candidate_ids(
        self,
        drug_name: str | None = None,
        condition: str | None = None,
        aspects: list[Aspect] | None = None,
        strata: list[Stratum] | None = None,
    ) -> np.ndarray | None:
        """Intersect metadata filters into a candidate id array.

        ``None`` means "no constraint" and lets the caller skip the intersection
        entirely. An *empty array* means "constraints matched nothing", which is
        a different thing and must not be conflated: the first is a wide search,
        the second is grounds for abstention.
        """
        sets: list[np.ndarray] = []
        if drug_name:
            sets.append(self.by_drug.get(drug_name.casefold(), np.zeros(0, dtype=np.int64)))
        if condition:
            sets.append(self.by_condition.get(condition.casefold(), np.zeros(0, dtype=np.int64)))
        if aspects:
            union = (
                np.concatenate(
                    [self.by_aspect.get(a.value, np.zeros(0, dtype=np.int64)) for a in aspects]
                )
                if aspects
                else np.zeros(0, dtype=np.int64)
            )
            sets.append(np.unique(union))
        if strata:
            union = np.concatenate(
                [self.by_stratum.get(s.value, np.zeros(0, dtype=np.int64)) for s in strata]
            )
            sets.append(np.unique(union))

        if not sets:
            return None
        out = sets[0]
        for other in sets[1:]:
            out = np.intersect1d(out, other, assume_unique=False)
        return out

    # ------------------------------------------------------------------ #
    def dense_search(
        self, query_vector: np.ndarray, k: int, candidates: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Exact inner-product search, optionally restricted to candidates.

        Exact rather than approximate. At this corpus size a full matrix-vector
        product is a few milliseconds, and exactness removes an entire class of
        confound from the ablation study: a recall difference between two
        configurations is a property of the retrieval strategy, not of an ANN
        graph's recall curve.
        """
        if len(self.units) == 0:
            return []
        if candidates is None:
            scores = self.vectors @ query_vector
            k = min(k, scores.shape[0])
            if k <= 0:
                return []
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [(int(i), float(scores[i])) for i in top]

        if candidates.size == 0:
            return []
        sub = self.vectors[candidates] @ query_vector
        k = min(k, sub.shape[0])
        top = np.argpartition(-sub, k - 1)[:k] if k < sub.shape[0] else np.arange(sub.shape[0])
        top = top[np.argsort(-sub[top], kind="stable")]
        return [(int(candidates[i]), float(sub[i])) for i in top]

    def lexical_search(
        self, query: str, k: int, candidates: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        return self.bm25.top_k(query, k, candidates)

    # ------------------------------------------------------------------ #
    def save(self, directory: Path) -> None:
        """Persist to disk.

        Vectors go to ``.npy`` (memory-mappable, and the pickle protocol should
        not be responsible for a 100 MB float array); everything else is
        pickled; build metadata is written as readable JSON so a human can see
        what produced an index without loading it.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "units.pkl").open("wb") as fh:
            pickle.dump(self.units, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with (directory / "bm25.pkl").open("wb") as fh:
            pickle.dump(self.bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with (directory / "encoder.pkl").open("wb") as fh:
            pickle.dump(self.encoder, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with (directory / "postings.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "by_drug": self.by_drug,
                    "by_condition": self.by_condition,
                    "by_aspect": self.by_aspect,
                    "by_stratum": self.by_stratum,
                    "by_review": self.by_review,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        (directory / "build_info.json").write_text(json.dumps(self.build_info, indent=2))

    @classmethod
    def load(cls, directory: Path) -> CorpusIndex:
        directory = Path(directory)
        required = ["vectors.npy", "units.pkl", "bm25.pkl", "encoder.pkl", "postings.pkl"]
        missing = [f for f in required if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"index at {directory} is incomplete (missing {', '.join(missing)}). "
                f"Run `pharos build-index` first."
            )
        vectors = np.load(directory / "vectors.npy")
        with (directory / "units.pkl").open("rb") as fh:
            units = pickle.load(fh)
        with (directory / "bm25.pkl").open("rb") as fh:
            bm25 = pickle.load(fh)
        with (directory / "encoder.pkl").open("rb") as fh:
            encoder = pickle.load(fh)
        with (directory / "postings.pkl").open("rb") as fh:
            postings = pickle.load(fh)
        info_path = directory / "build_info.json"
        build_info = json.loads(info_path.read_text()) if info_path.exists() else {}
        return cls(
            units=units,
            vectors=vectors,
            bm25=bm25,
            encoder=encoder,
            build_info=build_info,
            **postings,
        )
