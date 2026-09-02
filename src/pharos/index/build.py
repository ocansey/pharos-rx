"""Corpus and index construction.

Two entry points, each idempotent and each writing a provenance record:

* :func:`build_corpus` — raw TSVs to a cleaned, subsampled review table plus an
  evidence-unit table.
* :func:`build_index` — evidence units to a queryable :class:`CorpusIndex`.

They are separate on purpose. Cleaning is expensive and stable; encoding is
cheap and gets swept during ablation. Keeping them apart means changing an
encoder does not re-run de-duplication over 215,063 rows.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pharos import __version__
from pharos.config import PharosConfig
from pharos.data.clean import ConditionRepairer, clean_corpus, load_raw, subsample
from pharos.data.schema import Aspect, EvidenceUnit, Stratum
from pharos.index.bm25 import BM25Index
from pharos.index.embeddings import build_encoder
from pharos.index.store import CorpusIndex
from pharos.nlp.aspects import label_aspects
from pharos.nlp.lexicon import extract_adverse_concepts, is_hedged
from pharos.nlp.segment import segment_review

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
def build_corpus(
    cfg: PharosConfig, progress: ProgressFn = _noop
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean, subsample, segment, and label. Returns ``(reviews, units, report)``."""
    t0 = time.perf_counter()

    progress("loading raw TSVs")
    raw = load_raw(cfg)

    progress(f"cleaning {len(raw):,} reviews")
    repairer = ConditionRepairer()
    cleaned, report = clean_corpus(raw, cfg, repairer)

    progress(f"subsampling to {cfg.data.sample_size or 'all'} reviews")
    reviews = subsample(cleaned, cfg)

    progress(f"segmenting {len(reviews):,} reviews into evidence units")
    unit_rows: list[dict] = []
    for row in reviews.itertuples(index=False):
        segments = segment_review(
            row.text, min_chars=cfg.data.min_unit_chars, max_chars=cfg.data.max_unit_chars
        )
        for seg in segments:
            concepts, saw_negation = extract_adverse_concepts(seg.text)
            unit_rows.append(
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
                    "negated": saw_negation,
                    "hedged": is_hedged(seg.text),
                }
            )
    units = pd.DataFrame(unit_rows)

    elapsed = time.perf_counter() - t0
    meta = report.to_dict()
    meta.update(
        {
            "pharos_version": __version__,
            "config_fingerprint": cfg.fingerprint(),
            "reviews_after_subsample": len(reviews),
            "evidence_units": len(units),
            "units_per_review": round(len(units) / max(len(reviews), 1), 2),
            "distinct_drugs": int(reviews["drug_name"].nunique()),
            "distinct_conditions": int(reviews["condition"].nunique()),
            "build_seconds": round(elapsed, 2),
        }
    )
    progress(f"corpus built: {len(reviews):,} reviews -> {len(units):,} units ({elapsed:.1f}s)")
    return reviews, units, meta


def save_corpus(reviews: pd.DataFrame, units: pd.DataFrame, meta: dict, cfg: PharosConfig) -> None:
    out = cfg.paths.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(out / "reviews.parquet", index=False)
    units.to_parquet(out / "units.parquet", index=False)
    (out / "corpus_report.json").write_text(json.dumps(meta, indent=2, default=str))


def load_corpus(cfg: PharosConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = cfg.paths.processed_dir
    reviews_path, units_path = out / "reviews.parquet", out / "units.parquet"
    if not reviews_path.exists() or not units_path.exists():
        raise FileNotFoundError(f"no processed corpus at {out}. Run `pharos build-corpus` first.")
    reviews = pd.read_parquet(reviews_path)
    units = pd.read_parquet(units_path)
    report_path = out / "corpus_report.json"
    meta = json.loads(report_path.read_text()) if report_path.exists() else {}
    return reviews, units, meta


# --------------------------------------------------------------------------- #
def units_to_models(units: pd.DataFrame) -> list[EvidenceUnit]:
    """Materialise the unit frame as validated models.

    Validation is not ceremony here. A malformed rating or a lost stratum would
    corrupt the quota allocator silently — the panel would still be full, just
    wrong — and this is the last point at which the failure is cheap to catch.
    """
    out: list[EvidenceUnit] = []
    for row in units.itertuples(index=False):
        out.append(
            EvidenceUnit(
                unit_id=row.unit_id,
                review_id=int(row.review_id),
                ordinal=int(row.ordinal),
                text=row.text,
                aspects=[Aspect(a) for a in row.aspects],
                drug_name=row.drug_name,
                condition=row.condition,
                rating=float(row.rating),
                stratum=Stratum(row.stratum),
                review_date=row.review_date,
                useful_count=int(row.useful_count),
                adverse_terms=list(row.adverse_terms),
                negated=bool(row.negated),
                hedged=bool(row.hedged),
            )
        )
    return out


def build_index(
    units: pd.DataFrame, cfg: PharosConfig, progress: ProgressFn = _noop
) -> CorpusIndex:
    """Fit the encoder, embed, build BM25 and the metadata postings."""
    t0 = time.perf_counter()
    models = units_to_models(units)
    texts = [u.text for u in models]

    progress(f"fitting encoder '{cfg.index.encoder}' on {len(texts):,} units")
    encoder = build_encoder(cfg.index, seed=cfg.data.seed).fit(texts)

    progress("encoding units")
    vectors = encoder.encode(texts).astype(np.float32)

    progress("building BM25 inverted index")
    bm25 = BM25Index(k1=cfg.index.bm25_k1, b=cfg.index.bm25_b).fit(texts)

    progress("building metadata postings")
    postings = CorpusIndex.build_postings(models)

    elapsed = time.perf_counter() - t0
    info = {
        "pharos_version": __version__,
        "config_fingerprint": cfg.fingerprint(),
        "encoder": encoder.name,
        "embedding_dim": int(vectors.shape[1]) if vectors.size else 0,
        "n_units": len(models),
        "bm25_vocabulary": bm25.vocabulary_size(),
        "avg_unit_tokens": round(float(bm25.doc_len.mean()), 2) if bm25.n_docs else 0.0,
        "build_seconds": round(elapsed, 2),
    }
    if hasattr(encoder, "explained_variance"):
        info["lsa_explained_variance"] = round(encoder.explained_variance, 4)

    progress(f"index built in {elapsed:.1f}s ({info['encoder']}, dim={info['embedding_dim']})")
    return CorpusIndex(
        units=models,
        vectors=vectors,
        bm25=bm25,
        encoder=encoder,
        build_info=info,
        **postings,
    )


def load_index(cfg: PharosConfig) -> CorpusIndex:
    return CorpusIndex.load(Path(cfg.paths.index_dir))
