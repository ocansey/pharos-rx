"""Typed, layered configuration.

Precedence, lowest to highest:

    field defaults  <  configs/default.yaml  <  a named config file  <  environment
    variables (``PHAROS_*``)  <  explicit keyword overrides

Every stage of the pipeline reads from this object, and every artifact written to
disk records the config hash that produced it. That is what makes an ablation
table trustworthy: a result row can be traced back to the exact settings that
generated it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# Sub-configurations
# --------------------------------------------------------------------------- #
class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths resolve against the repository root."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")
    artifacts_dir: Path = Path("artifacts")

    @model_validator(mode="after")
    def _absolutise(self) -> PathsConfig:
        for name in ("raw_dir", "processed_dir", "index_dir", "artifacts_dir"):
            value = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, (REPO_ROOT / value).resolve())
        return self


class DataConfig(BaseModel):
    """Corpus construction."""

    #: Deterministic subsample size, in *reviews*. ``None`` uses the full corpus.
    #: The default keeps a clean-clone build to a couple of minutes on a laptop
    #: while preserving every condition with adequate support.
    sample_size: int | None = 25_000
    seed: int = 20_260_902

    #: Drop reviews whose condition label is a scraper artifact or is missing.
    drop_unlabelled_conditions: bool = True
    #: Conditions with fewer than this many reviews are excluded from the
    #: cohort-statistics surface, because their intervals are uninformative.
    min_condition_support: int = 25
    #: Drug/condition pairs below this are still retrievable but are flagged
    #: as low-support in every statistic that mentions them.
    min_cohort_support: int = 10

    #: Near-duplicate suppression via character 5-gram MinHash Jaccard.
    dedup_threshold: float = 0.92
    dedup_enabled: bool = True

    #: Evidence-unit segmentation bounds, in characters.
    min_unit_chars: int = 60
    max_unit_chars: int = 420

    @field_validator("dedup_threshold")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("dedup_threshold must lie in (0, 1]")
        return v


class IndexConfig(BaseModel):
    """Encoder and index construction."""

    #: ``lsa`` is the deterministic, dependency-light default: a TF-IDF matrix
    #: reduced by truncated SVD. ``sentence-transformers`` swaps in a neural
    #: encoder for the headline numbers. ``hashing`` exists only for tests.
    encoder: Literal["lsa", "sentence-transformers", "hashing"] = "lsa"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    st_batch_size: int = 64
    normalize: bool = True

    # TF-IDF settings used by both the LSA encoder and the lexical index.
    tfidf_min_df: int = 3
    tfidf_max_df: float = 0.55
    tfidf_ngram_max: int = 2

    # Okapi BM25.
    bm25_k1: float = 1.4
    bm25_b: float = 0.72


class RetrievalConfig(BaseModel):
    """Retrieval, fusion, and stratification."""

    #: Candidates drawn from each arm before fusion.
    candidate_k: int = 120
    #: Evidence units handed to the generator.
    final_k: int = 12

    mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    #: Reciprocal-rank-fusion damping constant.
    rrf_k: int = 60
    #: Weight on the lexical arm during fusion; the dense arm gets ``1 - w``.
    lexical_weight: float = 0.5

    #: Stratified Evidence Sampling. When disabled the retriever degrades to a
    #: conventional top-k, which is exactly the baseline the ablation measures.
    stratify: bool = True
    #: Reference distribution for the quota allocator.
    strata_source: Literal["cohort", "uniform"] = "cohort"
    #: Minimum slots reserved for any stratum with non-zero cohort mass. This is
    #: what guarantees a dissenting minority is never rounded out of the panel.
    min_slots_per_stratum: int = 1
    #: Cap on evidence units contributed by any single review, so one verbose
    #: narrator cannot colonise the panel.
    max_units_per_review: int = 2

    #: Maximal-marginal-relevance redundancy penalty applied within strata.
    mmr_lambda: float = 0.65
    rerank: Literal["none", "mmr", "cross-encoder"] = "mmr"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    #: Below this fused score the panel is judged too weak to answer from.
    abstain_below_score: float = 0.0
    #: Fewer supporting units than this triggers an insufficient-evidence answer.
    abstain_below_units: int = 3


class LLMConfig(BaseModel):
    """Generative back-end. ``mock`` is deterministic and offline."""

    provider: Literal["anthropic", "openai", "ollama", "mock"] = "mock"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1400
    timeout_s: int = 90
    max_retries: int = 3


class AgentConfig(BaseModel):
    """Graph behaviour."""

    #: Bounded self-correction. One repair pass is the sweet spot measured in
    #: docs/RESULTS.md; more passes cost latency without improving support.
    max_repair_rounds: int = 1
    #: Verification is what makes the citations mean something. Off only for
    #: ablation.
    verify_claims: bool = True
    #: Drop claims the verifier cannot ground, rather than emitting them hedged.
    strip_unsupported: bool = True
    #: Statistic tools available to the compute node.
    enable_cohort_stats: bool = True
    enable_disproportionality: bool = True


class SafetyConfig(BaseModel):
    """Triage policy."""

    enabled: bool = True
    #: Refuse individualised clinical requests outright rather than hedging.
    refuse_personal_medical_advice: bool = True
    #: Route explicit crisis language to a dedicated response.
    crisis_routing: bool = True
    #: Even a permitted answer carries the standing disclaimer.
    always_disclaim: bool = True


class EvalConfig(BaseModel):
    """Evaluation protocol."""

    n_queries: int = 300
    seed: int = 7
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    #: Red-team probe count per hazard category.
    redteam_per_category: int = 6
    #: Bootstrap resamples for confidence intervals on every reported metric.
    bootstrap_n: int = 1000
    bootstrap_alpha: float = 0.05


# --------------------------------------------------------------------------- #
# Root configuration
# --------------------------------------------------------------------------- #
class PharosConfig(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="PHAROS_",
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    run_name: str = "default"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #
    def fingerprint(self) -> str:
        """A short, stable hash of the settings that affect artifacts.

        Cosmetic fields (``run_name``, absolute paths) are excluded so that the
        same pipeline run on two machines produces the same fingerprint.
        """
        payload = self.model_dump(mode="json", exclude={"run_name", "paths"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def ensure_dirs(self) -> None:
        for path in (
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.index_dir,
            self.paths.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> PharosConfig:
    """Load configuration from YAML, environment, and keyword overrides.

    Args:
        path: A YAML file layered on top of ``configs/default.yaml``. Ablation
            configs are thin overlays that change one knob each.
        **overrides: Dotted or nested keyword overrides applied last, e.g.
            ``load_config(retrieval={"stratify": False})``.
    """
    merged: dict[str, Any] = {}

    if DEFAULT_CONFIG_PATH.exists():
        merged = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}

    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        merged = _deep_merge(merged, yaml.safe_load(p.read_text()) or {})

    if overrides:
        merged = _deep_merge(merged, overrides)

    cfg = PharosConfig(**merged)
    cfg.ensure_dirs()
    return cfg
