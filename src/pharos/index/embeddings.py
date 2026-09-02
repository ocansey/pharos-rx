"""Encoders.

Three implementations behind one interface, chosen by config.

``lsa`` — TF-IDF followed by truncated SVD, fitted on the corpus itself. This is
the default, and the choice deserves defending rather than apologising for. It
downloads nothing, runs on any machine in seconds, is deterministic to the last
bit given a seed, and — because it is fitted on this corpus — carries the
domain's own vocabulary structure rather than a general-web prior. Every number
in ``docs/RESULTS.md`` is reproducible from a clean clone because of it. On a
corpus of short, lexically dense, highly repetitive clinical narrative, latent
semantic indexing is a genuinely competitive baseline, not a toy.

``sentence-transformers`` — a neural bi-encoder, for users who want it and have
the bandwidth. The interface is identical, so switching is a one-line config
change and the ablation runner will happily produce a second results table.

``hashing`` — a signed random projection of character n-grams. Fast and
dependency-free, used only to keep the test suite from fitting an SVD.

All encoders return L2-normalised rows, so inner product *is* cosine similarity
and the retrieval code never has to ask which it is looking at.
"""

from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from pharos.config import IndexConfig


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class Encoder(ABC):
    """Text -> dense matrix."""

    dim: int

    @abstractmethod
    def fit(self, corpus: list[str]) -> Encoder: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    @property
    def name(self) -> str:
        return type(self).__name__

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> Encoder:
        with path.open("rb") as fh:
            return pickle.load(fh)


class LSAEncoder(Encoder):
    """TF-IDF + truncated SVD, fitted on the corpus.

    Sublinear term frequency is on: a review that says "pain" nine times is not
    nine times more about pain, and the raw count would let a single distressed
    narrator dominate a topic direction.
    """

    def __init__(self, cfg: IndexConfig, seed: int = 0) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.cfg = cfg
        self.dim = cfg.embedding_dim
        self.seed = seed
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=cfg.tfidf_min_df,
            max_df=cfg.tfidf_max_df,
            ngram_range=(1, cfg.tfidf_ngram_max),
            stop_words="english",
            dtype=np.float32,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> LSAEncoder:
        from sklearn.decomposition import TruncatedSVD

        tfidf = self._vectorizer.fit_transform(corpus)
        # SVD cannot ask for more components than the matrix has columns; a small
        # test corpus would otherwise crash here rather than degrade.
        n_components = int(min(self.dim, min(tfidf.shape) - 1))
        n_components = max(n_components, 2)
        self._svd = TruncatedSVD(
            n_components=n_components, random_state=self.seed, algorithm="randomized", n_iter=7
        )
        self._svd.fit(tfidf)
        self.dim = n_components
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self._svd is None:
            raise RuntimeError("LSAEncoder.encode called before fit")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        reduced = self._svd.transform(self._vectorizer.transform(texts)).astype(np.float32)
        return _l2_normalise(reduced) if self.cfg.normalize else reduced

    @property
    def explained_variance(self) -> float:
        """Fraction of TF-IDF variance retained. Reported in the data card."""
        if self._svd is None:
            return 0.0
        return float(self._svd.explained_variance_ratio_.sum())

    @property
    def name(self) -> str:
        return f"lsa-{self.dim}d"


class SentenceTransformerEncoder(Encoder):
    """A neural bi-encoder. Requires ``pip install 'pharos-rx[encoders]'``."""

    def __init__(self, cfg: IndexConfig) -> None:
        self.cfg = cfg
        self.model_name = cfg.st_model
        self._model = None
        self.dim = cfg.embedding_dim

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "encoder='sentence-transformers' requires the optional extra:\n"
                    "    pip install 'pharos-rx[encoders]'\n"
                    "Or keep the default encoder='lsa', which needs no download."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def fit(self, corpus: list[str]) -> SentenceTransformerEncoder:
        self._ensure_model()  # a bi-encoder is pre-trained; "fit" only warms it
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = model.encode(
            texts,
            batch_size=self.cfg.st_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def __getstate__(self):
        # The model is many hundreds of megabytes and is reconstructible from
        # its name; pickling it would make every saved index enormous.
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    @property
    def name(self) -> str:
        return self.model_name


class HashingEncoder(Encoder):
    """Signed random projection of character 4-grams. Deterministic, no fitting."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def fit(self, corpus: list[str]) -> HashingEncoder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            t = text.lower()
            for i in range(max(1, len(t) - 3)):
                gram = t[i : i + 4]
                h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _l2_normalise(out)

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}d"


def build_encoder(cfg: IndexConfig, seed: int = 0) -> Encoder:
    """Instantiate the encoder named in the config."""
    if cfg.encoder == "lsa":
        return LSAEncoder(cfg, seed=seed)
    if cfg.encoder == "sentence-transformers":
        return SentenceTransformerEncoder(cfg)
    if cfg.encoder == "hashing":
        return HashingEncoder(dim=min(cfg.embedding_dim, 256))
    raise ValueError(f"unknown encoder: {cfg.encoder}")
