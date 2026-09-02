"""Corpus cleaning, label repair, de-identification, and near-duplicate removal.

This module is where the project earns its claim to rigour. The Drugs.com review
corpus is widely used and, in our reading of the literature and of public
notebooks, almost universally used *raw*. It carries at least five defects that
change downstream numbers:

1. **HTML entity escaping.** 100,566 of the 161,297 training reviews — 62 % —
   contain numeric character references such as ``&#039;``. Left in place, every
   tokenizer sees a literal ``039`` token, and "can&#039;t" never matches
   "can't" in a lexical index. A further 6,436 rows carry ``&amp;``.
2. **Quotation wrapping.** All 215,063 reviews are wrapped in an extra pair of
   double quotes by the source export.
3. **Systematic label truncation.** See :mod:`configs/condition_repairs.yaml`;
   6.7 % of reviews are filed under a label with its final ``r`` deleted.
4. **Footer contamination.** 1,171 rows captured the review footer
   (``"3</span> users found this comment helpful."``) in the ``condition`` field.
5. **Cross-split duplication.** The published train/test split contains exact and
   near-duplicate reviews that straddle the boundary, which inflates any
   retrieval or classification metric computed across it.

Each is handled explicitly and each is *counted*, because a cleaning step whose
effect you cannot quantify is a cleaning step you cannot defend.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pharos.config import REPO_ROOT, PharosConfig

REPAIR_TABLE_PATH = REPO_ROOT / "configs" / "condition_repairs.yaml"

# --------------------------------------------------------------------------- #
# De-identification patterns
# --------------------------------------------------------------------------- #
# The corpus is public and already pseudonymous, but reviewers volunteer
# identifiers anyway — phone numbers in "call me at", email addresses, ages
# combined with a rare condition. We scrub the mechanically detectable classes
# before anything is indexed. This is defence in depth, not a compliance claim.
_DEID_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("URL", re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Long digit runs are never clinically meaningful here and are the usual
    # shape of a member ID or an order number.
    ("IDNUM", re.compile(r"\b\d{9,}\b")),
)

_WS = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([!?.,])\1{2,}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """A tally of every transformation applied, for the data card and the tests.

    Tests assert on these counts. If an upstream mirror of the corpus ever
    changes shape, the suite fails loudly rather than silently cleaning nothing.
    """

    rows_in: int = 0
    rows_out: int = 0
    entities_unescaped: int = 0
    quote_unwrapped: int = 0
    condition_missing: int = 0
    condition_artifact: int = 0
    condition_repaired_leading: int = 0
    condition_repaired_trailing: int = 0
    condition_unrepairable: int = 0
    deidentified_rows: int = 0
    deid_by_type: Counter[str] = field(default_factory=Counter)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    dropped_short: int = 0
    dropped_no_condition: int = 0
    dropped_low_support: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "deid_by_type"}
        d["deid_by_type"] = dict(self.deid_by_type)
        return d

    def summary_lines(self) -> list[str]:
        r = self
        pct = (100.0 * r.rows_out / r.rows_in) if r.rows_in else 0.0
        return [
            f"rows in .................. {r.rows_in:,}",
            f"rows out ................. {r.rows_out:,}  ({pct:.1f}% retained)",
            f"html entities decoded ... {r.entities_unescaped:,}",
            f"quote-unwrapped .......... {r.quote_unwrapped:,}",
            f"labels: missing .......... {r.condition_missing:,}",
            f"labels: footer artifact .. {r.condition_artifact:,}",
            f"labels: leading repair ... {r.condition_repaired_leading:,}",
            f"labels: trailing repair .. {r.condition_repaired_trailing:,}",
            f"labels: unrepairable ..... {r.condition_unrepairable:,}",
            f"de-identified rows ....... {r.deidentified_rows:,}  {dict(r.deid_by_type)}",
            f"exact duplicates ......... {r.exact_duplicates:,}",
            f"near duplicates .......... {r.near_duplicates:,}",
            f"dropped (too short) ...... {r.dropped_short:,}",
            f"dropped (no condition) ... {r.dropped_no_condition:,}",
            f"dropped (low support) .... {r.dropped_low_support:,}",
        ]


# --------------------------------------------------------------------------- #
# Label repair
# --------------------------------------------------------------------------- #
class ConditionRepairer:
    """Applies the documented repair table to condition labels."""

    def __init__(self, table_path: Path | None = None) -> None:
        path = table_path or REPAIR_TABLE_PATH
        table = yaml.safe_load(path.read_text()) if path.exists() else {}
        self.artifact_re = re.compile(
            table.get("artifact_pattern", r"^\s*\d+\s*</span>.*helpful\.?\s*$")
        )
        self.leading: dict[str, str] = table.get("leading", {}) or {}
        self.trailing: dict[str, str] = table.get("trailing", {}) or {}
        self.whitelist: set[str] = set(table.get("whitelist", []) or [])
        self.unrepairable: set[str] = set(table.get("unrepairable", []) or [])

    def repair(self, label: str | None) -> tuple[str | None, str]:
        """Return ``(repaired_label, status)``.

        Status is one of ``ok``, ``missing``, ``artifact``, ``leading``,
        ``trailing``, ``leading+trailing``, ``unrepairable``.
        """
        if label is None or (isinstance(label, float) and pd.isna(label)):
            return None, "missing"

        text = str(label).strip()
        if not text:
            return None, "missing"
        if self.artifact_re.match(text):
            return None, "artifact"
        if text in self.whitelist:
            return text, "ok"
        if text in self.unrepairable:
            return None, "unrepairable"

        tags: list[str] = []

        # The leading table stores fully-repaired values, so that an auditor
        # reading the YAML sees the destination rather than an intermediate. The
        # raw form is still inspected for a trailing deletion, so the status
        # reports both defects when both were present -- "zen Shoulde" lost a
        # leading "Fro" *and* a trailing "r", and saying only "leading" would
        # understate what the row had wrong with it.
        had_trailing_defect = any(text.endswith(suffix) for suffix in self.trailing)

        if text in self.leading:
            text = self.leading[text]
            tags.append("leading")
            if had_trailing_defect:
                tags.append("trailing")
            return text, "+".join(tags)

        for suffix, replacement in self.trailing.items():
            if text.endswith(suffix):
                text = text[: -len(suffix)] + replacement
                tags.append("trailing")
                break

        return text, "+".join(tags) if tags else "ok"

    def audit(self, labels: pd.Series) -> pd.DataFrame:
        """Per-label forensic report, used by ``pharos audit-labels``."""
        counts = labels.fillna("<missing>").astype(str).value_counts()
        rows = []
        for raw, n in counts.items():
            repaired, status = self.repair(None if raw == "<missing>" else raw)
            rows.append({"raw": raw, "n": int(n), "repaired": repaired, "status": status})
        return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
def unescape_fully(text: str, max_rounds: int = 3) -> tuple[str, bool]:
    """Unescape HTML entities to a fixed point.

    Iterating rather than calling ``html.unescape`` once is defensive: a
    re-scraped or re-exported mirror of this corpus may well be double-escaped
    (``&amp;#039;``), and a single pass would leave ``&#039;`` behind for the
    tokenizer to read as the token ``039``. Bounded at ``max_rounds`` so that
    adversarial input cannot spin here.

    Returns ``(text, changed)``.
    """
    rounds = 0
    current = text
    while rounds < max_rounds:
        nxt = html.unescape(current)
        if nxt == current:
            break
        current = nxt
        rounds += 1
    return current, rounds > 0


def strip_export_quotes(text: str) -> tuple[str, bool]:
    """Remove the wrapping quotation marks added by the source export."""
    stripped = text.strip()
    changed = False
    while len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        stripped = stripped[1:-1].strip()
        changed = True
    return stripped, changed


def deidentify(text: str) -> tuple[str, Counter[str]]:
    """Replace mechanically detectable identifiers with typed placeholders."""
    found: Counter[str] = Counter()
    out = text
    for tag, pattern in _DEID_PATTERNS:
        out, n = pattern.subn(f"[{tag}]", out)
        if n:
            found[tag] += n
    return out, found


def normalise_text(text: str) -> str:
    """Unicode-normalise, collapse whitespace, and tame runaway punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _REPEAT_PUNCT.sub(r"\1\1", text)
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Near-duplicate detection
# --------------------------------------------------------------------------- #
_SHINGLE_K = 5
_MERSENNE_61 = (1 << 61) - 1


def _shingles(text: str, k: int = _SHINGLE_K) -> set[str]:
    t = re.sub(r"\W+", " ", text.lower()).strip()
    if len(t) <= k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def _stable_hash(shingle: str) -> int:
    """A stable 64-bit hash.

    Python's built-in ``hash`` is salted per-process, which would make
    de-duplication — and therefore the corpus, and therefore every reported
    number — differ between runs. blake2b is deterministic across machines and
    interpreter versions.
    """
    return int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "little")


def shingle_hashes(text: str, k: int = _SHINGLE_K) -> set[int]:
    """Hashed shingle set.

    Hashes are retained instead of the shingle strings themselves. On the full
    215,063-review corpus the string form holds roughly 90 million short Python
    objects — enough to exhaust an 8 GB machine before de-duplication finishes.
    The 64-bit hashes carry the same Jaccard structure at a fraction of the cost,
    and collisions at this scale are negligible against a 0.92 threshold.
    """
    return {_stable_hash(s) for s in _shingles(text, k)}


def minhash_signature(shingles: set[int], coeffs: np.ndarray) -> np.ndarray:
    """Vectorised MinHash: hash each shingle once, then permute arithmetically.

    The naive formulation hashes every shingle once per permutation, which for a
    450-shingle review and 64 permutations is 28,800 cryptographic hashes — and
    720 million across a 25k-review corpus. Hashing once and applying affine
    permutations ``(a·h + b) mod p`` in NumPy turns that into 450 hashes plus one
    small matrix operation, which is the difference between a de-duplication pass
    that takes minutes and one that takes seconds.
    """
    if not shingles:
        return np.zeros(coeffs.shape[1], dtype=np.uint64)
    h = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    a, b = coeffs[0], coeffs[1]
    # Work in int64-safe space modulo a Mersenne prime to avoid overflow.
    hh = (h % _MERSENNE_61).astype(np.int64)
    perm = (np.multiply.outer(hh, a.astype(np.int64)) + b.astype(np.int64)) % _MERSENNE_61
    return perm.min(axis=0).astype(np.uint64)


def _dedup_block(
    texts: list[str],
    offsets: list[int],
    threshold: float,
    bands: int,
    num_perm: int,
    coeffs: np.ndarray,
    max_bucket: int,
) -> set[int]:
    """LSH + exact-Jaccard de-duplication within one block. Returns global indices."""
    import numpy as np

    shingle_sets = [shingle_hashes(t) for t in texts]
    sigs = np.stack([minhash_signature(s, coeffs) for s in shingle_sets])
    rows_per_band = max(1, num_perm // bands)

    buckets: dict[tuple[int, bytes], list[int]] = {}
    for b in range(bands):
        chunk = sigs[:, b * rows_per_band : (b + 1) * rows_per_band]
        for idx in range(len(texts)):
            buckets.setdefault((b, chunk[idx].tobytes()), []).append(idx)

    drop: set[int] = set()
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        # Oversized buckets are hash degeneracies (near-empty reviews colliding),
        # not real duplicate groups; verifying them is quadratic and unproductive.
        if len(members) < 2 or len(members) > max_bucket:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a_, b_ = members[i], members[j]
                if b_ in drop or (a_, b_) in checked:
                    continue
                checked.add((a_, b_))
                sa, sb = shingle_sets[a_], shingle_sets[b_]
                union = len(sa | sb)
                if union and len(sa & sb) / union >= threshold:
                    drop.add(max(a_, b_))
    return {offsets[i] for i in drop}


def find_near_duplicates(
    texts: list[str],
    threshold: float = 0.92,
    bands: int = 16,
    num_perm: int = 64,
    seed: int = 0,
    max_bucket: int = 256,
    blocks: list[str] | None = None,
) -> set[int]:
    """Return the indices to drop, keeping the first member of each group.

    Locality-sensitive hashing over MinHash signatures: a signature is split into
    ``bands`` bands and any pair colliding in a band becomes a candidate.
    Candidates are then verified with *exact* Jaccard on the hashed shingle sets,
    so ``threshold`` means what it says rather than being an LSH approximation.

    Args:
        blocks: Optional blocking key per text — in practice the drug name.
            Comparison is confined to within-block pairs. This is not only a
            memory bound (the full corpus does not fit in one pass on a laptop);
            it is also a correctness improvement, because two reviews of
            *different* drugs that happen to share phrasing are two genuine
            reports and collapsing them would understate a denominator.
    """
    if not texts:
        return set()

    rng = np.random.default_rng(seed)
    coeffs = np.stack(
        [
            rng.integers(1, _MERSENNE_61, size=num_perm, dtype=np.int64),
            rng.integers(0, _MERSENNE_61, size=num_perm, dtype=np.int64),
        ]
    )

    if blocks is None:
        grouped: dict[str, list[int]] = {"__all__": list(range(len(texts)))}
    else:
        grouped = {}
        for i, key in enumerate(blocks):
            grouped.setdefault(key, []).append(i)

    drop: set[int] = set()
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        drop |= _dedup_block(
            [texts[i] for i in indices],
            indices,
            threshold,
            bands,
            num_perm,
            coeffs,
            max_bucket,
        )
    return drop


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_raw(cfg: PharosConfig) -> pd.DataFrame:
    """Load and concatenate the published train/test TSVs."""
    frames = []
    for split, filename in (("train", "drugsComTrain_raw.tsv"), ("test", "drugsComTest_raw.tsv")):
        path = cfg.paths.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `pharos fetch-data` (or place the Kaggle "
                f"download there) before building the corpus."
            )
        df = pd.read_csv(path, sep="\t", quotechar='"', index_col=0)
        df.index.name = "review_id"
        df = df.reset_index()
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_corpus(
    df: pd.DataFrame, cfg: PharosConfig, repairer: ConditionRepairer | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the full cleaning pipeline and return the corpus plus its report."""
    repairer = repairer or ConditionRepairer()
    report = CleaningReport(rows_in=len(df))
    out = df.copy()

    # --- text ------------------------------------------------------------- #
    texts, deid_flags = [], []
    n_entities, n_quote, n_deid = 0, 0, 0
    for raw in out["review"].astype(str):
        text, had_entities = unescape_fully(raw)
        n_entities += int(had_entities)
        text, unquoted = strip_export_quotes(text)
        n_quote += int(unquoted)
        text = normalise_text(text)
        text, found = deidentify(text)
        if found:
            n_deid += 1
            report.deid_by_type.update(found)
        texts.append(text)
        deid_flags.append(bool(found))

    out["text"] = texts
    out["deidentified"] = deid_flags
    report.entities_unescaped = n_entities
    report.quote_unwrapped = n_quote
    report.deidentified_rows = n_deid

    # --- labels ----------------------------------------------------------- #
    repaired, statuses = [], []
    for label in out["condition"]:
        new, status = repairer.repair(label)
        repaired.append(new)
        statuses.append(status)
    out["condition"] = repaired
    out["label_status"] = statuses

    status_counts = Counter(statuses)
    report.condition_missing = status_counts["missing"]
    report.condition_artifact = status_counts["artifact"]
    report.condition_unrepairable = status_counts["unrepairable"]
    report.condition_repaired_leading = sum(n for s, n in status_counts.items() if "leading" in s)
    report.condition_repaired_trailing = sum(n for s, n in status_counts.items() if "trailing" in s)
    out["condition_repaired"] = out["label_status"].isin(
        {"leading", "trailing", "leading+trailing"}
    )

    # --- structural filters ------------------------------------------------ #
    before = len(out)
    out = out[out["text"].str.len() >= cfg.data.min_unit_chars]
    report.dropped_short = before - len(out)

    if cfg.data.drop_unlabelled_conditions:
        before = len(out)
        out = out[out["condition"].notna()]
        report.dropped_no_condition = before - len(out)

    # Conditions too rare to support an interval are excluded from the corpus
    # entirely; keeping them would mean shipping statistics we would have to
    # caveat into meaninglessness.
    if cfg.data.min_condition_support > 1:
        before = len(out)
        counts = out["condition"].value_counts()
        keep = set(counts[counts >= cfg.data.min_condition_support].index)
        out = out[out["condition"].isin(keep)]
        report.dropped_low_support = before - len(out)

    # --- de-duplication ---------------------------------------------------- #
    # Identity is (drug, text): the same sentence written about two different
    # drugs is two independent reports, and collapsing them would understate
    # every denominator downstream.
    before = len(out)
    out = out.drop_duplicates(subset=["drugName", "text"], keep="first")
    report.exact_duplicates = before - len(out)

    if cfg.data.dedup_enabled:
        before = len(out)
        out = out.reset_index(drop=True)
        drop_idx = find_near_duplicates(
            out["text"].tolist(),
            threshold=cfg.data.dedup_threshold,
            seed=cfg.data.seed,
            blocks=out["drugName"].astype(str).tolist(),
        )
        out = out.drop(index=list(drop_idx))
        report.near_duplicates = before - len(out)

    # --- typing ------------------------------------------------------------ #
    out = out.rename(columns={"drugName": "drug_name", "usefulCount": "useful_count"})
    out["review_date"] = pd.to_datetime(out["date"], format="%B %d, %Y").dt.date
    out["rating"] = out["rating"].astype(float)
    out["useful_count"] = out["useful_count"].astype(int)
    out["drug_name"] = out["drug_name"].astype(str).str.strip()

    keep_cols = [
        "review_id",
        "drug_name",
        "condition",
        "text",
        "rating",
        "review_date",
        "useful_count",
        "split",
        "condition_repaired",
        "deidentified",
        "label_status",
    ]
    out = out[keep_cols].reset_index(drop=True)
    report.rows_out = len(out)
    return out, report


def subsample(df: pd.DataFrame, cfg: PharosConfig) -> pd.DataFrame:
    """Deterministic, condition-stratified subsample.

    Plain random sampling would let the 38,436-review Birth Control cohort
    dominate a 25k draw and starve the long tail. We instead allocate slots by
    the square root of each condition's support — a compromise that keeps the
    head representative while giving small cohorts enough rows to compute an
    interval over.
    """
    n = cfg.data.sample_size
    if n is None or n >= len(df):
        return df.reset_index(drop=True)

    counts = df["condition"].value_counts()
    weights = counts.pow(0.5)
    quotas = (weights / weights.sum() * n).round().astype(int).clip(lower=1)

    rng_seed = cfg.data.seed
    parts = []
    for condition, quota in quotas.items():
        pool = df[df["condition"] == condition]
        take = min(int(quota), len(pool))
        parts.append(pool.sample(n=take, random_state=rng_seed))
    sampled = pd.concat(parts).sample(frac=1.0, random_state=rng_seed)
    return sampled.head(n).sort_values("review_id").reset_index(drop=True)
