"""Corpus acquisition.

The dataset is the Drugs.com patient review corpus published by Gräßer, Kallumadi,
Malberg and Zaunseder (2018), deposited at the UCI Machine Learning Repository and
mirrored on Kaggle. 215,063 reviews, 3,671 drugs, 916 condition labels,
February 2008 to December 2017.

Three sources are tried in order, because a portfolio repository that stops
working when one host changes a URL is a portfolio repository that stops working.
Whatever the source, the files are verified against recorded SHA-256 digests and
row counts before anything downstream trusts them — a silently truncated download
would otherwise surface as an inexplicable drop in a metric, three stages later.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from pharos.config import PharosConfig

FILES = ("drugsComTrain_raw.tsv", "drugsComTest_raw.tsv")

#: Digests of the files this project was built and evaluated against.
EXPECTED_SHA256: dict[str, str] = {
    "drugsComTrain_raw.tsv": "83992773f0f42b20b8c9798767e0f2ce5cf3428c721ee56176d65573d76455e4",
    "drugsComTest_raw.tsv": "1953e81451b3a43f162e3d438dc1f9c2ee83c9283968a19fce4d3736fdb03eac",
}

#: Row counts from the original publication. Checked even when a digest does not
#: match, because a *different but complete* mirror is usable and should be
#: allowed through with a warning; a truncated one must not be.
EXPECTED_ROWS: dict[str, int] = {
    "drugsComTrain_raw.tsv": 161_297,
    "drugsComTest_raw.tsv": 53_766,
}

#: Direct TSV mirrors, tried first.
MIRRORS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/semnan-university-ai/Drug-Review-Dataset/main/{name}",
)

#: The canonical UCI archive, as a zip containing both files.
UCI_ZIP = "https://archive.ics.uci.edu/static/public/462/drug+review+dataset+drugs+com.zip"

KAGGLE_URL = "https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018"

ProgressFn = Callable[[str], None]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    """Count data rows, honouring quoted newlines inside review text.

    A naive line count is wrong here by tens of thousands: review bodies contain
    embedded newlines inside quoted fields, and every one of them would be
    counted as a record.
    """
    import csv

    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader, None)
        return sum(1 for _ in reader)


def _download(url: str, destination: Path, progress: ProgressFn) -> bool:
    try:
        progress(f"fetching {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "pharos-rx/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            destination.write_bytes(response.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        progress(f"  failed: {exc}")
        destination.unlink(missing_ok=True)
        return False


def _download_uci_zip(raw_dir: Path, progress: ProgressFn) -> bool:
    try:
        progress(f"fetching {UCI_ZIP}")
        request = urllib.request.Request(UCI_ZIP, headers={"User-Agent": "pharos-rx/1.0"})
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                base = Path(member).name
                if base in FILES:
                    (raw_dir / base).write_bytes(archive.read(member))
        return all((raw_dir / name).exists() for name in FILES)
    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as exc:
        progress(f"  failed: {exc}")
        return False


def verify(path: Path, name: str, progress: ProgressFn) -> bool:
    """Verify a downloaded file. Returns False only when it is unusable."""
    digest = sha256_of(path)
    expected_digest = EXPECTED_SHA256.get(name)
    rows = count_rows(path)
    expected_rows = EXPECTED_ROWS.get(name)

    if expected_rows is not None and rows != expected_rows:
        progress(f"  {name}: {rows:,} rows, expected {expected_rows:,} — file is incomplete")
        return False

    if expected_digest and digest != expected_digest:
        progress(
            f"  {name}: {rows:,} rows (correct) but digest {digest[:12]}… differs from "
            f"the recorded {expected_digest[:12]}…. A different mirror of the same data; "
            f"results should still reproduce, but note the difference."
        )
    else:
        progress(f"  {name}: {rows:,} rows, digest verified")
    return True


def fetch_corpus(cfg: PharosConfig, force: bool = False, progress: ProgressFn = print) -> Path:
    """Download and verify the corpus. Returns the raw directory."""
    raw_dir = Path(cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    present = [name for name in FILES if (raw_dir / name).exists()]
    if len(present) == len(FILES) and not force:
        progress("corpus already present; verifying")
        if all(verify(raw_dir / name, name, progress) for name in FILES):
            return raw_dir
        progress("verification failed; re-downloading")

    for template in MIRRORS:
        ok = all(_download(template.format(name=name), raw_dir / name, progress) for name in FILES)
        if ok and all(verify(raw_dir / name, name, progress) for name in FILES):
            return raw_dir

    if _download_uci_zip(raw_dir, progress) and all(
        verify(raw_dir / name, name, progress) for name in FILES
    ):
        return raw_dir

    raise RuntimeError(
        "Could not download the corpus automatically.\n\n"
        f"Download it manually from {KAGGLE_URL} (or the UCI ML Repository, dataset 462) "
        f"and place these files in {raw_dir}:\n"
        + "\n".join(f"  - {name}  ({EXPECTED_ROWS[name]:,} rows)" for name in FILES)
    )
