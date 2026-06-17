"""
Automatic model and tokenizer downloader.

Files are cached in ``~/.cache/eka_ai/`` and only downloaded once.
Downloads use ``gdown`` for Google Drive hosted files with progress bars.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional


# ── Cache directory ───────────────────────────────────────────────────────────

def get_cache_dir() -> Path:
    """
    Return the local cache directory for EKA model files.

    Priority
    --------
    1. ``EKA_CACHE_DIR`` environment variable
    2. ``~/.cache/eka_ai``
    """
    env_dir = os.environ.get("EKA_CACHE_DIR")
    if env_dir:
        cache = Path(env_dir)
    else:
        cache = Path.home() / ".cache" / "eka_ai"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ── Google Drive file registry ────────────────────────────────────────────────

#: Google Drive file IDs for each artifact.
#: Update these when you push new checkpoints to Drive.
GDRIVE_IDS: dict[str, str] = {
    "eka_model.pt": "1tGwFyuoEAn7rVOVpQnRj2n_qBL9qGAqJ",
    "tokenizer.model": "1sgi5dOl2JXdBzMFTro-ojxxbFGe_IMTR",
}

#: Expected SHA-256 digests (first 16 hex chars) for integrity checking.
#: Set to None to skip verification for a specific file.
EXPECTED_SHA256_PREFIX: dict[str, Optional[str]] = {
    "eka_model.pt": None,       # populate after upload
    "tokenizer.model": None,    # populate after upload
}


# ── Download helpers ──────────────────────────────────────────────────────────

def _check_gdown() -> None:
    """Raise a helpful ImportError if gdown is not installed."""
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'gdown' package is required to download model files.\n"
            "Install it with:  pip install gdown"
        ) from exc


def _sha256_prefix(path: Path, n_chars: int = 16) -> str:
    """Return the first ``n_chars`` of the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]


def _verify(path: Path, expected: Optional[str]) -> bool:
    """Return True if the file passes the integrity check (or check is skipped)."""
    if expected is None:
        return True
    actual = _sha256_prefix(path)
    return actual == expected


def download_file(
    filename: str,
    dest: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download a single EKA model file from Google Drive.

    Parameters
    ----------
    filename : str
        One of ``"eka_model.pt"`` or ``"tokenizer.model"``.
    dest : Path, optional
        Directory to save the file. Defaults to the EKA cache directory.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Absolute path to the downloaded (or cached) file.

    Raises
    ------
    ValueError
        If ``filename`` is not in the registry.
    RuntimeError
        If the download fails.
    """
    if filename not in GDRIVE_IDS:
        raise ValueError(
            f"Unknown file '{filename}'. Known files: {list(GDRIVE_IDS.keys())}"
        )

    if dest is None:
        dest = get_cache_dir()

    out_path = dest / filename

    if out_path.exists() and not force:
        expected = EXPECTED_SHA256_PREFIX.get(filename)
        if _verify(out_path, expected):
            print(f"[EKA] Using cached {filename} -> {out_path}", flush=True)
            return out_path
        else:
            print(
                f"[EKA] Cached {filename} failed integrity check — re-downloading.",
                flush=True,
            )

    _check_gdown()
    import gdown  # type: ignore[import]

    file_id = GDRIVE_IDS[filename]
    print(f"[EKA] Downloading {filename} from Google Drive …", flush=True)

    tmp_path = out_path.with_suffix(".tmp")
    result = gdown.download(id=file_id, output=str(tmp_path), quiet=False)

    if not result or not tmp_path.exists():
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Download of '{filename}' failed. "
            "Check your internet connection and that the Google Drive link is public."
        )

    # Integrity check on the freshly downloaded file
    expected = EXPECTED_SHA256_PREFIX.get(filename)
    if not _verify(tmp_path, expected):
        tmp_path.unlink()
        raise RuntimeError(
            f"Integrity check failed for '{filename}'. "
            "The downloaded file may be corrupt or the expected hash is stale."
        )

    tmp_path.rename(out_path)
    print(f"[EKA] Saved {filename} -> {out_path}", flush=True)
    return out_path


def download_all(
    dest: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download both the model checkpoint and tokenizer.

    Parameters
    ----------
    dest : Path, optional
        Target directory. Defaults to ``~/.cache/eka_ai``.
    force : bool
        Re-download even if files already exist.

    Returns
    -------
    dict[str, Path]
        Mapping of filename → local path for each artifact.
    """
    paths: dict[str, Path] = {}
    for filename in GDRIVE_IDS:
        paths[filename] = download_file(filename, dest=dest, force=force)
    return paths


def get_model_path(auto_download: bool = True) -> Path:
    """
    Return the path to the model checkpoint, downloading it if necessary.

    Parameters
    ----------
    auto_download : bool
        If ``True`` (default) and the file is not cached, download it.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If ``auto_download=False`` and the file is not in the cache.
    """
    cache = get_cache_dir()
    path = cache / "eka_model.pt"
    if not path.exists():
        if auto_download:
            download_file("eka_model.pt", dest=cache)
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at '{path}'.\n"
                "Call EKA() or eka_ai.downloader.download_all() to download it."
            )
    return path


def get_tokenizer_path(auto_download: bool = True) -> Path:
    """
    Return the path to the tokenizer model, downloading it if necessary.

    Parameters
    ----------
    auto_download : bool
        If ``True`` (default) and the file is not cached, download it.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If ``auto_download=False`` and the file is not in the cache.
    """
    cache = get_cache_dir()
    path = cache / "tokenizer.model"
    if not path.exists():
        if auto_download:
            download_file("tokenizer.model", dest=cache)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at '{path}'.\n"
                "Call EKA() or eka_ai.downloader.download_all() to download it."
            )
    return path
