"""
Automatic model and tokenizer downloader.

Files are cached in ``~/.cache/eka_ai/`` and only downloaded once.
Downloads use ``gdown`` for Google Drive hosted files with progress bars.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

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

#: Clean HTTPS hosting URLs for model weights (e.g. GitHub Releases / HuggingFace).
#: Users can also override these by setting the EKA_MODEL_URL and EKA_TOKENIZER_URL env vars.
CLEAN_URLS: dict[str, str] = {
    "eka_model.pt": "https://github.com/subhash9705/eka-ai/releases/download/v1.0.2/eka_model.pt",
    "tokenizer.model": "https://github.com/subhash9705/eka-ai/releases/download/v1.0.2/tokenizer.model",
}

#: Expected SHA-256 digests (first 16 hex chars) for integrity checking.
#: Set to None to skip verification for a specific file.
EXPECTED_SHA256_PREFIX: dict[str, str | None] = {
    "eka_model.pt": None,  # populate after upload
    "tokenizer.model": None,  # populate after upload
}


# ── Download helpers ──────────────────────────────────────────────────────────


def _download_url(url: str, dest_path: Path) -> None:
    """Download a file from a generic HTTPS URL with a custom progress bar."""
    temp_path = dest_path.with_suffix(".tmp")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 8
            downloaded = 0
            
            with open(temp_path, "wb") as f:
                t0 = time.perf_counter()
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        speed = downloaded / (time.perf_counter() - t0 + 1e-9) / (1024 * 1024)
                        sys.stdout.write(
                            f"\rDownloading... {percent:.1f}% | "
                            f"{downloaded / (1024 * 1024):.1f}M/{total_size / (1024 * 1024):.1f}M | "
                            f"{speed:.1f} MB/s"
                        )
                        sys.stdout.flush()
                print()
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed from {url}: {e}") from e


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


def _verify(path: Path, expected: str | None) -> bool:
    """Return True if the file passes the integrity check (or check is skipped)."""
    if expected is None:
        return True
    actual = _sha256_prefix(path)
    return actual == expected


def download_file(
    filename: str,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download a single EKA model file. Tries to download from clean public URLs
    first, and falls back to a quiet Google Drive download if needed.

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
        raise ValueError(f"Unknown file '{filename}'. Known files: {list(GDRIVE_IDS.keys())}")

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

    tmp_path = out_path.with_suffix(".tmp")
    
    # 1. Resolve clean download URL (either env override or default CLEAN_URL)
    env_var = "EKA_MODEL_URL" if filename == "eka_model.pt" else "EKA_TOKENIZER_URL"
    url = os.environ.get(env_var) or CLEAN_URLS.get(filename)
    
    download_success = False
    
    if url:
        print(f"[EKA] Downloading {filename} from {url} …", flush=True)
        try:
            _download_url(url, out_path)
            download_success = True
        except Exception as e:
            print(f"[EKA] Public URL download failed, falling back to Google Drive: {e}", flush=True)

    # 2. Fallback to Google Drive using gdown in quiet mode (does not expose drive redirect links)
    if not download_success:
        _check_gdown()
        import gdown  # type: ignore[import]

        file_id = GDRIVE_IDS[filename]
        print(f"[EKA] Downloading {filename} from Google Drive (quiet mode) …", flush=True)
        
        result = gdown.download(id=file_id, output=str(tmp_path), quiet=True)
        
        if not result or not tmp_path.exists():
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"Download of '{filename}' failed. "
                "Check your internet connection and that the Google Drive link is public."
            )

    # Integrity check on the freshly downloaded file
    target_path = out_path if download_success else tmp_path
    expected = EXPECTED_SHA256_PREFIX.get(filename)
    if not _verify(target_path, expected):
        target_path.unlink()
        raise RuntimeError(
            f"Integrity check failed for '{filename}'. "
            "The downloaded file may be corrupt or the expected hash is stale."
        )

    if not download_success:
        tmp_path.rename(out_path)

    print(f"[EKA] Saved {filename} -> {out_path}", flush=True)
    return out_path


def download_all(
    dest: Path | None = None,
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
