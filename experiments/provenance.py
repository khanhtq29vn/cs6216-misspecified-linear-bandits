"""Deterministic provenance for the CS6216 numerical experiments.

A Git revision alone cannot identify uncommitted code or an unpacked source ZIP.
Results therefore record a SHA-256 digest of the exact experiment and model
sources, plus an explicit seed protocol. This is an integrity fingerprint, not a
substitute for retaining code and logs alongside published results.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 2
SEED_PROTOCOL = "paired-parameter-sweeps-v2"
ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "src/algorithms.py",
    "src/environments.py",
    "src/utils.py",
    "experiments/run_experiments.py",
    "experiments/run_main_only.py",
    "experiments/plot_results.py",
    "experiments/provenance.py",
    "requirements.txt",
)


def source_fingerprint() -> str:
    """Hash the source bytes used for policy, environment and experiment outputs."""
    digest = hashlib.sha256()
    for filename in _SOURCE_FILES:
        path = ROOT / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_revision() -> tuple[str, str]:
    """Return commit SHA and dirty state, or explicit unknowns outside a clone."""
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
        return revision, "dirty" if dirty else "clean"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "source-archive-without-git", "unknown"


def provenance_fields() -> dict[str, np.ndarray]:
    revision, dirty = git_revision()
    return {
        "meta_schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
        "meta_seed_protocol": np.asarray([SEED_PROTOCOL]),
        "meta_source_sha256": np.asarray([source_fingerprint()]),
        "meta_git_commit": np.asarray([revision]),
        "meta_git_worktree": np.asarray([dirty]),
    }


def verify_provenance(results: dict[str, np.ndarray], *, require_runtime: bool = False) -> None:
    """Fail closed on stale source/seed provenance (and optionally runtime versions)."""
    expected = provenance_fields()
    for key in ("meta_schema_version", "meta_seed_protocol", "meta_source_sha256"):
        if key not in results or not np.array_equal(results[key], expected[key]):
            raise ValueError(f"Archive is not compatible with current source and protocol: {key}")
    if require_runtime:
        import sys
        runtime = {
            "meta_python_version": sys.version.split()[0],
            "meta_numpy_version": np.__version__,
        }
        for key, value in runtime.items():
            if key not in results or list(results[key]) != [value]:
                raise ValueError(f"Archive was created under a different runtime: {key}")


def atomic_savez(path: Path, results: dict[str, np.ndarray]) -> None:
    """Write a result archive atomically so an interrupted run cannot corrupt it."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            np.savez_compressed(output, **results)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: Path, contents: str) -> None:
    """Atomically replace a UTF-8 text output after the write is complete."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
