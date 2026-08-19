"""Run manifests for auditable paper reproduction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(directory: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def dependency_versions(
    names: Iterable[str] = (
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "peft",
        "accelerate",
    ),
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(
    *,
    command: list[str],
    seed: int,
    config: Mapping[str, Any],
    input_paths: Iterable[str | os.PathLike[str]] = (),
    model_revisions: Mapping[str, str | None] | None = None,
    workspace: str | os.PathLike[str] = ".",
) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for raw_path in input_paths:
        path = Path(raw_path).resolve()
        paths[str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "seed": int(seed),
        "config": dict(config),
        "inputs": paths,
        "models": dict(model_revisions or {}),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": _git_revision(Path(workspace).resolve()),
    }


def write_manifest(manifest: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def set_reproducible_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

