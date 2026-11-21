"""Disk cache for the raw content bundle (per plan) and the processed chunk index (per CP)."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import orjson


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BundleCache:
    """Raw bundle bytes on disk: data/cache/bundles/{planId}.json (+ .meta.json)."""

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir) / "cache" / "bundles"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _bundle_path(self, plan_id: str) -> Path:
        return self._dir / f"{plan_id}.json"

    def _meta_path(self, plan_id: str) -> Path:
        return self._dir / f"{plan_id}.meta.json"

    def read(self, plan_id: str) -> bytes | None:
        path = self._bundle_path(plan_id)
        if not path.exists():
            return None
        return path.read_bytes()

    def read_meta(self, plan_id: str) -> dict[str, Any] | None:
        path = self._meta_path(plan_id)
        if not path.exists():
            return None
        return orjson.loads(path.read_bytes())

    def write(self, plan_id: str, data: bytes) -> dict[str, Any]:
        self._bundle_path(plan_id).write_bytes(data)
        meta = {
            "fetchedAt": time.time(),
            "sha256": sha256_hex(data),
            "sizeBytes": len(data),
        }
        self._meta_path(plan_id).write_bytes(orjson.dumps(meta))
        return meta

    def is_fresh(self, plan_id: str, ttl_minutes: int) -> bool:
        meta = self.read_meta(plan_id)
        if meta is None:
            return False
        age_seconds = time.time() - meta["fetchedAt"]
        return age_seconds < ttl_minutes * 60


class IndexCache:
    """Processed chunks on disk: data/index/{planId}/{cpId}.chunks.jsonl (+ manifest.json)."""

    def __init__(self, data_dir: str | Path) -> None:
        self._base = Path(data_dir) / "index"

    def _plan_dir(self, plan_id: str) -> Path:
        d = self._base / plan_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _manifest_path(self, plan_id: str) -> Path:
        return self._plan_dir(plan_id) / "manifest.json"

    def _chunks_path(self, plan_id: str, cp_id: str) -> Path:
        return self._plan_dir(plan_id) / f"{cp_id}.chunks.jsonl"

    def read_manifest(self, plan_id: str) -> dict[str, Any] | None:
        path = self._manifest_path(plan_id)
        if not path.exists():
            return None
        return orjson.loads(path.read_bytes())

    def write_manifest(self, plan_id: str, manifest: dict[str, Any]) -> None:
        self._manifest_path(plan_id).write_bytes(orjson.dumps(manifest))

    def read_chunks(self, plan_id: str, cp_id: str) -> list[dict[str, Any]] | None:
        path = self._chunks_path(plan_id, cp_id)
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        return [orjson.loads(line) for line in lines if line.strip()]

    def write_chunks(self, plan_id: str, cp_id: str, chunks: list[dict[str, Any]]) -> None:
        path = self._chunks_path(plan_id, cp_id)
        with path.open("wb") as f:
            for chunk in chunks:
                f.write(orjson.dumps(chunk))
                f.write(b"\n")
