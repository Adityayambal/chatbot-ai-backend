"""Owns the per-plan content lifecycle: fetch -> cache -> parse -> chunk -> index.

One KnowledgeBase per process, shared across requests. Each plan's CPIndex is swapped
atomically on refresh so in-flight requests never see a half-built index.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field

import orjson

from app.config import Settings
from app.content.cache import BundleCache, IndexCache, sha256_hex
from app.content.chunker import Chunk, SubConceptPosition, build_chunks, compute_positions
from app.content.cleaner import normalize_name
from app.content.client import ContentBundleClient, ContentBundleError
from app.content.language import medium_to_lang_key
from app.content.models import BundleResponse, ContentPackage, RetrievalConfiguration
from app.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)

PlanStatus = str  # "loading" | "ready" | "stale" | "error"


@dataclass
class CPMeta:
    cp_id: str
    cp_name: str
    plan_id: str
    plan_name: str
    board_id: str
    board_name: str
    grade_id: str
    grade_name: str
    medium_id: str
    medium_name: str
    subject_id: str
    subject_name: str
    is_active: bool
    chapter_count: int
    concept_count: int
    subconcept_count: int


@dataclass
class CPIndex:
    meta: CPMeta
    cp: ContentPackage
    positions: list[SubConceptPosition]
    outline_text: str
    code_to_subconcept_id: dict[str, str]
    concept_code_to_subconcept_codes: dict[str, list[str]]
    subconcept_id_to_chunk_ids: dict[str, list[str]]
    chunks: list[Chunk]
    bm25: BM25Index
    retrieval_config: RetrievalConfiguration
    sibling_cp_indices: list["CPIndex"] = field(default_factory=list)


def build_outline(
    cp: ContentPackage, positions: list[SubConceptPosition]
) -> tuple[str, dict[str, str], dict[str, list[str]]]:
    """Deterministic, byte-identical-per-bundle outline text plus code lookup maps."""
    lines: list[str] = []
    code_to_subconcept_id: dict[str, str] = {}
    concept_code_to_subconcept_codes: dict[str, list[str]] = {}

    for _chapter_pos, chapter_group_iter in itertools.groupby(positions, key=lambda p: p.chapter_pos):
        chapter_group = list(chapter_group_iter)
        first = chapter_group[0]
        lines.append(f"{first.chapter_pos} {first.chapter_name}")
        for _concept_pos, concept_group_iter in itertools.groupby(chapter_group, key=lambda p: p.concept_pos):
            concept_group = list(concept_group_iter)
            concept_first = concept_group[0]
            concept_code = concept_first.concept_code
            concept_code_to_subconcept_codes[concept_code] = [p.topic_code for p in concept_group]
            for p in concept_group:
                code_to_subconcept_id[p.topic_code] = p.subconcept_id

            collapse = len(concept_group) == 1 and (
                normalize_name(concept_group[0].subconcept_name).lower()
                == normalize_name(concept_first.concept_name).lower()
            )
            if collapse:
                p = concept_group[0]
                lines.append(f"  {p.topic_code} {p.subconcept_name}")
            else:
                lines.append(f"  {concept_code} {concept_first.concept_name}")
                for p in concept_group:
                    lines.append(f"    {p.topic_code} {p.subconcept_name}")

    return "\n".join(lines), code_to_subconcept_id, concept_code_to_subconcept_codes


class KnowledgeBase:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bundle_client = ContentBundleClient(settings.content_bundle_url, settings.content_bundle_token)
        self._bundle_cache = BundleCache(settings.data_dir)
        self._index_cache = IndexCache(settings.data_dir)
        self._plans: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._refresh_tasks: dict[str, asyncio.Task] = {}

    def _lock_for(self, plan_id: str) -> asyncio.Lock:
        if plan_id not in self._locks:
            self._locks[plan_id] = asyncio.Lock()
        return self._locks[plan_id]

    async def ensure_loaded(self, plan_ids: list[str]) -> None:
        for plan_id in plan_ids:
            await self._load_plan(plan_id)

    def ensure_loading_started(self, plan_id: str) -> None:
        """Fire-and-forget load for a plan_id the frontend asked for that we haven't seen yet."""
        if plan_id in self._plans:
            return
        self._plans[plan_id] = {"status": "loading"}
        asyncio.create_task(self._load_plan(plan_id))

    def maybe_schedule_refresh(self, plan_id: str) -> None:
        """Cheap TTL check; starts a background refresh if the cache has expired."""
        if plan_id not in self._plans or self._plans[plan_id]["status"] not in ("ready", "stale"):
            return
        if self._bundle_cache.is_fresh(plan_id, self._settings.bundle_cache_ttl_minutes):
            return
        existing_task = self._refresh_tasks.get(plan_id)
        if existing_task is not None and not existing_task.done():
            return
        self._refresh_tasks[plan_id] = asyncio.create_task(self._load_plan(plan_id))

    async def reload(self, plan_ids: list[str]) -> None:
        """Forced refresh (admin endpoint), bypassing the TTL."""
        for plan_id in plan_ids:
            await self._load_plan(plan_id, force=True)

    async def _load_plan(self, plan_id: str, *, force: bool = False) -> None:
        async with self._lock_for(plan_id):
            try:
                raw, used_stale_fallback = await self._get_bundle_bytes(plan_id, force=force)
                bundle = BundleResponse.model_validate(orjson.loads(raw))
                cp_indices = await self._build_cp_indices(plan_id, raw, bundle)
                self._plans[plan_id] = {
                    "status": "stale" if used_stale_fallback else "ready",
                    "cp_indices": cp_indices,
                    "fetched_at": time.time(),
                }
                if used_stale_fallback:
                    logger.warning(
                        "plan %s: content API unavailable, served stale cache (%d content package(s))",
                        plan_id,
                        len(cp_indices),
                    )
                else:
                    logger.info("plan %s ready: %d content package(s)", plan_id, len(cp_indices))
            except Exception as exc:  # noqa: BLE001 - convert any failure into plan status
                existing = self._plans.get(plan_id, {})
                if existing.get("cp_indices"):
                    existing["status"] = "stale"
                    logger.warning("plan %s refresh failed, serving stale cache: %s", plan_id, exc)
                else:
                    self._plans[plan_id] = {"status": "error", "error": str(exc)}
                    logger.error("plan %s failed to load: %s", plan_id, exc)

    async def _get_bundle_bytes(self, plan_id: str, *, force: bool = False) -> tuple[bytes, bool]:
        """Returns (bytes, used_stale_fallback). used_stale_fallback is True only when a
        live fetch was attempted and failed, and we fell back to the on-disk cache."""
        if not force and self._bundle_cache.is_fresh(plan_id, self._settings.bundle_cache_ttl_minutes):
            cached = self._bundle_cache.read(plan_id)
            if cached is not None:
                return cached, False
        try:
            data = await self._bundle_client.fetch([plan_id])
            self._bundle_cache.write(plan_id, data)
            return data, False
        except ContentBundleError:
            cached = self._bundle_cache.read(plan_id)
            if cached is not None:
                logger.warning("content API unavailable for plan %s; serving stale cache", plan_id)
                return cached, True
            raise

    async def _build_cp_indices(
        self, plan_id: str, raw_bytes: bytes, bundle: BundleResponse
    ) -> dict[str, CPIndex]:
        bundle_sha = sha256_hex(raw_bytes)
        manifest = self._index_cache.read_manifest(plan_id)
        reuse = manifest is not None and manifest.get("sha256") == bundle_sha

        cp_indices: dict[str, CPIndex] = {}
        counts: dict[str, int] = {}
        for cp in bundle.result:
            if not cp.is_active and not self._settings.include_inactive_cps:
                continue
            lang_key = medium_to_lang_key(cp.medium_name)
            raw_chunks = self._index_cache.read_chunks(plan_id, cp.cp_id) if reuse else None
            if raw_chunks is not None:
                chunks = [Chunk.from_dict(d) for d in raw_chunks]
            else:
                chunks = await build_chunks(
                    cp, lang_key, ingest_ebook_pdfs=self._settings.ingest_ebook_pdfs
                )
                self._index_cache.write_chunks(plan_id, cp.cp_id, [c.to_dict() for c in chunks])
            cp_indices[cp.cp_id] = self._build_cp_index(cp, chunks)
            counts[cp.cp_id] = len(chunks)

        all_indices = list(cp_indices.values())
        for idx in all_indices:
            idx.sibling_cp_indices = [other for other in all_indices if other is not idx]

        if not reuse:
            self._index_cache.write_manifest(
                plan_id, {"sha256": bundle_sha, "builtAt": time.time(), "counts": counts}
            )
        return cp_indices

    def _build_cp_index(self, cp: ContentPackage, chunks: list[Chunk]) -> CPIndex:
        positions = compute_positions(cp)
        outline_text, code_to_subconcept_id, concept_code_to_subconcept_codes = build_outline(cp, positions)

        subconcept_id_to_chunk_ids: dict[str, list[str]] = {}
        for c in chunks:
            subconcept_id_to_chunk_ids.setdefault(c.subconcept_id, []).append(c.chunk_id)

        meta = CPMeta(
            cp_id=cp.cp_id,
            cp_name=cp.cp_name,
            plan_id=cp.plan_id,
            plan_name=cp.plan_name,
            board_id=cp.board_id,
            board_name=cp.board_name,
            grade_id=cp.grade_id,
            grade_name=cp.grade_name,
            medium_id=cp.medium_id,
            medium_name=cp.medium_name,
            subject_id=cp.subject_id,
            subject_name=cp.subject_name,
            is_active=cp.is_active,
            chapter_count=len(cp.chapters),
            concept_count=sum(len(ch.concepts) for ch in cp.chapters),
            subconcept_count=len(positions),
        )
        return CPIndex(
            meta=meta,
            cp=cp,
            positions=positions,
            outline_text=outline_text,
            code_to_subconcept_id=code_to_subconcept_id,
            concept_code_to_subconcept_codes=concept_code_to_subconcept_codes,
            subconcept_id_to_chunk_ids=subconcept_id_to_chunk_ids,
            chunks=chunks,
            bm25=BM25Index(chunks),
            retrieval_config=cp.retrieval_configuration,
        )

    def get_plan_status(self, plan_id: str) -> PlanStatus:
        return self._plans.get(plan_id, {}).get("status", "loading")

    def get_plan(self, plan_id: str) -> dict[str, CPIndex] | None:
        state = self._plans.get(plan_id)
        if state is None or state["status"] not in ("ready", "stale"):
            return None
        return state["cp_indices"]

    def known_plan_ids(self) -> set[str]:
        return set(self._plans.keys())

    def get_fetched_at(self, plan_id: str) -> float | None:
        return self._plans.get(plan_id, {}).get("fetched_at")

    def health_snapshot(self) -> dict[str, str]:
        return {plan_id: state.get("status", "loading") for plan_id, state in self._plans.items()}
