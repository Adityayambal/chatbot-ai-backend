#!/usr/bin/env python3
"""Prints per-CP/per-type counts and sizes from the cached content bundle, and times
ingestion (parse -> chunk). No LLM calls. Run from backend/:

    python scripts/inspect_bundle.py --plan 6b00418010a08d28bb37e089
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orjson

from app.config import get_settings
from app.content.cache import BundleCache
from app.content.chunker import build_chunks, compute_positions
from app.content.language import medium_to_lang_key
from app.content.models import BundleResponse


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Plan ID to inspect (must already be cached)")
    args = parser.parse_args()

    settings = get_settings()
    cache = BundleCache(settings.data_dir)
    raw = cache.read(args.plan)
    if raw is None:
        print(f"No cached bundle for plan {args.plan}.")
        print("Start the app once (uvicorn app.main:app) so it fetches and caches the bundle, then retry.")
        sys.exit(1)

    t0 = time.perf_counter()
    bundle = BundleResponse.model_validate(orjson.loads(raw))
    parse_s = time.perf_counter() - t0
    print(
        f"Bundle size: {len(raw):,} bytes. Parsed in {parse_s:.2f}s. "
        f"Content packages: {len(bundle.result)}"
    )

    total_elapsed = 0.0
    for cp in bundle.result:
        t1 = time.perf_counter()
        lang_key = medium_to_lang_key(cp.medium_name)
        positions = compute_positions(cp)

        raw_notes = raw_transcripts = raw_questions = 0
        note_chars = transcript_chars = question_chars = 0
        for pos in positions:
            content = pos.sub_concept.content
            raw_notes += len(content.revision_notes)
            raw_transcripts += len(content.video_transcripts)
            raw_questions += len(content.questions)
            for note in content.revision_notes:
                for note_text in note.note_text:
                    text = note_text.content.get(lang_key) or note_text.content.get("en", "")
                    note_chars += len(text)
            for vt in content.video_transcripts:
                transcript_chars += len(vt.transcript_text)
            for q in content.questions:
                text = q.question_text.get(lang_key) or q.question_text.get("en", "")
                question_chars += len(text)

        chunks = await build_chunks(cp, lang_key, ingest_ebook_pdfs=settings.ingest_ebook_pdfs)
        elapsed = time.perf_counter() - t1
        total_elapsed += elapsed

        by_type: dict[str, int] = {}
        for c in chunks:
            by_type[c.content_type] = by_type.get(c.content_type, 0) + 1

        print(f"\nCP {cp.cp_id} — {cp.cp_name}")
        print(
            f"  {cp.medium_name} | Grade {cp.grade_name} | {cp.board_name} | {cp.subject_name} | "
            f"active={cp.is_active}"
        )
        print(
            f"  chapters={len(cp.chapters)} concepts={sum(len(ch.concepts) for ch in cp.chapters)} "
            f"subconcepts={len(positions)}"
        )
        print(
            f"  raw: notes={raw_notes} ({note_chars:,} chars)  "
            f"transcripts={raw_transcripts} ({transcript_chars:,} chars)  "
            f"questions={raw_questions} ({question_chars:,} chars)"
        )
        print(
            f"  chunks after cleaning/chunking/dedupe: {dict(sorted(by_type.items()))}  "
            f"total={len(chunks)}"
        )
        print(f"  ingestion time: {elapsed:.2f}s")

    print(f"\nTotal ingestion time across all CPs: {total_elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
