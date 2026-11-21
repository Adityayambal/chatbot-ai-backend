"""Turns a parsed ContentPackage into retrievable Chunks.

One chunking strategy per content type (revision notes, video transcripts, practice
questions, optional eBook PDFs), all producing the same Chunk shape so the retriever
doesn't need to know where a chunk came from.
"""
from __future__ import annotations

import hashlib
import io
import logging
import math
import re
from dataclasses import asdict, dataclass, field

import httpx

from app.content.cleaner import ImageRef, clean_html, clean_transcript, normalize_name
from app.content.language import pick_lang
from app.content.models import ContentPackage, SubConcept

logger = logging.getLogger(__name__)

NOTE_CHUNK_MAX_CHARS = 1800
TRANSCRIPT_WINDOW_CHARS = 1600
MIN_PDF_TEXT_CHARS = 200

__all__ = ["Chunk", "ImageRef", "SubConceptPosition", "compute_positions", "build_chunks"]


@dataclass
class Chunk:
    chunk_id: str
    cp_id: str
    chapter_id: str
    chapter_name: str
    chapter_pos: int
    concept_id: str
    concept_name: str
    concept_pos: int
    subconcept_id: str
    subconcept_name: str
    subconcept_pos: int
    topic_code: str
    content_type: str  # "revision_note" | "video_transcript" | "question" | "pdf"
    title: str
    text: str
    part: int
    total_parts: int
    images: list[ImageRef] = field(default_factory=list)
    video_url: str | None = None
    video_id: str = ""
    duration_sec: int | None = None
    question_id: str | None = None
    difficulty: int | None = None
    source_content_id: str = ""
    updated_at: str = ""
    est_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        d = dict(d)
        d["images"] = [ImageRef(**img) for img in d.get("images", [])]
        return cls(**d)


@dataclass
class SubConceptPosition:
    chapter_id: str
    chapter_name: str
    chapter_pos: int
    concept_id: str
    concept_name: str
    concept_pos: int
    subconcept_id: str
    subconcept_name: str
    subconcept_pos: int
    concept_code: str
    topic_code: str
    sub_concept: SubConcept


def compute_positions(cp: ContentPackage) -> list[SubConceptPosition]:
    """Sort chapters/concepts/sub-concepts by sequence and assign stable 1-based codes."""
    positions: list[SubConceptPosition] = []
    chapters_sorted = sorted(cp.chapters, key=lambda c: c.chapter_sequence)
    for ch_pos, chapter in enumerate(chapters_sorted, start=1):
        concepts_sorted = sorted(chapter.concepts, key=lambda c: c.concept_sequence)
        for co_pos, concept in enumerate(concepts_sorted, start=1):
            subs_sorted = sorted(concept.sub_concepts, key=lambda s: s.sub_concept_sequence)
            concept_code = f"{ch_pos}.{co_pos}"
            for sub_pos, sub in enumerate(subs_sorted, start=1):
                positions.append(
                    SubConceptPosition(
                        chapter_id=chapter.chapter_id,
                        chapter_name=normalize_name(chapter.chapter_name),
                        chapter_pos=ch_pos,
                        concept_id=concept.concept_id,
                        concept_name=normalize_name(concept.concept_name),
                        concept_pos=co_pos,
                        subconcept_id=sub.sub_concept_id,
                        subconcept_name=normalize_name(sub.sub_concept_name),
                        subconcept_pos=sub_pos,
                        concept_code=concept_code,
                        topic_code=f"{concept_code}.{sub_pos}",
                        sub_concept=sub,
                    )
                )
    return positions


def _chunk_id(cp_id: str, source_id: str, part: int) -> str:
    raw = f"{cp_id}:{source_id}:{part}".encode()
    return hashlib.sha1(raw).hexdigest()


def _est_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5) if text else 0


_LIST_MARKER_RE = re.compile(r"^(\s*-\s|\s*\d+\.\s)")


def _is_protected_block(block: str) -> bool:
    stripped = block.strip()
    if stripped.startswith("|"):
        return True
    if _LIST_MARKER_RE.match(stripped):
        return True
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return True
    return False


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_paragraph_by_budget(block: str, max_chars: int) -> list[str]:
    sentences = _split_sentences(block)
    if not sentences:
        return [block]
    pieces: list[str] = []
    current = ""
    for s in sentences:
        candidate = f"{current} {s}".strip() if current else s
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = s
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _pack_blocks(blocks: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
        current = []
        current_len = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if not _is_protected_block(block) and len(block) > max_chars:
            for piece in _split_paragraph_by_budget(block, max_chars):
                piece_len = len(piece) + 2
                if current and current_len + piece_len > max_chars:
                    flush()
                current.append(piece)
                current_len += piece_len
            continue
        block_len = len(block) + 2
        if current and current_len + block_len > max_chars:
            flush()
        current.append(block)
        current_len += block_len
    flush()
    return chunks


def _build_note_chunks(cp_id: str, pos: SubConceptPosition, lang_key: str) -> list[Chunk]:
    content = pos.sub_concept.content
    chunks: list[Chunk] = []
    breadcrumb = f"{pos.chapter_name} › {pos.concept_name} › {pos.subconcept_name}"
    for note in content.revision_notes:
        title = pick_lang(note.title, lang_key) or pos.subconcept_name
        all_blocks: list[str] = []
        all_images: list[ImageRef] = []
        for note_text in note.note_text:
            html = pick_lang(note_text.content, lang_key)
            if not html:
                continue
            result = clean_html(html)
            if result.text:
                all_blocks.extend(result.text.split("\n\n"))
            all_images.extend(result.images)
        if not all_blocks:
            continue
        packed = _pack_blocks(all_blocks, NOTE_CHUNK_MAX_CHARS)
        total_parts = len(packed)
        for i, body in enumerate(packed, start=1):
            prefixed = f"{breadcrumb} — Revision note (part {i}/{total_parts})\n\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(cp_id, note.content_id, i),
                    cp_id=cp_id,
                    chapter_id=pos.chapter_id,
                    chapter_name=pos.chapter_name,
                    chapter_pos=pos.chapter_pos,
                    concept_id=pos.concept_id,
                    concept_name=pos.concept_name,
                    concept_pos=pos.concept_pos,
                    subconcept_id=pos.subconcept_id,
                    subconcept_name=pos.subconcept_name,
                    subconcept_pos=pos.subconcept_pos,
                    topic_code=pos.topic_code,
                    content_type="revision_note",
                    title=title,
                    text=prefixed,
                    part=i,
                    total_parts=total_parts,
                    images=all_images if i == 1 else [],
                    source_content_id=note.content_id,
                    updated_at=note.updated_at,
                    est_tokens=_est_tokens(prefixed),
                )
            )
    return chunks


def _vimeo_url(video_urls: list) -> str | None:
    for vu in video_urls:
        if vu.content_type == "Video":
            url = vu.content.get("en") or next(iter(vu.content.values()), "")
            if url:
                return url
    return None


def _build_transcript_chunks(cp_id: str, pos: SubConceptPosition, lang_key: str) -> list[Chunk]:
    content = pos.sub_concept.content
    chunks: list[Chunk] = []
    breadcrumb = f"{pos.chapter_name} › {pos.concept_name} › {pos.subconcept_name}"
    for vt in content.video_transcripts:
        raw_text = vt.transcript_text
        if not raw_text:
            continue
        cleaned = clean_transcript(raw_text)
        if not cleaned:
            continue
        windows = _split_transcript_windows(cleaned, TRANSCRIPT_WINDOW_CHARS)
        title = pick_lang(vt.title, lang_key) or pos.subconcept_name
        video_url = _vimeo_url(vt.video_url)
        total_parts = len(windows)
        for i, body in enumerate(windows, start=1):
            prefixed = f"{breadcrumb} — Video lesson: {title} (part {i}/{total_parts})\n\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(cp_id, vt.content_id, i),
                    cp_id=cp_id,
                    chapter_id=pos.chapter_id,
                    chapter_name=pos.chapter_name,
                    chapter_pos=pos.chapter_pos,
                    concept_id=pos.concept_id,
                    concept_name=pos.concept_name,
                    concept_pos=pos.concept_pos,
                    subconcept_id=pos.subconcept_id,
                    subconcept_name=pos.subconcept_name,
                    subconcept_pos=pos.subconcept_pos,
                    topic_code=pos.topic_code,
                    content_type="video_transcript",
                    title=title,
                    text=prefixed,
                    part=i,
                    total_parts=total_parts,
                    video_url=video_url,
                    video_id=vt.video_id,
                    duration_sec=vt.duration or None,
                    source_content_id=vt.content_id,
                    updated_at=vt.updated_at,
                    est_tokens=_est_tokens(prefixed),
                )
            )
    return chunks


def _split_transcript_windows(text: str, window: int) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    windows: list[str] = []
    current = [sentences[0]]
    current_len = len(sentences[0])
    for s in sentences[1:]:
        if current_len + 1 + len(s) > window:
            windows.append(" ".join(current))
            overlap = current[-1]
            current = [overlap, s]
            current_len = len(overlap) + 1 + len(s)
        else:
            current.append(s)
            current_len += 1 + len(s)
    windows.append(" ".join(current))
    return windows


@dataclass
class _RawQuestion:
    pos: SubConceptPosition
    question_id: str
    difficulty: int | None
    updated_at: str
    q_text: str
    sol_text: str


def _collect_raw_questions(positions: list[SubConceptPosition], lang_key: str) -> list[_RawQuestion]:
    raw: list[_RawQuestion] = []
    for pos in positions:
        for q in pos.sub_concept.content.questions:
            q_html = pick_lang(q.question_text, lang_key)
            sol_html = pick_lang(q.solution_description, lang_key)
            q_text = clean_html(q_html).text if q_html else ""
            sol_text = clean_html(sol_html).text if sol_html else ""
            if not q_text:
                continue
            raw.append(
                _RawQuestion(
                    pos=pos,
                    question_id=q.question_id,
                    difficulty=q.difficulty_level or None,
                    updated_at=q.updated_at,
                    q_text=q_text,
                    sol_text=sol_text,
                )
            )
    return raw


def _dedupe_raw_questions(raw: list[_RawQuestion]) -> list[_RawQuestion]:
    """Dedupe on normalised question text only, keeping the longer explanation."""
    best: dict[str, _RawQuestion] = {}
    order: list[str] = []
    for r in raw:
        key = normalize_name(r.q_text).lower()
        existing = best.get(key)
        if existing is None:
            best[key] = r
            order.append(key)
        elif len(r.sol_text) > len(existing.sol_text):
            best[key] = r
    return [best[k] for k in order]


def _build_question_chunk(cp_id: str, r: _RawQuestion) -> Chunk:
    pos = r.pos
    breadcrumb = f"{pos.chapter_name} › {pos.concept_name} › {pos.subconcept_name}"
    body = f"Q: {r.q_text} (options not available)\nExplanation: {r.sol_text}"
    prefixed = f"{breadcrumb} — Practice question\n\n{body}"
    return Chunk(
        chunk_id=_chunk_id(cp_id, r.question_id, 1),
        cp_id=cp_id,
        chapter_id=pos.chapter_id,
        chapter_name=pos.chapter_name,
        chapter_pos=pos.chapter_pos,
        concept_id=pos.concept_id,
        concept_name=pos.concept_name,
        concept_pos=pos.concept_pos,
        subconcept_id=pos.subconcept_id,
        subconcept_name=pos.subconcept_name,
        subconcept_pos=pos.subconcept_pos,
        topic_code=pos.topic_code,
        content_type="question",
        title=pos.subconcept_name,
        text=prefixed,
        part=1,
        total_parts=1,
        question_id=r.question_id,
        difficulty=r.difficulty,
        source_content_id=r.question_id,
        updated_at=r.updated_at,
        est_tokens=_est_tokens(prefixed),
    )


async def _fetch_pdf_text(url: str) -> str:
    from pypdf import PdfReader

    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


async def build_chunks(
    cp: ContentPackage, lang_key: str, *, ingest_ebook_pdfs: bool = False
) -> list[Chunk]:
    positions = compute_positions(cp)
    note_chunks: list[Chunk] = []
    transcript_chunks: list[Chunk] = []

    for pos in positions:
        note_chunks.extend(_build_note_chunks(cp.cp_id, pos, lang_key))
        transcript_chunks.extend(_build_transcript_chunks(cp.cp_id, pos, lang_key))

    raw_questions = _collect_raw_questions(positions, lang_key)
    deduped_questions = _dedupe_raw_questions(raw_questions)
    question_chunks = [_build_question_chunk(cp.cp_id, r) for r in deduped_questions]

    pdf_chunks: list[Chunk] = []
    if ingest_ebook_pdfs:
        seen_chapters: dict[str, SubConceptPosition] = {}
        for pos in positions:
            seen_chapters.setdefault(pos.chapter_id, pos)
        for chapter_id, pos in seen_chapters.items():
            ebook_pdf = next(
                (c.ebook_pdf for c in cp.chapters if c.chapter_id == chapter_id and c.ebook_pdf),
                "",
            )
            if not ebook_pdf:
                continue
            try:
                text = await _fetch_pdf_text(ebook_pdf)
            except Exception as exc:  # noqa: BLE001 - best-effort optional ingestion
                logger.warning("failed to fetch/parse eBook PDF for chapter %s: %s", chapter_id, exc)
                continue
            if len(text.strip()) < MIN_PDF_TEXT_CHARS:
                logger.info("eBook PDF for chapter %s yielded too little text, skipping", chapter_id)
                continue
            blocks = [b for b in text.split("\n\n") if b.strip()]
            packed = _pack_blocks(blocks, NOTE_CHUNK_MAX_CHARS)
            total_parts = len(packed)
            for i, body in enumerate(packed, start=1):
                prefixed = f"{pos.chapter_name} — eBook (part {i}/{total_parts})\n\n{body}"
                pdf_chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(cp.cp_id, chapter_id, i),
                        cp_id=cp.cp_id,
                        chapter_id=pos.chapter_id,
                        chapter_name=pos.chapter_name,
                        chapter_pos=pos.chapter_pos,
                        concept_id="",
                        concept_name="",
                        concept_pos=0,
                        subconcept_id="",
                        subconcept_name="",
                        subconcept_pos=0,
                        topic_code=f"{pos.chapter_pos}",
                        content_type="pdf",
                        title=pos.chapter_name,
                        text=prefixed,
                        part=i,
                        total_parts=total_parts,
                        source_content_id=chapter_id,
                        est_tokens=_est_tokens(prefixed),
                    )
                )

    return note_chunks + transcript_chunks + question_chunks + pdf_chunks
