"""Pure-Python retrieval: BM25 fusion + weighting + dedupe + diversification + budget.

No network and no LLM calls here — this whole module runs in a few milliseconds.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import Settings
from app.content.chunker import Chunk
from app.content.cleaner import normalize_name
from app.content.models import RetrievalConfiguration
from app.retrieval.tokenize import tokenize

if TYPE_CHECKING:
    from app.content.knowledge_base import CPIndex
    from app.llm.schemas import RouterDecision

_DEFAULT_WEIGHTS: dict[str, float] = {
    "revision_note": 0.9,
    "video_transcript": 0.7,
    "question": 0.5,
    "solution": 0.4,
    "pdf": 1.0,
}

_TYPE_ORDER = {"revision_note": 0, "pdf": 0, "video_transcript": 1, "question": 2}

_RRF_K = 60
_LIST_A_TOP_N = 200
_LIST_B_TOP_N = 40
_CROSS_SUBJECT_PENALTY = 0.5
_JACCARD_DEDUPE_THRESHOLD = 0.85
_MAX_CHUNKS_PER_SUBCONCEPT = 4
_MAX_QUESTION_CHUNKS = 5


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source_id: str = ""


def weight_lookup(config: RetrievalConfiguration):
    overrides = {p.content_type: p.weight for p in config.content_priority_order if p.content_type}

    def weight_for(content_type: str) -> float:
        return overrides.get(content_type, _DEFAULT_WEIGHTS.get(content_type, 1.0))

    return weight_for


def fuse(
    list_a: list[tuple[Chunk, float]],
    list_b: list[tuple[Chunk, float]],
    weight_for,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion across the two ranked lists, then content-type weighting."""
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for lst in (list_a, list_b):
        for rank, (chunk, _raw_score) in enumerate(lst, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            chunk_by_id[chunk.chunk_id] = chunk

    results = [
        RetrievedChunk(chunk=chunk_by_id[cid], score=rrf * weight_for(chunk_by_id[cid].content_type))
        for cid, rrf in rrf_scores.items()
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def dedupe(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Exact normalised-text hash, plus token-set Jaccard >= 0.85 against already-selected."""
    selected: list[RetrievedChunk] = []
    selected_token_sets: list[set[str]] = []
    seen_hashes: set[str] = set()
    for r in results:
        norm = normalize_name(r.chunk.text).lower()
        text_hash = hashlib.sha1(norm.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        token_set = set(tokenize(r.chunk.text))
        is_near_dupe = any(
            _jaccard(token_set, existing) >= _JACCARD_DEDUPE_THRESHOLD for existing in selected_token_sets
        )
        if is_near_dupe:
            continue
        seen_hashes.add(text_hash)
        selected.append(r)
        selected_token_sets.append(token_set)
    return selected


def diversify(results: list[RetrievedChunk], enable: bool) -> list[RetrievedChunk]:
    """Best note + best transcript + 2 best questions first, then fill by score,
    capped at 4 chunks/sub-concept and 5 question chunks overall."""
    if not enable:
        return results

    ordered = sorted(results, key=lambda r: r.score, reverse=True)
    chosen: list[RetrievedChunk] = []
    chosen_ids: set[str] = set()

    def pick_best(content_type: str) -> RetrievedChunk | None:
        for r in ordered:
            if r.chunk.chunk_id in chosen_ids:
                continue
            if r.chunk.content_type == content_type and r.score > 0:
                return r
        return None

    best_note = pick_best("revision_note")
    if best_note:
        chosen.append(best_note)
        chosen_ids.add(best_note.chunk.chunk_id)

    best_transcript = pick_best("video_transcript")
    if best_transcript:
        chosen.append(best_transcript)
        chosen_ids.add(best_transcript.chunk.chunk_id)

    question_count = 0
    for r in ordered:
        if question_count >= 2:
            break
        if r.chunk.chunk_id in chosen_ids:
            continue
        if r.chunk.content_type == "question" and r.score > 0:
            chosen.append(r)
            chosen_ids.add(r.chunk.chunk_id)
            question_count += 1

    subconcept_counts: dict[str, int] = {}
    for r in chosen:
        subconcept_counts[r.chunk.subconcept_id] = subconcept_counts.get(r.chunk.subconcept_id, 0) + 1

    for r in ordered:
        if r.chunk.chunk_id in chosen_ids:
            continue
        if subconcept_counts.get(r.chunk.subconcept_id, 0) >= _MAX_CHUNKS_PER_SUBCONCEPT:
            continue
        if r.chunk.content_type == "question" and question_count >= _MAX_QUESTION_CHUNKS:
            continue
        chosen.append(r)
        chosen_ids.add(r.chunk.chunk_id)
        subconcept_counts[r.chunk.subconcept_id] = subconcept_counts.get(r.chunk.subconcept_id, 0) + 1
        if r.chunk.content_type == "question":
            question_count += 1

    return chosen


def apply_budget(results: list[RetrievedChunk], top_k: int, max_context_tokens: int) -> list[RetrievedChunk]:
    chosen: list[RetrievedChunk] = []
    total_tokens = 0
    for r in results:
        if len(chosen) >= top_k:
            break
        if chosen and total_tokens + r.chunk.est_tokens > max_context_tokens:
            break
        chosen.append(r)
        total_tokens += r.chunk.est_tokens
    return chosen


def _apply_threshold(results: list[RetrievedChunk], min_similarity: float) -> list[RetrievedChunk]:
    if min_similarity <= 0 or not results:
        return results
    max_score = max(r.score for r in results)
    if max_score <= 0:
        return results
    return [r for r in results if (r.score / max_score) >= min_similarity]


def _order_for_prompt(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    ordered = sorted(
        results,
        key=lambda r: (
            r.chunk.chapter_pos,
            r.chunk.concept_pos,
            r.chunk.subconcept_pos,
            _TYPE_ORDER.get(r.chunk.content_type, 3),
            r.chunk.part,
        ),
    )
    for i, r in enumerate(ordered, start=1):
        r.source_id = f"S{i}"
    return ordered


def _expand_topic_codes(cp_index: "CPIndex", codes: list[str]) -> set[str] | None:
    if not codes:
        return None
    subconcept_ids: set[str] = set()
    for code in codes:
        sub_id = cp_index.code_to_subconcept_id.get(code)
        if sub_id:
            subconcept_ids.add(sub_id)
            continue
        for sub_code in cp_index.concept_code_to_subconcept_codes.get(code, []):
            sub_id = cp_index.code_to_subconcept_id.get(sub_code)
            if sub_id:
                subconcept_ids.add(sub_id)
    if not subconcept_ids:
        return None
    chunk_ids: set[str] = set()
    for sub_id in subconcept_ids:
        chunk_ids.update(cp_index.subconcept_id_to_chunk_ids.get(sub_id, []))
    return chunk_ids or None


def _build_query(standalone_question: str, search_queries: list[str]) -> str:
    return (standalone_question + " " + " ".join(search_queries)).strip()


def retrieve(cp_index: "CPIndex", decision: "RouterDecision", settings: Settings) -> list[RetrievedChunk]:
    query = _build_query(decision.standalone_question, decision.search_queries)
    query_tokens = tokenize(query)
    config = cp_index.retrieval_config
    weight_for = weight_lookup(config)

    topic_chunk_ids = _expand_topic_codes(cp_index, decision.matched_topic_codes)
    list_a = (
        cp_index.bm25.search(query_tokens, top_n=_LIST_A_TOP_N, restrict_to=topic_chunk_ids)
        if topic_chunk_ids
        else []
    )
    list_b = cp_index.bm25.search(query_tokens, top_n=_LIST_B_TOP_N, restrict_to=None)

    candidates = fuse(list_a, list_b, weight_for)

    if config.allow_cross_subject_search:
        for sibling in getattr(cp_index, "sibling_cp_indices", []):
            sib_list_b = sibling.bm25.search(query_tokens, top_n=_LIST_B_TOP_N, restrict_to=None)
            sib_weight_for = weight_lookup(sibling.retrieval_config)
            sib_candidates = fuse(sib_list_b, [], sib_weight_for)
            for r in sib_candidates:
                r.score *= _CROSS_SUBJECT_PENALTY
            candidates.extend(sib_candidates)
        candidates.sort(key=lambda r: r.score, reverse=True)

    candidates = _apply_threshold(candidates, config.minimum_similarity_threshold)

    if config.deduplicate_results:
        candidates = dedupe(candidates)

    candidates = diversify(candidates, config.enable_source_diversification)

    top_k = config.top_k if config.top_k > 0 else settings.retrieval_top_k
    candidates = apply_budget(candidates, top_k, settings.max_context_tokens)

    return _order_for_prompt(candidates)
