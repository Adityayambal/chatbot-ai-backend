import pytest

from app.content.chunker import Chunk
from app.content.models import ContentTypePriority, RetrievalConfiguration
from app.retrieval.retriever import (
    RetrievedChunk,
    _apply_threshold,
    apply_budget,
    dedupe,
    diversify,
    fuse,
    weight_lookup,
)


def make_chunk(
    chunk_id: str,
    content_type: str = "revision_note",
    text: str = "some lesson text about acceleration",
    subconcept_id: str = "sub-1",
    chapter_pos: int = 1,
    concept_pos: int = 1,
    subconcept_pos: int = 1,
    part: int = 1,
    est_tokens: int = 100,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        cp_id="cp-1",
        chapter_id="ch-1",
        chapter_name="Chapter",
        chapter_pos=chapter_pos,
        concept_id="co-1",
        concept_name="Concept",
        concept_pos=concept_pos,
        subconcept_id=subconcept_id,
        subconcept_name="Subconcept",
        subconcept_pos=subconcept_pos,
        topic_code=f"{chapter_pos}.{concept_pos}.{subconcept_pos}",
        content_type=content_type,
        title="Title",
        text=text,
        part=part,
        total_parts=1,
        est_tokens=est_tokens,
    )


def test_weight_lookup_defaults():
    config = RetrievalConfiguration()
    weight_for = weight_lookup(config)
    assert weight_for("revision_note") == 0.9
    assert weight_for("video_transcript") == 0.7
    assert weight_for("question") == 0.5
    assert weight_for("solution") == 0.4
    assert weight_for("pdf") == 1.0


def test_weight_lookup_uses_config_overrides():
    config = RetrievalConfiguration(
        content_priority_order=[ContentTypePriority(content_type="question", priority=1, weight=0.95)]
    )
    weight_for = weight_lookup(config)
    assert weight_for("question") == 0.95
    assert weight_for("revision_note") == 0.9  # falls back to default


def test_fuse_combines_ranks_with_rrf_and_weight():
    a = make_chunk("a", content_type="revision_note")
    b = make_chunk("b", content_type="question")
    list_a = [(a, 10.0), (b, 5.0)]
    list_b = [(b, 8.0)]

    def weight_for(ct: str) -> float:
        return {"revision_note": 0.9, "question": 0.5}[ct]

    results = fuse(list_a, list_b, weight_for)
    by_id = {r.chunk.chunk_id: r.score for r in results}

    expected_a = (1.0 / 61) * 0.9  # rank 1 in list_a only
    expected_b = (1.0 / 62 + 1.0 / 61) * 0.5  # rank 2 in list_a, rank 1 in list_b
    assert by_id["a"] == pytest.approx(expected_a)
    assert by_id["b"] == pytest.approx(expected_b)
    # sorted descending by score
    assert results[0].score >= results[1].score


def test_apply_threshold_drops_low_relative_scores():
    results = [RetrievedChunk(make_chunk("a"), score=1.0), RetrievedChunk(make_chunk("b"), score=0.1)]
    kept = _apply_threshold(results, min_similarity=0.5)
    assert [r.chunk.chunk_id for r in kept] == ["a"]


def test_apply_threshold_noop_when_disabled():
    results = [RetrievedChunk(make_chunk("a"), score=1.0), RetrievedChunk(make_chunk("b"), score=0.1)]
    kept = _apply_threshold(results, min_similarity=0.0)
    assert len(kept) == 2


def test_dedupe_drops_exact_text_duplicates():
    a = make_chunk("a", text="Acceleration is the rate of change of velocity.")
    b = make_chunk("b", text="Acceleration is the rate of change of velocity.")
    results = [RetrievedChunk(a, score=2.0), RetrievedChunk(b, score=1.0)]
    kept = dedupe(results)
    assert len(kept) == 1
    assert kept[0].chunk.chunk_id == "a"


def test_dedupe_drops_near_duplicates_by_jaccard():
    a = make_chunk("a", text="the rate of change of velocity is called acceleration in physics")
    b = make_chunk("b", text="the rate of change of velocity is called acceleration in physics today")
    results = [RetrievedChunk(a, score=2.0), RetrievedChunk(b, score=1.0)]
    kept = dedupe(results)
    assert len(kept) == 1


def test_dedupe_keeps_distinct_text():
    a = make_chunk("a", text="Newton's second law relates force, mass and acceleration.")
    b = make_chunk("b", text="Ohm's law relates voltage, current and resistance.")
    results = [RetrievedChunk(a, score=2.0), RetrievedChunk(b, score=1.0)]
    kept = dedupe(results)
    assert len(kept) == 2


def test_diversify_prioritises_note_transcript_and_two_questions():
    note = make_chunk("note", content_type="revision_note", subconcept_id="s1")
    transcript = make_chunk("transcript", content_type="video_transcript", subconcept_id="s1")
    q1 = make_chunk("q1", content_type="question", subconcept_id="s1")
    q2 = make_chunk("q2", content_type="question", subconcept_id="s2")
    q3 = make_chunk("q3", content_type="question", subconcept_id="s2")
    results = [
        RetrievedChunk(q3, score=5.0),
        RetrievedChunk(q2, score=4.0),
        RetrievedChunk(q1, score=3.0),
        RetrievedChunk(transcript, score=2.0),
        RetrievedChunk(note, score=1.0),
    ]
    ordered = diversify(results, enable=True)
    first_four_ids = [r.chunk.chunk_id for r in ordered[:4]]
    assert first_four_ids == ["note", "transcript", "q3", "q2"]


def test_diversify_noop_when_disabled():
    results = [RetrievedChunk(make_chunk("a"), score=1.0)]
    assert diversify(results, enable=False) == results


def test_diversify_caps_chunks_per_subconcept():
    chunks = [make_chunk(f"n{i}", content_type="revision_note", subconcept_id="s1") for i in range(6)]
    results = [RetrievedChunk(c, score=6.0 - i) for i, c in enumerate(chunks)]
    ordered = diversify(results, enable=True)
    assert len(ordered) == 4


def test_diversify_caps_question_chunks_overall():
    chunks = [make_chunk(f"q{i}", content_type="question", subconcept_id=f"s{i}") for i in range(8)]
    results = [RetrievedChunk(c, score=8.0 - i) for i, c in enumerate(chunks)]
    ordered = diversify(results, enable=True)
    assert sum(1 for r in ordered if r.chunk.content_type == "question") == 5


def test_apply_budget_respects_top_k():
    results = [RetrievedChunk(make_chunk(f"c{i}", est_tokens=10), score=1.0) for i in range(5)]
    kept = apply_budget(results, top_k=3, max_context_tokens=10_000)
    assert len(kept) == 3


def test_apply_budget_respects_token_ceiling():
    results = [RetrievedChunk(make_chunk(f"c{i}", est_tokens=500), score=1.0) for i in range(5)]
    kept = apply_budget(results, top_k=10, max_context_tokens=1200)
    assert len(kept) == 2  # 500 + 500 = 1000 fits, + another 500 would exceed 1200
