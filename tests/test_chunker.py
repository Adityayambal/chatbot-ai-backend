import pytest

from app.content.chunker import build_chunks, compute_positions
from tests.fakes import load_fixture_bundle


@pytest.fixture
def cp():
    bundle = load_fixture_bundle()
    return bundle.result[0]


def test_compute_positions_matches_fixture_shape(cp):
    positions = compute_positions(cp)
    assert len(positions) == 4  # 4 sub-concepts in the fixture
    codes = [p.topic_code for p in positions]
    assert codes == ["1.1.1", "1.2.1", "1.2.2", "2.1.1"]


@pytest.mark.asyncio
async def test_build_chunks_counts_and_types(cp):
    chunks = await build_chunks(cp, "en")
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.content_type] = by_type.get(c.content_type, 0) + 1

    assert by_type["revision_note"] == 3  # SC 1.1.1, 1.2.1, 2.1.1 each have one note
    assert by_type["video_transcript"] == 2  # SC 1.1.1, 1.2.1 have transcripts
    assert by_type["question"] == 5  # 6 raw questions, one exact duplicate removed


@pytest.mark.asyncio
async def test_empty_subconcept_produces_no_chunks(cp):
    chunks = await build_chunks(cp, "en")
    assert all(c.subconcept_id != "sc0000000000000000000003" for c in chunks)


@pytest.mark.asyncio
async def test_chunks_are_prefixed_with_breadcrumb_and_type_header(cp):
    chunks = await build_chunks(cp, "en")
    note = next(c for c in chunks if c.content_type == "revision_note" and c.topic_code == "1.1.1")
    assert "Laws of Motion" in note.text
    assert "Revision note (part 1/1)" in note.text

    transcript = next(c for c in chunks if c.content_type == "video_transcript" and c.topic_code == "1.1.1")
    assert "Video lesson" in transcript.text
    assert transcript.video_url == "https://player.vimeo.com/video/100010001"
    assert transcript.duration_sec == 180

    question = next(c for c in chunks if c.content_type == "question")
    assert "Practice question" in question.text
    assert "(options not available)" in question.text


@pytest.mark.asyncio
async def test_question_dedupe_keeps_longer_explanation(cp):
    chunks = await build_chunks(cp, "en")
    speed_questions = [c for c in chunks if "SI unit of speed" in c.text]
    assert len(speed_questions) == 1
    assert "meters per second" in speed_questions[0].text.lower()


@pytest.mark.asyncio
async def test_note_chunk_respects_max_chars_budget(cp):
    chunks = await build_chunks(cp, "en")
    for c in chunks:
        if c.content_type == "revision_note":
            # breadcrumb header adds some overhead on top of the 1800-char body budget
            assert len(c.text) < 1800 + 200


@pytest.mark.asyncio
async def test_est_tokens_is_positive_and_proportional_to_length(cp):
    chunks = await build_chunks(cp, "en")
    for c in chunks:
        assert c.est_tokens > 0
        assert c.est_tokens == pytest.approx(len(c.text) / 3.5, abs=1)
