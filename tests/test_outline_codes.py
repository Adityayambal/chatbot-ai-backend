from app.content.chunker import compute_positions
from app.content.knowledge_base import build_outline
from tests.fakes import load_fixture_bundle


def test_outline_is_byte_identical_across_builds():
    bundle = load_fixture_bundle()
    cp = bundle.result[0]
    positions_a = compute_positions(cp)
    positions_b = compute_positions(cp)
    text_a, _, _ = build_outline(cp, positions_a)
    text_b, _, _ = build_outline(cp, positions_b)
    assert text_a == text_b


def test_outline_text_matches_expected_shape():
    bundle = load_fixture_bundle()
    cp = bundle.result[0]
    positions = compute_positions(cp)
    text, code_to_subconcept_id, concept_code_to_subconcept_codes = build_outline(cp, positions)

    lines = text.splitlines()
    assert lines[0] == "1 Laws of Motion"
    # single sub-concept whose name matches the concept name collapses to one line
    assert lines[1] == "  1.1.1 introduction to motion"
    # concept with two differently-named sub-concepts prints both levels
    assert lines[2] == "  1.2 Distance and displacement"
    assert lines[3] == "    1.2.1 distance and displacement"
    assert lines[4] == "    1.2.2 numerical on distance and displacement"
    assert lines[5] == "2 Force and Laws of Motion"
    assert lines[6] == "  2.1.1 newton's second law of motion"

    assert code_to_subconcept_id["1.2.2"] == "sc0000000000000000000003"
    assert concept_code_to_subconcept_codes["1.2"] == ["1.2.1", "1.2.2"]
    assert concept_code_to_subconcept_codes["1.1"] == ["1.1.1"]


def test_topic_codes_are_stable_1_based_positions():
    bundle = load_fixture_bundle()
    cp = bundle.result[0]
    positions = compute_positions(cp)
    codes = [p.topic_code for p in positions]
    assert codes == ["1.1.1", "1.2.1", "1.2.2", "2.1.1"]
