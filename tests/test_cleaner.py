from app.content.cleaner import clean_html, clean_transcript, normalize_name


def test_img_becomes_figure_marker_and_records_ref():
    result = clean_html('<p>See below.</p><p><img src="https://x/y.png" alt="a diagram" /></p>')
    assert "[Figure 1]" in result.text
    assert len(result.images) == 1
    assert result.images[0].url == "https://x/y.png"
    assert result.images[0].alt == "a diagram"


def test_table_becomes_markdown_pipe_table_with_colspan_flattened():
    html = (
        "<table><tbody>"
        "<tr><td colspan='2'>Header</td></tr>"
        "<tr><td>A</td><td>B</td></tr>"
        "</tbody></table>"
    )
    result = clean_html(html)
    lines = result.text.splitlines()
    assert lines[0] == "| Header | Header |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| A | B |"


def test_list_becomes_markdown_with_nested_indent():
    html = "<ul><li>First<ul><li>Nested</li></ul></li><li>Second</li></ul>"
    result = clean_html(html)
    lines = result.text.splitlines()
    assert lines[0] == "- First"
    assert lines[1] == "  - Nested"
    assert lines[2] == "- Second"


def test_ordered_list_numbers_items():
    html = "<ol><li>One</li><li>Two</li></ol>"
    result = clean_html(html)
    assert result.text.splitlines() == ["1. One", "2. Two"]


def test_nbsp_and_whitespace_collapse():
    result = clean_html("<p>A&nbsp;&nbsp;B&nbsp;C</p>")
    assert result.text == "A B C"


def test_sub_and_sup_unicode_for_digits():
    result = clean_html("<p>H<sub>2</sub>O and 10<sup>3</sup></p>")
    assert result.text == "H₂O and 10³"


def test_sub_fallback_for_non_digit_content():
    result = clean_html("<p>x<sub>max</sub></p>")
    assert result.text == "x_(max)"


def test_sup_fallback_for_non_digit_content():
    result = clean_html("<p>x<sup>th</sup></p>")
    assert result.text == "x^(th)"


def test_strong_and_em_markers():
    result = clean_html("<p><strong>bold</strong> and <em>italic</em></p>")
    assert result.text == "**bold** and _italic_"


def test_clean_transcript_strips_greeting_and_outro():
    text = (
        "Hi friends. Welcome to Top Scholars, the smart learning app. "
        "Today we learn about motion. "
        "Thanks for watching, see you in the next video."
    )
    assert clean_transcript(text) == "Today we learn about motion."


def test_normalize_name_collapses_whitespace():
    assert normalize_name("Effect of speed and direction on velocity\n\n") == (
        "Effect of speed and direction on velocity"
    )
