from sclm.ncert_clean import clean_document, clean_page_lines, join_wrapped_lines


def test_drops_reprint_stamp_and_edge_page_number():
    raw = "Ganita Prakash | Grade 6\n2\nPatterns exist.\nReprint 2026-27\n"
    lines = clean_page_lines(raw)
    assert "Reprint 2026-27" not in lines
    assert "2" not in lines
    assert "Patterns exist." in lines


def test_keeps_a_number_that_is_not_an_edge_page_number():
    raw = "Count the cats.\n3\nThen draw them.\n"
    lines = [line for line in clean_page_lines(raw) if line]
    assert lines == ["Count the cats.", "3", "Then draw them."]


def test_drops_production_marks():
    raw = "Prelims.indd 1\n1/20/2025 11:58:58 AM\nPD 110T BS\nA real sentence.\n"
    lines = [line for line in clean_page_lines(raw) if line]
    assert lines == ["A real sentence."]


def test_collapses_immediate_duplicate_lines():
    raw = "Shapes Around Us\nShapes Around Us\nLook at the blocks.\n"
    lines = [line for line in clean_page_lines(raw) if line]
    assert lines == ["Shapes Around Us", "Look at the blocks."]


def test_joins_long_wrapped_lines_and_dehyphenates():
    lines = [
        "Mathematics is, in large part, the search for patterns, and for",
        "the explanations.",
        "A well-",
        "known fact.",
    ]
    text = join_wrapped_lines(lines)
    assert "patterns, and for the explanations." in text
    assert "wellknown fact." in text


def test_running_header_kept_once():
    page = "Joyful Mathematics\nThe cat is under the bed. " + ("word " * 20)
    text = clean_document([page, page, page])
    assert text.count("Joyful Mathematics") == 1
    assert text.count("The cat is under the bed.") == 3


def test_does_not_swallow_a_section_heading():
    long = [
        "calendars, clocks, and other ordinary tools used in daily life and work",
        "1.2 Patterns in Numbers",
    ]
    text = join_wrapped_lines(long)
    assert "work 1.2" not in text
    assert "1.2 Patterns in Numbers" in text
    numbered = join_wrapped_lines(["2.", "How has mathematics helped?"])
    assert numbered.startswith("2. How has mathematics helped?")
