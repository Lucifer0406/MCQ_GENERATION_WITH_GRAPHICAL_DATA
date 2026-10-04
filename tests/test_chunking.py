"""Chunker tests on a small synthetic chapter (no NCERT text: the repo is public)."""

from mcq.retrieval.chunking import MAX_CHARS, chunk_markdown

CHAPTER = """<!-- page 1 -->
# MOTION ALONG A LINE

Opening paragraph before any numbered section.

## 3.1 INTRODUCTION

Motion is change of position with time.

<!-- page 2 -->
## 3.2 Uniform acceleration

For constant acceleration the velocity changes linearly:

$$
v = u + at
\\quad (3.1)
$$

Squaring and eliminating time gives the third relation.

<!-- page 3 -->
### 3.2.1 Graphs

The area under a velocity-time graph is the displacement.

## SUMMARY

1. This recap must not be chunked.

## EXERCISES

3.1 A ready-made question that must not be chunked.
"""


def chunks():
    return chunk_markdown(CHAPTER, "physics", "test101")


def test_sections_and_ids():
    got = [(c.chunk_id, c.section) for c in chunks()]
    assert got == [
        ("physics/test101/0/000", "Introduction"),
        ("physics/test101/3.1/000", "3.1 Introduction"),   # ALL CAPS heading normalised
        ("physics/test101/3.2/000", "3.2 Uniform acceleration"),
        ("physics/test101/3.2.1/000", "3.2.1 Graphs"),
    ]


def test_chapter_title_is_not_content():
    assert all("MOTION ALONG A LINE" not in c.text for c in chunks())


def test_summary_and_exercises_are_skipped():
    text = " ".join(c.text for c in chunks())
    assert "recap" not in text and "ready-made question" not in text


def test_display_equation_kept_whole_and_pages_tracked():
    c = next(c for c in chunks() if c.section.startswith("3.2 "))
    assert "$$\nv = u + at\n\\quad (3.1)\n$$" in c.text
    assert (c.page_start, c.page_end) == (2, 2)


def test_chunk_text_starts_with_section_title():
    assert all(c.text.startswith(c.section + "\n\n") for c in chunks())


def test_long_section_is_packed_without_splitting_paragraphs():
    para = "Sentence about kinematics. " * 20   # ~540 chars
    md = "## 1.1 Long\n\n" + "\n\n".join(f"P{i} {para}" for i in range(8))
    result = chunk_markdown(md, "physics", "t")
    assert len(result) > 1
    assert all(len(c.text) <= MAX_CHARS + len("1.1 Long\n\n") for c in result)
    joined = "\n\n".join(c.text.split("\n\n", 1)[1] for c in result)
    assert all(f"P{i} {para.strip()}" in joined for i in range(8))   # nothing lost or cut


def test_repeated_heading_keeps_ids_unique():
    md = "## 1.1 A\n\nfirst part\n\n## 1.1 A\n\nsecond part"
    ids = [c.chunk_id for c in chunk_markdown(md, "physics", "t")]
    assert ids == ["physics/t/1.1/000", "physics/t/1.1/001"]


# ── extraction helpers (no API) ──

from mcq.retrieval.extract import _batches, _fix_markers


def test_batches_group_consecutive_pages():
    assert _batches([1, 2, 3, 4, 5, 9, 10]) == [[1, 2, 3, 4], [5], [9, 10]]


def test_printed_page_numbers_are_remapped_in_order():
    text = "<!-- page 184 -->\na\n<!-- page 185 -->\nb"
    assert _fix_markers(text, [9, 10]) == "<!-- page 9 -->\na\n<!-- page 10 -->\nb"


def test_ambiguous_markers_are_left_for_the_check_to_reject():
    assert _fix_markers("<!-- page 1 -->\na", [9, 10]) == "<!-- page 1 -->\na"          # too few
    assert _fix_markers("<!-- page 5 -->\n<!-- page 4 -->", [9, 10]).count("page 9") == 0  # not increasing
