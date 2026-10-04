"""Split a processed chapter (Markdown + LaTeX, with <!-- page N --> markers) into chunks.

Strategy (structure-aware, unlike EduRAG's "group every 5 Whisper segments"):
- A chunk never crosses a numbered section heading, so every chunk has one clean citation.
- Inside a section, whole paragraphs are packed up to MAX_CHARS. Paragraphs are never
  split, and a $$...$$ display equation is never cut in half.
- End-of-chapter material (Summary, Exercises, Points to Ponder) is skipped: it is either
  a recap or ready-made questions the generator would just copy.
"""

import re
from dataclasses import dataclass

MAX_CHARS = 1500   # ~350-400 tokens: one focused idea plus its equations
MIN_CHARS = 300    # a smaller leftover is merged into the previous chunk of the same section

PAGE_RE = re.compile(r"^<!-- page (\d+) -->$")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(.+)$")   # "2.4 Kinematic ..." / "10.6.1 ..."
SKIP_RE = re.compile(r"summary|exercise|points to ponder|additional exercise", re.I)


@dataclass
class Chunk:
    chunk_id: str
    subject: str
    section: str          # "2.4 Kinematic Equations For Uniformly Accelerated Motion"
    text: str             # section title line + body; this is also what gets embedded
    page_start: int
    page_end: int


def _clean_title(title: str) -> str:
    title = title.strip().strip("*").strip()
    return title.title() if title.isupper() else title


def _blocks(lines: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Group lines into paragraphs (blank-line separated), keeping $$...$$ blocks whole.

    Returns (first_page, last_page, text) per paragraph.
    """
    blocks, cur, pages, in_display = [], [], [], False
    for page, line in lines:
        if not line.strip() and not in_display:
            if cur:
                blocks.append((pages[0], pages[-1], "\n".join(cur).strip()))
            cur, pages = [], []
            continue
        cur.append(line)
        pages.append(page)
        if line.count("$$") % 2 == 1:   # an opening or closing $$ on this line
            in_display = not in_display
    if cur:
        blocks.append((pages[0], pages[-1], "\n".join(cur).strip()))
    return blocks


def _pack(blocks: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Greedily pack paragraphs into chunks of at most MAX_CHARS (a single huge paragraph stays whole)."""
    chunks: list[list] = []
    for first, last, text in blocks:
        if chunks and len(chunks[-1][2]) + len(text) + 2 <= MAX_CHARS:
            chunks[-1][1] = last
            chunks[-1][2] += "\n\n" + text
        else:
            chunks.append([first, last, text])
    if len(chunks) > 1 and len(chunks[-1][2]) < MIN_CHARS:
        first, last, text = chunks.pop()
        chunks[-1][1] = last
        chunks[-1][2] += "\n\n" + text
    return [tuple(c) for c in chunks]


def chunk_markdown(markdown: str, subject: str, file_stem: str) -> list[Chunk]:
    sections: list[tuple[str, str, list[tuple[int, str]]]] = []   # (number, title, lines)
    number, title, lines = "0", "Introduction", []
    page, skipping = 1, False

    def close():
        if lines and not skipping:
            sections.append((number, title, lines))

    for raw in markdown.splitlines():
        if m := PAGE_RE.match(raw.strip()):
            page = int(m.group(1))
            continue
        if h := HEADING_RE.match(raw):
            heading = _clean_title(h.group(2))
            if n := NUMBERED_RE.match(heading):
                close()
                number, title, lines, skipping = n.group(1), _clean_title(n.group(2)), [], False
                continue
            if SKIP_RE.search(heading):
                close()
                lines, skipping = [], True
                continue
            if h.group(1) == "#":   # chapter title: not content
                continue
        if not skipping:
            lines.append((page, raw))
    close()

    chunks: list[Chunk] = []
    per_section: dict[str, int] = {}   # ids stay unique even if a heading appears twice
    for number, title, sec_lines in sections:
        section = title if number == "0" else f"{number} {title}"
        for first, last, body in _pack(_blocks(sec_lines)):
            i = per_section.get(number, 0)
            per_section[number] = i + 1
            chunks.append(Chunk(
                chunk_id=f"{subject}/{file_stem}/{number}/{i:03d}",
                subject=subject,
                section=section,
                text=f"{section}\n\n{body}",
                page_start=first,
                page_end=last,
            ))
    return chunks
