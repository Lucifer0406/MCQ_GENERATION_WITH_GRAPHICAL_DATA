"""One-time conversion of NCERT chapter PDFs into clean Markdown with LaTeX math.

Why an LLM and not plain text extraction: PyMuPDF keeps prose but scrambles
equations (fractions, subscripts and stacked terms come out as scattered tokens).
Gemini reads each rendered page and transcribes it with LaTeX math.

Run once (needs GOOGLE_API_KEY), output is cached per page:
    python -m mcq.retrieval.extract                # all subjects
    python -m mcq.retrieval.extract physics        # one subject

Output: Knowledgebase/Processed/<subject>/pages_AAA-BBB.md (one file per API call, several
pages each, every page starting with a <!-- page N --> marker) plus <subject>.md (everything
joined in page order) for the chunker. Pages already transcribed are never re-sent.

Free-tier quota (2026-10-03): 20 generate requests per day, per model, per project. So pages
are batched (PAGES_PER_CALL per request), and a model that reports its daily quota is used
up is skipped for the rest of the run instead of being retried.
"""

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf
from google import genai
from google.genai import types

from mcq.config import get_api_key
from mcq.retrieval.config import EXTRACTION_MODELS, PROCESSED_DIR, RAW_DIR, SOURCES

PROMPT = """Transcribe this textbook page into GitHub-flavoured Markdown.

Rules:
- Transcribe faithfully. Do not summarise, paraphrase, explain, or add anything.
- Write ALL mathematics in LaTeX: inline as $...$, displayed equations as $$...$$.
  Keep equation numbers like (2.4) as text after the equation.
- Write chemical formulas with LaTeX subscripts/superscripts, e.g. $\\mathrm{H_2O}$, $\\mathrm{SO_4^{2-}}$.
- Numbered section headings become Markdown headings on one line:
  "2.4" -> "## 2.4 Title", "2.4.1" -> "### 2.4.1 Title". The chapter title -> "# Title".
- Drop running headers/footers, page numbers, and lines like "Reprint 2026-27".
- Replace each figure with one line: [Figure X.Y: caption and any labels needed to understand it].
- Tables become Markdown tables. Boxed notes become > blockquotes.
- If the page has two columns, transcribe the left column fully, then the right.
- Output only the Markdown, no code fences, no commentary.
"""

PAGE_MARKER = "<!-- page {n} -->"
PAGES_PER_CALL = 4
BATCH_NOTE = """
This PDF contains {count} page(s): textbook pages {first} to {last}.
Start the transcription of each page with its marker line on its own line. Use exactly these
numbers, not the page numbers printed on the pages:
{markers}
"""
FILE_RE = re.compile(r"^pages?_(\d{3})(?:-(\d{3}))?\.md$")   # page_007.md or pages_007-010.md


def _pages_pdf_bytes(doc: pymupdf.Document, first: int, last: int) -> bytes:
    """0-based inclusive page range as a standalone PDF."""
    part = pymupdf.open()
    part.insert_pdf(doc, from_page=first, to_page=last)
    return part.tobytes()


_exhausted: set[str] = set()   # models whose daily quota ran out during this run


def _transcribe(client: genai.Client, pdf_bytes: bytes, prompt: str, retries: int = 6) -> str:
    """Each retry moves to the next model in the fallback chain (503s and quotas are per-model)."""
    for attempt in range(retries):
        available = [m for m in EXTRACTION_MODELS if m not in _exhausted]
        if not available:
            raise RuntimeError("All extraction models have used up today's free-tier quota; rerun tomorrow.")
        model = available[attempt % len(available)]
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            )
            text = (response.text or "").strip()
            text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text)  # in case the model adds fences anyway
            if text:
                return text
            raise RuntimeError("empty response")
        except Exception as e:  # overload / rate limit / transient: back off, try the next model
            if "PerDay" in str(e):
                _exhausted.add(model)
                print(f"    {model}: daily quota used up, skipping it for this run")
                continue
            if attempt == retries - 1:
                raise
            wait = 2 * 2**attempt
            print(f"    {model} failed ({type(e).__name__}: {str(e)[:160]}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("transcription failed on every attempt")


def _fix_markers(text: str, pages: list[int]) -> str:
    """Models sometimes number markers with the book's printed page numbers (e.g. 184-187).
    If there is exactly one marker per page, strictly increasing, map them onto our pages in order."""
    found = [int(n) for n in re.findall(r"<!-- page (\d+) -->", text)]
    if found == pages or len(found) != len(pages) or any(b <= a for a, b in zip(found, found[1:])):
        return text
    ours = iter(pages)
    return re.sub(r"<!-- page \d+ -->", lambda _: PAGE_MARKER.format(n=next(ours)), text)


def _covered_pages(out_dir) -> set[int]:
    pages = set()
    for f in out_dir.glob("page*.md"):
        if m := FILE_RE.match(f.name):
            first = int(m.group(1))
            pages.update(range(first, int(m.group(2) or first) + 1))
    return pages


def _batches(todo: list[int]) -> list[list[int]]:
    """Group missing 1-based page numbers into consecutive runs of at most PAGES_PER_CALL."""
    out: list[list[int]] = []
    for p in todo:
        if out and p == out[-1][-1] + 1 and len(out[-1]) < PAGES_PER_CALL:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def _with_single_marker(f, text: str) -> str:
    """Old single-page files (page_NNN.md) have no marker inside; batch files do."""
    m = FILE_RE.match(f.name)
    return text if m.group(2) else PAGE_MARKER.format(n=int(m.group(1))) + "\n" + text


def extract_subject(subject: str, client: genai.Client, workers: int = 2) -> None:
    source = SOURCES[subject]
    out_dir = PROCESSED_DIR / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(RAW_DIR / source.pdf)
    todo = sorted(set(range(1, len(doc) + 1)) - _covered_pages(out_dir))
    batches = _batches(todo)
    print(f"{subject}: {len(doc)} pages, {len(todo)} to transcribe in {len(batches)} requests")

    def work(pages: list[int]) -> None:
        first, last = pages[0], pages[-1]
        markers = "\n".join(PAGE_MARKER.format(n=p) for p in pages)
        prompt = PROMPT + BATCH_NOTE.format(count=len(pages), first=first, last=last, markers=markers)
        text = _transcribe(client, _pages_pdf_bytes(doc, first - 1, last - 1), prompt)
        text = _fix_markers(text, pages)
        found = [int(n) for n in re.findall(r"<!-- page (\d+) -->", text)]
        if found != pages:   # wrong page markers would give wrong citations: don't save it
            raise RuntimeError(f"{subject} pages {first}-{last}: expected markers {pages}, got {found}")
        (out_dir / f"pages_{first:03d}-{last:03d}.md").write_text(text, encoding="utf-8")
        print(f"  {subject} pages {first}-{last} done")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, batches))  # list() re-raises any worker exception

    if _covered_pages(out_dir) != set(range(1, len(doc) + 1)):
        raise RuntimeError(f"{subject}: some pages are still missing; rerun to continue")
    files = sorted((f for f in out_dir.glob("page*.md") if FILE_RE.match(f.name)),
                   key=lambda f: int(FILE_RE.match(f.name).group(1)))
    joined = "\n\n".join(_with_single_marker(f, f.read_text(encoding="utf-8")) for f in files)
    (PROCESSED_DIR / f"{subject}.md").write_text(joined + "\n", encoding="utf-8")


def main(subjects: list[str]) -> None:
    client = genai.Client(api_key=get_api_key())
    for subject in subjects or list(SOURCES):
        try:
            extract_subject(subject, client)
        except RuntimeError as e:   # keep going: other subjects may still have pages to do
            print(f"!! {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
