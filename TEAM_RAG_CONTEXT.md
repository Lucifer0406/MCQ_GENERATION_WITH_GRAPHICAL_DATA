# TEAM_RAG_CONTEXT — Retrieval (RAG) component

Context for teammates and their Claude Code sessions. Status as of 2026-10-03 (deadline 2026-10-05).
For the shared data contracts and how to use them, read [CONTRACTS.md](CONTRACTS.md) first.

## 1. Project in one paragraph

A Streamlit app for students. The user picks **one subject** (`physics`, `chemistry`, `mathematics`,
`biology`) and **one difficulty** (`easy`, `medium`, `hard`) and clicks Generate. The app shows
**10 MCQs**, each with the correct answer, an explanation, and (when relevant) a rendered visual
below it. No uploads, no accounts, no answer submission. Knowledge comes from one NCERT Class XI
chapter per subject, prepared offline.

```
Streamlit (Member 3) ── subject, difficulty ──► Retrieval (Member 1) ── QuizContext ──►
Generation (Member 2, Gemini) ── list[MCQ] ──► Streamlit renders question + answer + visual
```

| Member | Owns | Code |
|---|---|---|
| 1 | Knowledge prep + retrieval | `mcq/retrieval/`, `eval/` |
| 2 | Prompting, MCQ generation, validation, visual specs | `mcq/generation/` |
| 3 | Streamlit UI, KaTeX / Plotly / RDKit rendering | `app.py`, `mcq/rendering/` |
| all | Contracts | `mcq/schemas.py` (change only after telling the team) |

## 2. Knowledge source

| Subject | NCERT file | Chapter | Pages | Chunks | Topics |
|---|---|---|---|---|---|
| physics | keph102.pdf | Motion in a Straight Line | 14 | 24 | 3 |
| chemistry | kech104.pdf | Chemical Bonding and Molecular Structure | 36 | 85 | 35 |
| mathematics | kemh110.pdf | Conic Sections | 32 | 32 | 16 |
| biology | kebo102.pdf | Biological Classification | 13 | 27 | 17 |

**Copyright:** NCERT text is copyrighted and this repo is public. `Knowledgebase/Raw`, `Processed`,
`Index` and `*.zip` are gitignored. The built index is shared privately as `Knowledgebase_index.zip`.

## 3. Pipeline

```
OFFLINE (done once; outputs cached)
Raw PDF ─► extract.py: Gemini transcribes pages (4 per request) to Markdown + LaTeX
        ─► Processed/<subject>.md  (with <!-- page N --> markers)
        ─► chunking.py: split on numbered section headings, pack paragraphs ≤1500 chars,
           never split $$…$$, skip Summary / Exercises
        ─► build_index.py: gemini-embedding-2 (768-d, unit vectors) ─► FAISS IndexFlatIP per subject
           + topic vectors precomputed + manifest.json

ONLINE (per Generate click, ~1 ms, NO API call)
retrieve_quiz_context(subject, difficulty)
   pick 10 different "seed" chunks, spread round-robin across the chapter's sections
   for each seed: use its stored vector as the query ─► nearest neighbours in that subject's index
   ─► item = [seed (the question's focus), neighbours…], k = 3 / 4 / 5 by difficulty
   ─► QuizContext (validated Pydantic)
```

### Key decisions and why

| Decision | Why |
|---|---|
| Gemini page transcription, not PyMuPDF text | PyMuPDF scrambles equations, and **kech104's text layer is corrupted** ("valence electrons" is stored as `YalenFe eleFtrons`). Gemini reads the rendered page. |
| One FAISS index per subject | Hard guarantee of no subject mixing; flat exact search is instant at <100 chunks. |
| Seed-and-expand for quizzes | With one chapter there are only 3–35 topics; topic queries alone repeated questions (found in testing). Seeds give 10 distinct focuses per click. |
| Precomputed vectors, no API call at quiz time | Free tier is tiny (see §7); retrieval stays fast, free, deterministic, offline. |
| Difficulty is NOT in the query | Embeddings encode topic, not difficulty. Difficulty only sets context size; question difficulty is the prompt's job. |
| Sections, not fixed windows, define chunks | Every chunk has one clean citation (`§2.4`, pages). |

## 4. Interface (what Member 2 calls)

```python
from mcq.retrieval.retriever import retrieve_quiz_context, retrieve_context, format_context_for_prompt

quiz = retrieve_quiz_context(subject, difficulty, n_questions=10, avoid_chunk_ids=frozenset(), seed=None)
# quiz.items[i].chunks[0]  = focus chunk: the question must be about this
# quiz.items[i].chunks[1:] = supporting chunks (may repeat across items)
# avoid_chunk_ids = {it.chunks[0].chunk_id for it in previous_quiz.items}  -> next click gets fresh focuses

ctx = retrieve_context(subject, topic="Hybridisation", difficulty="medium")   # one question on a named topic
# custom topic text not in the catalog -> one live embedding API call
text = format_context_for_prompt(item)   # chunks as prompt text tagged [chunk_id: ...]
```

Output schema: `QuizContext` → `RetrievedContext` → `ContextChunk` → `SourceRef` in `mcq/schemas.py`.
Each `ContextChunk` has `chunk_id` (e.g. `chemistry/kech104/4.6/002`), `text` (Markdown + LaTeX,
starting with the section title), `source` (document, file, chapter, section, page_start, page_end)
and `score` (cosine similarity to the query; the focus chunk is 1.0).

Grounding check for generated MCQs: `mcq.schemas.check_citations(mcq, item)`.

**Not owned by retrieval:** whether a question gets a visual, the visual spec, rendering. Retrieval
returns text only and has no knowledge of KaTeX / Plotly / RDKit.

## 5. Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14 tested
cp .env.example .env            # add your own GOOGLE_API_KEY (only needed to rebuild or for custom topics)
unzip Knowledgebase_index.zip   # from the team's private share -> Knowledgebase/Index/
.venv/bin/python -c "from mcq.retrieval.retriever import retrieve_quiz_context as r; print(len(r('physics','easy').items))"
```

Rebuild from scratch (needs NCERT PDFs in `Knowledgebase/Raw/<subject>/` and a key; ~45 generate
requests + ~170 embeddings; resumable):

```bash
python -m mcq.retrieval.extract          # PDF -> Processed/ (cached per page batch)
python -m mcq.retrieval.build_index      # Processed/ -> Index/
```

## 6. Testing and verification

```bash
.venv/bin/python -m pytest -q            # 68 tests, offline (fake embeddings, synthetic text)
python -m eval.run_retrieval_eval        # live; needs index + key (26 embedding calls)
```

Unit tests cover: schemas, config/key loading, chunking (sections, equations kept whole, skipped
Summary/Exercises, unique ids, page tracking), extraction helpers (batching, page-marker repair),
index build/reload, subject isolation, k by difficulty, per-section cap, exclusions, FAISS `-1`
padding, stale-manifest rejection, quiz uniqueness/cycling/avoidance/reproducibility.

### Verification results (2026-10-03)

| Check | Method | Result |
|---|---|---|
| Extraction fidelity | share of PyMuPDF words present in Gemini page text | physics 0.99, maths 0.97, biology 0.96 mean. Chemistry 0.73 is an artifact of its corrupted text layer; spot-checked correct |
| Retrieval quality | 26 hand-written paraphrased queries with expected sections | **Hit@4 1.00**, Hit@1 0.92, MRR 0.96 (all subjects Hit@4 1.00) |
| Quiz output | all 4 subjects × 3 difficulties | 10 items, 10 unique focus chunks, k = 3/4/5, 0 cross-subject chunks, valid `QuizContext`, ~1 ms |
| Repeat clicks | second quiz with `avoid_chunk_ids` | 10/10 fresh focus chunks (physics) |
| Reproducibility | `seed=` | identical output; index reload gives identical results |

Metric choice: Hit@k matters most (the generator *saw* the right material); MRR rewards ranking;
Precision@k is not reported because neighbouring sections are often legitimately useful context.
The eval set is small (26 queries): it rules out obvious failures, it does not prove quality.

## 7. Known limitations

- **Gemini free tier:** 20 generate requests/day per model per project; embeddings 100 texts/minute.
  Models also return `503 high demand` at random. This mainly threatens **Member 2's live demo**
  (see CONTRACTS.md). Billing on one project removes it.
- **Physics has only 3 topics** (the rationalised chapter has sections 2.2–2.4), so its 10 questions
  share 3 topic labels; focus chunks still differ.
- **Page numbers are approximate in a few places:** the model occasionally puts the top of a page
  before that page's marker (seen on biology p.12), so a citation can be one page early.
- Extracted text is LLM output: faithful in checks so far, but not proof-read line by line.
  Figures are replaced by one-line descriptions; diagrams themselves are not available.
- Topics are section headings; worked "Examples" are part of their section, end-of-chapter exercises
  and Maths "Miscellaneous Examples" are excluded.
- `SUFFICIENT_SCORE = 0.6` is a rough calibration (unrelated text ≈ 0.5, relevant ≈ 0.75–0.9);
  only relevant for custom-topic `retrieve_context`.

## 8. Status and next steps

Done: extraction, chunking, indexes for all 4 subjects, retrieval API, tests, eval.
Next:
1. Member 2: integrate `retrieve_quiz_context` → generate 10 MCQs in 1–2 Gemini calls → `check_citations`.
2. End-to-end test together: all 4 subjects × 3 difficulties through generation and the UI.
3. Manual grounding review of ~20 generated MCQs ("is the answer supported by the cited chunk?").
4. Only if time allows: more chapters (add to `SOURCES` in `mcq/retrieval/config.py`, rerun both scripts).
