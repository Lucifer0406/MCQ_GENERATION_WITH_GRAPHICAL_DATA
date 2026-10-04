# CLAUDE.md — working notes for Claude Code sessions

Educational MCQ app: pick subject + difficulty → retrieve NCERT context → Gemini generates 10 MCQs →
Streamlit shows each with answer, explanation and a visual. 3-person intern team, **deadline 2026-10-05**.
Keep it lean: no uploads, no answer submission, no extra infrastructure.

Read first: `TEAM_RAG_CONTEXT.md` (retrieval design, results, limitations) and `CONTRACTS.md`
(shared schemas, how each member uses them, Gemini quota rules).

## Status (end of session 1, 2026-10-03)

- Member 1 (repo owner, retrieval): **done and verified.** All 4 subjects extracted + indexed;
  68 tests pass; retrieval eval Hit@4 1.00 / MRR 0.96 on 26 queries. Work is on branch `lucifer`.
- Member 2 (generation) and Member 3 (UI/visuals): building against `examples/*.json`.
  `mcq/generation/` and `mcq/rendering/` are still empty placeholders in this repo.
- Not yet done: end-to-end test (retrieval → generation → UI), manual grounding review of ~20 MCQs.

## Next steps

1. Once Member 2's generation exists: run 4 subjects × 3 difficulties end to end; check every MCQ
   validates and passes `check_citations`.
2. Manual grounding review: is each correct answer supported by its focus chunk?
3. Help with integration bugs; keep `TEAM_RAG_CONTEXT.md` / `CONTRACTS.md` current.

## Commands

```bash
.venv/bin/python -m pytest -q                              # offline, no key needed
PYTHONPATH=. .venv/bin/python -m eval.run_retrieval_eval   # live: needs key + index
PYTHONPATH=. .venv/bin/python -m mcq.retrieval.extract     # only to rebuild (resumable)
PYTHONPATH=. .venv/bin/python -m mcq.retrieval.build_index
```

## Rules and gotchas (learned the hard way)

- **Public repo:** never commit `.env`, `Knowledgebase/{Raw,Processed,Index}` or `*.zip` (NCERT is
  copyrighted). Before any commit, scan staged files for the key value. Built index is shared
  privately as `Knowledgebase_index.zip`.
- **Gemini free tier:** 20 generate requests/day **per model per project**; embeddings 100 texts/min.
  Random `503 high demand` per model → always use a fallback chain. Don't rotate teammates' keys to
  evade quotas (ToS); batching or billing are the fixes.
- `gemini-2.5-flash*` return 404 for this key; `gemini-3.x-flash` models work when not overloaded.
- `gemini-embedding-2` **merges a list of strings into one embedding** — send each text as its own
  `types.Content` (done in `mcq/retrieval/embeddings.py`, with a count check).
- Structured output: `response_json_schema=MCQ.model_json_schema()` works; `response_schema=MCQ` rejects
  the visual union.
- Retrieval needs **no API call** at quiz time (precomputed vectors); `chunks[0]` of each item is the
  question's focus chunk.
- `load_dotenv()` without a path crashes in stdin scripts — always use `mcq.config.get_api_key()`.
- Changing `mcq/schemas.py` changes teammates' contracts: tell the team, update `examples/`, log it in
  CONTRACTS.md.
- The user is an intern who wants to learn: explain genuinely new concepts briefly, inspect before
  changing, test before claiming something works, and don't push or publish without asking.
