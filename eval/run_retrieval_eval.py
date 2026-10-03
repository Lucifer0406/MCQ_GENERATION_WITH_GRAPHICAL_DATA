"""Small retrieval evaluation: do hand-written queries find the right NCERT section?

    python -m eval.run_retrieval_eval          (needs the built index + an API key: each query is embedded live)

Each item in eval/retrieval_eval.json is a query written in our own words (not copied from
the book, so it tests meaning, not word overlap) plus the section(s) that should answer it.

Metrics, per subject and overall:
  Hit@1  - the top chunk is from a correct section
  Hit@k  - some chunk in the top k is from a correct section (k = medium difficulty's chunk count).
           This is what matters most: it means the generator SAW the right material.
  MRR    - average of 1/rank of the first correct chunk (rewards ranking it high)
Precision@k is left out on purpose: neighbouring sections are often useful context,
so counting them as "wrong" would make the number misleading.
"""

import json
from pathlib import Path

from mcq.retrieval.config import CHUNKS_PER_QUESTION
from mcq.retrieval.retriever import retrieve_context

EVAL_FILE = Path(__file__).with_name("retrieval_eval.json")


def evaluate(items: list[dict], k: int = CHUNKS_PER_QUESTION["medium"]) -> dict:
    by_subject: dict[str, list[tuple[int | None, dict]]] = {}
    for item in items:
        ctx = retrieve_context(item["subject"], item["query"], "medium")
        sections = [c.source.section.split(" ")[0] for c in ctx.chunks[:k]]   # "2.4.1 Title" -> "2.4.1"
        rank = next((r for r, s in enumerate(sections, 1)
                     if any(s == e or s.startswith(e + ".") for e in item["expected_sections"])), None)
        by_subject.setdefault(item["subject"], []).append((rank, {**item, "got": sections}))

    def summary(rows):
        n = len(rows)
        return {
            "n": n,
            "hit@1": sum(r == 1 for r, _ in rows) / n,
            f"hit@{k}": sum(r is not None for r, _ in rows) / n,
            "mrr": sum(1 / r for r, _ in rows if r) / n,
        }

    report = {s: summary(rows) for s, rows in by_subject.items()}
    report["overall"] = summary([row for rows in by_subject.values() for row in rows])
    misses = [item for rows in by_subject.values() for r, item in rows if r is None]
    return {"metrics": report, "misses": misses}


def main() -> None:
    result = evaluate(json.loads(EVAL_FILE.read_text(encoding="utf-8")))
    print(f"{'subject':12s} {'n':>3s} {'hit@1':>6s} {'hit@4':>6s} {'mrr':>6s}")
    for subject, m in result["metrics"].items():
        print(f"{subject:12s} {m['n']:3d} {m['hit@1']:6.2f} {m['hit@4']:6.2f} {m['mrr']:6.2f}")
    for miss in result["misses"]:
        print(f"MISS [{miss['subject']}] {miss['query']!r}: expected {miss['expected_sections']}, got {miss['got']}")


if __name__ == "__main__":
    main()
