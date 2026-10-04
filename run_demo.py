#!/usr/bin/env python3
"""End-to-end pipeline: Retrieval (Member 1) → Generation (Member 2).

Usage:
    venv/bin/python run_demo.py                     # physics medium (default)
    venv/bin/python run_demo.py chemistry hard       # any subject + difficulty
    venv/bin/python run_demo.py --all                # all 4 subjects × medium
"""

import asyncio
import json
import sys
import time

from mcq.retrieval.retriever import retrieve_quiz_context
from mcq.generation import generate_mcqs
from mcq.schemas import SUBJECTS, DIFFICULTIES


async def run_pipeline(subject: str, difficulty: str, total_mcqs: int = 10) -> dict:
    """Run retrieval → generation for one subject+difficulty and return stats."""
    print(f"\n{'=' * 70}")
    print(f" 🎯  {subject.upper()} | {difficulty.upper()}")
    print(f"{'=' * 70}")

    # ── Step 1: Retrieval (Member 1) — offline, ~1 ms ────────────────────
    t0 = time.perf_counter()
    ctx = retrieve_quiz_context(subject, difficulty, n_questions=total_mcqs)
    t_retrieval = time.perf_counter() - t0

    print(f"\n[1/2] Retrieval completed in {t_retrieval * 1000:.1f} ms")
    print(f"      • {len(ctx.items)} items, topics:")
    for i, item in enumerate(ctx.items):
        print(f"        {i+1:2d}. {item.topic} ({len(item.chunks)} chunks, "
              f"sufficient={item.sufficient})")

    # ── Step 2: Generation (Member 2) — async LLM calls ─────────────────
    from mcq.config import get_default_provider
    provider = get_default_provider()
    print(f"\n[2/2] Generating MCQs via {provider.upper()} (async, target={total_mcqs})...")
    t1 = time.perf_counter()
    mcqs = await generate_mcqs(ctx, total_mcqs=total_mcqs)
    t_generation = time.perf_counter() - t1
    t_total = t_retrieval + t_generation

    # ── Display results ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"✅ Generated {len(mcqs)} MCQs in {t_generation:.2f}s "
          f"(retrieval: {t_retrieval*1000:.1f}ms + generation: {t_generation:.2f}s = {t_total:.2f}s total)")
    if mcqs:
        print(f"   Throughput: {len(mcqs)/t_generation:.2f} MCQs/sec | "
              f"Effective: {t_generation/len(mcqs):.2f}s per MCQ")

    visual_types = {}
    for i, mcq in enumerate(mcqs, 1):
        vtype = mcq.visual.type if mcq.visual else "none"
        visual_types[vtype] = visual_types.get(vtype, 0) + 1

        print(f"\n🔹 Q{i} [{mcq.difficulty}] — {mcq.topic}")
        print(f"   {mcq.question[:120]}{'…' if len(mcq.question) > 120 else ''}")
        letters = "ABCD"
        for j, opt in enumerate(mcq.options):
            marker = "✅" if j == mcq.correct_index else "  "
            print(f"   {marker} {letters[j]}) {opt}")
        print(f"   💡 {mcq.explanation[:100]}{'…' if len(mcq.explanation) > 100 else ''}")
        if mcq.visual:
            print(f"   📊 Visual: {mcq.visual.type}")
        print(f"   🔗 Sources: {mcq.source_chunk_ids}")

    print(f"\n📊 Visual distribution: {visual_types}")

    return {
        "subject": subject,
        "difficulty": difficulty,
        "mcqs_generated": len(mcqs),
        "retrieval_ms": round(t_retrieval * 1000, 1),
        "generation_s": round(t_generation, 2),
        "total_s": round(t_total, 2),
        "visual_types": visual_types,
    }


async def main():
    args = sys.argv[1:]

    if "--all" in args:
        # Run all 4 subjects at medium difficulty
        subjects = list(SUBJECTS)
        difficulty = "medium"
        results = []
        for subj in subjects:
            try:
                result = await run_pipeline(subj, difficulty, total_mcqs=4)
                results.append(result)
            except Exception as e:
                print(f"\n❌ {subj} failed: {e}")
                results.append({"subject": subj, "error": str(e)})

        # Summary table
        print(f"\n\n{'=' * 70}")
        print(" 📋 END-TO-END SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Subject':<14} {'MCQs':>5} {'Retrieval':>10} {'Generation':>11} {'Total':>8} {'Visuals'}")
        print(f"{'─'*14} {'─'*5} {'─'*10} {'─'*11} {'─'*8} {'─'*20}")
        for r in results:
            if "error" in r:
                print(f"{r['subject']:<14} {'ERROR':>5}   {r['error'][:40]}")
            else:
                print(f"{r['subject']:<14} {r['mcqs_generated']:>5} "
                      f"{r['retrieval_ms']:>8.1f}ms {r['generation_s']:>9.2f}s "
                      f"{r['total_s']:>6.2f}s {r['visual_types']}")

    else:
        subject = args[0] if args else "physics"
        difficulty = args[1] if len(args) > 1 else "medium"
        if subject not in SUBJECTS:
            print(f"❌ Unknown subject '{subject}'. Choose from: {', '.join(SUBJECTS)}")
            sys.exit(1)
        if difficulty not in DIFFICULTIES:
            print(f"❌ Unknown difficulty '{difficulty}'. Choose from: {', '.join(DIFFICULTIES)}")
            sys.exit(1)
        await run_pipeline(subject, difficulty)


if __name__ == "__main__":
    asyncio.run(main())
