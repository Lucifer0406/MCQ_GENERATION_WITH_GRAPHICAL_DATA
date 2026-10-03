#!/usr/bin/env python3
"""Quick demo script to generate MCQs from the sample quiz context."""

import asyncio
import json
from pathlib import Path

from mcq.schemas import QuizContext
from mcq.generation import generate_mcqs


async def main():
    sample_file = Path("examples/sample_quiz_context.json")
    if not sample_file.exists():
        print(f"Error: {sample_file} not found.")
        return

    print("=" * 70)
    print(" 🎯 MCQ GENERATION DEMO (Member 2)")
    print("=" * 70)

    # 1. Load sample context (simulating Member 1 retrieval output)
    print("\n[1/3] Loading sample retrieval context from Member 1...")
    ctx = QuizContext.model_validate_json(sample_file.read_text(encoding="utf-8"))
    print(f"      • Subject   : {ctx.subject.title()}")
    print(f"      • Difficulty: {ctx.difficulty.title()}")
    print(f"      • Topics    : {len(ctx.items)} topic(s):")
    for item in ctx.items:
        print(f"        - {item.topic} ({len(item.chunks)} chunk(s))")

    # 2. Call generation
    total = 6  # 3 per topic for a quick clean demo
    print(f"\n[2/3] Calling Gemini API (generating {total} MCQs asynchronously)...")
    mcqs = await generate_mcqs(ctx, total_mcqs=total)

    # 3. Display output
    print(f"\n[3/3] Successfully generated and validated {len(mcqs)} MCQs!\n")
    print("=" * 70)

    for i, mcq in enumerate(mcqs, 1):
        print(f"\n🔹 Question {i} of {len(mcqs)} [{mcq.subject.upper()} | {mcq.difficulty.upper()}]")
        print(f"Topic: {mcq.topic}")
        print("-" * 70)
        print(f"Q: {mcq.question}\n")

        letters = ["A", "B", "C", "D"]
        for idx, opt in enumerate(mcq.options):
            marker = "✅" if idx == mcq.correct_index else "  "
            print(f"   {marker} {letters[idx]}) {opt}")

        print(f"\n💡 Explanation: {mcq.explanation}")
        
        if mcq.visual:
            print(f"\n📊 Visual Attached: [{mcq.visual.type}]")
            v_dict = mcq.visual.model_dump(exclude_none=True)
            print(f"   Details: {json.dumps(v_dict, indent=4)}")
        else:
            print("\n📊 Visual Attached: None")

        print(f"🔗 Cited Chunk IDs: {mcq.source_chunk_ids}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
