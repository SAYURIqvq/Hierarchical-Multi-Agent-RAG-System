# tests/evaluation/test_ragas_real.py

"""
Real RAGAS evaluation — uses actual WriterAgent to generate answers,
then RAGAS scores the generated answers.
"""

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.ragas_evaluator import RAGASEvaluator
from langchain_anthropic import ChatAnthropic


# ========== GENERATE ANSWER USING WRITER PROMPT ==========

def generate_answer(query: str, contexts: list[str]) -> str:
    """
    Generate answer using the same prompt as WriterAgent.
    This ensures RAGAS scores the ACTUAL output of our system.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.0
    )

    # Format context dengan numbering (sama seperti WriterAgent)
    context = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(contexts)
    )

    # ← Ini prompt yang sama dari WriterAgent
    prompt = f"""You are a precise and faithful assistant. Your ONLY job is to answer questions using the provided context. You must NEVER add information that is not explicitly stated in the context.

User Question: {query}

Context (with source references):
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING RULES (most important):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EVERY claim in your answer MUST come directly from the context above
2. Do NOT infer, assume, or add information beyond what the context states
3. Do NOT paraphrase in a way that changes the meaning
4. If the context does not contain the answer, say:
   "The provided documents do not contain information about [topic]."
5. Do NOT use your general knowledge — ONLY the context matters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Cite ONLY the specific chunk(s) that directly support EACH statement
2. Use inline citations: [1], [2], [3] — NOT grouped like [1][2][3][4]
3. Each sentence should cite ONLY the chunks it actually uses
4. If a statement uses ONLY chunk 2, cite [2] alone
5. If combining info from chunks 2 and 5, cite [2][5]
6. Different paragraphs will naturally cite DIFFERENT chunks
7. DO NOT cite all chunks in every paragraph

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ WRONG — adds info not in context (hallucination):
"Machine learning was invented in 1950 by Alan Turing [1]."
(If context does not explicitly say this, do NOT write it)

❌ WRONG — groups citations:
"ML is AI subset [1][2]. Uses algorithms [1][2]."

❌ WRONG — uses general knowledge:
"As is commonly known, neural networks have multiple layers."
(Only write this if the context actually states it)

✅ CORRECT — grounded + proper citation:
"Machine learning is a subset of artificial intelligence [1]. 
It uses algorithms to learn patterns from data [2]."

✅ CORRECT — honest when info is missing:
"The provided documents do not contain information about 
the history of machine learning."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer using ONLY information from context
2. Cite specific chunks per statement
3. Be comprehensive but concise
4. If context lacks info, state clearly using the template above
5. Write naturally but stay strictly grounded
6. DO NOT add "Sources:" section — ONLY inline citations
7. End your answer immediately after the last sentence

Answer (inline citations only, no Sources section):"""

    response = llm.invoke(prompt)
    return response.content


# ========== TEST CASES ==========

def get_test_cases() -> list[dict]:
    """
    Test cases: context sengaja dirancang untuk test grounding.
    answer akan di-GENERATE, bukan hardcoded.
    """
    return [
        {
            "label": "GOOD — complete info available",
            "question": "What is artificial intelligence?",
            "contexts": [
                "Artificial intelligence (AI) is intelligence demonstrated by machines, "
                "as opposed to the natural intelligence displayed by animals including humans.",
                "AI systems are designed to simulate human intelligence processes "
                "including learning, reasoning, and self-correction.",
                "Applications of AI include robotics, natural language processing, "
                "and computer vision."
            ],
            "ground_truth": (
                "Artificial intelligence is intelligence demonstrated by machines. "
                "It simulates human processes like learning, reasoning, and self-correction. "
                "Applications include robotics, NLP, and computer vision."
            )
        },
        {
            "label": "PARTIAL — missing info in context",
            "question": "What are the main types of machine learning and who invented them?",
            "contexts": [
                "Machine learning is broadly divided into three types: "
                "supervised learning, unsupervised learning, and reinforcement learning.",
                "Supervised learning uses labeled training data to learn input-output mappings.",
                "Unsupervised learning finds hidden patterns in data without labeled responses.",
                "Reinforcement learning learns through trial and error, receiving rewards for correct actions."
            ],
            "ground_truth": (
                "The three main types are supervised learning, "
                "unsupervised learning, and reinforcement learning. "
                "Supervised learning uses labeled data. "
                "Unsupervised learning finds patterns without labels. "
                "Reinforcement learning uses rewards."
            )
        },
        {
            "label": "MISSING — context tidak ada info",
            "question": "What is the population of Jakarta in 2024?",
            "contexts": [
                "Jakarta is the capital city of Indonesia.",
                "Indonesia is an archipelago nation in Southeast Asia.",
                "Jakarta has been the center of Indonesian politics and economy."
            ],
            "ground_truth": (
                "The provided documents do not contain information "
                "about the population of Jakarta in 2024."
            )
        }
    ]


# ========== MAIN ==========

def run_real_evaluation():
    evaluator = RAGASEvaluator()
    test_cases = get_test_cases()

    print("=" * 60)
    print("REAL RAGAS EVALUATION (Claude API)")
    print("Answers generated by WriterAgent prompt")
    print("=" * 60)

    all_scores = []

    for i, case in enumerate(test_cases):
        print(f"\n--- Case {i+1}: {case['label']} ---")

        # ← GENERATE answer, bukan hardcoded
        print("   Generating answer with WriterAgent prompt...")
        generated_answer = generate_answer(
            query=case["question"],
            contexts=case["contexts"]
        )
        print(f"   Generated: {generated_answer[:100]}...")

        # ← Score the generated answer
        scores = evaluator.evaluate_single_case(
            question=case["question"],
            answer=generated_answer,          # ← Generated, bukan hardcoded
            contexts=case["contexts"],
            ground_truth=case["ground_truth"]
        )

        all_scores.append({
            "label": case["label"],
            "scores": scores,
            "generated_answer": generated_answer
        })

    # ========== SUMMARY ==========

    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Label':<45} {'Faith':>6} {'Overall':>8}")
    print("-" * 60)
    for item in all_scores:
        print(
            f"{item['label']:<45} "
            f"{item['scores']['faithfulness']:>6.3f} "
            f"{item['scores']['overall']:>8.3f}"
        )

    # ========== PRODUCTION GATE ==========

    print("\n📋 PRODUCTION GATE")
    print("-" * 60)
    print("  Faithfulness threshold: >= 0.5 (hard gate)")
    print("  Overall threshold:      >= 0.7 (soft gate)")
    print("  Honest non-answer:      skip overall check")
    print("-" * 60)

    for item in all_scores:
        gate = evaluator.check_production_gate(
            scores=item["scores"],
            answer=item["generated_answer"]   # ← Pass answer
        )
        status = "✅ PASS" if gate["passed"] else "❌ FAIL"
        print(f"\n  {status} — {item['label']}")
        for reason in gate["reasons"]:
            print(f"      {reason}")


    # ========== GENERATED ANSWERS ==========

    print("\n" + "=" * 60)
    print("📝 GENERATED ANSWERS")
    print("=" * 60)
    for item in all_scores:
        print(f"\n--- {item['label']} ---")
        print(item["generated_answer"])


if __name__ == "__main__":
    run_real_evaluation()