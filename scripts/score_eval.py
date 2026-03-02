#!/usr/bin/env python3
"""Score evaluation runs using LLM-as-judge.

Reads logs/runs.jsonl + eval/questions.jsonl → calls LLM judge → writes eval/scores.csv

Metrics:
  1. Retrieval recall (local) — gold_passages vs retrieved sources (skipped when null)
  2. Claim recall (1 LLM call) — key_claims present in generated answer
  3. Faithfulness (1 LLM call) — answer claims supported by retrieved passages
  4. Overall quality (1 LLM call) — 1-5 score with rubric

Usage:
    python scripts/score_eval.py                          # score all unscored
    python scripts/score_eval.py --fresh                  # re-score everything
    python scripts/score_eval.py --judge-model openai/gpt-4o
    python scripts/score_eval.py --verbose
"""

import argparse
import csv
import json
import os
import re
import sys
from io import StringIO

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

RUNS_FILE = "logs/runs.jsonl"
QUESTIONS_FILE = "eval/questions.jsonl"
SCORES_FILE = "eval/scores.csv"

CSV_COLUMNS = [
    "question",
    "num_hops",
    "retrieval_recall",
    "claim_recall",
    "faithfulness",
    "judge_score",
    "judge_reason",
    "agent_cost",
    "judge_cost",
    "latency_ms",
    "run_id",
]

# ── LLM Judge Prompts ──────────────────────────────────────────────────────────

CLAIM_RECALL_SYSTEM = """You are an evaluation judge. Given a generated answer and a list of key claims, determine which claims are PRESENT in the answer.

A claim is PRESENT if the answer conveys the same factual information, even if worded differently.
A claim is ABSENT if the answer does not mention or contradicts the information.

Respond with JSON: {"verdicts": [{"claim": "...", "verdict": "PRESENT" or "ABSENT"}]}"""

FAITHFULNESS_SYSTEM = """You are an evaluation judge. Given a generated answer and the retrieved passages used to generate it, assess faithfulness.

1. Decompose the answer into atomic factual claims.
2. For each claim, judge if it is SUPPORTED, NOT_SUPPORTED, or CONTRADICTED by the passages.
   - SUPPORTED: The passages contain evidence for this claim.
   - NOT_SUPPORTED: The passages neither support nor contradict this claim.
   - CONTRADICTED: The passages contain evidence against this claim.

Respond with JSON: {"claims": [{"claim": "...", "verdict": "SUPPORTED" or "NOT_SUPPORTED" or "CONTRADICTED"}]}"""

QUALITY_SYSTEM = """You are an evaluation judge. Given a question, expected answer, and generated answer, score the quality of the generated answer on a 1-5 scale.

Rubric:
  5 = Excellent: Complete, accurate, well-structured answer covering all key points
  4 = Good: Mostly complete and accurate, may miss minor details
  3 = Acceptable: Partially correct, misses some important points
  2 = Poor: Significant inaccuracies or major omissions
  1 = Very Poor: Mostly incorrect or irrelevant

Respond with JSON: {"score": <1-5>, "reason": "<brief explanation>"}"""


# ── Helpers ─────────────────────────────────────────────────────────────────────

def call_judge(system: str, user_msg: str, model: str, api_key: str, base_url: str) -> tuple[dict, float]:
    """Call the LLM judge. Returns (parsed_json, cost)."""
    resp = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    cost = data.get("usage", {}).get("cost", 0.0) or 0.0

    # Parse JSON with regex fallback
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse judge response: {content[:200]}")

    return result, cost


def retrieval_recall(gold_passages: list[dict] | None, sources: list[dict]) -> float | None:
    """Compute set intersection recall on (title, chunk_index) tuples."""
    if gold_passages is None:
        return None
    if not gold_passages:
        return None

    gold_set = {(p["title"], p.get("chunk_index")) for p in gold_passages}
    retrieved_set = {(s["title"], s.get("chunk_index")) for s in sources}
    if not gold_set:
        return None
    return len(gold_set & retrieved_set) / len(gold_set)


def score_claim_recall(answer: str, key_claims: list[str], model: str, api_key: str, base_url: str) -> tuple[float, float]:
    """Score claim recall. Returns (score, cost)."""
    claims_str = "\n".join(f"- {c}" for c in key_claims)
    user_msg = f"Generated answer:\n{answer}\n\nKey claims to check:\n{claims_str}"

    result, cost = call_judge(CLAIM_RECALL_SYSTEM, user_msg, model, api_key, base_url)
    verdicts = result.get("verdicts", [])
    if not verdicts:
        return 0.0, cost

    present = sum(1 for v in verdicts if v.get("verdict", "").upper() == "PRESENT")
    return present / len(key_claims), cost


def score_faithfulness(answer: str, passages: list[str], model: str, api_key: str, base_url: str) -> tuple[float, float]:
    """Score faithfulness. Returns (score, cost)."""
    passages_text = "\n\n---\n\n".join(passages[:20])  # Limit to avoid token overflow
    user_msg = f"Generated answer:\n{answer}\n\nRetrieved passages:\n{passages_text}"

    result, cost = call_judge(FAITHFULNESS_SYSTEM, user_msg, model, api_key, base_url)
    claims = result.get("claims", [])
    if not claims:
        return 1.0, cost  # No claims = vacuously faithful

    supported = sum(1 for c in claims if c.get("verdict", "").upper() == "SUPPORTED")
    return supported / len(claims), cost


def score_quality(question: str, expected: str, answer: str, model: str, api_key: str, base_url: str) -> tuple[int, str, float]:
    """Score overall quality. Returns (score, reason, cost)."""
    user_msg = f"Question: {question}\n\nExpected answer: {expected}\n\nGenerated answer: {answer}"

    result, cost = call_judge(QUALITY_SYSTEM, user_msg, model, api_key, base_url)
    score = int(result.get("score", 3))
    reason = result.get("reason", "")
    return score, reason, cost


def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score eval runs with LLM judge")
    parser.add_argument("--fresh", action="store_true", help="Re-score everything from scratch")
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini"))
    parser.add_argument("--verbose", action="store_true", help="Print per-question details")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    if not api_key:
        print("Error: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load questions (keyed by question text)
    questions = {}
    with open(QUESTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                questions[q["question"]] = q

    # Load runs (take the latest run per question)
    if not os.path.exists(RUNS_FILE):
        print(f"Error: {RUNS_FILE} not found. Run `cargo run -- eval` first.", file=sys.stderr)
        sys.exit(1)

    runs = {}
    with open(RUNS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                run = json.loads(line)
                runs[run["question"]] = run  # Latest run wins

    # Load existing scores for resumability
    scored = set()
    existing_rows = []
    if not args.fresh and os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scored.add(row["question"])
                existing_rows.append(row)

    # Find questions to score
    to_score = []
    for q_text, q_data in questions.items():
        if q_text in runs and q_text not in scored:
            to_score.append((q_text, q_data, runs[q_text]))

    if not to_score:
        print("All questions already scored (or no matching runs found).")
        if existing_rows:
            print_summary(existing_rows)
        return

    print(f"Scoring {len(to_score)} questions with {args.judge_model}...")
    if scored:
        print(f"  Resuming — {len(scored)} already scored")

    # Write CSV header if starting fresh or file doesn't exist
    write_header = args.fresh or not os.path.exists(SCORES_FILE)
    if args.fresh and os.path.exists(SCORES_FILE):
        os.remove(SCORES_FILE)
        existing_rows = []

    all_rows = list(existing_rows)

    with open(SCORES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, (q_text, q_data, run) in enumerate(to_score):
            print(f"  [{len(scored) + i + 1}/{len(questions)}] {q_text[:70]}...")

            total_judge_cost = 0.0

            # 1. Retrieval recall
            rr = retrieval_recall(q_data.get("gold_passages"), run.get("sources", []))

            # 2. Claim recall
            key_claims = q_data.get("key_claims")
            if key_claims:
                cr, cost = score_claim_recall(run["final_answer"], key_claims, args.judge_model, api_key, base_url)
                total_judge_cost += cost
            else:
                cr = None

            # 3. Faithfulness
            passages = run.get("passages", [])
            if passages:
                faith, cost = score_faithfulness(run["final_answer"], passages, args.judge_model, api_key, base_url)
                total_judge_cost += cost
            else:
                faith = None

            # 4. Overall quality
            js, jr, cost = score_quality(q_text, q_data["expected_answer"], run["final_answer"], args.judge_model, api_key, base_url)
            total_judge_cost += cost

            row = {
                "question": q_text,
                "num_hops": len(run.get("hops", [])),
                "retrieval_recall": f"{rr:.3f}" if rr is not None else "",
                "claim_recall": f"{cr:.3f}" if cr is not None else "",
                "faithfulness": f"{faith:.3f}" if faith is not None else "",
                "judge_score": js,
                "judge_reason": jr,
                "agent_cost": f"{run.get('total_cost', 0):.6f}",
                "judge_cost": f"{total_judge_cost:.6f}",
                "latency_ms": run.get("total_latency_ms", 0),
                "run_id": run.get("id", ""),
            }

            writer.writerow(row)
            f.flush()
            all_rows.append(row)

            if args.verbose:
                print(f"    Claim recall: {cr:.3f}" if cr is not None else "    Claim recall: N/A")
                print(f"    Faithfulness: {faith:.3f}" if faith is not None else "    Faithfulness: N/A (no passages)")
                print(f"    Judge score:  {js}/5 — {jr}")
                print(f"    Judge cost:   ${total_judge_cost:.4f}")

    print(f"\nScores written to {SCORES_FILE}")
    print_summary(all_rows)


def print_summary(rows: list[dict]):
    """Print aggregate summary with bootstrap CIs."""
    n = len(rows)
    if n == 0:
        return

    def parse_floats(key):
        return [float(r[key]) for r in rows if r.get(key) not in (None, "")]

    claim_recalls = parse_floats("claim_recall")
    faithfulness_vals = parse_floats("faithfulness")
    judge_scores = parse_floats("judge_score")
    agent_costs = parse_floats("agent_cost")
    judge_costs = parse_floats("judge_cost")
    latencies = parse_floats("latency_ms")
    hops = parse_floats("num_hops")

    print(f"\n{'=' * 55}")
    print(f"  Evaluation Summary ({n} questions)")
    print(f"{'=' * 55}")
    print(f"{'':>16} {'Mean':>7}  {'95% CI':>16}  {'Min':>6}  {'Max':>6}")
    print(f"{'-' * 55}")

    for label, vals in [
        ("Claim Recall", claim_recalls),
        ("Faithfulness", faithfulness_vals),
        ("Judge Score", judge_scores),
    ]:
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        print(f"{label:>16} {np.mean(vals):>7.2f}  [{lo:.2f}, {hi:.2f}]  {min(vals):>6.2f}  {max(vals):>6.2f}")

    print(f"{'-' * 55}")
    avg_hops = np.mean(hops) if hops else 0
    avg_latency = np.mean(latencies) / 1000 if latencies else 0
    total_agent = sum(agent_costs)
    total_judge = sum(judge_costs)
    print(f"Avg hops: {avg_hops:.1f} | Avg latency: {avg_latency:.1f}s | Agent cost: ${total_agent:.2f} | Judge cost: ${total_judge:.2f}")


if __name__ == "__main__":
    main()
