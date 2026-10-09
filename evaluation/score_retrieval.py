"""Score aligned retrieval runs using binary relevance labels, never internal scores.

JSONL row: {"query_id":"q1", "relevant_ids":["d1"],
            "retrieved_ids":["d2","d1"], "latency_ms":12.0}
Use stable document/chunk IDs and the same labels for every compared run.
"""
import argparse
import json
import math
from pathlib import Path


def score_rows(rows, k=10):
    if k < 1 or not rows:
        raise ValueError("k must be positive and the run must be nonempty")
    seen = set()
    details = []
    for row in rows:
        qid = row['query_id']
        relevant_list = row['relevant_ids']
        retrieved = row['retrieved_ids']
        latency = row['latency_ms']
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("query_id must be a unique nonempty string")
        seen.add(qid)
        for ids in [relevant_list, retrieved]:
            if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
                raise ValueError("IDs must be lists of nonempty strings")
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate IDs would bias retrieval metrics")
        relevant = set(relevant_list)
        if not relevant:
            raise ValueError("retrieval relevance requires labels; evaluate unanswerable queries separately")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("latency_ms must be finite and nonnegative")
        hits = [int(doc in relevant) for doc in retrieved[:k]]
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
        details.append({
            'query_id': qid,
            'recall_at_k': sum(hits) / len(relevant),
            'mrr_at_k': next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
            'ndcg_at_k': dcg / ideal,
            'latency_ms': latency,
        })
    return {'queries': len(details), 'k': k,
            'mean': {key: sum(d[key] for d in details) / len(details)
                     for key in ['recall_at_k', 'mrr_at_k', 'ndcg_at_k', 'latency_ms']},
            'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.run.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(score_rows(rows, args.k), indent=2) + '\n')


if __name__ == '__main__':
    main()
