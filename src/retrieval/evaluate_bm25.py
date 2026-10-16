import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


QUESTIONS_PATH = Path("data/eval/questions.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def retrieve(
    question: str,
    chunks: list[dict],
    bm25: BM25Okapi,
    top_k: int = 10,
) -> list[dict]:

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def main() -> None:
    print("Loading evaluation questions...")
    questions = load_json(QUESTIONS_PATH)
    print(f"Loaded questions: {len(questions)}")

    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    recall_at_5_hits = 0
    recall_at_10_hits = 0
    reciprocal_ranks = []

    print("\nRunning BM25 evaluation...\n")

    for question in questions:

        qid = question["id"]
        text = question["question"]

        expected_sections = {
            str(section)
            for section in question["expected_sections"]
        }

        results = retrieve(
            question=text,
            chunks=chunks,
            bm25=bm25,
            top_k=10,
        )

        returned_sections = [
            str(result["section_number"])
            for result in results
        ]

        # Recall@5
        top_5_sections = set(returned_sections[:5])

        hit_at_5 = bool(
            expected_sections.intersection(top_5_sections)
        )

        if hit_at_5:
            recall_at_5_hits += 1

        # Recall@10
        top_10_sections = set(returned_sections[:10])

        hit_at_10 = bool(
            expected_sections.intersection(top_10_sections)
        )

        if hit_at_10:
            recall_at_10_hits += 1

        # MRR
        rank = None

        for i, section in enumerate(returned_sections, start=1):
            if section in expected_sections:
                rank = i
                break

        reciprocal_ranks.append(
            0.0 if rank is None else 1.0 / rank
        )

        print(
            f"{qid} | "
            f"R@5={'YES' if hit_at_5 else 'NO'} | "
            f"R@10={'YES' if hit_at_10 else 'NO'} | "
            f"rank={rank}"
        )

    total = len(questions)

    recall_at_5 = recall_at_5_hits / total
    recall_at_10 = recall_at_10_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 60)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Questions   : {total}")
    print(f"Recall@5    : {recall_at_5:.4f}")
    print(f"Recall@10   : {recall_at_10:.4f}")
    print(f"MRR         : {mrr:.4f}")


if __name__ == "__main__":
    main()