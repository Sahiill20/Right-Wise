from __future__ import annotations

import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)


def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def main() -> None:
    chunks = load_chunks()

    lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"Total chunks : {len(chunks)}")
    print(f"Min chars    : {min(lengths)}")
    print(f"Max chars    : {max(lengths)}")
    print(f"Average      : {statistics.mean(lengths):.2f}")
    print(f"Median       : {statistics.median(lengths):.2f}")

    print(
        f"Chunks > 3000 chars : "
        f"{sum(length > 3000 for length in lengths)}"
    )

    print(
        f"Chunks > 5000 chars : "
        f"{sum(length > 5000 for length in lengths)}"
    )

    print("\nLargest chunks:\n")

    largest = sorted(
        chunks,
        key=lambda chunk: len(chunk["text"]),
        reverse=True,
    )[:10]

    for chunk in largest:
        print(
            f"{chunk['chunk_id']} | "
            f"Section {chunk['section_number']} | "
            f"{len(chunk['text'])} chars"
        )


if __name__ == "__main__":
    main()