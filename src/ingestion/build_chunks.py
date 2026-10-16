from __future__ import annotations

import json
from pathlib import Path

from chunker import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_sections,)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sections.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_sections() -> list[dict]:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    sections = load_sections()
    print(f"Loaded sections: {len(sections)}")
    print(
        f"Chunk settings: size={DEFAULT_CHUNK_SIZE}, "
        f"overlap={DEFAULT_CHUNK_OVERLAP}"
    )

    chunks = chunk_sections(
        sections,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Generated chunks: {len(chunks)}")
    print(f"Written to: {OUTPUT_PATH}")

    for chunk in chunks[:5]:
        print(
            f"{chunk['chunk_id']} | Section {chunk['section_number']} | "
            f"{len(chunk['text'])} chars | "
            f"Pages {chunk['page_start']}-{chunk['page_end']}"
        )


if __name__ == "__main__":
    main()
