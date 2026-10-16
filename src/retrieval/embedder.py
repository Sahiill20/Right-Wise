import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(path: Path) -> list[dict]:
    """Load chunk records from JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    print(f"Loading chunks from: {CHUNKS_PATH}")

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Embedding dimension: {embeddings.shape[1]}")

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as f:
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                "chunk_id": chunk["chunk_id"],
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "chapter": chunk["chapter"],
                "part_index": chunk["part_index"],
                "text": chunk["text"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "embedding": embedding.tolist(),
            }

            f.write(json.dumps(record) + "\n")

    print(f"Written embeddings to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()