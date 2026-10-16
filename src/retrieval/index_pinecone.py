"""Index the current Consumer Protection Act chunks in Pinecone.

Run from the RightWise project root:
    python -m src.retrieval.index_pinecone --replace

This follows the LangChain PineconeVectorStore pattern used in the IT Support
Agentic RAG project. Documents are embedded by HuggingFaceEmbeddings during
upsert, using the same BGE model used by RightWise.
"""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.retrieval.vector_store import (
    add_docs,
    create_index,
    get_namespace,
    get_vectorstore,
)

load_dotenv()

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Build the Act corpus and chunks first."
        )

    with path.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if not chunks:
        raise ValueError(f"No chunks were found in {path}.")

    return chunks


def chunk_to_document(chunk: dict) -> Document:
    """Convert a RightWise chunk into a Pinecone-safe LangChain Document."""
    metadata = {
        "chunk_id": str(chunk["chunk_id"]),
        "source_type": "act",
        "source_name": "Consumer Protection Act, 2019",
        "section_number": str(chunk.get("section_number") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "part_index": int(chunk.get("part_index") or 1),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
    }

    if chunk.get("chapter") is not None:
        metadata["chapter"] = str(chunk["chapter"])

    if chunk.get("act") is not None:
        metadata["act"] = str(chunk["act"])

    return Document(
        page_content=chunk["text"],
        metadata=metadata,
    )


def get_namespace_count(index, namespace: str) -> int:
    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", None)

    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})

    namespace_stats = (namespaces or {}).get(namespace)
    if namespace_stats is None:
        return 0

    count = getattr(namespace_stats, "vector_count", None)
    if count is None and isinstance(namespace_stats, dict):
        count = namespace_stats.get("vector_count", 0)

    return int(count or 0)


def wait_for_namespace_count(
    index,
    namespace: str,
    expected_count: int,
    timeout_seconds: int = 90,
) -> int:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        count = get_namespace_count(index, namespace)
        if count >= expected_count:
            return count
        print(
            f"Waiting for Pinecone indexing: {count}/{expected_count} records..."
        )
        time.sleep(3)

    raise TimeoutError(
        f"Pinecone namespace '{namespace}' contains "
        f"{get_namespace_count(index, namespace)} records; "
        f"expected at least {expected_count}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index RightWise's Consumer Protection Act chunks in Pinecone."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Delete all records in the configured RightWise namespace before indexing. "
            "Only use this namespace for the Act corpus."
        ),
    )
    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    ids = [str(chunk["chunk_id"]) for chunk in chunks]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk_id values were found in chunks.jsonl.")

    print(f"Loaded Act chunks: {len(chunks)}")
    print(f"Target Pinecone namespace: {get_namespace()}")
    print("Embedding model: BAAI/bge-small-en-v1.5")

    index = create_index()

    if args.replace:
        print(
            "Clearing only the configured RightWise namespace "
            f"'{get_namespace()}'..."
        )
        index.delete(delete_all=True, namespace=get_namespace())

    # Construct the LangChain vector store after index creation/clearing.
    vector_store = get_vectorstore()
    documents = [chunk_to_document(chunk) for chunk in chunks]

    for start in range(0, len(documents), BATCH_SIZE):
        batch_documents = documents[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        add_docs(batch_documents, ids=batch_ids)
        end = min(start + BATCH_SIZE, len(documents))
        print(f"Indexed {end}/{len(documents)} chunks")

    count = wait_for_namespace_count(
        index=index,
        namespace=get_namespace(),
        expected_count=len(documents),
    )

    print(f"Pinecone namespace record count: {count}")

    print("\nRunning a small vector-search smoke test...")
    results = vector_store.similarity_search_with_score(
        query="consumer definition under the Consumer Protection Act",
        k=min(3, len(documents)),
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(
            f"{rank}. chunk={document.metadata.get('chunk_id')} | "
            f"section={document.metadata.get('section_number')} | "
            f"score={float(score):.4f}"
        )

    print("\nPinecone indexing completed.")
    print(
        "Next: run `python -m src.retrieval.evaluate_hybrid_pinecone` "
        "before switching DENSE_BACKEND to pinecone."
    )


if __name__ == "__main__":
    main()
