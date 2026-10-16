"""Pinecone vector-store helpers for RightWise.

The structure intentionally follows the vectorStore.py pattern used in the
IT Support Agentic RAG project, but keeps RightWise's BGE embedding model and
legal corpus metadata.
"""

from __future__ import annotations

import os
import time
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = {
    "baai/bge-small-en-v1.5": 384,
}

DEFAULT_INDEX_NAME = "rightwise-consumer-act"
DEFAULT_NAMESPACE = "rightwise-act-v1"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"

embeddings = None
vectorStore = None
pineconeClient = None
pineconeIndex = None


def get_embedding_dimensions(model_name: str | None = None) -> int:
    """Return the known vector dimension for the configured embedding model."""
    name = (model_name or EMBEDDING_MODEL).strip().lower()

    if name not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unsupported embedding model '{model_name or EMBEDDING_MODEL}'. "
            "Add its output dimension to EMBEDDING_DIMENSIONS first."
        )

    return EMBEDDING_DIMENSIONS[name]


def get_index_name() -> str:
    return os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME).strip()


def get_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE).strip()


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local Hugging Face embeddings object once and reuse it."""
    global embeddings

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return embeddings


def _get_pinecone_client() -> Pinecone:
    global pineconeClient

    if pineconeClient is None:
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is missing. Add it to your .env file."
            )

        pineconeClient = Pinecone(api_key=api_key)

    return pineconeClient


def _read_value(obj, key: str, default=None):
    """Read a field from either a Pinecone model object or dictionary."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    value = getattr(obj, key, default)
    if value is not default:
        return value

    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return default


def _get_index_names(client: Pinecone) -> set[str]:
    names = set()

    for item in client.list_indexes():
        name = _read_value(item, "name")
        if name:
            names.add(str(name))

    return names


def create_index():
    """Create the configured 384-dimensional cosine index if it is absent.

    An existing index with an incompatible dimension is not deleted. Instead,
    this raises a clear error so RightWise cannot destroy another index by
    silently recreating it.
    """
    global pineconeIndex

    if pineconeIndex is not None:
        return pineconeIndex

    client = _get_pinecone_client()
    index_name = get_index_name()
    dimension = get_embedding_dimensions()
    cloud = os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD).strip()
    region = os.getenv("PINECONE_REGION", DEFAULT_REGION).strip()

    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME cannot be empty.")

    existing_names = _get_index_names(client)

    if index_name in existing_names:
        description = client.describe_index(index_name)
        current_dimension = _read_value(description, "dimension")

        if current_dimension is not None and int(current_dimension) != dimension:
            raise RuntimeError(
                f"Pinecone index '{index_name}' has dimension "
                f"{current_dimension}; RightWise requires {dimension}. "
                "Choose a new PINECONE_INDEX_NAME or recreate the intended index "
                "manually after confirming it contains no data you need."
            )

        print(f"Using existing Pinecone index: {index_name}")

    else:
        print(f"Creating Pinecone index: {index_name}")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        # This SDK version returns only after creation is requested; wait until
        # Pinecone reports the index ready before opening a data-plane client.
        while True:
            description = client.describe_index(index_name)
            status = _read_value(description, "status", {}) or {}
            ready = _read_value(status, "ready", False)
            if ready:
                break
            time.sleep(1)

        print("Pinecone index is ready.")

    pineconeIndex = client.Index(index_name)
    return pineconeIndex


def get_index():
    """Return the configured Pinecone index connection."""
    return create_index()


def get_vectorstore() -> PineconeVectorStore:
    """Create and cache the LangChain PineconeVectorStore."""
    global vectorStore

    if vectorStore is None:
        vectorStore = PineconeVectorStore(
            index=get_index(),
            embedding=get_embeddings(),
            text_key="text",
            namespace=get_namespace(),
        )

    return vectorStore


def get_retriever(k: int = 5):
    """Return a standard LangChain retriever for simple vector-only searches."""
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k}
    )


def add_docs(
    documents: Iterable[Document],
    ids: list[str] | None = None,
):
    """Add LangChain Documents to the configured index."""
    docs = list(documents)
    if not docs:
        return []

    if ids is not None and len(ids) != len(docs):
        raise ValueError("The number of document IDs must match the documents.")

    return get_vectorstore().add_documents(
        documents=docs,
        ids=ids,
    )
