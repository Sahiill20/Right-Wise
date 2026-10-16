import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.generation.llm_client import generate_answer
from src.generation.prompt_builder import build_prompt
from src.generation.citation_resolver import resolve_citations
from src.rag.pdf_ingestion import extract_pdf_chunks
from src.rag.query_utils import (
    build_retrieval_query,
    is_likely_follow_up,
    prioritize_explicit_section_chunks,
)

load_dotenv()


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RRF_K = 60
TOP_K = 10
COMBINED_PER_SOURCE_TOP_K = 5
MAX_CONTEXT_CHUNKS = 10


class RAGService:
    def __init__(self):
        print("Initializing RightWise RAG service...")

        self.chunks = self._load_jsonl(CHUNKS_PATH)
        self.dense_backend = os.getenv("DENSE_BACKEND", "numpy").strip().lower()

        if self.dense_backend not in {"numpy", "pinecone"}:
            raise ValueError(
                "DENSE_BACKEND must be either 'numpy' or 'pinecone'."
            )

        self.vector_store = None

        if self.dense_backend == "numpy":
            # Load the locally persisted vectors for the NumPy baseline.
            embeddings = self._load_jsonl(EMBEDDINGS_PATH)

            if len(self.chunks) != len(embeddings):
                raise ValueError(
                    "Chunk/embedding count mismatch: "
                    f"{len(self.chunks)} chunks vs {len(embeddings)} embeddings. "
                    "Rebuild the embeddings before starting RightWise."
                )

            for chunk, embedding in zip(self.chunks, embeddings):
                if chunk["chunk_id"] != embedding["chunk_id"]:
                    raise ValueError(
                        "Chunk/embedding order mismatch at "
                        f"{chunk['chunk_id']} vs {embedding['chunk_id']}. "
                        "Rebuild the embeddings before starting RightWise."
                    )
                chunk["embedding"] = embedding["embedding"]

            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Dense backend: local NumPy similarity search")

        else:
            # Pinecone + LangChain handle dense indexing/search for the
            # permanent Act corpus. User-uploaded PDF chunks are still embedded
            # locally with this same HuggingFaceEmbeddings object.
            from src.retrieval.vector_store import get_embeddings, get_vectorstore

            self.embedding_model = get_embeddings()
            self.vector_store = get_vectorstore()
            print("Dense backend: Pinecone + LangChain PineconeVectorStore")

        self.bm25 = self._build_bm25(self.chunks)
        print(f"RightWise RAG ready with {len(self.chunks)} Act chunks.")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _build_bm25(self, chunks: list[dict]) -> BM25Okapi:
        tokenized_chunks = [self._tokenize(chunk["text"]) for chunk in chunks]
        if not tokenized_chunks:
            raise ValueError("Cannot build BM25 over an empty corpus.")
        return BM25Okapi(tokenized_chunks)

    def _encode_query(self, query: str) -> np.ndarray:
        """Embed a query with the currently configured local embedding object."""
        if hasattr(self.embedding_model, "encode"):
            vector = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
            )
        else:
            vector = self.embedding_model.embed_query(query)
        return np.asarray(vector, dtype=np.float32)

    def _pinecone_dense_ranking(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[int]:
        """Return chunk indices in Pinecone similarity order.

        The current corpus is small (202 chunks), so retrieve all current Act
        records to preserve the same full dense ranking used by the NumPy
        baseline and keep the RRF comparison straightforward.
        """
        if self.vector_store is None:
            raise RuntimeError("Pinecone vector store is not initialized.")

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=len(chunks),
        )

        chunk_index = {
            str(chunk["chunk_id"]): index
            for index, chunk in enumerate(chunks)
        }
        ranked_indices = []
        seen = set()

        for document, _score in results:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            index = chunk_index.get(chunk_id)
            if index is not None and index not in seen:
                ranked_indices.append(index)
                seen.add(index)

        if len(ranked_indices) != len(chunks):
            raise RuntimeError(
                "Pinecone did not return all current RightWise Act chunks "
                f"(found {len(ranked_indices)} of {len(chunks)}). "
                "Run `python -m src.retrieval.index_pinecone --replace` "
                "to re-index the current chunks in the configured namespace."
            )

        return ranked_indices

    def _dense_ranking(self, query: str, chunks: list[dict]) -> list[int]:
        if not chunks:
            return []

        # Pinecone stores the permanent Act corpus. Uploaded PDFs remain
        # temporary in-session chunks and continue using local similarity.
        if self.dense_backend == "pinecone" and chunks is self.chunks:
            return self._pinecone_dense_ranking(query, chunks)

        query_embedding = self._encode_query(query)
        chunk_embeddings = np.asarray(
            [chunk["embedding"] for chunk in chunks],
            dtype=np.float32,
        )
        scores = chunk_embeddings @ query_embedding
        return list(np.argsort(scores)[::-1])

    def _bm25_ranking(self, query: str, bm25: BM25Okapi) -> list[int]:
        scores = bm25.get_scores(self._tokenize(query))
        return list(np.argsort(scores)[::-1])

    @staticmethod
    def _rrf_fusion(
        dense_indices: list[int],
        bm25_indices: list[int],
        top_k: int = TOP_K,
    ) -> list[int]:
        scores: dict[int, float] = {}

        for rank, index in enumerate(dense_indices, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (RRF_K + rank)

        for rank, index in enumerate(bm25_indices, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (RRF_K + rank)

        ranked_indices = sorted(scores, key=scores.get, reverse=True)
        return ranked_indices[:top_k]

    def _build_retrieval_query(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Include prior user-question context only for likely follow-ups."""
        return build_retrieval_query(query, conversation_history)

    @staticmethod
    def _source_targets(
        query: str,
        has_uploaded_pdf: bool,
    ) -> tuple[bool, bool]:
        """Return (search_act, search_uploaded_pdf) using lightweight intent cues.

        Clear document-only questions search the uploaded PDF. Clear Act/legal
        questions search the Act. Explicit mixed questions search both. Ambiguous
        consumer-problem questions search both when an upload is available.
        """
        if not has_uploaded_pdf:
            return True, False

        text = query.casefold()

        act_markers = (
            "consumer protection act", "under the act", "under this act",
            "under the law", "section ", "district commission",
            "state commission", "national commission", "consumer commission",
            "legal remedy", "legal remedies", "statutory", "file a complaint",
            "consumer complaint", "unfair trade practice", "product liability",
            "consumer rights", "limitation period", "time limit for filing",
        )
        document_markers = (
            "warranty", "warrantee", "uploaded pdf", "uploaded document",
            "this document", "the document", "this receipt", "my receipt",
            "receipt", "invoice", "bill", "agreement", "e-commerce policy",
            "this policy", "the policy", "according to the document",
            "according to this document",
        )

        act_specific = any(marker in text for marker in act_markers)
        document_specific = any(marker in text for marker in document_markers)

        if act_specific and document_specific:
            return True, True
        if act_specific:
            return True, False
        if document_specific:
            return False, True

        # Generic problems may need both the document and statutory context.
        return True, True

    def _retrieve_from_corpus(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        bm25: BM25Okapi | None = None,
    ) -> list[dict]:
        """Run dense + BM25 + RRF over one source corpus only."""
        if not chunks:
            return []

        active_bm25 = bm25 if bm25 is not None else self._build_bm25(chunks)
        dense_indices = self._dense_ranking(query, chunks)
        bm25_indices = self._bm25_ranking(query, active_bm25)
        fused_indices = self._rrf_fusion(
            dense_indices,
            bm25_indices,
            top_k=top_k,
        )
        ranked_chunks = [chunks[index] for index in fused_indices]

        # Only prioritize a section when the retrieval query explicitly names
        # it. This is a general section-number lookup, not a Section 39 rule.
        return prioritize_explicit_section_chunks(
            query=query,
            corpus_chunks=chunks,
            ranked_chunks=ranked_chunks,
            top_k=top_k,
        )

    @staticmethod
    def _interleave_sources(
        act_chunks: list[dict],
        uploaded_chunks: list[dict],
        max_chunks: int = MAX_CONTEXT_CHUNKS,
    ) -> list[dict]:
        """Combine source rankings without letting either corpus dominate by size."""
        combined: list[dict] = []
        seen_ids: set[str] = set()
        maximum_rank = max(len(act_chunks), len(uploaded_chunks))

        # Alternate sources by rank, uploaded document first for equal ranks.
        for rank in range(maximum_rank):
            for source_results in (uploaded_chunks, act_chunks):
                if rank >= len(source_results):
                    continue
                chunk = source_results[rank]
                chunk_id = chunk.get("chunk_id")
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                combined.append(chunk)
                if len(combined) >= max_chunks:
                    return combined

        return combined

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """Retrieve from the Consumer Protection Act corpus only."""
        return self._retrieve_from_corpus(
            query=query,
            chunks=self.chunks,
            top_k=top_k,
            bm25=self.bm25,
        )

    def answer_query(
        self,
        query: str,
        extra_chunks: list[dict] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        uploaded_chunks = extra_chunks or []

        retrieval_query = self._build_retrieval_query(
            query,
            conversation_history,
        )
        print("\nRetrieval query:")
        print(retrieval_query)

        search_act, search_uploaded = self._source_targets(
            retrieval_query,
            has_uploaded_pdf=bool(uploaded_chunks),
        )

        print(
            "\nSource routing: "
            f"Act={'ON' if search_act else 'OFF'}, "
            f"Uploaded PDF={'ON' if search_uploaded else 'OFF'}"
        )

        act_results = []
        uploaded_results = []

        if search_act:
            # Use more candidates for an Act-only query so Section 2 can still
            # be found when it ranks below five, as in the current benchmark.
            act_top_k = (
                COMBINED_PER_SOURCE_TOP_K
                if search_uploaded
                else TOP_K
            )
            act_results = self._retrieve_from_corpus(
                query=retrieval_query,
                chunks=self.chunks,
                top_k=act_top_k,
                bm25=self.bm25,
            )

        if search_uploaded and uploaded_chunks:
            upload_bm25 = self._build_bm25(uploaded_chunks)
            uploaded_results = self._retrieve_from_corpus(
                query=retrieval_query,
                chunks=uploaded_chunks,
                top_k=(
                    COMBINED_PER_SOURCE_TOP_K
                    if search_act
                    else TOP_K
                ),
                bm25=upload_bm25,
            )

        if search_act and search_uploaded:
            retrieved_chunks = self._interleave_sources(
                act_chunks=act_results,
                uploaded_chunks=uploaded_results,
                max_chunks=MAX_CONTEXT_CHUNKS,
            )
        elif search_uploaded:
            retrieved_chunks = uploaded_results[:MAX_CONTEXT_CHUNKS]
        else:
            retrieved_chunks = act_results[:MAX_CONTEXT_CHUNKS]

        print("\nRetrieved chunks:")
        for index, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.get("source_type", "act") == "uploaded_pdf":
                print(
                    f"{index}. {chunk.get('source_name', 'Uploaded PDF')} "
                    f"(page {chunk['page_start']}) — {chunk['chunk_id']}"
                )
            else:
                print(
                    f"{index}. Section {chunk['section_number']} - "
                    f"{chunk['section_title']} — {chunk['chunk_id']}"
                )

        # Prior turns clarify references in follow-up questions only. The prompt
        # builder explicitly treats history as conversational context, not legal
        # evidence; all legal claims must still be grounded in retrieved chunks.
        prompt_history = (
            conversation_history
            if is_likely_follow_up(query)
            else None
        )
        prompt = build_prompt(
            query,
            retrieved_chunks,
            conversation_history=prompt_history,
        )
        answer = generate_answer(prompt)
        # Use the normalized answer returned by the resolver so citation
        # formatting is consistent and the UI's disclaimer appears only once.
        answer, citations = resolve_citations(answer, retrieved_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }

    def prepare_uploaded_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> list[dict]:
        """Extract, chunk, and embed a user-uploaded PDF."""
        chunks = extract_pdf_chunks(
            file_bytes=file_bytes,
            filename=filename,
        )
        if not chunks:
            raise ValueError(
                "No readable text was found in the uploaded PDF. "
                "RightWise currently supports text-based PDFs only."
            )

        texts = [chunk["text"] for chunk in chunks]
        if hasattr(self.embedding_model, "encode"):
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
            )
        else:
            # LangChain HuggingFaceEmbeddings is used in Pinecone mode; this
            # keeps uploaded-PDF embeddings in the same BGE vector space.
            embeddings = self.embedding_model.embed_documents(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = np.asarray(
                embedding,
                dtype=np.float32,
            ).tolist()

        return chunks
