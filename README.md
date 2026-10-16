# Right-Wise

RightWise is a retrieval-augmented generation (RAG) application that helps users explore consumer-rights information in India using the **Consumer Protection Act, 2019** and a text-based PDF they upload, such as a warranty card, invoice, or agreement.

> **Disclaimer:** RightWise provides legal information, not legal advice. Its responses can be incomplete or mistaken. Verify important points against the cited source and consult a qualified professional for advice about a specific dispute.

## Features

- **Act corpus:** the law is parsed into sections and split into retrieval chunks with section/page metadata.
- **Hybrid retrieval:** dense-vector search through Pinecone, lexical search with BM25, and Reciprocal Rank Fusion (RRF).
- **PDF question answering:** users can upload a text-based PDF; extracted chunks retain document names and page numbers.
- **Grounded generation:** the prompt asks the LLM to answer from retrieved evidence and cite temporary source IDs.
- **Source references:** the UI maps citations to an Act section or uploaded PDF page and chunk ID.
- **Section-aware queries:** when a user explicitly names a section, RightWise can prioritize chunks tagged with that section.
- **Follow-up context:** recent conversation context can be used to resolve references in follow-up questions. History is context, not legal evidence.

## Architecture

```text
Consumer Protection Act PDF
          |
          v
Custom section parser --> Section-aware chunking --> Chunk corpus
                                                   |            |
                                                   v            v
                                             Pinecone       Local BM25
                                                   \            /
                                                    \          /
                                                     RRF fusion
                                                         |
Uploaded PDF --> PyMuPDF extraction --> text chunks ---+
                                                         |
                                                         v
                                               Evidence prompt
                                                         |
                                                         v
                                                   Groq LLM
                                                         |
                                                         v
                                         Citation resolver + Streamlit UI
```

The persistent Act corpus is indexed in Pinecone when `DENSE_BACKEND=pinecone`. Uploaded PDFs are ingested per session and searched locally; they are not added to the permanent Pinecone namespace.

## Technology stack

- Python 3.12
- Streamlit
- LangChain text splitters and document objects
- Sentence Transformers: `BAAI/bge-small-en-v1.5` (384-dimensional embeddings)
- Pinecone for persistent dense retrieval
- `rank-bm25` plus Reciprocal Rank Fusion for hybrid retrieval
- Groq API for answer generation
- PyMuPDF for PDF text extraction

## Project structure

```text
RightWise/
├── frontend/
│   └── app.py
├── src/
│   ├── generation/
│   │   ├── citation_resolver.py
│   │   ├── llm_client.py
│   │   └── prompt_builder.py
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── build_corpus.py
│   │   └── build_chunks.py
│   ├── rag/
│   │   ├── pdf_ingestion.py
│   │   ├── query_utils.py
│   │   └── rag_service.py
│   └── retrieval/
│       ├── vector_store.py
│       ├── index_pinecone.py
│       ├── embedder.py
│       └── evaluate_hybrid_pinecone.py
├── data/
│   ├── raw/                 # Source Act PDF
│   ├── processed/           # Generated sections, chunks, embeddings
│   └── eval/                # Retrieval benchmark questions
├── .env.example
├── requirements.txt
└── README.md
```

`query_utils.py` is included in the current focused patch for explicit section references and follow-up detection. If using an older checkout without it, apply the non-combined-query patch before relying on those behaviors.

## Setup (Windows PowerShell)

### 1. Create and activate a virtual environment

Run these commands from the project root (`D:\RightWise` in the current local setup):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use your machine's normal virtual-environment activation policy or run `.venv\Scripts\python.exe` explicitly.

### 2. Configure environment variables

Copy the sample file and fill in your own credentials:

```powershell
Copy-Item .env.example .env
```

Required for the default Pinecone + Groq setup:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `DENSE_BACKEND=pinecone`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`
- `PINECONE_CLOUD` and `PINECONE_REGION`

`GROQ_MODEL` is optional; if omitted, the code's configured default is used. Choose a model available to your Groq account. Do not commit `.env` or real API keys to source control.

### 3. Build local corpus artifacts (only if missing)

The source Act PDF is expected at `data/raw/consumer_protection_act_2019.pdf`. The generated files under `data/processed/` are ignored by Git, so rebuild them if they are absent:

```powershell
python src/ingestion/build_corpus.py
python src/ingestion/build_chunks.py
```

To create the local embedding file as well (needed for the NumPy dense backend and some local evaluation workflows):

```powershell
python -m src.retrieval.embedder
```

The current corpus baseline contains 107 parsed Act sections and 202 chunks; the BGE embedding dimension is 384.

### 4. Index the Act chunks in Pinecone

Set the index name, namespace, cloud, and region in `.env`, then run from the project root:

```powershell
python -m src.retrieval.index_pinecone
```

The script creates or reuses the configured index and upserts the current chunks. Do not use `--replace` casually: it deletes all vectors in the configured namespace before indexing. The index/namespace should be dedicated to the RightWise Act corpus.

### 5. Evaluate retrieval

```powershell
python -m src.retrieval.evaluate_hybrid_pinecone
```

This script sets `DENSE_BACKEND=pinecone` for its own process, runs the 20-question expected-section benchmark, and prints Recall@5, Recall@10, and MRR. It does not modify `.env`.

### 6. Run the Streamlit application

```powershell
$env:PYTHONPATH = (Get-Location).Path
streamlit run frontend/app.py --server.fileWatcherType none
```

Open the local Streamlit URL printed in the terminal. Upload a text-based PDF and ask a focused question. The application currently limits uploads to 10 MB and does not run OCR on scanned PDFs.

## Evaluation snapshot

The 20-question Pinecone + BM25 + RRF retrieval benchmark matched the existing NumPy baseline:

| Metric | NumPy baseline | Pinecone hybrid |
|---|---:|---:|
| Recall@5 | 0.9500 | 0.9500 |
| Recall@10 | 1.0000 | 1.0000 |
| MRR | 0.8338 | 0.8338 |

These metrics measure retrieval of expected sections for a small benchmark. They do **not** prove that every answer is legally correct, complete, or fully supported by its citations.

Recent manual smoke tests passed for: explicit Section 39 lookup; a follow-up asking whether punitive damages may be awarded; the Section 69 limitation-period question; and a warranty-only first-year coverage question. See `docs/MANUAL_EVALUATION.md` for details and limitations.

## Known limitations

- **Combined warranty + Act questions:** a known retrieval limitation remains. In the tested combined query, the final context included warranty chunks and Sections 84–86, but omitted Section 39. This has deliberately been deferred rather than addressed with section-forcing rules or a custom 7/3 source allocation.
- **Answer completeness:** finding a relevant section does not guarantee that every applicable rule or remedy is represented in the answer.
- **Citation validation:** citations are mapped to retrieved chunk metadata, but the resolver does not semantically prove that every cited passage entails its associated claim.
- **Text-based PDFs only:** scanned PDFs are not OCR-processed. PDF extraction may miss text in unusual layouts.
- **Model variability:** answers may vary between runs and can still be wrong. Manual review is necessary.
- **Legal scope:** the app is an information tool, not a substitute for legal advice.

## Manual test checklist

- [ ] Start Streamlit with `DENSE_BACKEND=pinecone`.
- [ ] Ask “What's Section 39 about?” and verify Section 39 chunks appear in `Retrieved chunks:`.
- [ ] Ask “Can it also award punitive damages?” and confirm the follow-up uses the correct section evidence.
- [ ] Ask “What is the limitation period for filing a consumer complaint under the Consumer Protection Act, 2019?” and verify Section 69 is cited.
- [ ] Upload the IFB refrigerator warranty card and ask “What does this warranty cover during the first year?”; verify document name and page/chunk references.
- [ ] Confirm the disclaimer appears once and sources map to the right Act section or uploaded PDF page.
- [ ] Record the combined-source retrieval issue as deferred; do not claim it has been solved.
