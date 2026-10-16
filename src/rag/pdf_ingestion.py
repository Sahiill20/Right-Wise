import re
from collections import defaultdict

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    normalized_lines = []
    previous_line_was_blank = False

    for line in lines:
        if not line:
            if normalized_lines and not previous_line_was_blank:
                normalized_lines.append("")

            previous_line_was_blank = True

        else:
            normalized_lines.append(line)
            previous_line_was_blank = False

    return "\n".join(normalized_lines).strip()


def extract_pdf_chunks(
    file_bytes: bytes,
    filename: str,
) -> list[dict]:
    """
    Extract text from an uploaded PDF and split it with LangChain.

    Each chunk preserves:
    - Unique chunk ID
    - Original filename
    - PDF page number
    - Source type
    """

    # ---------------------------------------------
    # Validate the uploaded file
    # ---------------------------------------------

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Please upload a PDF file.")

    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            "The PDF exceeds the maximum allowed size of 10 MB."
        )

    # ---------------------------------------------
    # Extract text page by page
    # ---------------------------------------------

    try:
        pdf = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )

    except Exception as exc:
        raise ValueError(
            "The uploaded file could not be opened as a PDF."
        ) from exc

    page_documents = []

    try:
        for page_number, page in enumerate(pdf, start=1):

            page_text = normalize_text(
                page.get_text("text")
            )

            if not page_text:
                continue

            document = Document(
                page_content=page_text,
                metadata={
                    "source": filename,
                    "source_name": filename,
                    "page_start": page_number,
                    "page_end": page_number,
                },
            )

            page_documents.append(document)

    finally:
        pdf.close()

    # No readable text was extracted.
    # The caller already handles an empty list.
    if not page_documents:
        return []

    # ---------------------------------------------
    # Split using LangChain
    # ---------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    split_documents = splitter.split_documents(
        page_documents
    )

    # ---------------------------------------------
    # Convert LangChain Documents to RightWise schema
    # ---------------------------------------------

    page_part_counts = defaultdict(int)
    chunks = []

    for document in split_documents:

        page_number = int(
            document.metadata["page_start"]
        )

        page_part_counts[page_number] += 1

        part_index = page_part_counts[page_number]

        text = document.page_content.strip()

        if not text:
            continue

        chunk = {
            "chunk_id": (
                f"upload_p{page_number}_{part_index:02d}"
            ),
            "act": None,
            "chapter": None,
            "section_number": None,
            "section_title": None,
            "part_index": part_index,
            "text": text,
            "page_start": page_number,
            "page_end": page_number,
            "source_type": "uploaded_pdf",
            "source_name": filename,
        }

        chunks.append(chunk)

    return chunks