from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
ACT_NAME = "Consumer Protection Act, 2019"


def chunk_sections(
    sections: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split legal sections while retaining their source metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = []

    for section in sections:
        text = section["text"].strip()

        if not text:
            continue

        # Process each section separately so chunks never cross sections.
        document = Document(
            page_content=text,
            metadata={
                "act": ACT_NAME,
                "chapter": section.get("chapter"),
                "section_number": str(section["section_number"]),
                "section_title": section.get("section_title"),
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            },
        )

        section_chunks = splitter.split_documents([document])

        for part_index, chunk in enumerate(section_chunks, start=1):
            section_number = chunk.metadata["section_number"]

            chunks.append(
                {
                    "chunk_id": f"cpa_s{section_number}_{part_index:02d}",
                    **chunk.metadata,
                    "part_index": part_index,
                    "text": chunk.page_content.strip(),
                }
            )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique.")

    return chunks
