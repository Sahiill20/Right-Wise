import re


# Accept common formatting variants emitted by the LLM, then normalize them.
SOURCE_PATTERN = re.compile(
    r"(?:\[\s*S(\d+)\s*\]|【\s*S(\d+)\s*】)",
    re.IGNORECASE,
)

# The UI displays a fixed disclaimer. Remove any LLM-generated copy so it
# does not appear twice. Handles plain text and common Markdown emphasis.
DISCLAIMER_PATTERN = re.compile(
    r"(?:\s*[*_`]{0,3}\s*This response provides legal information,\s*"
    r"not legal advice\.\s*[*_`]{0,3}\s*)+$",
    re.IGNORECASE,
)


def normalize_answer_format(answer: str) -> str:
    """Normalize citation spacing and remove a trailing duplicate disclaimer."""

    def normalize_source(match: re.Match) -> str:
        number_text = match.group(1) or match.group(2)
        return f"[S{int(number_text)}]"

    answer = SOURCE_PATTERN.sub(normalize_source, answer)
    answer = DISCLAIMER_PATTERN.sub("", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def resolve_citations(
    answer: str,
    retrieved_chunks: list[dict],
) -> tuple[str, list[dict]]:
    """Normalize model citations and map source IDs to retrieved chunk metadata.

    This validates that a cited source ID exists in the retrieved list; it does
    not prove that the cited text semantically supports the associated claim.
    """

    normalized_answer = normalize_answer_format(answer)
    citations = []
    seen_source_numbers = set()

    for match in SOURCE_PATTERN.finditer(normalized_answer):
        source_number = int(match.group(1) or match.group(2))

        if not 1 <= source_number <= len(retrieved_chunks):
            continue

        if source_number in seen_source_numbers:
            continue
        seen_source_numbers.add(source_number)

        chunk = retrieved_chunks[source_number - 1]

        citation = {
            "source_id": f"S{source_number}",
            "source_type": chunk.get("source_type", "act"),
            "source_name": chunk.get("source_name"),
            "section_number": chunk.get("section_number"),
            "section_title": chunk.get("section_title"),
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
        }
        citations.append(citation)

    return normalized_answer, citations
