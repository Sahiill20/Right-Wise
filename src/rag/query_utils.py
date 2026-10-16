"""Small, deterministic query helpers for RightWise retrieval and follow-ups."""

from __future__ import annotations

import re


_SECTION_REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sec\.?)\s*(\d+[a-z]?)\b",
    re.IGNORECASE,
)

_FOLLOW_UP_STARTERS = (
    "what about",
    "how about",
    "what if",
    "and what about",
    "and what if",
    "does that",
    "is that",
    "are they",
    "can they",
    "does it",
    "can it",
    "will that",
    "would that",
    "what happens then",
    "how does that",
    "why did you",
    "why didn't you",
    "why did not you",
    "why was that",
    "why is that",
    "why not",
    "what do you mean",
    "what's the reason",
    "what is the reason",
    "can you explain that",
    "please explain that",
)

_REFERENCE_WORDS = {
    "it",
    "that",
    "they",
    "them",
    "those",
    "these",
    "there",
    "same",
    "above",
    "previous",
    "earlier",
}


# These phrases commonly refer to a prior assistant response rather than
# introduce an entirely new question.
_FOLLOW_UP_META_PATTERN = re.compile(
    r"\byou\b.{0,70}\b(?:mention(?:ed)?|include(?:d)?|cite(?:d)?|"
    r"refer(?:red)? to|say|said|state(?:d)?|explain(?:ed)?)\b"
    r"|\b(?:your answer|the previous answer|previous answer|"
    r"that answer|the answer above|your previous response)\b",
    re.IGNORECASE,
)


def extract_requested_sections(query: str) -> list[str]:
    """Return explicitly requested section numbers in first-mention order."""
    result: list[str] = []
    seen: set[str] = set()
    for match in _SECTION_REFERENCE_PATTERN.finditer(query or ""):
        number = match.group(1).casefold()
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result


def is_likely_follow_up(query: str) -> bool:
    """Detect clear conversational follow-ups without an LLM call."""
    normalized = re.sub(r"\s+", " ", (query or "").casefold()).strip()
    if not normalized:
        return False

    if normalized.startswith(_FOLLOW_UP_STARTERS):
        return True

    if _FOLLOW_UP_META_PATTERN.search(normalized):
        return True

    words = re.findall(r"\b\w+\b", normalized)
    if len(words) <= 7 and any(word in _REFERENCE_WORDS for word in words):
        return True

    # Short replies such as "Why?" or "How?" are follow-ups when history exists.
    return normalized.rstrip(".!? ") in {"why", "how", "why exactly", "and why", "and how"}


def build_retrieval_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """Add the latest user question only when the current query is a follow-up."""
    if not conversation_history or not is_likely_follow_up(query):
        return query

    previous_user_query = None
    for message in reversed(conversation_history):
        if message.get("role") == "user":
            previous_user_query = (message.get("content") or "").strip()
            if previous_user_query:
                break

    if not previous_user_query:
        return query

    if previous_user_query.casefold() == query.strip().casefold():
        return query

    return (
        f"Previous user question: {previous_user_query}\n"
        f"Current user question: {query}"
    )


def prioritize_explicit_section_chunks(
    query: str,
    corpus_chunks: list[dict],
    ranked_chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Prioritize a section only when the user explicitly names it.

    This is a metadata lookup, not a Section 39 rule: the same behavior applies
    to any section number explicitly present in the current retrieval query.
    Remaining slots preserve the existing hybrid ranking.
    """
    requested_sections = extract_requested_sections(query)
    if not requested_sections:
        return ranked_chunks[:top_k]

    matching_by_section: dict[str, list[dict]] = {}
    for chunk in corpus_chunks:
        section_number = str(chunk.get("section_number", "")).strip().casefold()
        if section_number in requested_sections:
            matching_by_section.setdefault(section_number, []).append(chunk)

    prioritized: list[dict] = []
    seen_ids: set[str] = set()

    def add_chunk(chunk: dict) -> None:
        chunk_id = str(chunk.get("chunk_id", id(chunk)))
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            prioritized.append(chunk)

    for section_number in requested_sections:
        section_chunks = matching_by_section.get(section_number, [])
        section_chunks = sorted(
            section_chunks,
            key=lambda item: (
                int(item.get("part_index", 0) or 0),
                str(item.get("chunk_id", "")),
            ),
        )
        for chunk in section_chunks:
            add_chunk(chunk)

    for chunk in ranked_chunks:
        add_chunk(chunk)

    return prioritized[:top_k]
