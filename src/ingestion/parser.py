from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pymupdf  # PyMuPDF


# Matches legal section starts such as:
# "1. (1) This Act..."
# "35. (1) A complaint..."
# "107. (1) The Consumer Protection Act..."
SECTION_PATTERN = re.compile(
    r"^\s*(\d{1,3})\.\s+(.*)$"
)

# Matches chapter headings such as:
# "CHAPTER I"
# "CHAPTER IV"
CHAPTER_PATTERN = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+)\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    """
    Structured representation of one section of the Act.

    This structure will later become the foundation of our RAG chunks.
    """

    section_number: str
    section_title: Optional[str]
    chapter: Optional[str]

    page_start: int
    page_end: int

    text: str

def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_page_header_or_footer(text: str) -> bool:
    """
    Identify recurring Gazette/page header text that should not become
    part of a legal section's content.
    """

    normalized = normalize_text(text)

    if not normalized:
        return True

    # Examples from the PDF:
    # "8 THE GAZETTE OF INDIA EXTRAORDINARY [PART II—"
    # "SEC. 1] THE GAZETTE OF INDIA EXTRAORDINARY 9"
    if "THE GAZETTE OF INDIA EXTRAORDINARY" in normalized:
        return True

    if normalized.startswith("SEC. 1]"):
        return True

    # Standalone page numbers such as "18"
    if re.fullmatch(r"\d{1,3}", normalized):
        return True

    return False


def is_title_block(x0: float, text: str) -> bool:
    """
    Section titles in this particular PDF appear in the outer margins.

    Most are on the left (x < 110) or right (x > 480).
    """

    normalized = normalize_text(text)

    if not normalized:
        return False

    # Section titles are placed outside the main body text column.
    return x0 < 110 or x0 > 480


def extract_title_for_section(
    blocks: list[tuple],
    section_y: float,
) -> Optional[str]:
    """
    Find the title block corresponding to a section.

    The title is visually aligned with the start of the section but is
    extracted as a separate PDF text block.
    """

    candidates: list[tuple[float, str]] = []

    for block in blocks:
        x0, y0, x1, y1, text, *_ = block

        normalized = normalize_text(text)

        if not normalized:
            continue

        if not is_title_block(x0, normalized):
            continue

        # A section title is very close vertically to the section's
        # first body block.
        distance = abs(y0 - section_y)

        if distance <= 6:
            candidates.append((distance, normalized))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]


def clean_section_start(text: str, section_number: str) -> str:
    """
    Remove the section number from the first text block.

    Example:

        "35. (1) A complaint..."

    becomes:

        "(1) A complaint..."
    """

    pattern = re.compile(
        rf"^\s*{re.escape(section_number)}\.\s*"
    )

    return pattern.sub("", text, count=1).strip()


def parse_pdf(pdf_path: str | Path) -> list[Section]:
    """
    Parse the Consumer Protection Act PDF into structured sections.

    Each section keeps:
      - section number
      - section title
      - chapter
      - starting/ending page
      - full section text
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    sections: list[Section] = []

    current_section: Optional[dict] = None
    current_chapter: Optional[str] = None
    # True when we have encountered "CHAPTER X" and are waiting
    # to skip the chapter title that follows it.
    skip_chapter_title = False

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):
            page_number = page_index + 1

            # PDF text blocks preserve spatial information.
            blocks = page.get_text("blocks")

            # Reading order: top → bottom, left → right.
            blocks = sorted(
                blocks,
                key=lambda block: (block[1], block[0]),
            )

            for block in blocks:
                x0, y0, x1, y1, raw_text, *_ = block

                text = normalize_text(raw_text)

                if not text:
                    continue

                # ---------------------------------------------------------
                # Chapter detection
                # ---------------------------------------------------------
            
                chapter_match = CHAPTER_PATTERN.match(text)

                if chapter_match:
                    current_chapter = chapter_match.group(1)
                    skip_chapter_title = True
                    continue

                # Skip the title immediately following "CHAPTER X".
                if skip_chapter_title:
                    skip_chapter_title = False
                    continue

                # ---------------------------------------------------------
                # Ignore recurring page headers / footers
                # ---------------------------------------------------------
                if is_page_header_or_footer(text):
                    continue

                # ---------------------------------------------------------
                # Section detection
                # ---------------------------------------------------------
                section_match = SECTION_PATTERN.match(text)

                # Section body is inside the main text column.
                # This prevents unrelated margin text from being treated
                # as a new legal section.
                is_main_body = x0 >= 110

                if section_match and is_main_body:
                    section_number = section_match.group(1)

                    # Save previous section.
                    if current_section is not None:
                        sections.append(
                            Section(**current_section)
                        )

                    section_title = extract_title_for_section(
                        blocks,
                        section_y=y0,
                    )

                    first_text = clean_section_start(
                        text,
                        section_number,
                    )

                    current_section = {
                        "section_number": section_number,
                        "section_title": section_title,
                        "chapter": current_chapter,
                        "page_start": page_number,
                        "page_end": page_number,
                        "text": first_text,
                    }

                    continue

                # ---------------------------------------------------------
                # Add text to current section
                # ---------------------------------------------------------
                if current_section is not None:

                    # Don't include margin titles in the section body.
                    if is_title_block(x0, text):
                        continue

                    # Don't accidentally include chapter headings.
                    if CHAPTER_PATTERN.match(text):
                        continue

                    # Append the block with a space.
                    current_section["text"] += " " + text
                    current_section["page_end"] = page_number

        # Save final section.
        if current_section is not None:
            sections.append(
                Section(**current_section)
            )

    return sections


def validate_sections(sections: list[Section]) -> None:
    """
    Basic sanity checks for the parsed corpus.

    These checks are intentionally strict because errors in the
    source corpus will directly affect RAG retrieval and citations.
    """

    if not sections:
        raise ValueError("No sections were extracted from the PDF.")

    numbers = [int(section.section_number) for section in sections]

    # The Act should have sequentially numbered sections.
    expected = list(range(1, max(numbers) + 1))

    if numbers != expected:
        raise ValueError(
            "Section numbering is not sequential.\n"
            f"Extracted: {numbers}"
        )

    for section in sections:

        if not section.text.strip():
            raise ValueError(
                f"Section {section.section_number} has empty text."
            )

        if section.page_start > section.page_end:
            raise ValueError(
                f"Invalid page range for section "
                f"{section.section_number}."
            )


def section_to_dict(section: Section) -> dict:
    """
    Convert a Section dataclass into a JSON-serializable dictionary.
    """

    return asdict(section)