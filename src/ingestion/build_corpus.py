from __future__ import annotations

import json
from pathlib import Path

from parser import parse_pdf, section_to_dict, validate_sections


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = (PROJECT_ROOT/ "data"/ "raw" / "consumer_protection_act_2019.pdf")

OUTPUT_PATH = (PROJECT_ROOT/ "data"/ "processed"/ "sections.jsonl")


def main() -> None:
    print(f"Reading PDF: {PDF_PATH}")

    sections = parse_pdf(PDF_PATH)

    print(f"Extracted sections: {len(sections)}")

    validate_sections(sections)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True,)

    with OUTPUT_PATH.open("w", encoding="utf-8",) as file:

        for section in sections:
            record = section_to_dict(section)

            file.write(json.dumps(record, ensure_ascii=False,)+ "\n")

    print(f"Written to: {OUTPUT_PATH}")

    print("\nFirst 3 sections:\n")

    for section in sections[:3]:
        print(
            f"Section {section.section_number}: "
            f"{section.section_title}"
        )

    print("\nLast 3 sections:\n")

    for section in sections[-3:]:
        print(
            f"Section {section.section_number}: "
            f"{section.section_title}"
        )


if __name__ == "__main__":
    main()