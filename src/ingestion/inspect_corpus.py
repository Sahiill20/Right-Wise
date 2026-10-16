from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sections.jsonl"
)


def load_sections() -> list[dict]:
    sections = []

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            sections.append(json.loads(line))

    return sections


def print_section(section: dict) -> None:
    print("=" * 80)

    print(f"Section      : {section['section_number']}")
    print(f"Title        : {section['section_title']}")
    print(f"Chapter      : {section['chapter']}")
    print(
        f"Pages        : "
        f"{section['page_start']} - {section['page_end']}"
    )

    print("\nText preview:")
    print(section["text"][:1000])

    print()


def main() -> None:
    sections = load_sections()

    print(f"Total sections: {len(sections)}")

    # Inspect representative sections.
    section_numbers = ["1", "2", "35", "38", "107"]

    for number in section_numbers:
        matching = [
            section
            for section in sections
            if section["section_number"] == number
        ]

        if not matching:
            print(f"WARNING: Section {number} not found.")
            continue

        print_section(matching[0])


if __name__ == "__main__":
    main()