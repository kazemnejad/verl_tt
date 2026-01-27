#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent / "agent-docs"

EXCLUDED_DIRS = {"archive", "research"}


def compact_strings(values: list) -> list[str]:
    result = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            result.append(normalized)
    return result


def walk_markdown_files(directory: Path, base: Optional[Path] = None) -> list[str]:
    if base is None:
        base = directory

    files = []
    try:
        entries = sorted(directory.iterdir())
    except PermissionError:
        return files

    for entry in entries:
        if entry.name.startswith("."):
            continue

        if entry.is_dir():
            if entry.name in EXCLUDED_DIRS:
                continue
            files.extend(walk_markdown_files(entry, base))
        elif entry.is_file() and entry.suffix == ".md":
            files.append(str(entry.relative_to(base)))

    return sorted(files, key=str.lower)


def extract_metadata(full_path: Path) -> dict:
    try:
        content = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"summary": None, "read_when": [], "error": str(e)}

    if not content.startswith("---"):
        return {"summary": None, "read_when": [], "error": "missing front matter"}

    end_index = content.find("\n---", 3)
    if end_index == -1:
        return {"summary": None, "read_when": [], "error": "unterminated front matter"}

    front_matter = content[3:end_index].strip()
    lines = front_matter.split("\n")

    summary_line = None
    read_when: list[str] = []
    collecting_field = None

    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith("summary:"):
            summary_line = line
            collecting_field = None
            continue

        if line.startswith("read_when:"):
            collecting_field = "read_when"
            inline = line[len("read_when:") :].strip()
            if inline.startswith("[") and inline.endswith("]"):
                try:
                    parsed = json.loads(inline.replace("'", '"'))
                    if isinstance(parsed, list):
                        read_when.extend(compact_strings(parsed))
                except json.JSONDecodeError:
                    pass
            continue

        if collecting_field == "read_when":
            if line.startswith("- "):
                hint = line[2:].strip()
                if hint:
                    read_when.append(hint)
            elif line == "":
                pass
            else:
                collecting_field = None

    if not summary_line:
        return {"summary": None, "read_when": read_when, "error": "summary key missing"}

    summary_value = summary_line[len("summary:") :].strip()
    normalized = re.sub(r"\s+", " ", summary_value.strip("'\"")).strip()

    if not normalized:
        return {"summary": None, "read_when": read_when, "error": "summary is empty"}

    return {"summary": normalized, "read_when": read_when}


def main():
    print("Listing all markdown files in docs folder:")

    markdown_files = walk_markdown_files(DOCS_DIR)

    for relative_path in markdown_files:
        full_path = DOCS_DIR / relative_path
        meta = extract_metadata(full_path)
        summary = meta.get("summary")
        read_when = meta.get("read_when", [])
        error = meta.get("error")

        if summary:
            print(f"{relative_path} - {summary}")
            if read_when:
                print(f"  Read when: {'; '.join(read_when)}")
        else:
            reason = f" - [{error}]" if error else ""
            print(f"{relative_path}{reason}")

    print(
        '\nReminder: keep docs up to date as behavior changes. When your task matches any "Read when" hint above (React hooks, cache directives, database work, tests, etc.), read that doc before coding, and suggest new coverage when it is missing.'
    )


if __name__ == "__main__":
    main()
