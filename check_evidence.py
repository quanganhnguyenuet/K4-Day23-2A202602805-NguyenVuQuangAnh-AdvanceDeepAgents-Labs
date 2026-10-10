"""Validate source provenance INSIDE the sandbox using host-retrieved evidence only."""
import json
import re
import sys

ROOT = "/tmp/work/research/"


def check(sources, ledger, report):
    actual = {(r["source"], r["url"]): r for r in ledger}
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["sources.json must be a nonempty array"]
    for source in sources:
        if not isinstance(source, dict):
            problems.append("source is not an object")
            continue
        found = actual.get((source.get("source"), source.get("url")))
        if not found:
            problems.append(f"unretrieved source/family: {source.get('url')}")
            continue
        for field in ("id", "title", "date"):
            normalize = lambda value: " ".join(str(value or "").split()).casefold()
            if normalize(source.get(field)) != normalize(found.get(field)):
                problems.append(f"{source['url']}: incorrect {field}; expected {found.get(field)!r}")
    body = re.split(r"(?m)^## References\s*$", report)[0]
    words = len(re.findall(r"\b[\w'-]+\b", body))
    if words < 1000:
        problems.append(f"survey body too brief ({words} words; need >= 1000 for substantive synthesis)")
    return problems


def main():
    try:
        with open(ROOT + "sources.json", encoding="utf-8") as file:
            sources = json.load(file)
        with open(ROOT + "evidence.json", encoding="utf-8") as file:
            ledger = json.load(file)
        with open("/tmp/work/report/report.md", encoding="utf-8") as file:
            report = file.read()
        problems = check(sources, ledger, report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"cannot check evidence: {exc}")
        return 1
    if problems:
        print("\n".join(problems))
        return 1
    print(f"EVIDENCE OK: {len(sources)} sources match actual retrieval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
