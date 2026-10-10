"""Convert exact title/year citations to [n], only INSIDE the sandbox.

This mechanical repair never adds sources or rewrites research claims. Ambiguous
titles and incorrect years fail closed; the provided finalizer runs afterwards.
"""
import json
import re
import sys
from pathlib import Path

REPORT = Path("/tmp/work/report/report.md")
SOURCES = Path("/tmp/work/research/sources.json")


def key(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def normalize(report, sources):
    body = re.split(r"(?m)^##\s+References\s*$", report)[0].rstrip()
    titles = {}
    for source in sources:
        title = re.sub(r"^\[\d{4}\.\d+\]\s*", "", source["title"])
        for alias in {title, title.split(" | ", 1)[0]}:
            titles.setdefault(key(alias), {})[source["n"]] = source
    problems = []
    urls = {s["url"]: s["n"] for s in sources}

    def replace(match):
        title, year = match.groups()
        matches = list(titles.get(key(title), {}).values())
        if len(matches) != 1:
            problems.append(f"Citation title is unknown or ambiguous: {title!r}; use an exact numbered citation")
            return match.group(0)
        source = matches[0]
        date = source.get("date") or "n.d."
        if (year == "n.d." and date != "n.d.") or (year != "n.d." and not date.startswith(year)):
            problems.append(f"Citation year differs from retrieved metadata for {title!r}: {year} vs {date}")
            return match.group(0)
        return f"[{source['n']}]"

    # Leave fenced/inline code intact. The finalizer will rebuild References.
    segments = re.split(r"(```.*?```|`[^`\n]*`)", body, flags=re.S)
    for i in range(0, len(segments), 2):
        segments[i] = re.sub(r"\(([^()\n]{8,}?),\s*(\d{4}|n\.d\.)\)", replace, segments[i])
        def replace_link(match):
            url = match.group(2)
            if url not in urls:
                problems.append(f"Citation URL absent from sources.json: {url}")
                return match.group(0)
            return f"[{urls[url]}]"
        segments[i] = re.sub(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", replace_link, segments[i])
        segments[i] = re.sub(r"\(\[(\d+)\]\)", r"[\1]", segments[i])
    return "".join(segments) + "\n", list(dict.fromkeys(problems))


def main():
    try:
        text, errors = normalize(REPORT.read_text(encoding="utf-8"), json.loads(SOURCES.read_text(encoding="utf-8")))
        if errors:
            print("\n".join(errors))
            return 1
        REPORT.write_text(text, encoding="utf-8")
        print("NORMALIZED: exact retrieved title/year citations; run the provided finalizer next")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot normalize: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
