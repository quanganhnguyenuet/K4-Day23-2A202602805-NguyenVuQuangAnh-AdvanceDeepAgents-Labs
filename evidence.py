"""Run-scoped evidence ledger and guards against invented research artifacts."""
import json
import re
import threading

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

_lock = threading.Lock()
_records = {}


def clean(value):
    return " ".join(str(value or "").split())


def reset():
    with _lock:
        _records.clear()


def records():
    with _lock:
        return [dict(record) for record in _records.values()]


def register(family, entries):
    with _lock:
        for entry in entries:
            record = {
                "id": entry.get("id", ""), "url": entry["url"], "title": clean(entry.get("title")),
                "date": entry.get("published") or entry.get("date") or "n.d.", "source": family,
                "summary": clean(entry.get("summary"))[:600],
                "text": str(entry.get("text") or entry.get("summary") or "")[:12000],
            }
            _records[(family, record["url"])] = record


def register_web(text, url=None):
    entries = []
    if url:
        heading = re.search(r"(?m)^#+\s+(.+)$", text)
        title = heading.group(1) if heading else url
        entries.append({"url": url, "title": title, "summary": text, "text": text})
    else:
        for block in re.split(r"(?m)(?=^Title:)", text):
            match = re.search(r"(?m)^URL:\s*(https?://\S+)", block)
            title = re.search(r"(?m)^Title:\s*(.+)", block)
            if not match or not title:
                continue
            published = re.search(r"(?m)^Published:\s*(\d{4}-\d{2}-\d{2})", block)
            entries.append({"url": match.group(1), "title": title.group(1),
                            "published": published.group(1) if published else "n.d.",
                            "summary": block.split("Highlights:", 1)[-1], "text": block})
    # Fetching a previously discovered paper extends evidence without changing its metadata.
    for entry in entries:
        with _lock:
            existing = [key for key in _records if key[1] == entry["url"]]
            for key in existing:
                _records[key]["text"] = (_records[key]["text"] + "\n" + entry["text"])[:24000]
        if not url or not existing:
            register("web", [entry])


def source_problems(sources, ledger=None):
    actual = {(r["source"], r["url"]): r for r in (records() if ledger is None else ledger)}
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["sources must be a nonempty array of retrieved records"]
    for source in sources:
        if not isinstance(source, dict):
            problems.append("source must be an object")
            continue
        found = actual.get((source.get("source"), source.get("url")))
        if not found:
            problems.append(f"source was not retrieved by its declared tool family: {source.get('url')}")
            continue
        for field in ("id", "title", "date"):
            if clean(source.get(field)).casefold() != clean(found[field]).casefold():
                problems.append(f"{source['url']}: {field} differs from retrieved metadata; expected {found[field]!r}")
    return problems


def note_problems(content):
    blocks = re.split(r"(?m)^###\s+", content)[1:]
    if not blocks:
        return ["notes need ### source blocks with retrieved metadata and evidence quotations"]
    problems = []
    for block in blocks:
        source = {"title": block.splitlines()[0]}
        for field in ("id", "url", "date", "source"):
            match = re.search(rf"(?m)^{field}:[ \t]*(.*)$", block)
            source[field] = match.group(1).strip() if match else ""
        found_problems = source_problems([source])
        problems.extend(found_problems)
        if found_problems:
            continue
        found = next(r for r in records() if (r["source"], r["url"]) == (source["source"], source["url"]))
        quotations = re.findall(r'(?m)^\s*-\s*["“](.+?)["”]\s*$', block)
        if not quotations:
            problems.append(f"{source['url']}: include a verbatim Evidence quotation")
        for quotation in quotations:
            pieces = re.split(r"\.\.\.|…", quotation)
            if not all(clean(piece).casefold() in clean(found["text"]).casefold() for piece in pieces if clean(piece)):
                problems.append(f"{source['url']}: evidence quotation does not appear in retrieved text")
    return problems


class EvidenceGuard(AgentMiddleware):
    """Validate notes/source writes before the filesystem tool performs them."""

    def __init__(self, role):
        self.role = role

    def _reject(self, request):
        call = request.tool_call
        args = call.get("args", {})
        if call["name"] != "write_file":
            return None
        path, content = args.get("file_path", ""), args.get("content", "")
        if path.startswith("/tmp/work/research/notes/"):
            if self.role == "lead":
                problems = ["Lead must NOT pre-create notes or write placeholder evidence. Delegate researcher first."]
            else:
                problems = note_problems(content)
        elif path == "/tmp/work/research/sources.json":
            if self.role == "lead":
                return ToolMessage(content="ERROR: use compile_sources; do not hand-write sources.json.",
                                   tool_call_id=call["id"], name=call["name"], status="error")
            try:
                problems = source_problems(json.loads(content))
            except ValueError:
                problems = ["sources.json must contain one valid JSON array, no Markdown or trailing data"]
        else:
            return None
        if not problems:
            return None
        return ToolMessage(
            content="ERROR: evidence write rejected; no file was written. " + "; ".join(problems[:8])
                    + "\nUse retrieved_sources for exact metadata and save_research_notes for verified notes.",
            tool_call_id=call["id"], name=call["name"], status="error",
        )

    def wrap_tool_call(self, request, handler):
        rejected = self._reject(request)
        return rejected if rejected is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        rejected = self._reject(request)
        return rejected if rejected is not None else await handler(request)
