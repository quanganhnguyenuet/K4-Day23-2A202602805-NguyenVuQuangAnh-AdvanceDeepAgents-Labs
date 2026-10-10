"""Host-side source tools. Network failures become strings, never agent crashes."""
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

import evidence

load_dotenv()
ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
RETRY_STATUSES = {429, 500, 502, 503, 504}
_arxiv_lock = threading.Lock()
_last_arxiv_call = None
_arxiv_cooldown_until = 0.0


class RetryableError(Exception):
    """Temporary failure; retry_after is an optional delay in seconds."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Retry only RetryableError, with capped exponential backoff and jitter."""
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("attempts must be positive; base and cap must be non-negative")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = max(0.0, float(exc.retry_after))
            else:
                delay = base * 2**attempt + random.uniform(0, base)
            time.sleep(min(cap, delay))


def redact(text):
    """Remove configured secrets, including URL-encoded values, from diagnostics."""
    text = str(text)
    for name, value in os.environ.items():
        if value and len(value) >= 4 and (
            name.endswith(("API_KEY", "SECRET", "TOKEN", "PASSWORD")) or name == "OPENAI_KEY"
        ):
            for variant in {value, quote(value, safe=""), quote_plus(value)}:
                text = text.replace(variant, "[REDACTED]")
    return re.sub(r"""(?i)(exaApiKey=)[^\s&"'<>]+""", r"\1[REDACTED]", text)


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {redact(exc)}"


def _retry_after(value):
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _request(method, url, **kwargs):
    """One HTTP attempt; callers wrap this function in with_retry."""
    try:
        timeout = kwargs.pop("timeout", 45)
        response = httpx.request(method, url, timeout=timeout, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(redact(exc)) from exc
    if response.status_code in RETRY_STATUSES:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response.headers.get("Retry-After")))
    response.raise_for_status()
    return response


def _clean(value):
    return " ".join(str(value or "").split())


def _records_text(records, family=None):
    if not records:
        return "NO RESULTS"
    if family:
        evidence.register(family, records)
    return redact(json.dumps([dict(record, summary=record["summary"][:600]) for record in records], ensure_ascii=False))


def _arxiv_request(params):
    global _last_arxiv_call
    # arXiv requires both three-second spacing and only one active connection.
    with _arxiv_lock:
        if time.monotonic() < _arxiv_cooldown_until:
            raise RuntimeError("arXiv temporarily unavailable; reuse retrieved_sources or query HF/web; preserve actual discovery family")
        if _last_arxiv_call is not None:
            time.sleep(max(0.0, 3.0 - (time.monotonic() - _last_arxiv_call)))
        _last_arxiv_call = time.monotonic()
        return _request("GET", ARXIV_URL, params=params, timeout=httpx.Timeout(20, connect=10))


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv keywords, newest first. JSON: id, url, published, title, summary.
    Abstracts are truncated to 600 characters. Returns NO RESULTS or ERROR on failure.
    """
    global _arxiv_cooldown_until
    try:
        query = re.sub(r"\b(?:all|ti|au|abs|cat)\s*:", " ", query, flags=re.I)
        stopwords = {"a", "an", "and", "about", "for", "from", "in", "of", "on", "or", "survey", "the", "to", "with"}
        terms = [term for term in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query)
                 if term.upper() not in {"AND", "OR", "NOT"} and term.casefold() not in stopwords][:6]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{term}" for term in terms),
                  "sortBy": "submittedDate", "sortOrder": "descending", "start": 0,
                  "max_results": max(1, min(30, max_results))}
        response = with_retry(lambda: _arxiv_request(params), attempts=3, base=3, cap=10)
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = entry.findtext("a:id", "", ns)
            if "/abs/" not in raw_id:
                if "error" in raw_id.lower():
                    raise ValueError(_clean(entry.findtext("a:summary", "arXiv query failed", ns)))
                continue
            paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                            "published": entry.findtext("a:published", "", ns)[:10],
                            "title": _clean(entry.findtext("a:title", "", ns)),
                            "summary": _clean(entry.findtext("a:summary", "", ns))})
        return _records_text(records, "arxiv")
    except RetryableError as exc:
        with _arxiv_lock:
            _arxiv_cooldown_until = time.monotonic() + 120
        return _error(exc) + "; arXiv paused for 120s; use retrieved_sources or HF/web instead."
    except Exception as exc:
        return _error(exc)


def _hf_records(items, *, prefer_ai=False):
    if not isinstance(items, list):
        raise ValueError("Hugging Face returned a non-list response")
    records = []
    for item in items:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict) or not paper.get("id"):
            continue
        paper_id = re.sub(r"v\d+$", "", str(paper["id"]))
        summary = (paper.get("ai_summary") or item.get("ai_summary")) if prefer_ai else None
        records.append({"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
                        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                        "title": _clean(paper.get("title") or item.get("title")),
                        "summary": _clean(summary or paper.get("summary") or item.get("summary")),
                        "upvotes": int(paper.get("upvotes") or item.get("upvotes") or 0),
                        "github": paper.get("githubRepo") or item.get("githubRepo") or "",
                        "stars": int(paper.get("githubStars") or item.get("githubStars") or 0)})
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Trending Hugging Face Daily Papers, sorted by upvotes; not a topic search.
    Optional date is YYYY-MM-DD. keyword filters titles/abstracts locally.
    JSON records: id, url, published, title, summary, upvotes, github, stars.
    """
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            datetime.strptime(date, "%Y-%m-%d")
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = _hf_records(response.json())
        if keyword.strip():
            term = keyword.strip().casefold()
            records = [r for r in records if term in (r["title"] + " " + r["summary"]).casefold()]
        return _records_text(sorted(records, key=lambda r: r["upvotes"], reverse=True), "hf-daily")
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Prefer AI summaries when available.
    JSON records: id, url, published, title, summary, upvotes, github, stars.
    Returns NO RESULTS or ERROR on failure.
    """
    try:
        if not query.strip():
            return "NO RESULTS"
        params = {"q": query.strip(), "limit": max(1, min(50, limit))}
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        return _records_text(_hf_records(response.json(), prefer_ai=True), "hf-search")
    except Exception as exc:
        return _error(exc)


def _is_rate_limited(value):
    """Exa can report throttling inside HTTP 200 JSON-RPC metadata/text."""
    if isinstance(value, dict):
        for key, item in value.items():
            name = re.sub(r"[^a-z]", "", key.lower())
            if "ratelimit" in name and item is True:
                return True
            if name in {"ratelimited", "ratelimitexceeded", "isratelimited"} and item:
                return True
            if _is_rate_limited(item):
                return True
    elif isinstance(value, list):
        return any(_is_rate_limited(item) for item in value)
    elif isinstance(value, str):
        return bool(re.search(r"rate[ _-]*limit(?:ed|\s+exceeded|\s+reached)?|too many requests", value, re.I))
    return False


def _mcp_message(response):
    try:
        payload = response.json()
    except ValueError:
        # Assemble multiline SSE data fields, ignoring comments and keep-alives.
        payload = None
        for event in re.split(r"\r?\n\r?\n", response.text):
            data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
            if data and data != "[DONE]":
                candidate = json.loads(data)
                if "result" in candidate or "error" in candidate:
                    payload = candidate
        if payload is None:
            raise ValueError("Exa returned no JSON-RPC result in its event stream")
    return payload


def _exa_call(name, arguments):
    def attempt():
        key = os.getenv("EXA_API_KEY", "").strip()
        endpoint = EXA_URL + ("?exaApiKey=" + quote(key, safe="") if key else "")
        response = _request("POST", endpoint,
                            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": name, "arguments": arguments}})
        payload = _mcp_message(response)
        result = payload.get("result", {})
        content = result.get("content", [])
        text = "\n".join(str(part.get("text", "")) for part in content if part.get("type") == "text")
        # Check metadata first; an article about rate limits is not a server error.
        throttled = _is_rate_limited(result.get("_meta", {})) or _is_rate_limited(payload.get("error", {}))
        if not throttled and (result.get("isError") or len(text) < 1800):
            throttled = bool(re.search(
                r"rate[ _-]*limit(?:ed|\s+(?:exceeded|reached))|too many requests|"
                r"(?:hit|exceeded|reached).{0,80}rate.{0,10}limit", text, re.I))
        if throttled:
            raise RetryableError("Exa rate limit exceeded", _retry_after(response.headers.get("Retry-After")))
        if "error" in payload:
            raise ValueError(redact(json.dumps(payload["error"], ensure_ascii=False)))
        if result.get("isError"):
            raise ValueError(redact(text or "Exa tool failed"))
        if "result" not in payload:
            raise ValueError("Exa response has no result")
        text = redact(text.strip())
        if not text or re.fullmatch(r"no (?:search )?results(?: found)?[.!]?", text, re.I):
            return "NO RESULTS"
        return text
    return with_retry(attempt, attempts=7, base=3, cap=60)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search Exa for web sources with URLs and excerpts. objective describes ideal sources.
    Prefer original papers, official project pages and author posts. Returns NO RESULTS or ERROR on failure.
    """
    try:
        if not query.strip():
            return "NO RESULTS"
        result = _exa_call("web_search_exa", {"query": query.strip(),
                         "objective": objective.strip() or f"Find authoritative sources about {query.strip()}",
                         "numResults": max(1, min(10, num_results))})
        evidence.register_web(result)
        return result
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Fetch one HTTP(S) page with Exa as text; truncate to 12000 characters.
    Use to verify claims and read beyond search excerpts. Returns NO RESULTS or ERROR on failure.
    """
    try:
        parsed = httpx.URL(url)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("url must be an absolute HTTP(S) URL")
        result = _exa_call("web_fetch_exa", {"urls": [url]})[:12000]
        if result != "NO RESULTS":
            evidence.register_web(result, url)
        return result
    except Exception as exc:
        return _error(exc)


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


@tool
def retrieved_sources(source_family: str = "", limit: int = 48) -> str:
    """Read the actual metadata/excerpts retrieved by this run's source tools.
    This is the authoritative catalog for notes and sources.json; never invent entries.
    source_family optionally filters arxiv, hf-daily, hf-search or web. No network calls.
    """
    try:
        records = evidence.records()
        if source_family:
            records = [r for r in records if r["source"] == source_family]
        else:
            # Do not let one discovery family hide the others in a capped catalog.
            groups = {family: [r for r in records if r["source"] == family]
                      for family in ("web", "hf-search", "arxiv", "hf-daily")}
            groups["web"].sort(key=lambda r: (r["date"] == "n.d.", r["date"]))
            records = [group[i] for i in range(max((len(g) for g in groups.values()), default=0))
                       for group in groups.values() if i < len(group)]
        return redact(json.dumps([{k: r[k] for k in ("id", "url", "title", "date", "source", "summary")}
                                  for r in records[:max(1, min(60, limit))]], ensure_ascii=False)) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
