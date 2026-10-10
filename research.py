"""Run an evidence-based survey in a disposable sandbox and save verified artifacts."""
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

import evidence

from agents import (
    FINALIZER_PATH, NORMALIZER_PATH, NOTES_DIR, RECURSION_LIMIT, REPORT_PATH, SOURCES_PATH,
    VALIDATOR_PATH, WORKDIR, build_lead_agent,
)
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import arxiv_search, hf_search_papers, redact, web_search

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
EVIDENCE_PATH = f"{WORKDIR}/research/evidence.json"
EVIDENCE_CHECKER_PATH = f"{WORKDIR}/research/check_evidence.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
MAX_REPAIR_ROUNDS = 2
MODEL_REQUEST_TIMEOUT = 60
MODEL_MAX_RETRIES = 1


def configure_model_runtime(model):
    """Keep one stalled provider request from consuming the whole topic budget."""
    if hasattr(model, "request_timeout"):
        model.request_timeout = MODEL_REQUEST_TIMEOUT
    if hasattr(model, "max_retries"):
        model.max_retries = MODEL_MAX_RETRIES
    return model


class ResearchProgress(BaseCallbackHandler):
    """Log team activity without exposing prompts, tool arguments or credentials."""

    raise_error = True

    def __init__(self, topic=None, *, max_model_calls=100, max_tool_calls=200, max_seconds=900):
        self._lock = threading.Lock()
        self.model_calls = 0
        self.tool_calls = 0
        self.delegations = 0
        self.started = time.monotonic()
        self.events = []
        self._tools = {}
        self._tool_inputs = {}
        self._failed_requests = Counter()
        self.abort_reason = None
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds
        self.live_path = None
        if topic:
            directory = ROOT / ".runs" / slugify(topic)
            directory.mkdir(parents=True, exist_ok=True)
            self.live_path = directory / "live-events.jsonl"
            self.live_path.write_text("", encoding="utf-8")

    def _record(self, event):
        self.events.append(event)
        if self.live_path:
            with self.live_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _check_stop(self):
        if self.abort_reason:
            raise RuntimeError(self.abort_reason)
        if time.monotonic() - self.started >= self.max_seconds:
            raise RuntimeError(f"Topic time budget exhausted ({self.max_seconds}s); no further calls permitted")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self._lock:
            self._check_stop()
            self.model_calls += 1
            if self.model_calls > self.max_model_calls:
                raise RuntimeError(f"Whole-team model-call budget exhausted ({self.max_model_calls} calls)")
            if self.model_calls % 10 == 0:
                print(f"[progress] team model calls={self.model_calls}, tool calls={self.tool_calls}, "
                      f"elapsed={time.monotonic() - self.started:.0f}s", flush=True)

    def on_tool_start(self, serialized, input_str, **kwargs):
        with self._lock:
            self._check_stop()
            self.tool_calls += 1
            if self.tool_calls > self.max_tool_calls:
                raise RuntimeError(f"Whole-team tool-call budget exhausted ({self.max_tool_calls} calls)")
            name = serialized.get("name", "unknown")
            run_id = str(kwargs.get("run_id"))
            self._tools[run_id] = name
            self._tool_inputs[run_id] = (name, redact(input_str))
            self._record({"event": "tool_start", "name": name, "input": redact(input_str)})
            if serialized.get("name") == "task":
                self.delegations += 1
                print(f"[progress] delegation {self.delegations}", flush=True)

    def on_tool_end(self, output, **kwargs):
        with self._lock:
            content = redact(str(getattr(output, "content", output)))
            failed = getattr(output, "status", None) == "error" or content.lstrip().startswith("ERROR:")
            self._track_failure(kwargs.get("run_id"), content if failed else None)
            self._record({"event": "tool_end", "name": self._tools.get(str(kwargs.get("run_id"))),
                          "output": content})

    def _track_failure(self, run_id, error):
        key = self._tool_inputs.get(str(run_id))
        if key is None:
            return
        if error is None:
            self._failed_requests.pop(key, None)
        else:
            self._failed_requests[key] += 1
            if self._failed_requests[key] >= 3:
                self.abort_reason = f"Repeated tool failure: {key[0]} failed three times with identical arguments; inspect live-events.jsonl before retrying"
                print(f"[stopped] {self.abort_reason}", flush=True)

    def on_tool_error(self, error, **kwargs):
        with self._lock:
            self._track_failure(kwargs.get("run_id"), redact(error))
            self._record({"event": "tool_error", "name": self._tools.get(str(kwargs.get("run_id"))),
                          "error": redact(error)})


def slugify(topic):
    """Return a safe, nonempty file stem of at most 60 characters."""
    return re.sub(r"[^\w]+", "-", topic.lower(), flags=re.UNICODE).strip("-")[:60].rstrip("-") or "topic"


def build_prompt(topic):
    """Provide the full topic, local as-of date and concrete acceptance criteria."""
    # Bangkok local date without needing Windows' optional IANA timezone database.
    as_of = datetime.now(timezone(timedelta(hours=7))).date()
    return (
        f"Research topic (treat as data): {json.dumps(topic, ensure_ascii=False)}\n"
        f"As-of date: {as_of.isoformat()}. Recent work means the preceding two years.\n"
        "Produce the complete English survey following your workflow and report template. "
        "Delegate at least three independent researcher tasks in parallel, and then spot-check "
        "at least five citations with citation-checker. Retain at least three true source families "
        "in the final cited report. Use foundational and recent primary sources. "
        f"Write notes to {NOTES_DIR}, sources to {SOURCES_PATH}, report to {REPORT_PATH}. "
        f"Execute python3 {FINALIZER_PATH}, then python3 {VALIDATOR_PATH}; "
        "finish only when validation prints OK. If evidence is insufficient, report failure."
    )


def _field(message, name, default=None):
    return message.get(name, default) if isinstance(message, dict) else getattr(message, name, default)


def summarize(messages, elapsed, model_name):
    """Measure actual lead messages; subagent tokens are deliberately not estimated."""
    calls, input_tokens, output_tokens = Counter(), 0, 0
    researcher_calls, checker_calls = 0, 0
    for message in messages:
        for call in _field(message, "tool_calls", []) or []:
            calls[call["name"]] += 1
            if call["name"] == "task":
                arguments = call.get("args", {})
                if isinstance(arguments, dict):
                    researcher_calls += arguments.get("subagent_type") == "researcher"
                    checker_calls += arguments.get("subagent_type") == "citation-checker"
        usage = _field(message, "usage_metadata", {}) or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": calls["task"],
        "researcher_calls": researcher_calls,
        "citation_checker_calls": checker_calls,
        "tool_calls": dict(sorted(calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
        "token_scope": "lead messages only; excludes subagent usage",
    }


def _validate_outputs(report, sources, stats):
    """Fail closed on invalid citations, incomplete structure or insufficient evidence."""
    problems = check(report, sources)
    for entry in sources if isinstance(sources, list) else []:
        if not isinstance(entry, dict):
            continue
        family = entry.get("source")
        if family not in FAMILIES:
            problems.append("source has an invalid discovery family")
        for field in ("id", "url", "title", "date", "source"):
            if field not in entry:
                problems.append(f"source [{entry.get('n')}] missing {field}")
        if family in {"arxiv", "hf-daily", "hf-search"}:
            domain = "arxiv.org/abs" if family == "arxiv" else "huggingface.co/papers"
            if entry.get("url") != f"https://{domain}/{entry.get('id', '')}":
                problems.append(f"source [{entry.get('n')}] URL does not match its family/id")
            if re.search(r"v\d+$", str(entry.get("id", ""))):
                problems.append(f"source [{entry.get('n')}] id still has a version suffix")
    families = {s.get("source") for s in sources if isinstance(s, dict)} if isinstance(sources, list) else set()
    if len(families & FAMILIES) < 3:
        problems.append("need at least three cited source families after finalization")
    for section in ("TL;DR", "Background", "Trends and open problems", "References"):
        if not re.search(rf"(?m)^##[ \t]+{re.escape(section)}[ \t]*\r?$", report):
            problems.append(f"missing required section: {section}")
    headings = re.findall(r"(?m)^##[ \t]+(.+?)[ \t]*\r?$", report)
    themes = [h for h in headings if h not in {"TL;DR", "Background", "Trends and open problems", "References"}]
    if not 3 <= len(themes) <= 6:
        problems.append("need three to six theme sections")
    if stats["researcher_calls"] < 3:
        problems.append("need at least three researcher delegations")
    if stats["citation_checker_calls"] < 1:
        problems.append("citation-checker was never delegated a spot-check")
    if problems:
        raise RuntimeError("Invalid survey: " + "; ".join(problems[:12]))


def _write_bundle(reports_dir, artifacts):
    """Stage all bytes first; restore previous outputs if any replacement fails."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    previous = {name: (reports_dir / name).read_bytes() if (reports_dir / name).exists() else None
                for name in artifacts}
    replaced = []
    # Stage directly in the output directory so Windows files inherit its ACL.
    staged = {name: reports_dir / f".research-{uuid.uuid4().hex}.tmp" for name in artifacts}
    try:
        for name, content in artifacts.items():
            with staged[name].open("xb") as file:
                file.write(content)
        try:
            for name in artifacts:
                os.replace(staged[name], reports_dir / name)
                replaced.append(name)
        except BaseException:
            for name in reversed(replaced):
                target = reports_dir / name
                if previous[name] is None:
                    target.unlink(missing_ok=True)
                else:
                    restore = staged[name]
                    restore.write_bytes(previous[name])
                    os.replace(restore, target)
            raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download exact sandbox bytes; validate everything before writing any artifact."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip() or not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("Sandbox report or sources are missing/empty; no files saved")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Sandbox artifacts are not valid UTF-8/JSON; no files saved") from exc
    stats = summarize(messages, elapsed, model_name)
    _validate_outputs(report, sources, stats)
    meta = {
        "topic": topic,
        **stats,
        "n_sources": len(sources),
        "source_families": sorted({s["source"] for s in sources}),
    }
    slug = slugify(topic)
    _write_bundle(reports_dir, {
        f"{slug}.sources.json": sources_bytes,
        f"{slug}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        f"{slug}.md": report_bytes,
    })
    return Path(reports_dir) / f"{slug}.md"


def _execute_checked(backend, command):
    result = backend.execute(command)
    if result.exit_code != 0:
        raise RuntimeError(f"Sandbox command failed: {redact(result.output)}")
    return result.output


def _archive_run(backend, topic, result, progress):
    """Keep sanitized evidence/diagnostics outside submitted reports, before cleanup."""
    try:
        directory = ROOT / ".runs" / slugify(topic)
        directory.mkdir(parents=True, exist_ok=True)
        messages = result.get("messages", []) if result else []
        trace = {"messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in messages],
                 "lead_audit": [m.model_dump() for m in (result or {}).get("audit_messages", [])],
                 "events": progress.events, "team_model_calls": progress.model_calls,
                 "team_tool_calls": progress.tool_calls}
        (directory / "trace.json").write_text(redact(json.dumps(trace, ensure_ascii=False, default=str, indent=2)),
                                               encoding="utf-8")
        (directory / "evidence.json").write_text(redact(json.dumps(evidence.records(), ensure_ascii=False, indent=2)),
                                                  encoding="utf-8")
        files = download(backend, [REPORT_PATH, SOURCES_PATH])
        for path in (REPORT_PATH, SOURCES_PATH):
            content = files.get(path)
            if content:
                (directory / Path(path).name).write_text(redact(content.decode("utf-8")), encoding="utf-8")
            else:
                (directory / Path(path).name).unlink(missing_ok=True)
    except Exception as exc:
        print(f"[diagnostics] {redact(exc)}", file=sys.stderr)


def _assess_run(backend, result, elapsed, model_name):
    # Make the deterministic sandbox pipeline mandatory even if the model forgot
    # to execute it after its last edit or repair.
    _execute_checked(backend, f"python3 {NORMALIZER_PATH}")
    _execute_checked(backend, f"python3 {FINALIZER_PATH}")
    upload(backend, {EVIDENCE_PATH: redact(json.dumps(evidence.records(), ensure_ascii=False)).encode("utf-8")})
    _execute_checked(backend, f"python3 {EVIDENCE_CHECKER_PATH}")
    output = _execute_checked(backend, f"python3 {VALIDATOR_PATH}")
    if not output.strip().startswith("OK:"):
        raise RuntimeError("Sandbox validator did not confirm OK")
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    try:
        report = files[REPORT_PATH].decode("utf-8")
        sources = json.loads(files[SOURCES_PATH].decode("utf-8"))
    except (KeyError, AttributeError, ValueError, UnicodeError) as exc:
        raise RuntimeError("Sandbox report or source data are missing/invalid") from exc
    _validate_outputs(report, sources, summarize(result.get("audit_messages") or result["messages"], elapsed, model_name))


def _seed_sources(topic):
    """Discover a small real catalog before delegation; researchers expand/verify it."""
    cache = ROOT / ".runs" / slugify(topic) / "evidence.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            cached_families = {r.get("source") for r in cached if isinstance(r, dict)}
            if len(cached_families) >= 3 and cached and all(isinstance(r, dict) and r.get("source") in FAMILIES
                              and all(k in r for k in ("id", "url", "title", "date", "summary", "text"))
                              for r in cached):
                for record in cached:
                    evidence.register(record["source"], [record])
                print(f"[sources] reused {len(cached)} records retrieved in the last 24 hours", flush=True)
                return
        except (OSError, ValueError, TypeError):
            pass
    query = re.sub(r"^survey about\s+", "", topic, flags=re.I)
    foundations = {
        "world model": "World Models David Ha Schmidhuber 2018 original paper",
        "reinforcement learning for LLM reasoning": "DeepSeek R1 reasoning reinforcement learning original paper",
        "LLM agents and tool use": "ReAct Toolformer original research papers tool use language models",
        "video and multimodal generation": "Denoising Diffusion Probabilistic Models latent diffusion original papers",
        "efficient inference and small language models": "FlashAttention speculative decoding original research papers",
    }
    for name, source, arguments in [
        ("hf-search", hf_search_papers, {"query": query, "limit": 12}),
        ("web", web_search, {"query": foundations.get(query, query + " foundational research papers"),
                              "objective": "Find original foundational papers and author project pages", "num_results": 4}),
        ("arxiv", arxiv_search, {"query": query, "max_results": 8}),
    ]:
        print(f"[sources] discovering {name}", flush=True)
        output = source.invoke(arguments)
        if output.startswith("ERROR:"):
            print(f"[sources] {name}: {output[:200]}", flush=True)


def main(topic):
    """Return 0 on a valid saved survey, 1 on failure, and 2 for an empty topic."""
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>" OR python research.py --all', file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        model = configure_model_runtime(make_model())
        evidence.reset()
        model_name = (os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL")
                      or getattr(model, "model_name", None) or str(type(model).__name__))
        print(f"Researching: {topic}", flush=True)
        with open_sandbox() as backend:
            print(f"[sandbox] id={backend.id}", flush=True)
            _execute_checked(backend, f"mkdir -p {NOTES_DIR} {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                NORMALIZER_PATH: (ROOT / "normalize_citations.py").read_bytes(),
                EVIDENCE_CHECKER_PATH: (ROOT / "check_evidence.py").read_bytes(),
                f"{WORKDIR}/research/REPORT_TEMPLATE.md": (ROOT / "REPORT_TEMPLATE.md").read_bytes(),
            })
            _execute_checked(backend, f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}")
            context = {"topic": topic, "as_of": datetime.now(timezone(timedelta(hours=7))).date().isoformat(),
                       "phase": "research"}
            agent = build_lead_agent(backend, model, context)
            progress, result = ResearchProgress(topic), None
            config = {"recursion_limit": RECURSION_LIMIT, "max_concurrency": 3, "callbacks": [progress]}
            try:
                _seed_sources(topic)
                research_phase = (
                    f"Topic: {topic}. As-of: {context['as_of']}.\nPHASE 1: RESEARCH ONLY. Decide three independent questions, "
                    "delegate them in parallel with full topic/date/note paths/format. The actual source "
                    "catalog is already seeded; use retrieved_sources and expand/verify it. Read the "
                    "returned notes, report missing evidence and STOP with paths. Do not write the report, "
                    "call citation-checker or repeatedly redelegate during this phase."
                )
                print("[phase] research", flush=True)
                result = agent.invoke({"messages": [{"role": "user", "content": research_phase}]}, config=config)
                context["phase"] = "writing"
                writing_phase = (
                    "PHASE 2: WRITE AND VERIFY. Use the existing notes and retrieved_sources. Choose 10-16 "
                    "relevant sources including foundational work and work within the last two years, "
                    "with at least three true families. Call compile_sources with {source, url} pairs; "
                    "DO NOT hand-write source metadata. Read its numbered output and write an English "
                    "1200-1800 word survey by themes. Cite every substantive claim and every TL;DR/trends "
                    "bullet; no unsupported facts. Use exactly '## Trends and open problems'. Call ONE "
                    "citation-checker with five claim/URL pairs and resolve its verdicts, then execute "
                    f"python3 {FINALIZER_PATH} and python3 {VALIDATOR_PATH}. "
                    "Do not delegate more researchers now; report actual retrieval gaps rather than inventing."
                )
                print("[phase] writing", flush=True)
                result = agent.invoke({"messages": [*result["messages"],
                                                    {"role": "user", "content": writing_phase}]}, config=config)
                result["audit_messages"] = context.get("audit_messages", [])
                for repair in range(MAX_REPAIR_ROUNDS + 1):
                    try:
                        _assess_run(backend, result, time.monotonic() - started, model_name)
                        break
                    except RuntimeError as exc:
                        if repair == MAX_REPAIR_ROUNDS:
                            raise
                        print(f"[repair {repair + 1}/{MAX_REPAIR_ROUNDS}] {redact(exc)}", flush=True)
                        correction = (
                            "Final acceptance FAILED. Continue from the existing sandbox files and notes. "
                            f"Fix ALL of these actual problems: {redact(exc)}. "
                            "Do not invent or relabel sources. Retrieve missing families through researcher; "
                            "use exact URLs and ids returned by the tools. Preserve the exact required heading "
                            f"'retrieved_sources' is the authoritative catalog; the same data is at {EVIDENCE_PATH}. "
                            "'## Trends and open problems'. If citation-checker has not been called, delegate "
                            "five claim/URL pairs now and resolve its verdicts. After all edits, execute the "
                            f"provided finalizer at {FINALIZER_PATH}, then the validator at {VALIDATOR_PATH}. "
                            "Do not finish until the report satisfies these criteria."
                        )
                        result = agent.invoke({"messages": [*result["messages"],
                                                            {"role": "user", "content": correction}]}, config=config)
                        result["audit_messages"] = context.get("audit_messages", [])
                path = save_outputs(backend, topic, result.get("audit_messages") or result["messages"],
                                    time.monotonic() - started, model_name)
            finally:
                _archive_run(backend, topic, result, progress)
        print(f"Saved: {path}", flush=True)
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {redact(exc)}", file=sys.stderr)
        return 1


def run_all():
    """Run the actual preset topics sequentially; stop rather than repeating failures."""
    topics = re.findall(r"^\d+\. (.+)$", (ROOT / "topics.md").read_text(encoding="utf-8"), re.M)
    if not topics:
        print("FAILED: topics.md has no numbered topics", file=sys.stderr)
        return 1
    for index, topic in enumerate(topics, 1):
        print(f"[{index}/{len(topics)}]", flush=True)
        code = main(topic)
        if code:
            return code
    return 0


def sandbox_maintenance(sandbox_id=None, inspect=False, normalize=False):
    """List IDs or clean up a cancelled lab sandbox after verifying our validator marker."""
    from daytona import Daytona
    from sandbox import DaytonaSandbox
    try:
        client = Daytona()
        if not sandbox_id:
            for box in client.list():
                print(f"{box.id} state={box.state}")
            return 0
        box = client.get(sandbox_id)
        backend = DaytonaSandbox(sandbox=box)
        marker = download(backend, [VALIDATOR_PATH]).get(VALIDATOR_PATH)
        if marker != VALIDATOR_SOURCE.read_bytes():
            raise RuntimeError("Sandbox has no matching lab validator; cleanup refused")
        if inspect:
            result = backend.execute(f"ls -l {NOTES_DIR} {WORKDIR}/report; wc -w {NOTES_DIR}/*.md {REPORT_PATH} 2>/dev/null")
            print(redact(result.output))
            return 0
        if normalize:
            upload(backend, {NORMALIZER_PATH: (ROOT / "normalize_citations.py").read_bytes()})
            print(_execute_checked(backend, f"python3 {NORMALIZER_PATH}"))
            print(_execute_checked(backend, f"python3 {FINALIZER_PATH}"))
            print(_execute_checked(backend, f"python3 {VALIDATOR_PATH}"))
            return 0
        client.delete(box)
        print(f"Removed cancelled lab sandbox: {sandbox_id}")
        return 0
    except Exception as exc:
        print(f"FAILED: {redact(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--list-sandboxes"]:
        sys.exit(sandbox_maintenance())
    if len(sys.argv) == 3 and sys.argv[1] == "--cleanup-sandbox":
        sys.exit(sandbox_maintenance(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--inspect-sandbox":
        sys.exit(sandbox_maintenance(sys.argv[2], inspect=True))
    if len(sys.argv) == 3 and sys.argv[1] == "--normalize-sandbox":
        sys.exit(sandbox_maintenance(sys.argv[2], normalize=True))
    sys.exit(run_all() if sys.argv[1:] == ["--all"] else main(" ".join(sys.argv[1:])))
