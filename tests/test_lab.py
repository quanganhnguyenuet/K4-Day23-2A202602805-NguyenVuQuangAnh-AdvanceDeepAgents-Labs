"""Offline regression tests: no paid model calls, no submitted report fixtures."""
import contextlib
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import agents
import evidence
import research
import tools
from check_citations import check
from finalize_citations import finalize


def response(payload=None, *, status=200, text=None, headers=None):
    return httpx.Response(status, json=payload if text is None else None,
                          text=text, headers=headers, request=httpx.Request("GET", "https://example.org"))


def survey_fixture():
    sources = [
        {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001",
         "title": "Test A", "date": "2025-01-01", "source": "arxiv"},
        {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002",
         "title": "Test B", "date": "2025-01-02", "source": "hf-search"},
        {"n": 3, "id": "", "url": "https://example.org/c",
         "title": "Test C", "date": "n.d.", "source": "web"},
    ]
    body = ("# Test survey\n\n## TL;DR\n- Finding [1].\n## Background\nBackground [2].\n"
            "## Theme A\nComparison [1][2].\n## Theme B\nEvidence [3].\n"
            "## Theme C\nLimitations [2][3].\n## Trends and open problems\nOpen problem [3].\n")
    report, sources, problems = finalize(body, sources)
    assert not problems
    calls = [{"name": "task", "args": {"subagent_type": name}, "id": str(i)}
             for i, name in enumerate(["researcher"] * 3 + ["citation-checker"])]
    messages = [AIMessage(content="", tool_calls=calls,
                          usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})]
    return report, sources, messages


class CitationTests(unittest.TestCase):
    def test_adjacent_citations_are_all_counted(self):
        sources = survey_fixture()[1][:2]
        report = "# Title\n\nClaim [1][2].\n\n## References\n[1] A. https://arxiv.org/abs/2501.00001\n[2] B. https://huggingface.co/papers/2501.00002\n"
        self.assertEqual(check(report, sources), [])

    def setUp(self):
        self.report, self.sources, _ = survey_fixture()

    def test_valid_finalized_report_and_idempotence(self):
        self.assertEqual(check(self.report, self.sources), [])
        report, sources, problems = finalize(self.report, self.sources)
        self.assertEqual((report, sources, problems), (self.report, self.sources, []))

    def test_empty_and_malformed_sources(self):
        for sources in ([], {}, [None], [{"n": True, "url": "file:///x"}]):
            with self.subTest(sources=sources):
                self.assertTrue(check(self.report, sources))

    def test_missing_heading(self):
        self.assertTrue(check(self.report.replace("## References", "## Bibliography"), self.sources))

    def test_unknown_body_number(self):
        self.assertIn("[99] cited but missing from sources.json", check("[99]\n" + self.report, self.sources))

    def test_reference_is_not_a_body_citation(self):
        body, refs = self.report.split("## References")
        self.assertTrue(any("never cited" in p for p in check(body.replace("[1]", "") + "## References" + refs, self.sources)))

    def test_grouped_citations_and_ranges(self):
        refs = self.report.split("## References")[1]
        for citations in ("[1, 2, 3]", "[1-3]", "[1–3]", "[1, 2-3]"):
            with self.subTest(citations=citations):
                self.assertEqual(check(citations + "\n## References" + refs, self.sources), [])

    def test_code_and_numeric_links_do_not_count(self):
        self.assertEqual(check("`[99]`\n```python\n[98]\n```\n~~~\n[97]\n~~~\n[96](https://x.org)\n" + self.report,
                               self.sources), [])

    def test_duplicate_source_numbers_and_urls(self):
        for key in ("n", "url"):
            sources = copy.deepcopy(self.sources)
            sources[1][key] = sources[0][key]
            self.assertTrue(any("duplicate" in p for p in check(self.report, sources)))

    def test_missing_duplicate_unknown_references(self):
        line = self.report.split("## References\n")[1].splitlines()[0]
        for report in (self.report.replace(line, ""), self.report + line + "\n",
                       self.report + "[99] Unknown https://example.org/z (n.d.)\n"):
            self.assertTrue(check(report, self.sources))

    def test_reference_url_mismatch_and_bundling(self):
        for replacement in ("https://wrong.org", "https://arxiv.org/abs/2501.00001 https://other.org"):
            body, refs = self.report.split("## References")
            refs = refs.replace(self.sources[0]["url"], replacement)
            self.assertTrue(check(body + "## References" + refs, self.sources))

    def test_extra_section_and_bad_ranges(self):
        self.assertTrue(check(self.report + "\n## Appendix\n", self.sources))
        self.assertTrue(check("[3-1]\n" + self.report, self.sources))


class RetryTests(unittest.TestCase):
    @patch("tools.random.uniform", return_value=0.25)
    @patch("tools.time.sleep")
    def test_exponential_jitter_cap_and_no_final_sleep(self, sleep, jitter):
        fn = Mock(side_effect=tools.RetryableError("temporary"))
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fn, attempts=4, base=1, cap=3)
        self.assertEqual(fn.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.25, 2.25, 3])

    @patch("tools.time.sleep")
    def test_retry_after_and_success(self, sleep):
        fn = Mock(side_effect=[tools.RetryableError("temporary", retry_after=8), "ok"])
        self.assertEqual(tools.with_retry(fn, cap=5), "ok")
        sleep.assert_called_once_with(5)

    @patch("tools.time.sleep")
    def test_other_errors_are_not_retried(self, sleep):
        fn = Mock(side_effect=ValueError("bad input"))
        with self.assertRaises(ValueError):
            tools.with_retry(fn)
        fn.assert_called_once()
        sleep.assert_not_called()

    def test_invalid_retry_options(self):
        with self.assertRaises(ValueError):
            tools.with_retry(lambda: None, attempts=0)

    @patch("tools.httpx.request")
    def test_http_and_transport_classification(self, request):
        for status in tools.RETRY_STATUSES:
            request.return_value = response({}, status=status, headers={"Retry-After": "4"})
            with self.assertRaises(tools.RetryableError) as error:
                tools._request("GET", "https://example.org")
            self.assertEqual(error.exception.retry_after, 4)
        request.return_value = response({}, status=401)
        with self.assertRaises(httpx.HTTPStatusError):
            tools._request("GET", "https://example.org")
        request.side_effect = httpx.ConnectError("offline")
        with self.assertRaises(tools.RetryableError):
            tools._request("GET", "https://example.org")

    def test_retry_after_date_and_invalid_header(self):
        self.assertEqual(tools._retry_after("Wed, 01 Jan 2020 00:00:00 GMT"), 0)
        self.assertIsNone(tools._retry_after("invalid"))


class SourceTests(unittest.TestCase):
    @patch("tools._arxiv_request")
    def test_arxiv_normalization_and_clean_query(self, request):
        request.return_value = response(text='''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2501.00001v3</id><published>2025-01-01T00:00:00Z</published>
        <title> A\n title </title><summary> A   summary </summary></entry></feed>''')
        result = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model:', "max_results": 100}))
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2501.00001")
        search_query = request.call_args.args[0]["search_query"]
        self.assertNotIn("all:for", search_query)
        self.assertNotIn("all:survey", search_query)
        self.assertEqual(result[0]["title"], "A title")
        self.assertEqual(request.call_args.args[0]["search_query"], "all:world AND all:model")
        self.assertEqual(request.call_args.args[0]["max_results"], 30)

    @patch("tools._request")
    def test_empty_query_avoids_network(self, request):
        self.assertEqual(tools.arxiv_search.invoke({"query": '\" : AND OR'}), "NO RESULTS")
        self.assertEqual(tools.hf_search_papers.invoke({"query": " "}), "NO RESULTS")
        request.assert_not_called()

    @patch("tools._request")
    @patch("tools.time.monotonic", side_effect=[10.0, 10.0, 11.0, 11.0, 13.0])
    @patch("tools.time.sleep")
    def test_arxiv_spacing(self, sleep, monotonic, request):
        with patch.object(tools, "_last_arxiv_call", None):
            tools._arxiv_request({})
            tools._arxiv_request({})
        sleep.assert_called_once_with(2.0)
        self.assertTrue(all(c.args[1].startswith("https://") for c in request.call_args_list))

    @patch("tools.time.sleep")
    def test_arxiv_exhausted_retries_pause_further_network_calls(self, sleep):
        real_request = tools._arxiv_request
        with patch.object(tools, "_arxiv_cooldown_until", 0.0), \
                patch("tools._arxiv_request", side_effect=tools.RetryableError("read operation timed out")) as request:
            result = tools.arxiv_search.invoke({"query": "world model"})
            self.assertIn("paused for 120s", result)
            self.assertEqual(request.call_count, 3)
            with patch("tools._request") as network:
                with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                    real_request({})
                network.assert_not_called()

    @patch("tools._request")
    def test_hf_filter_sort_fields_and_ai_summary(self, request):
        items = [{"paper": {"id": "2501.00001", "title": "World model", "summary": "A" * 700,
                            "ai_summary": "Concise evidence", "upvotes": 2, "githubRepo": "https://github.com/a/b",
                            "githubStars": 7, "publishedAt": "2025-01-01T01:00:00Z"}},
                 {"paper": {"id": "2501.00002", "title": "WORLD dynamics", "summary": "Test", "upvotes": 9}},
                 {"paper": {"title": "Missing id"}}]
        request.return_value = response(items)
        records = json.loads(tools.hf_daily_papers.invoke({"keyword": "world", "date": "2025-01-01"}))
        self.assertEqual([r["upvotes"] for r in records], [9, 2])
        self.assertEqual(records[1]["stars"], 7)
        self.assertEqual(len(records[1]["summary"]), 600)
        records = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
        self.assertEqual(records[0]["summary"], "Concise evidence")

    @patch("tools._request")
    def test_empty_hf_and_source_failures_are_strings(self, request):
        request.return_value = response([])
        self.assertEqual(tools.hf_daily_papers.invoke({}), "NO RESULTS")
        self.assertEqual(tools.hf_search_papers.invoke({"query": "x"}), "NO RESULTS")
        request.side_effect = ValueError("invalid response")
        for fn, args in [(tools.arxiv_search, {"query": "x"}), (tools.hf_daily_papers, {}),
                         (tools.hf_search_papers, {"query": "x"}), (tools.web_search, {"query": "x"}),
                         (tools.web_fetch, {"url": "https://example.org"})]:
            with self.subTest(tool=fn.name):
                self.assertTrue(fn.invoke(args).startswith("ERROR:"))

    @patch("tools._request")
    @patch("tools.time.sleep")
    def test_exa_http_200_metadata_throttle_is_retried(self, sleep, request):
        request.side_effect = [response({"result": {"_meta": {"rateLimitExceeded": True},
                                                      "content": [{"type": "text", "text": "Wait"}]}}),
                               response({"result": {"content": [{"type": "text", "text": "Real result"}]}})]
        self.assertEqual(tools.web_search.invoke({"query": "x"}), "Real result")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()
        args = request.call_args.kwargs["json"]["params"]["arguments"]
        self.assertTrue(args["objective"])

    @patch("tools._request")
    def test_exa_sse_fetch_array_and_truncation(self, request):
        payload = {"result": {"content": [{"type": "text", "text": "X" * 13000}]}}
        request.return_value = response(text="event: message\ndata: " + json.dumps(payload) + "\n\n")
        self.assertEqual(len(tools.web_fetch.invoke({"url": "https://example.org"})), 12000)
        self.assertEqual(request.call_args.kwargs["json"]["params"]["arguments"], {"urls": ["https://example.org"]})

    @patch("tools._request")
    def test_exa_rpc_error_and_empty_results(self, request):
        request.return_value = response({"error": {"code": -32602, "message": "Invalid arguments"}})
        self.assertTrue(tools.web_search.invoke({"query": "x"}).startswith("ERROR:"))
        request.return_value = response({"result": {"content": []}})
        self.assertEqual(tools.web_search.invoke({"query": "x"}), "NO RESULTS")

    @patch("tools._request")
    @patch("tools.time.sleep")
    def test_exa_text_throttle_and_article_about_rate_limits(self, sleep, request):
        request.side_effect = [response({"result": {"content": [{"type": "text", "text": "You have hit Exa's MCP rate limit."}]}}),
                               response({"result": {"content": [{"type": "text", "text": "No results found."}]}})]
        self.assertEqual(tools.web_search.invoke({"query": "x"}), "NO RESULTS")
        sleep.assert_called_once()
        request.side_effect = None
        request.return_value = response({"result": {"content": [{"type": "text", "text": "Article: How rate limits work"}]}})
        self.assertEqual(tools.web_search.invoke({"query": "x"}), "Article: How rate limits work")

    @patch("tools._request")
    def test_secrets_are_redacted_from_errors_and_successes(self, request):
        # Assemble a synthetic value; never include a real credential in fixtures.
        secret = "test" + "/ key+value"
        with patch.dict(os.environ, {"EXA_API_KEY": secret}):
            request.side_effect = ValueError("URL https://mcp.exa.ai/mcp?exaApiKey=" + tools.quote(secret, safe=""))
            output = tools.web_search.invoke({"query": "x"})
            self.assertNotIn(secret, output)
            self.assertNotIn(tools.quote(secret, safe=""), output)
            self.assertIn("REDACTED", output)
            request.side_effect = None
            request.return_value = response({"result": {"content": [{"type": "text", "text": secret}]}})
            self.assertEqual(tools.web_search.invoke({"query": "x"}), "[REDACTED]")


class ResearchTests(unittest.TestCase):
    def test_model_runtime_timeout_and_retry_are_bounded(self):
        model = SimpleNamespace(request_timeout=None, max_retries=None)
        self.assertIs(research.configure_model_runtime(model), model)
        self.assertEqual(model.request_timeout, 60)
        self.assertEqual(model.max_retries, 1)

    @patch("research._validate_outputs")
    @patch("research.download")
    @patch("research.upload")
    @patch("research._execute_checked")
    def test_assessment_always_normalizes_and_finalizes_before_validation(self, execute, upload, download, validate):
        report = survey_fixture()[0]
        sources = survey_fixture()[1]
        download.return_value = {agents.REPORT_PATH: report.encode(), agents.SOURCES_PATH: json.dumps(sources).encode()}
        execute.side_effect = ["NORMALIZED", "FINALIZED", "EVIDENCE OK", "OK: citations"]
        research._assess_run(Mock(), {"messages": []}, 1.0, "test-model")
        commands = [call.args[1] for call in execute.call_args_list]
        self.assertEqual(commands[:2], [f"python3 {agents.NORMALIZER_PATH}", f"python3 {agents.FINALIZER_PATH}"])
        self.assertIn("check_evidence.py", commands[2])
        self.assertIn("check_citations.py", commands[3])

    def test_slug_is_safe_bounded_and_nonempty(self):
        for topic in ("../../x", "", "...", "A " * 100, "Survey About World Model"):
            slug = research.slugify(topic)
            self.assertTrue(slug)
            self.assertLessEqual(len(slug), 60)
            self.assertNotIn("/", slug)
            self.assertNotIn("\\", slug)
        self.assertEqual(research.slugify("survey about world model"), "survey-about-world-model")

    def test_metadata_counts_actual_calls_and_usage(self):
        _, _, messages = survey_fixture()
        stats = research.summarize(messages, 1.234, "test-model")
        self.assertEqual(stats["subagent_calls"], 4)
        self.assertEqual(stats["researcher_calls"], 3)
        self.assertEqual(stats["tokens"], {"input": 11, "output": 7})
        self.assertEqual(stats["elapsed_s"], 1.2)

    def test_save_preserves_exact_downloaded_bytes(self):
        report, sources, messages = survey_fixture()
        files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode()}
        with tempfile.TemporaryDirectory() as directory, patch("research.download", return_value=files):
            path = research.save_outputs(None, "Test", messages, 1, "test-model", directory)
            self.assertEqual(path.read_bytes(), files[research.REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), files[research.SOURCES_PATH])
            meta = json.loads(path.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])

    def test_bad_download_or_citations_write_nothing(self):
        report, sources, messages = survey_fixture()
        for files in ({}, {research.REPORT_PATH: b"", research.SOURCES_PATH: b"[]"},
                      {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: b"{"},
                      {research.REPORT_PATH: b"invalid report", research.SOURCES_PATH: json.dumps(sources).encode()}):
            with tempfile.TemporaryDirectory() as directory, patch("research.download", return_value=files):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "Test", messages, 1, "test-model", directory)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_insufficient_delegation_or_wrong_families_fail(self):
        report, sources, messages = survey_fixture()
        stats = research.summarize([], 0, "test-model")
        with self.assertRaises(RuntimeError):
            research._validate_outputs(report, sources, stats)
        wrong = copy.deepcopy(sources)
        wrong[0]["source"] = "hf-daily"
        with self.assertRaises(RuntimeError):
            research._validate_outputs(report, wrong, research.summarize(messages, 0, "test-model"))

    def test_bundle_rolls_back_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"old")
            original_replace = os.replace
            calls = 0

            def failing_replace(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated failure")
                return original_replace(source, target)

            with patch("research.os.replace", side_effect=failing_replace):
                with self.assertRaises(OSError):
                    research._write_bundle(root, {"a": b"new", "b": b"new"})
            self.assertEqual((root / "a").read_bytes(), b"old")
            self.assertFalse((root / "b").exists())

    def test_main_failure_cleans_up_and_never_saves(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        cleaned = []

        @contextlib.contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                cleaned.append(True)

        agent = Mock()
        agent.invoke.side_effect = RuntimeError("model failure")
        with patch("research.make_model", return_value=Mock()), patch("research.open_sandbox", sandbox), \
                patch("research.upload") as upload, patch("research.build_lead_agent", return_value=agent), \
                patch("research.save_outputs") as save, patch("research._archive_run"), patch("research._seed_sources"):
            self.assertEqual(research.main("test topic"), 1)
            save.assert_not_called()
            seeded = upload.call_args.args[1]
            self.assertEqual(set(seeded), {research.VALIDATOR_PATH, research.FINALIZER_PATH, research.NORMALIZER_PATH,
                                          research.EVIDENCE_CHECKER_PATH,
                                          "/tmp/work/research/REPORT_TEMPLATE.md"})
        self.assertEqual(cleaned, [True])

    def test_empty_topic_returns_usage_code(self):
        self.assertEqual(research.main("  "), 2)

    def test_failed_acceptance_resumes_same_conversation(self):
        _, _, messages = survey_fixture()
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: 3 sources")
        agent = Mock()
        agent.invoke.return_value = {"messages": messages}
        with patch("research.make_model", return_value=Mock()), \
                patch("research.open_sandbox", return_value=contextlib.nullcontext(backend)), \
                patch("research.upload"), patch("research.build_lead_agent", return_value=agent), \
                patch("research._assess_run", side_effect=[RuntimeError("need missing family"), None]), \
                patch("research.save_outputs", return_value=Path("test.md")) as save, \
                patch("research._archive_run"), patch("research._seed_sources"):
            self.assertEqual(research.main("test topic"), 0)
        self.assertEqual(agent.invoke.call_count, 3)
        retry_input = agent.invoke.call_args.args[0]["messages"]
        self.assertEqual(retry_input[:-1], messages)
        self.assertIn("need missing family", retry_input[-1]["content"])
        save.assert_called_once()

    def test_repairs_stop_at_configured_limit(self):
        _, _, messages = survey_fixture()
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: 3 sources")
        agent = Mock()
        agent.invoke.return_value = {"messages": messages}
        with patch("research.make_model", return_value=Mock()), \
                patch("research.open_sandbox", return_value=contextlib.nullcontext(backend)), \
                patch("research.upload"), patch("research.build_lead_agent", return_value=agent), \
                patch("research._assess_run", side_effect=RuntimeError("invalid survey")), \
                patch("research.save_outputs") as save, patch("research._archive_run"), patch("research._seed_sources"):
            self.assertEqual(research.main("test topic"), 1)
        self.assertEqual(agent.invoke.call_count, 2 + research.MAX_REPAIR_ROUNDS)
        save.assert_not_called()

    def test_shared_budget_applies_across_all_repair_invocations(self):
        progress = research.ResearchProgress()
        progress.model_calls = progress.max_model_calls
        with self.assertRaises(RuntimeError):
            progress.on_chat_model_start({}, [])
        progress.tool_calls = progress.max_tool_calls
        with self.assertRaises(RuntimeError):
            progress.on_tool_start({"name": "web_fetch"}, "{}")

    def test_identical_failed_tool_calls_stop_next_model_call(self):
        progress = research.ResearchProgress()
        for number in range(3):
            progress.on_tool_start({"name": "compile_sources"}, '{"selections": []}', run_id=number)
            progress.on_tool_end(SimpleNamespace(content="ERROR: missing sources", status="error"), run_id=number)
        with self.assertRaisesRegex(RuntimeError, "Repeated tool failure"):
            progress.on_chat_model_start({}, [])

    def test_success_clears_failed_request_count(self):
        progress = research.ResearchProgress()
        for number, failed in enumerate([True, True, False, True, True]):
            progress.on_tool_start({"name": "compile_sources"}, "{}", run_id=number)
            progress.on_tool_end(SimpleNamespace(content="ERROR: failed" if failed else "Saved", status="error" if failed else "success"), run_id=number)
        progress.on_chat_model_start({}, [])
        self.assertIsNone(progress.abort_reason)

    def test_topic_time_budget_stops_new_calls(self):
        progress = research.ResearchProgress(max_seconds=15)
        progress.started -= 16
        with self.assertRaisesRegex(RuntimeError, "time budget"):
            progress.on_chat_model_start({}, [])
        with self.assertRaisesRegex(RuntimeError, "time budget"):
            progress.on_tool_start({"name": "web_fetch"}, "{}")

    def test_real_agent_graph_compiles_with_bounded_workers(self):
        class ToolAwareFakeModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = ToolAwareFakeModel(responses=[AIMessage(content="Done")],
                                   custom_get_token_ids=lambda text: list(range(len(text.split()))))
        with patch("agents.create_deep_agent", wraps=agents.create_deep_agent) as create:
            graph = agents.build_lead_agent(None, model)
        self.assertTrue(callable(graph.invoke))
        result = graph.invoke({"messages": [{"role": "user", "content": "Smoke test"}]},
                              config={"recursion_limit": agents.RECURSION_LIMIT})
        self.assertEqual(result["messages"][-1].content, "Done")
        specs = create.call_args.kwargs["subagents"]
        self.assertEqual({s["name"] for s in specs}, {"researcher", "citation-checker", "general-purpose"})
        for spec in specs:
            self.assertEqual(len(spec["middleware"]), 2 if spec["name"] == "citation-checker" else 3)
        lead = create.call_args.kwargs["middleware"]
        self.assertEqual(len(lead), 5)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        evidence.reset()
        evidence.register("arxiv", [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001",
                                     "title": "Real retrieved title", "published": "2025-01-01",
                                     "summary": "The method learns predictive dynamics from observations."}])

    def tearDown(self):
        evidence.reset()

    def test_invented_urls_and_relabelled_families_are_rejected(self):
        real = evidence.records()[0]
        self.assertEqual(evidence.source_problems([real]), [])
        for altered in (dict(real, url="https://arxiv.org/abs/12345"), dict(real, source="hf-search"),
                        dict(real, title="Invented title"), dict(real, date="2020-01-01")):
            self.assertTrue(evidence.source_problems([altered]))

    def test_notes_require_retrieved_metadata_and_real_quotations(self):
        note = ('# Scope\n### Real retrieved title\nid: 2501.00001\n'
                'url: https://arxiv.org/abs/2501.00001\ndate: 2025-01-01\nsource: arxiv\n'
                'Evidence:\n- "The method learns predictive dynamics from observations."\nFindings:\n- A finding.\n')
        self.assertEqual(evidence.note_problems(note), [])
        self.assertTrue(evidence.note_problems(note.replace("predictive dynamics", "perfectly accurate physics")))

    def test_lead_cannot_precreate_researcher_notes(self):
        request = SimpleNamespace(tool_call={"name": "write_file", "id": "x", "args": {
            "file_path": "/tmp/work/research/notes/01.md", "content": "placeholder"}})
        handler = Mock()
        result = evidence.EvidenceGuard("lead").wrap_tool_call(request, handler)
        self.assertEqual(result.status, "error")
        handler.assert_not_called()

    def test_phase_guard_enriches_three_tasks_and_blocks_checker_during_research(self):
        from langgraph.prebuilt.tool_node import ToolCallRequest
        context = {"phase": "research", "topic": "world model", "as_of": "2026-10-10"}
        guard = agents.WorkflowGuard(context)
        request = ToolCallRequest(tool_call={"name": "task", "id": "x", "args": {
            "subagent_type": "researcher", "description": "Compare representations"}},
            tool=None, state={}, runtime=None)
        for index in range(1, 4):
            enriched, error = guard._task_request(request)
            self.assertIsNone(error)
            description = enriched.tool_call["args"]["description"]
            self.assertIn("world model", description)
            self.assertIn("2026-10-10", description)
            self.assertIn(f"/{index:02d}-research.md", description)
        self.assertIsNotNone(guard._task_request(request)[1])
        context["phase"] = "writing"
        self.assertIsNotNone(guard._task_request(request)[1])

    def test_checker_requires_actual_claims_and_five_urls(self):
        from langgraph.prebuilt.tool_node import ToolCallRequest
        guard = agents.WorkflowGuard({"phase": "writing"})
        request = ToolCallRequest(tool_call={"name": "task", "id": "x", "args": {
            "subagent_type": "citation-checker", "description": "Verify citations"}},
            tool=None, state={}, runtime=None)
        self.assertIsNotNone(guard._task_request(request)[1])
        claims = "\n".join(f"{i}. Claim: a substantial supported finding with the retrieved evidence excerpt. "
                           f"Source: https://example.org/{i} Evidence: a method learns predictive dynamics." for i in range(5))
        request = request.override(tool_call={**request.tool_call, "args": {
            "subagent_type": "citation-checker", "description": claims}})
        self.assertIsNone(guard._task_request(request)[1])
        self.assertIsNotNone(guard._task_request(request)[1])

    def test_research_phase_hides_report_tools_and_removes_task_after_three_calls(self):
        from langchain.agents.middleware.types import ModelRequest
        from langchain_core.messages import SystemMessage
        guard = agents.WorkflowGuard({"phase": "research"})
        request = ModelRequest(model=Mock(), system_message=SystemMessage(content="Workflow"),
                               messages=[], tools=[SimpleNamespace(name=n) for n in
                                                   ("task", "read_file", "write_file", "execute", "compile_sources")])
        self.assertEqual([t.name for t in guard._model_request(request).tools], ["task", "read_file"])
        guard.research_calls = 3
        self.assertEqual([t.name for t in guard._model_request(request).tools], ["read_file"])

    def test_lead_audit_preserves_real_usage_before_history_is_summarized(self):
        context = {"phase": "research"}
        guard = agents.WorkflowGuard(context)
        message = AIMessage(content="", id="real-call", tool_calls=[{
            "id": "t", "name": "task", "args": {"subagent_type": "researcher"}}],
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
        guard.after_model({"messages": [message]}, None)
        guard.after_model({"messages": [message]}, None)
        self.assertEqual(len(context["audit_messages"]), 1)
        stats = research.summarize(context["audit_messages"], 1, "model")
        self.assertEqual(stats["researcher_calls"], 1)
        self.assertEqual(stats["tokens"], {"input": 100, "output": 20})

    def test_note_compiler_saves_exact_evidence_and_scope_in_sandbox(self):
        entries = [{"id": f"2501.0000{i}", "url": f"https://arxiv.org/abs/2501.0000{i}",
                    "title": f"Retrieved method {i}", "published": "2025-01-01",
                    "summary": "The method learns predictive dynamics from observations."} for i in range(2, 5)]
        evidence.register("arxiv", entries)
        evidence.register("web", [{"url": "https://example.org/paper", "title": "Foundation",
                                   "published": "2018-01-01", "summary": "A compact predictive representation."}])
        backend = Mock()
        backend.upload_files.return_value = [SimpleNamespace(error=None)]
        spec = agents.build_subagents(backend, {"topic": "world model", "as_of": "2026-10-10"})[0]
        compiler = next(t for t in spec["tools"] if t.name == "save_research_notes")
        selected = [{"source": r["source"], "url": r["url"], "finding": "A predictive representation."}
                    for r in evidence.records()[1:]]
        output = compiler.invoke({"file_path": "/tmp/work/research/notes/01-research.md",
                                  "question": "Compare representations", "selections": selected,
                                  "comparison": "Evidence is limited to retrieved abstracts."})
        self.assertTrue(output.startswith("Saved"))
        path, content = backend.upload_files.call_args.args[0][0]
        self.assertEqual(path, "/tmp/work/research/notes/01-research.md")
        self.assertIn("As-of: 2026-10-10", content.decode())
        self.assertEqual(evidence.note_problems(content.decode()), [])

    def test_note_compiler_corrects_a_unique_actual_family(self):
        evidence.reset()
        entries = [{"id": str(i), "url": f"https://example.org/{i}", "title": f"Source {i}",
                    "published": "2025-01-01", "summary": "Retrieved evidence."} for i in range(3)]
        evidence.register("web", entries)
        evidence.register("arxiv", [{"id": "4", "url": "https://arxiv.org/abs/4", "title": "Source 4",
                                     "published": "2025-01-01", "summary": "Retrieved evidence."}])
        backend = Mock()
        backend.upload_files.return_value = [SimpleNamespace(error=None)]
        spec = agents.build_subagents(backend, {"topic": "test", "as_of": "2026-10-10"})[0]
        compiler = next(t for t in spec["tools"] if t.name == "save_research_notes")
        selected = [{"source": "arxiv" if r["source"] == "web" else r["source"],
                     "url": r["url"], "finding": "Supported."} for r in evidence.records()]
        output = compiler.invoke({"file_path": "/tmp/work/research/notes/unique.md", "question": "Q",
                                  "selections": selected, "comparison": "Comparison"})
        self.assertIn("'web'", output)
        self.assertIn("source: web", backend.upload_files.call_args.args[0][0][1].decode())

    def test_web_catalog_uses_actual_title_url_and_date(self):
        evidence.register_web("Title: Actual project\nURL: https://example.org/project\nPublished: 2025-02-03T00:00:00Z\nHighlights:\nActual evidence.")
        found = next(r for r in evidence.records() if r["source"] == "web")
        self.assertEqual(found["title"], "Actual project")
        self.assertEqual(found["date"], "2025-02-03")

    def test_sandbox_evidence_checker_rejects_placeholders_and_short_body(self):
        from check_evidence import check as check_evidence
        real = evidence.records()[0]
        self.assertEqual(check_evidence([real], evidence.records(), "word " * 1000), [])
        self.assertTrue(check_evidence([dict(real, url="https://arxiv.org/abs/12345")], evidence.records(), "word " * 1000))
        self.assertTrue(check_evidence([real], evidence.records(), "too short"))

    def test_compile_sources_copies_metadata_and_deduplicates(self):
        backend = Mock()
        backend.upload_files.return_value = [SimpleNamespace(error=None)]
        with patch("agents.create_deep_agent") as create:
            agents.build_lead_agent(backend, Mock())
        compiler = next(tool for tool in create.call_args.kwargs["tools"] if tool.name == "compile_sources")
        selection = {"source": "arxiv", "url": "https://arxiv.org/abs/2501.00001"}
        result = json.loads(compiler.invoke({"selections": [selection, selection]}))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["n"], 1)
        self.assertEqual(result[0]["title"], "Real retrieved title")
        self.assertEqual(result[0]["date"], "2025-01-01")
        self.assertEqual(json.loads(backend.upload_files.call_args.args[0][0][1]), result)

    def test_compile_sources_never_writes_an_unretrieved_url(self):
        backend = Mock()
        with patch("agents.create_deep_agent") as create:
            agents.build_lead_agent(backend, Mock())
        compiler = next(tool for tool in create.call_args.kwargs["tools"] if tool.name == "compile_sources")
        result = compiler.invoke({"selections": [{"source": "arxiv", "url": "https://arxiv.org/abs/12345"}]})
        self.assertTrue(result.startswith("ERROR:"))
        backend.upload_files.assert_not_called()

    def test_selection_schema_canonicalizes_only_known_tool_names(self):
        from pydantic import ValidationError
        self.assertEqual(agents.SourceSelection(source="hf_search_papers", url="https://example.org").source,
                         "hf-search")
        with self.assertRaises(ValidationError):
            agents.SourceSelection(source="A paper title", url="https://example.org")


class CitationNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"n": 3, "title": "A Retrieved Method | Author Project", "date": "2025-01-01",
                         "url": "https://example.org/paper", "source": "web", "id": ""}]

    def test_exact_title_year_converts_and_finalizer_rebuilds_references(self):
        from normalize_citations import normalize
        from finalize_citations import finalize
        body, problems = normalize("# Survey\nA supported claim (A Retrieved Method, 2025).\n\n## References\nwrong", self.sources)
        self.assertEqual(problems, [])
        self.assertIn("claim [3]", body)
        report, sources, errors = finalize(body, self.sources)
        self.assertEqual(errors, [])
        self.assertEqual(check(report, sources), [])

    def test_unknown_ambiguous_and_wrong_year_citations_fail_closed(self):
        from normalize_citations import normalize
        for body, sources in [
            ("Claim (Invented Method, 2025).", self.sources),
            ("Claim (A Retrieved Method, 2024).", self.sources),
            ("Claim (A Retrieved Method, 2025).", self.sources + [dict(self.sources[0], n=4)]),
        ]:
            normalized, problems = normalize(body, sources)
            self.assertTrue(problems)
            self.assertEqual(normalized.strip(), body)

    def test_code_spans_are_preserved(self):
        from normalize_citations import normalize
        body = "`(A Retrieved Method, 2025)`\n```\n(A Retrieved Method, 2025)\n```"
        self.assertEqual(normalize(body, self.sources), (body + "\n", []))

    def test_citation_links_convert_only_for_exact_catalog_urls(self):
        from normalize_citations import normalize
        self.assertEqual(normalize("Claim ([method](https://example.org/paper)).", self.sources),
                         ("Claim [3].\n", []))
        self.assertTrue(normalize("Claim [method](https://example.org/invented).", self.sources)[1])

    def test_sandbox_adapter_prepends_repair_only_for_finalizer_execution(self):
        backend = Mock()
        adapter = agents.CitationSandbox(backend)
        command = f"python3 {agents.FINALIZER_PATH} && python3 {agents.VALIDATOR_PATH}"
        adapter.execute(command)
        self.assertEqual(backend.execute.call_args.args[0], f"python3 {agents.NORMALIZER_PATH} && " + command)
        adapter.execute("ls /tmp/work")
        self.assertEqual(backend.execute.call_args.args[0], "ls /tmp/work")


if __name__ == "__main__":
    unittest.main()
