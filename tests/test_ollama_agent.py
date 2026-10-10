"""Ollama request contracts and shared evidence validation without inference."""

import copy
import json
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import requests

from src.models.config import OLLAMA_MODEL_ID
from src.models.model import AgentOutputError, JobSummary
from src.models.ollama_agent import OllamaJobMatchAgent
from test_model import JOB, RESUME, SUMMARY, FIT


def response_for(content, *, done_reason="stop", done=True, prompt_tokens=1000, output_tokens=200):
    return {"message": {"content": json.dumps(content)}, "done": done, "done_reason": done_reason,
            "prompt_eval_count": prompt_tokens, "eval_count": output_tokens}


def http_session(response):
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    session.post.return_value = response
    return session


class OllamaTests(unittest.TestCase):
    def test_default_uses_exact_qwen_q4_and_does_not_import_inference_stack(self):
        agent = OllamaJobMatchAgent()
        self.assertEqual(agent.model_id, OLLAMA_MODEL_ID)
        self.assertEqual(agent.context_size, 8192)
        result = subprocess.run([sys.executable, "-c", "import sys; from src.models.ollama_agent import OllamaJobMatchAgent; "
                                 "OllamaJobMatchAgent(); assert not {'torch','transformers','accelerate'} & sys.modules.keys()"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schema_thinking_context_and_sampling_payload(self):
        agent = OllamaJobMatchAgent()
        with patch.object(agent, "_request", side_effect=[response_for(SUMMARY), response_for(FIT)]) as request:
            report = agent.assess_resume(JOB, RESUME)
        self.assertEqual(report.fit, "strong_fit")
        first = request.call_args_list[0]
        self.assertEqual(first.args[0], "/api/chat")
        payload = first.args[1]
        self.assertEqual(payload["model"], OLLAMA_MODEL_ID)
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertIs(payload["truncate"], False)
        self.assertIs(payload["shift"], False)
        self.assertEqual(payload["format"], JobSummary.model_json_schema())
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertIn(JOB, payload["messages"][1]["content"])
        self.assertIn(RESUME, request.call_args_list[1].args[1]["messages"][1]["content"])

    def test_schema_and_evidence_errors_fail_closed(self):
        invalid = copy.deepcopy(FIT)
        invalid["requirements"][0]["resume_evidence"] = "An invented resume quote"
        agent = OllamaJobMatchAgent()
        with patch.object(agent, "_request", side_effect=[response_for(SUMMARY), response_for(invalid)]):
            with self.assertRaisesRegex(AgentOutputError, "quote"):
                agent.assess_resume(JOB, RESUME)
        invalid_summary = copy.deepcopy(SUMMARY)
        invalid_summary["requirements"][0]["job_evidence"] = "An invented job quote"
        with patch.object(agent, "_request", return_value=response_for(invalid_summary)):
            with self.assertRaisesRegex(AgentOutputError, "quote"):
                agent.summarize_job(JOB)
        with patch.object(agent, "_request", return_value=response_for({"unexpected": "field"})):
            with self.assertRaises(AgentOutputError):
                agent.summarize_job(JOB)

    def test_incomplete_truncated_and_empty_answers_rejected(self):
        agent = OllamaJobMatchAgent()
        for data in (response_for(SUMMARY, done=False), response_for(SUMMARY, done_reason="length"),
                     {**response_for(SUMMARY), "message": {"content": ""}},
                     {**response_for(SUMMARY), "message": {"content": "{broken"}}):
            with self.subTest(data=data), patch.object(agent, "_request", return_value=data):
                with self.assertRaises(AgentOutputError):
                    agent.summarize_job(JOB)

    def test_input_budget_and_invalid_config_fail_before_requests(self):
        agent = OllamaJobMatchAgent()
        with patch.object(agent, "_request") as request:
            for job in ("", "a" * 30000, "你好" * 5000):
                with self.assertRaises(ValueError):
                    agent.summarize_job(job)
            request.assert_not_called()
        for kwargs in ({"base_url": "https://example.com"}, {"base_url": "http://localhost:11434/api"},
                       {"context_size": True}, {"max_new_tokens": 9000}, {"max_new_tokens": False},
                       {"timeout": 0}, {"timeout": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OllamaJobMatchAgent(**kwargs)

    def test_realistic_documents_fit_the_context_and_are_sent_whole(self):
        # About 3 KB of job ad and 6 KB of resume: rejected when bytes were counted as tokens.
        job = JOB + " Maintain dashboards and explain sales trends to stakeholders." * 50
        resume = RESUME + " Automated weekly reporting for regional managers." * 120
        agent = OllamaJobMatchAgent()
        with patch.object(agent, "_request", side_effect=[response_for(SUMMARY), response_for(FIT)]) as request:
            self.assertEqual(agent.assess_resume(job, resume).fit, "strong_fit")
        content = request.call_args_list[1].args[1]["messages"][1]["content"]
        self.assertIn(job, content)
        self.assertIn(resume, content)

    def test_prompt_rejected_by_ollama_is_actionable_and_model_is_still_unloaded(self):
        agent = OllamaJobMatchAgent()
        # Body returned by Ollama 0.40.2 for truncate=False with a 12,208-token prompt.
        response = Mock(status_code=400, text=json.dumps({"error": json.dumps({"error": {
            "code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 12208, "n_ctx": 8192,
            "message": "request (12208 tokens) exceeds the available context size (8192 tokens), try increasing it",
        }})}))
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
        with patch("src.models.ollama_agent.requests.Session", return_value=http_session(response)),                 self.assertRaisesRegex(ValueError, "need 12,208 tokens, more than the 8,192-token context") as caught:
            agent.summarize_job(JOB)
        self.assertNotIsInstance(caught.exception, AgentOutputError)
        with patch.object(agent, "_request") as request:
            agent.unload()
        request.assert_called_once()
        response.text = json.dumps({"error": "invalid options"})
        with patch("src.models.ollama_agent.requests.Session", return_value=http_session(response)),                 self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            agent.summarize_job(JOB)

    def test_missing_token_usage_and_context_overflow_fail_closed(self):
        agent = OllamaJobMatchAgent()
        for field in ("prompt_eval_count", "eval_count"):
            data = response_for(SUMMARY)
            del data[field]
            with self.subTest(field=field), patch.object(agent, "_request", return_value=data),                     self.assertRaisesRegex(AgentOutputError, "token usage"):
                agent.summarize_job(JOB)
        # Seen when context shifting is left on: the answer completes after prompt tokens were dropped.
        shifted = response_for(SUMMARY, prompt_tokens=7512, output_tokens=900)
        with patch.object(agent, "_request", return_value=shifted),                 self.assertRaisesRegex(ValueError, "exceed the 8,192-token context"):
            agent.summarize_job(JOB)
        full = response_for(SUMMARY, prompt_tokens=7512, output_tokens=680)
        with patch.object(agent, "_request", return_value=full):
            self.assertEqual(agent.summarize_job(JOB).title, "Data Analyst")

    def test_unload_is_sent_only_after_use_and_releases_memory(self):
        agent = OllamaJobMatchAgent()
        with patch.object(agent, "_request", return_value=response_for(SUMMARY)) as request:
            agent.unload()
            request.assert_not_called()
            agent.summarize_job(JOB)
            agent.unload()
            agent.unload()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args, ("/api/generate", {"model": OLLAMA_MODEL_ID, "keep_alive": 0, "stream": False}))

    def test_local_http_disables_proxies_and_closes_response(self):
        agent = OllamaJobMatchAgent()
        response = Mock(status_code=200)
        response.json.return_value = {"done": True}
        session = http_session(response)
        with patch("src.models.ollama_agent.requests.Session", return_value=session):
            self.assertEqual(agent._request("/api/chat", {"model": "local"}), {"done": True})
        self.assertFalse(session.trust_env)
        self.assertFalse(session.post.call_args.kwargs["allow_redirects"])
        response.__exit__.assert_called_once()
        session.__exit__.assert_called_once()

    def test_connection_timeout_and_missing_model_errors_are_actionable(self):
        agent = OllamaJobMatchAgent()
        for error, message in ((requests.ConnectionError(), "Start Ollama"),
                               (requests.Timeout(), "timed out"),
                               (requests.HTTPError(response=Mock(status_code=404)), "ollama pull")):
            session = Mock()
            session.__enter__ = Mock(return_value=session)
            session.__exit__ = Mock(return_value=False)
            session.post.side_effect = error
            with patch("src.models.ollama_agent.requests.Session", return_value=session), \
                    self.assertRaisesRegex(RuntimeError, message):
                agent._request("/api/chat", {})


if __name__ == "__main__":
    unittest.main()
