"""Offline agent tests: no Transformers imports, weight downloads or GPU needed."""

import copy
import json
import subprocess
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.models.model import AgentOutputError, JobMatchAgent, JobSummary, ResumeFit


JOB = "Data Analyst. Requirements: Python experience. SQL experience."
RESUME = "Built Python pipelines. Created SQL reports."
SUMMARY = {
    "title": "Data Analyst", "summary": "Analyze data using Python and SQL.",
    "responsibilities": [], "skills": ["Python", "SQL"],
    "requirements": [
        {"requirement": "Python experience", "importance": "required", "job_evidence": "Python experience."},
        {"requirement": "SQL experience", "importance": "required", "job_evidence": "SQL experience."},
    ],
}
FIT = {
    "fit": "strong_fit", "reasoning": "The resume supplies evidence for both requirements.",
    "requirements": [
        {"requirement_index": 0, "status": "met", "resume_evidence": "Built Python pipelines.", "explanation": "Python project experience."},
        {"requirement_index": 1, "status": "met", "resume_evidence": "Created SQL reports.", "explanation": "SQL reporting experience."},
    ],
    "follow_up_questions": [],
}


def fake_agent(response: str, *, input_tokens: int = 10, context_limit: int = 256000):
    """Exercise the real generation path with small stand-ins for HF objects."""
    agent = JobMatchAgent()
    inputs = {"input_ids": SimpleNamespace(shape=(1, input_tokens))}
    batch = Mock()
    batch.__getitem__ = Mock(side_effect=inputs.__getitem__)
    batch.to.return_value = inputs
    agent._processor = Mock()
    agent._processor.apply_chat_template.return_value = batch
    agent._processor.decode.return_value = response
    agent._model = Mock()
    agent._model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=context_limit))
    agent._model.device = "cpu"
    agent._model.generate.return_value = [list(range(input_tokens + 2))]
    agent._torch = SimpleNamespace(inference_mode=nullcontext)
    return agent


class AgentTests(unittest.TestCase):
    def compare(self, fit, summary=None, resume=RESUME):
        agent = JobMatchAgent()
        outputs = [JobSummary.model_validate(summary or SUMMARY), ResumeFit.model_validate(fit)]
        with patch.object(agent, "_generate", side_effect=outputs):
            return agent.assess_resume(JOB, resume)

    def test_import_and_construction_do_not_import_inference_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; from src.models.model import JobMatchAgent; JobMatchAgent(); "
             "assert not {'torch', 'transformers', 'accelerate'} & sys.modules.keys()"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_inputs_fail_before_loading(self):
        agent = JobMatchAgent()
        with patch.object(agent, "_load") as load:
            for job, resume in [(" ", RESUME), (JOB, ""), (JOB, None)]:
                with self.subTest(job=job, resume=resume), self.assertRaises(ValueError):
                    agent.assess_resume(job, resume)
            load.assert_not_called()

    def test_summary_and_assessment_return_typed_serializable_results(self):
        result = self.compare(FIT)
        self.assertEqual(result.fit, "strong_fit")
        self.assertEqual(result.job_summary.title, "Data Analyst")
        self.assertEqual(json.loads(result.model_dump_json())["requirements"][0]["status"], "met")

    def test_fabricated_job_quote_is_rejected(self):
        summary = copy.deepcopy(SUMMARY)
        summary["requirements"][0]["job_evidence"] = "Ten years of Python experience."
        with self.assertRaisesRegex(AgentOutputError, "job description"):
            self.compare(FIT, summary)

    def test_resume_quote_must_exist_but_whitespace_may_differ(self):
        self.assertEqual(self.compare(FIT, resume="Built Python\npipelines. Created SQL reports.").fit, "strong_fit")
        fit = copy.deepcopy(FIT)
        fit["requirements"][0]["resume_evidence"] = "Led a Python engineering team."
        with self.assertRaisesRegex(AgentOutputError, "resume"):
            self.compare(fit)

    def test_missing_duplicate_and_out_of_range_requirement_indices_fail(self):
        for indices in ([0], [0, 0], [0, 2]):
            fit = copy.deepcopy(FIT)
            fit["requirements"] = [dict(FIT["requirements"][0], requirement_index=i) for i in indices]
            with self.subTest(indices=indices), self.assertRaisesRegex(AgentOutputError, "exactly once"):
                self.compare(fit)

    def test_missing_evidence_does_not_prove_a_candidate_lacks_skills(self):
        fit = copy.deepcopy(FIT)
        fit["fit"] = "insufficient_information"
        for item in fit["requirements"]:
            item.update(status="not_evidenced", resume_evidence=None, explanation="Not mentioned.")
        self.assertEqual(self.compare(fit).fit, "insufficient_information")
        fit["fit"] = "low_fit"
        with self.assertRaisesRegex(AgentOutputError, "relevant resume evidence"):
            self.compare(fit)

    def test_strong_fit_is_rejected_when_a_core_requirement_is_unresolved(self):
        fit = copy.deepcopy(FIT)
        fit["requirements"][1].update(status="not_evidenced", resume_evidence=None)
        with self.assertRaisesRegex(AgentOutputError, "every core requirement"):
            self.compare(fit)
        fit["fit"] = "partial_fit"
        self.assertEqual(self.compare(fit).fit, "partial_fit")

    def test_low_fit_requires_explicit_conflicting_core_evidence(self):
        fit = copy.deepcopy(FIT)
        fit["fit"] = "low_fit"
        with self.assertRaisesRegex(AgentOutputError, "explicit conflicting"):
            self.compare(fit)
        fit["requirements"][1].update(
            status="contradicted", resume_evidence="No SQL experience.",
            explanation="The resume explicitly states no SQL experience.",
        )
        result = self.compare(fit, resume="Built Python pipelines. No SQL experience.")
        self.assertEqual(result.fit, "low_fit")

    def test_unknown_evidence_is_null_and_other_statuses_need_quotes(self):
        for status, evidence in [("not_evidenced", "Created SQL reports."), ("met", None), ("partial", None), ("contradicted", None)]:
            fit = copy.deepcopy(FIT)
            fit["requirements"][1].update(status=status, resume_evidence=evidence)
            with self.subTest(status=status), self.assertRaises(AgentOutputError):
                self.compare(fit)

    def test_jobs_without_requirements_need_insufficient_information(self):
        summary = dict(SUMMARY, requirements=[])
        fit = dict(FIT, fit="insufficient_information", requirements=[])
        self.assertEqual(self.compare(fit, summary).fit, "insufficient_information")
        with self.assertRaises(AgentOutputError):
            self.compare(dict(fit, fit="strong_fit"), summary)

    def test_real_generation_path_uses_documents_and_slices_prompt_tokens(self):
        agent = fake_agent(json.dumps(SUMMARY))
        result = agent.summarize_job(JOB)
        self.assertEqual(result.title, "Data Analyst")
        messages = agent._processor.apply_chat_template.call_args.args[0]
        self.assertIn(JOB, messages[1]["content"])
        self.assertIn("untrusted data", messages[0]["content"])
        self.assertFalse(agent._processor.apply_chat_template.call_args.kwargs["enable_thinking"])
        agent._processor.decode.assert_called_once_with([10, 11], skip_special_tokens=True)
        agent._model.generate.assert_called_once()
        agent._model.to.assert_not_called()

    def test_malformed_truncated_and_extra_field_outputs_fail_closed(self):
        for response in ('{"title":', 'Here is your answer: ' + json.dumps(SUMMARY), json.dumps(dict(SUMMARY, unexpected=True))):
            with self.subTest(response=response), self.assertRaises(AgentOutputError):
                fake_agent(response).summarize_job(JOB)
        self.assertEqual(fake_agent('```json\n' + json.dumps(SUMMARY) + '\n```').summarize_job(JOB).title, "Data Analyst")

    def test_large_input_or_context_overflow_fails_before_generation(self):
        for tokens, limit in [(16385, 256000), (10, 4096)]:
            agent = fake_agent(json.dumps(SUMMARY), input_tokens=tokens, context_limit=limit)
            with self.subTest(tokens=tokens, limit=limit), self.assertRaisesRegex(ValueError, "limit"):
                agent.summarize_job(JOB)
            agent._model.generate.assert_not_called()

    def test_load_uses_dispatch_and_is_reused(self):
        agent = JobMatchAgent()
        processor_type, model_type, model = Mock(), Mock(), Mock()
        model_type.from_pretrained.return_value = model
        transformers = SimpleNamespace(AutoProcessor=processor_type, AutoModelForMultimodalLM=model_type)
        with patch.dict(sys.modules, {"torch": Mock(), "accelerate": Mock(), "transformers": transformers}):
            agent._load()
            agent._load()
        model_type.from_pretrained.assert_called_once_with(agent.model_id, dtype="auto", device_map="auto")
        model.eval.assert_called_once()
        model.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
