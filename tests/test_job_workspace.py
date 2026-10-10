"""Offline UI integration through real crawlers and validated AI assessments."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import requests
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.main import MainWindow
from src.models.model import JobSummary, ResumeFit
from test_jobstreet import SEARCH_HTML, DETAIL_HTML, response_for
from test_model import JOB, RESUME, SUMMARY, FIT
from test_resume_parser import write_pdf


class JobWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.resume = self.root / "resume.txt"
        self.resume.write_text(RESUME, encoding="utf-8")
        self.window = MainWindow()
        self.panel = self.window.jobs_panel
        # Preserve coverage for the existing Transformers integration.
        self.panel.backend_input.setCurrentIndex(1)
        self.session = Mock(spec=requests.Session)

    def tearDown(self):
        for worker in (self.window.worker, self.panel.worker):
            if worker is not None:
                worker.wait(5000)
        self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def wait_for_operation(self):
        # Yield Python's GIL while servicing Qt events during cold imports.
        deadline = time.monotonic() + 20
        while self.window.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(self.window.busy, "Desktop operation did not finish")

    def prepare_match(self):
        self.window.parse_resume(self.resume)
        self.wait_for_operation()
        self.panel.job_text.setPlainText(JOB)

    def generated_outputs(self):
        return [JobSummary.model_validate(SUMMARY), ResumeFit.model_validate(FIT)]

    def run_match(self):
        with patch("src.models.model.JobMatchAgent._generate", side_effect=self.generated_outputs()):
            self.panel.match_button.click()
            self.wait_for_operation()

    def test_search_and_full_fetch_use_real_crawler_and_close_sessions(self):
        self.session.get.side_effect = [response_for(SEARCH_HTML), response_for(DETAIL_HTML)]
        self.panel.title_input.setText("Data Scientist")
        self.panel.page_input.setValue(2)
        with patch("src.crawler.jobstreet.requests.Session", return_value=self.session):
            self.panel.search_button.click()
            self.wait_for_operation()
            self.assertEqual(self.panel.results_list.count(), 2)
            self.assertEqual(self.panel.url_input.text(), "https://my.jobstreet.com/job/12345678")
            self.assertEqual(self.panel.job_text.toPlainText(), "")
            self.panel.fetch_button.click()
            self.wait_for_operation()
        self.assertIn("Build dashboards.", self.panel.job_text.toPlainText())
        self.assertNotIn("Sign in", self.panel.job_text.toPlainText())
        self.assertIn("Example Sdn Bhd", self.panel.job_source.text())
        self.assertIn("page=2", self.session.get.call_args_list[0].args[0])
        self.assertEqual(self.session.close.call_count, 2)
        self.panel.results_list.setCurrentRow(1)
        self.assertEqual(self.panel.job_text.toPlainText(), "")

    def test_no_results_and_blocked_search_clear_previous_listings(self):
        self.session.get.side_effect = [response_for(SEARCH_HTML), response_for("No matching jobs")]
        with patch("src.crawler.jobstreet.requests.Session", return_value=self.session):
            self.panel.search_jobs()
            self.wait_for_operation()
            self.panel.search_jobs()
            self.wait_for_operation()
        self.assertEqual(self.panel.results_list.count(), 0)
        self.assertFalse(self.panel.fetch_button.isEnabled())
        self.assertIn("No jobs found", self.panel.results_hint.text())
        self.session.get.side_effect = requests.HTTPError("403 Forbidden")
        with patch("src.crawler.jobstreet.requests.Session", return_value=self.session), \
                patch("src.ui.job_workspace.QMessageBox.warning") as warning:
            self.panel.search_jobs()
            self.wait_for_operation()
        self.assertIn("paste", warning.call_args.args[2])
        self.assertIn("failed", self.panel.results_hint.text())
        self.assertTrue(self.panel.search_button.isEnabled())

    def test_failed_fetch_removes_previous_ad_and_recovers_to_paste(self):
        self.prepare_match()
        self.run_match()
        self.panel.url_input.setText("https://example.com/job/123")
        with patch("src.crawler.jobstreet.requests.Session", return_value=self.session), \
                patch("src.ui.job_workspace.QMessageBox.warning") as warning:
            self.panel.fetch_url_button.click()
            self.wait_for_operation()
        warning.assert_called_once()
        self.session.get.assert_not_called()
        self.assertEqual(self.panel.job_text.toPlainText(), "")
        self.assertEqual(self.panel.report_json, "")
        self.assertFalse(self.panel.match_button.isEnabled())
        self.panel.job_text.setPlainText(JOB)
        self.assertTrue(self.panel.match_button.isEnabled())

    def test_ai_match_passes_extracted_text_and_reuses_agent(self):
        self.prepare_match()
        with patch("src.models.model.JobMatchAgent._generate", side_effect=self.generated_outputs() * 2) as generate:
            self.panel.match_button.click()
            self.wait_for_operation()
            agent = self.panel.session.agent
            self.panel.match_button.click()
            self.wait_for_operation()
        self.assertIs(self.panel.session.agent, agent)
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(generate.call_args.args[1]["resume_text"], RESUME)
        self.assertEqual(generate.call_args.args[1]["job_description"], JOB)
        self.assertEqual(self.panel.fit_label.text(), "Strong Fit")
        self.assertIn("Job evidence: Python experience.", self.panel.report_preview.toPlainText())
        self.assertIn("Resume evidence: Built Python pipelines.", self.panel.report_preview.toPlainText())
        self.assertTrue(self.panel.save_report_button.isEnabled())

    def test_report_copy_export_and_original_protection(self):
        self.prepare_match()
        self.run_match()
        self.panel.copy_report_button.click()
        self.assertIn("REQUIREMENT EVIDENCE", self.app.clipboard().text())
        output = self.root / "report.json"
        with patch("src.ui.job_workspace.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            self.panel.save_report_button.click()
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["fit"], "strong_fit")
        self.assertEqual(len(report["requirements"]), 2)
        with patch("src.ui.job_workspace.QFileDialog.getSaveFileName", return_value=(str(self.resume), "")), \
                patch("src.ui.job_workspace.QMessageBox.warning") as warning:
            self.panel.save_report()
        warning.assert_called_once()
        self.assertEqual(self.resume.read_text(encoding="utf-8"), RESUME)

    def test_input_changes_invalidate_report_and_gate_matching(self):
        self.panel.job_text.setPlainText(JOB)
        self.assertFalse(self.panel.match_button.isEnabled())
        self.prepare_match()
        self.run_match()
        self.panel.job_text.appendPlainText("Updated requirements")
        self.assertEqual(self.panel.report_json, "")
        self.assertFalse(self.panel.save_report_button.isEnabled())
        self.panel.job_text.setPlainText(JOB)
        self.run_match()
        self.panel.model_input.setText("another-model")
        self.assertEqual(self.panel.report_preview.toPlainText(), "")
        self.window.clear_selection()
        self.assertFalse(self.panel.match_button.isEnabled())
        self.assertFalse(self.panel.similarity_button.isEnabled())

    def test_missing_dependencies_and_invalid_ai_output_are_recoverable(self):
        self.prepare_match()
        self.run_match()
        with patch("src.models.model.JobMatchAgent._generate", side_effect=RuntimeError("Install requirements-model.txt")), \
                patch("src.ui.job_workspace.QMessageBox.warning") as warning:
            self.panel.match_button.click()
            self.wait_for_operation()
        self.assertIn("requirements-model.txt", warning.call_args.args[2])
        self.assertEqual(self.panel.report_json, "")
        self.assertEqual(self.panel.fit_label.text(), "AI assessment failed")
        invalid = copy.deepcopy(FIT)
        invalid["requirements"][0]["resume_evidence"] = "Invented evidence"
        outputs = [JobSummary.model_validate(SUMMARY), ResumeFit.model_validate(invalid)]
        with patch("src.models.model.JobMatchAgent._generate", side_effect=outputs), \
                patch("src.ui.job_workspace.QMessageBox.warning") as warning:
            self.panel.match_button.click()
            self.wait_for_operation()
        self.assertIn("quote", warning.call_args.args[2])
        self.assertEqual(self.panel.report_preview.toPlainText(), "")
        self.assertTrue(self.panel.match_button.isEnabled())

    def test_model_change_creates_new_agent(self):
        self.prepare_match()
        self.run_match()
        first = self.panel.session.agent
        self.panel.model_input.setText("local-model-path")
        self.run_match()
        self.assertIsNot(self.panel.session.agent, first)
        self.assertEqual(self.panel.session.agent.model_id, "local-model-path")

    def test_ollama_is_desktop_default_and_backend_changes_clear_results(self):
        from src.models.config import OLLAMA_MODEL_ID

        fresh = MainWindow()
        self.assertEqual(fresh.jobs_panel.backend_input.currentData(), "ollama")
        self.assertEqual(fresh.jobs_panel.model_input.text(), OLLAMA_MODEL_ID)
        fresh.close()
        fresh.deleteLater()
        self.prepare_match()
        self.run_match()
        self.panel.backend_input.setCurrentIndex(0)
        self.assertEqual(self.panel.report_json, "")
        with patch("src.models.ollama_agent.OllamaJobMatchAgent._generate", side_effect=self.generated_outputs()):
            self.panel.match_button.click()
            self.wait_for_operation()
        self.assertEqual(self.panel.session.backend, "ollama")
        self.assertEqual(self.panel.session.agent.model_id, OLLAMA_MODEL_ID)
        self.assertEqual(self.panel.fit_label.text(), "Strong Fit")

    def test_ollama_is_unloaded_before_resume_ocr(self):
        self.prepare_match()
        self.panel.backend_input.setCurrentIndex(0)
        with patch("src.models.ollama_agent.OllamaJobMatchAgent._generate", side_effect=self.generated_outputs()):
            self.panel.match_button.click()
            self.wait_for_operation()
        agent = self.panel.session.agent
        pdf = self.root / "scan.pdf"
        write_pdf(pdf, ["<scan>"])
        self.window.select_file(pdf)
        self.window.ocr_checkbox.setChecked(True)

        def extract(*args, **kwargs):
            agent.unload.assert_called_once()
            return RESUME

        with patch.object(agent, "unload"), patch("src.main.extract_resume_text", side_effect=extract):
            self.window.parse_button.click()
            self.wait_for_operation()
        self.assertIsNone(self.panel.session.agent)

    def test_ocr_releases_cached_ai_memory_before_extraction(self):
        self.prepare_match()
        self.run_match()
        self.assertIsNotNone(self.panel.session.agent)
        pdf = self.root / "scan.pdf"
        write_pdf(pdf, ["<scan>"])
        self.window.select_file(pdf)
        self.window.ocr_checkbox.setChecked(True)
        torch = Mock()
        torch.cuda.is_available.return_value = True
        observed_threads = []

        def extract(*args, **kwargs):
            self.assertIsNone(self.panel.session.agent)
            torch.cuda.empty_cache.assert_called_once()
            observed_threads.append(threading.get_ident())
            return RESUME

        with patch.dict(sys.modules, {"torch": torch}), \
                patch("src.main.extract_resume_text", side_effect=extract) as extractor:
            self.window.parse_button.click()
            self.wait_for_operation()
        extractor.assert_called_once_with(pdf, ocr=True, password=None)
        self.assertNotEqual(observed_threads[0], threading.get_ident())
        self.assertEqual(self.window.text_preview.toPlainText(), RESUME)

    def test_ai_worker_keeps_gui_responsive_and_serializes_all_operations(self):
        self.prepare_match()
        release = threading.Event()
        gui_thread = threading.get_ident()
        worker_threads = []

        def generate(*args):
            worker_threads.append(threading.get_ident())
            release.wait(3)
            return self.generated_outputs()[0 if len(worker_threads) == 1 else 1]

        with patch("src.models.model.JobMatchAgent._generate", side_effect=generate):
            try:
                self.panel.match_button.click()
                QTest.qWait(30)
                self.assertFalse(self.window.browse_button.isEnabled())
                self.assertTrue(self.panel.job_text.isReadOnly())
                self.assertFalse(self.panel.search_button.isEnabled())
                event = QCloseEvent()
                self.window.closeEvent(event)
                self.assertFalse(event.isAccepted())
                self.window.clear_selection()
                self.assertEqual(self.window.selected_path, self.resume)
                self.panel.search_jobs()
            finally:
                release.set()
                self.wait_for_operation()
        self.assertEqual(len(worker_threads), 2)
        self.assertTrue(all(thread != gui_thread for thread in worker_threads))
        self.assertTrue(self.window.browse_button.isEnabled())
        self.assertFalse(self.panel.job_text.isReadOnly())

    def test_text_similarity_runs_without_model_and_is_labeled_separately(self):
        self.prepare_match()
        with patch("src.models.model.JobMatchAgent._load", side_effect=AssertionError("Model should not load")):
            self.panel.similarity_button.click()
            self.wait_for_operation()
        self.assertIn("TF-IDF", self.panel.similarity_label.text())
        self.assertIn("not a hiring probability", self.panel.similarity_label.text())
        self.assertEqual(self.panel.report_json, "")
        self.panel.job_text.clear()
        self.assertEqual(self.panel.similarity_label.text(), "Text similarity: —")

    def test_startup_does_not_import_model_runtime_or_crawl(self):
        result = subprocess.run(
            [sys.executable, "-c", "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; "
             "import sys; from PySide6.QtWidgets import QApplication; from src.main import MainWindow; "
             "app=QApplication([]); w=MainWindow(); "
             "assert not {'torch','transformers','accelerate','src.crawler.jobstreet','src.models.model'} & sys.modules.keys(); w.close()"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
