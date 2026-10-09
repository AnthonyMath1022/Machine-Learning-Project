"""Desktop workflow regressions with a real Qt event loop and local documents."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from docx import Document
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.main import MainWindow
from src.resume.parser import EncryptedPDFError, OCRRequiredError
from test_resume_parser import write_pdf


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.resume = self.root / "resume.txt"
        self.text = "Data analyst\nPython, SQL and Power BI\nBuilt reporting pipelines."
        self.resume.write_text(self.text, encoding="utf-8")
        self.window = MainWindow()

    def tearDown(self):
        # A failed assertion must not leave a live QThread behind.
        if self.window.worker is not None:
            self.window.worker.wait(5000)
        self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def wait_for_extraction(self):
        deadline = time.monotonic() + 5
        while self.window.worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window.worker, "Background extraction did not finish")

    def extract(self, path=None):
        self.window.select_file(path or self.resume)
        self.window.parse_button.click()
        self.wait_for_extraction()

    def test_native_documents_copy_save_and_clear(self):
        pdf = self.root / "resume.pdf"
        write_pdf(pdf, ["Python SQL Power BI"])
        docx = self.root / "resume.docx"
        document = Document()
        document.add_paragraph(self.text)
        document.save(docx)
        for path in (self.resume, pdf, docx):
            with self.subTest(format=path.suffix):
                self.extract(path)
                self.assertIn("Python", self.window.text_preview.toPlainText())
                self.assertIn("Power BI", self.window.skills_label.text())
                self.assertTrue(self.window.save_button.isEnabled())
        self.window.copy_button.click()
        self.assertEqual(self.app.clipboard().text(), self.text)
        output = self.root / "export.txt"
        with patch("src.main.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            self.window.save_button.click()
        self.assertEqual(output.read_text(encoding="utf-8"), self.text)
        self.window.clear_button.click()
        self.assertIsNone(self.window.selected_path)
        self.assertFalse(self.window.parse_button.isEnabled())
        self.assertFalse(self.window.save_action.isEnabled())
        self.assertEqual(self.window.text_preview.toPlainText(), "")

    def test_cancelled_dialogs_preserve_current_resume(self):
        self.extract()
        with patch("src.main.QFileDialog.getOpenFileName", return_value=("", "")):
            self.window.browse_button.click()
        with patch("src.main.QFileDialog.getSaveFileName", return_value=("", "")):
            self.window.save_button.click()
        self.assertEqual(self.window.selected_path, self.resume)
        self.assertEqual(self.window.text_preview.toPlainText(), self.text)

    def test_save_cannot_overwrite_original(self):
        self.extract()
        with patch("src.main.QFileDialog.getSaveFileName", return_value=(str(self.resume), "")), \
                patch("src.main.QMessageBox.warning") as warning:
            self.window.save_text()
        warning.assert_called_once()
        self.assertEqual(self.resume.read_text(encoding="utf-8"), self.text)

    def test_failed_extraction_clears_stale_text_and_recovers(self):
        self.extract()
        with patch("src.main.extract_resume_text", side_effect=OCRRequiredError("Enable OCR")), \
                patch("src.main.QMessageBox.warning") as warning:
            self.window.parse_button.click()
            self.wait_for_extraction()
        warning.assert_called_once()
        self.assertEqual(self.window.text_preview.toPlainText(), "")
        self.assertFalse(self.window.copy_button.isEnabled())
        self.assertTrue(self.window.parse_button.isEnabled())

    def test_background_work_keeps_event_loop_alive_and_prevents_close(self):
        release = threading.Event()
        gui_thread = threading.get_ident()
        threads = []

        def slow_extract(*args, **kwargs):
            threads.append(threading.get_ident())
            release.wait(3)
            return self.text

        with patch("src.main.extract_resume_text", side_effect=slow_extract):
            try:
                self.window.parse_resume(self.resume)
                QTest.qWait(30)
                self.assertFalse(self.window.browse_button.isEnabled())
                event = QCloseEvent()
                self.window.closeEvent(event)
                self.assertFalse(event.isAccepted())
                self.window.clear_selection()
                self.assertEqual(self.window.selected_path, self.resume)
                self.assertEqual(len(threads), 1)
                self.assertNotEqual(threads[0], gui_thread)
            finally:
                release.set()
                self.wait_for_extraction()
        self.assertTrue(self.window.browse_button.isEnabled())

    def test_encrypted_pdf_retries_with_password(self):
        pdf = self.root / "encrypted.pdf"
        write_pdf(pdf, ["Python SQL"], password="secret")
        with patch("src.main.QInputDialog.getText", return_value=("secret", True)) as prompt:
            self.extract(pdf)
        prompt.assert_called_once()
        self.assertEqual(self.window.text_preview.toPlainText(), "Python SQL")

    def test_ocr_is_explicit_and_only_available_for_pdf(self):
        self.window.select_file(self.resume)
        self.assertFalse(self.window.ocr_checkbox.isEnabled())
        pdf = self.root / "scan.pdf"
        write_pdf(pdf, ["<scan>"])
        self.window.select_file(pdf)
        self.assertFalse(self.window.ocr_checkbox.isChecked())
        self.assertTrue(self.window.ocr_checkbox.isEnabled())
        self.window.ocr_checkbox.setChecked(True)
        with patch("src.main.extract_resume_text", return_value="Python SQL") as extractor:
            self.window.parse_button.click()
            self.wait_for_extraction()
        extractor.assert_called_once_with(pdf, ocr=True, password=None)

    def test_open_original_uses_desktop_file_url(self):
        self.window.select_file(self.resume)
        with patch("src.main.QDesktopServices.openUrl", return_value=True) as open_url:
            self.window.open_button.click()
        self.assertEqual(open_url.call_args.args[0].toLocalFile(), str(self.resume).replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
