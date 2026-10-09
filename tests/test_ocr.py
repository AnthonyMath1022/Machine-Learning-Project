"""Offline OCR adapter, page-order, cleanup and backend-contract regressions."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.resume.ocr import (
    MODEL_ID, MODEL_REVISION, OCRExecutionError, UnlimitedOCR,
    _recognize_pdf_pages, clean_ocr_output,
)
from src.resume.parser import extract_resume_text
from test_resume_parser import write_pdf


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_only_scanned_pages_are_ocrd_and_text_order_is_preserved(self):
        path = self.root / "mixed.pdf"
        write_pdf(path, ["Python skills.", "<scan>", None, "SQL experience.", "<scan>"])
        with patch.object(UnlimitedOCR, "extract_pages", return_value={2: "Excel skills.", 5: "Power BI skills."}) as backend:
            text = extract_resume_text(path, ocr=True)
        self.assertEqual(text, "Python skills.\n\nExcel skills.\n\nSQL experience.\n\nPower BI skills.")
        backend.assert_called_once_with(path, [2, 5], password=None)

    def test_native_pdf_and_docx_do_not_start_ocr_even_when_enabled(self):
        from docx import Document

        pdf, docx = self.root / "native.pdf", self.root / "native.docx"
        write_pdf(pdf, ["Python"])
        document = Document()
        document.add_paragraph("SQL")
        document.save(docx)
        with patch.object(UnlimitedOCR, "extract_pages") as backend:
            self.assertEqual(extract_resume_text(pdf, ocr=True), "Python")
            self.assertEqual(extract_resume_text(docx, ocr=True), "SQL")
            backend.assert_not_called()

    def test_ocr_failure_never_returns_partial_native_resume_text(self):
        path = self.root / "mixed.pdf"
        write_pdf(path, ["Python", "<scan>"])
        with patch.object(UnlimitedOCR, "extract_pages", side_effect=OCRExecutionError("Out of memory")):
            with self.assertRaisesRegex(OCRExecutionError, "Out of memory"):
                extract_resume_text(path, ocr=True)

    def test_layout_markers_are_removed_and_image_blocks_omitted(self):
        raw = (
            "<PAGE>\n<|det|>title [0, 0, 100, 20]<|/det|>Resume\n"
            "<|det|>text [0, 20, 100, 50]<|/det|>Python SQL\nPower BI\n"
            "<|det|>image [0, 50, 100, 90]<|/det|>photo\n![photo](image.jpg)\n"
            "<|det|>text [0, 90, 100, 120]<|/det|>Education\n<｜end▁of▁sentence｜>"
        )
        self.assertEqual(clean_ocr_output(raw), "Resume\n\nPython SQL\nPower BI\n\nEducation")
        for invalid in ("", "<|det|>text [", "<|det|>image [0,0,1,1]<|/det|>photo", None):
            with self.subTest(raw=invalid), self.assertRaises(OCRExecutionError):
                clean_ocr_output(invalid)

    def test_worker_protocol_keeps_password_off_argv_and_cleans_temp_files(self):
        requests = []

        def fake_run(command, **kwargs):
            request = Path(command[command.index("--request") + 1])
            response = Path(command[command.index("--response") + 1])
            payload = json.loads(request.read_text(encoding="utf-8"))
            self.assertEqual(payload["password"], "secret-password")
            self.assertNotIn("secret-password", command)
            self.assertTrue(kwargs["capture_output"])
            requests.append(request)
            response.write_text(json.dumps({"pages": {"2": "Python", "4": "SQL"}}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with patch("src.resume.ocr.subprocess.run", side_effect=fake_run):
            result = UnlimitedOCR(sys.executable).extract_pages(self.root / "resume.pdf", [2, 4], password="secret-password")
        self.assertEqual(result, {2: "Python", 4: "SQL"})
        self.assertFalse(requests[0].parent.exists())

    def test_missing_empty_duplicate_and_malformed_results_fail(self):
        for payload in ({"pages": {}}, {"pages": {"1": ""}}, {"pages": {"1": "Python", "2": "extra"}}, [], {"error": "GPU unavailable"}):
            def fake_run(command, **kwargs):
                Path(command[-1]).write_text(json.dumps(payload), encoding="utf-8")
                return SimpleNamespace(returncode=0)
            with self.subTest(payload=payload), patch("src.resume.ocr.subprocess.run", side_effect=fake_run):
                with self.assertRaises(OCRExecutionError):
                    UnlimitedOCR(sys.executable).extract_pages(self.root / "resume.pdf", [1])
        with self.assertRaises(ValueError):
            UnlimitedOCR(sys.executable).extract_pages(self.root / "resume.pdf", [1, 1])

    def test_missing_runtime_timeout_and_crashed_worker_fail_clearly(self):
        with self.assertRaisesRegex(OCRExecutionError, "environment is missing"):
            UnlimitedOCR(self.root / "missing-python").extract_pages(self.root / "resume.pdf", [1])
        for outcome, message in [(subprocess.TimeoutExpired("OCR", 1), "timed out"), (SimpleNamespace(returncode=1), "without a result")]:
            run = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
            with self.subTest(message=message), patch("src.resume.ocr.subprocess.run", run):
                with self.assertRaisesRegex(OCRExecutionError, message):
                    UnlimitedOCR(sys.executable).extract_pages(self.root / "resume.pdf", [1])

    def test_worker_pins_model_revision_and_uses_returned_text_without_saved_results(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True), bfloat16="bf16")
        document = Mock()
        document.__enter__ = Mock(return_value=document)
        document.__exit__ = Mock(return_value=False)
        document.__len__ = Mock(return_value=3)
        page = Mock()
        document.__getitem__ = Mock(return_value=page)
        document.needs_pass = False
        fitz = SimpleNamespace(open=Mock(return_value=document))
        tokenizer_type, model_type, model = Mock(), Mock(), Mock()
        tokenizer_type.from_pretrained.return_value.eos_token_id = 17
        original_generate = model.generate
        generated = MagicMock()
        generated.__getitem__.return_value.item.return_value = 17
        original_generate.return_value = generated
        model_type.from_pretrained.return_value.eval.return_value.cuda.return_value = model
        model.infer_multi.return_value = ("<PAGE>\nPython SQL", 2)
        transformers = SimpleNamespace(AutoModel=model_type, AutoTokenizer=tokenizer_type)
        with patch("src.resume.ocr.version", return_value="4.57.1"), patch.dict(sys.modules, {"torch": torch, "pymupdf": fitz, "transformers": transformers}):
            result = _recognize_pdf_pages({"pdf": "resume.pdf", "pages": [2], "password": None}, self.root)
        self.assertEqual(result, {"2": "Python SQL"})
        load = model_type.from_pretrained.call_args
        self.assertEqual(load.args[0], MODEL_ID)
        self.assertEqual(load.kwargs["revision"], MODEL_REVISION)
        self.assertTrue(load.kwargs["trust_remote_code"])
        self.assertTrue(load.kwargs["use_safetensors"])
        page.get_pixmap.assert_called_once_with(dpi=300, alpha=False)
        self.assertFalse(model.infer_multi.call_args.kwargs["save_results"])
        self.assertEqual(model.infer_multi.call_args.kwargs["image_size"], 1024)
        self.assertIs(model.generate(max_length=8192), generated)
        generated.__getitem__.return_value.item.return_value = 16
        with self.assertRaisesRegex(OCRExecutionError, "truncated"):
            model.generate(max_length=8192)

    def test_wrong_transformers_runtime_is_rejected_before_loading_weights(self):
        with patch("src.resume.ocr.version", return_value="5.7.0"):
            with self.assertRaisesRegex(OCRExecutionError, "4.57.1"):
                _recognize_pdf_pages({}, self.root)


if __name__ == "__main__":
    unittest.main()
