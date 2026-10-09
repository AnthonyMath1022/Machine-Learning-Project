"""Baidu Unlimited-OCR adapter and isolated CUDA inference worker.

The worker uses Baidu's Transformers 4.57.1 environment, separate from Gemma.
Importing this module never imports Torch or downloads model weights.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


MODEL_ID = "baidu/Unlimited-OCR"
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OCRExecutionError(RuntimeError):
    """OCR setup or inference failed; partial resume text must not be used."""


def clean_ocr_output(raw: str) -> str:
    """Strip layout annotations, retaining text/Markdown in reading order."""
    if not isinstance(raw, str):
        raise OCRExecutionError("Unlimited-OCR returned an invalid text result.")
    raw = raw.replace("<PAGE>", "\n\n").replace("<｜end▁of▁sentence｜>", "")
    blocks, current, skip_image = [], [], False
    pattern = re.compile(r"^<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?<\|/det\|>(.*)$")
    for line in raw.splitlines():
        match = pattern.match(line.strip())
        if match:
            if current:
                blocks.append("\n".join(current))
            current = []
            skip_image = match.group(1).casefold() == "image"
            line = match.group(2).strip()
        if line.strip() and not skip_image:
            current.append(line.rstrip())
    if current:
        blocks.append("\n".join(current))
    text = "\n\n".join(blocks).strip()
    if "<|det|>" in text or "<|/det|>" in text:
        raise OCRExecutionError("Unlimited-OCR returned incomplete layout annotations; retry OCR.")
    if not text:
        raise OCRExecutionError("Unlimited-OCR found no readable text on a requested page.")
    return text


class UnlimitedOCR:
    """Run one short-lived OCR process, releasing GPU memory before matching."""

    def __init__(self, python: str | Path | None = None, *, timeout: float = 1800):
        executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        self.python = Path(python or os.environ.get("UNLIMITED_OCR_PYTHON")
                           or PROJECT_ROOT / ".venv-ocr" / executable).resolve()
        self.timeout = timeout

    def extract_pages(self, path: Path, pages: list[int], *, password: str | None = None) -> dict[int, str]:
        """Recognize selected one-based PDF pages; keep credentials off argv."""
        if not self.python.is_file():
            raise OCRExecutionError(
                "Unlimited-OCR environment is missing. Create .venv-ocr and install "
                "requirements-ocr.txt; see README.md. Or set UNLIMITED_OCR_PYTHON."
            )
        if not pages or any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages):
            raise ValueError("OCR pages must be a nonempty list of positive page numbers.")
        if len(set(pages)) != len(pages):
            raise ValueError("OCR page numbers must be unique.")
        with tempfile.TemporaryDirectory(prefix="resume_ocr_") as directory:
            request = Path(directory) / "request.json"
            output = Path(directory) / "response.json"
            request.write_text(json.dumps({
                "pdf": str(Path(path).resolve()), "pages": pages, "password": password,
            }), encoding="utf-8")
            try:
                process = subprocess.run(
                    [str(self.python), "-B", str(Path(__file__).resolve()),
                     "--request", str(request), "--response", str(output)],
                    capture_output=True, timeout=self.timeout,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except subprocess.TimeoutExpired as error:
                raise OCRExecutionError("Unlimited-OCR timed out; no partial resume text was returned.") from error
            except OSError as error:
                raise OCRExecutionError("Could not start the Unlimited-OCR Python environment.") from error
            if not output.is_file():
                raise OCRExecutionError(
                    "Unlimited-OCR worker stopped without a result. Check its dependencies and available GPU memory."
                )
            try:
                result = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("Invalid OCR response")
                if process.returncode or "error" in result:
                    raise OCRExecutionError(str(result.get("error", "Unlimited-OCR worker failed.")))
                values = result["pages"]
                if not isinstance(values, dict) or set(values) != {str(page) for page in pages}:
                    raise ValueError("Missing or unexpected OCR pages")
                if any(not isinstance(text, str) or not text.strip() for text in values.values()):
                    raise ValueError("Empty OCR page")
                return {page: values[str(page)].strip() for page in pages}
            except (KeyError, ValueError) as error:
                raise OCRExecutionError("Unlimited-OCR returned an incomplete or invalid page result.") from error


def _recognize_pdf_pages(request: dict, directory: Path) -> dict[str, str]:
    """Worker-only inference, following the pinned Baidu model's infer_multi API."""
    try:
        if version("transformers") != "4.57.1":
            raise OCRExecutionError("Unlimited-OCR requires Transformers 4.57.1 in .venv-ocr; keep Gemma separate.")
        import torch
        import pymupdf
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise OCRExecutionError("Install requirements-ocr.txt in .venv-ocr before enabling OCR.") from error
    if not torch.cuda.is_available():
        raise OCRExecutionError("Unlimited-OCR requires CUDA PyTorch and an NVIDIA GPU; the current runtime has no CUDA.")
    if not torch.cuda.is_bf16_supported():
        raise OCRExecutionError("This Unlimited-OCR backend requires a GPU with bfloat16 support.")

    image_paths = []
    with pymupdf.open(request["pdf"]) as document:
        if document.needs_pass and not document.authenticate(request.get("password") or ""):
            raise OCRExecutionError("The OCR renderer needs the correct PDF password.")
        for number in request["pages"]:
            if number < 1 or number > len(document):
                raise OCRExecutionError("An OCR page number is outside the PDF page range.")
            image = directory / f"page_{number:04d}.png"
            document[number - 1].get_pixmap(dpi=300, alpha=False).save(str(image))
            image_paths.append(image)

    cache = PROJECT_ROOT / "data" / "model-cache" / "unlimited-ocr"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, cache_dir=str(cache),
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, cache_dir=str(cache),
    ).eval().cuda()
    generate = model.generate

    def generate_complete(**kwargs):
        output = generate(**kwargs)
        # infer_multi strips the end token before returning text; inspect it here
        # so hitting max_length cannot silently produce an incomplete resume.
        if output[0, -1].item() != tokenizer.eos_token_id:
            raise OCRExecutionError("OCR generation was truncated before the end token; no partial text was returned.")
        return output

    model.generate = generate_complete
    recognized = {}
    # One page per call limits VRAM pressure on the user's 8 GB laptop GPU.
    for number, image in zip(request["pages"], image_paths):
        raw, _tokens = model.infer_multi(
            tokenizer, prompt="<image>Multi page parsing.", image_files=[str(image)],
            output_path=str(directory / f"output_{number}"), image_size=1024,
            max_length=8192, no_repeat_ngram_size=35, ngram_window=1024,
            save_results=False,
        )
        recognized[str(number)] = clean_ocr_output(raw)
    return recognized


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Unlimited-OCR inference worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        pages = _recognize_pdf_pages(request, args.request.parent)
        result, code = {"pages": pages}, 0
    except Exception as error:
        # Return errors through a small JSON envelope; never print resume content.
        result, code = {"error": f"{type(error).__name__}: {error}"}, 1
    args.response.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
