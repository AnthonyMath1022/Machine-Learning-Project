"""Extract local PDF, DOCX and UTF-8 resumes with optional Baidu PDF OCR."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import BadZipFile


class ResumeExtractionError(ValueError):
    """The resume could not be read completely as a supported document."""


class NoResumeTextError(ResumeExtractionError):
    """The document has no extractable text."""


class OCRRequiredError(ResumeExtractionError):
    """A PDF page contains images without a text layer."""


class EncryptedPDFError(ResumeExtractionError):
    """The PDF requires a correct password."""


def _clean(text: str) -> str:
    # Preserve line breaks and Unicode; do not rewrite candidate statements.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _has_pdf_images(node, reader, seen=None) -> bool:
    """Inspect image references/inline operators without decoding image pixels."""
    from pypdf.generic import ContentStream

    if seen is None:
        seen = set()
    if id(node) in seen:
        return False
    seen.add(id(node))
    resources = node.get("/Resources", {})
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    objects = resources.get("/XObject", {})
    if hasattr(objects, "get_object"):
        objects = objects.get_object()
    for reference in objects.values():
        obj = reference.get_object()
        if obj.get("/Subtype") == "/Image":
            return True
        if obj.get("/Subtype") == "/Form" and _has_pdf_images(obj, reader, seen):
            return True
    stream = node.get_contents() if hasattr(node, "get_contents") else ContentStream(node, reader)
    return stream is not None and any(op == b"INLINE IMAGE" for _, op in stream.operations)


def _extract_pdf(path: Path, password: str | None, ocr: bool, ocr_python: str | Path | None) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import DependencyError, PdfReadError, PdfStreamError
    except ImportError as error:
        raise ResumeExtractionError("PDF extraction requires pypdf: install -r requirements-resume.txt.") from error

    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted and not reader.decrypt(password if password is not None else ""):
                raise EncryptedPDFError("The PDF is password protected; provide its correct password.")
            parts, scanned_pages = [], []
            for number, page in enumerate(reader.pages, 1):
                text = _clean(page.extract_text() or "")
                if text:
                    parts.append((number, text))
                elif _has_pdf_images(page, reader):
                    scanned_pages.append(number)
            if scanned_pages:
                if ocr:
                    from src.resume.ocr import UnlimitedOCR

                    recognized = UnlimitedOCR(ocr_python).extract_pages(path, scanned_pages, password=password)
                    parts.extend((number, _clean(recognized[number])) for number in scanned_pages)
                    return "\n\n".join(text for _, text in sorted(parts))
                pages = ", ".join(str(number) for number in scanned_pages)
                raise OCRRequiredError(
                    f"PDF page(s) {pages} contain images but no extractable text. "
                    "Enable Baidu Unlimited-OCR with ocr=True / --ocr, or export a searchable PDF."
                )
            return "\n\n".join(text for _, text in parts)
    except DependencyError as error:
        raise ResumeExtractionError(
            "This encrypted PDF needs the optional crypto dependency: install pypdf[crypto]."
        ) from error
    except (PdfReadError, PdfStreamError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ResumeExtractionError):
            raise
        raise ResumeExtractionError("Could not read the PDF; it may be damaged or not a valid PDF.") from error


def _docx_blocks(container):
    """Walk paragraphs and tables in document order, including nested tables."""
    from docx.table import Table

    for block in container.iter_inner_content():
        if not isinstance(block, Table):
            text = _clean(block.text)
            if text:
                yield text
            continue
        seen_cells = set()
        for row in block.rows:
            cells = []
            for cell in row.cells:
                # Merged cells may be returned more than once by python-docx.
                if cell._tc in seen_cells:
                    continue
                seen_cells.add(cell._tc)
                text = "\n".join(_docx_blocks(cell))
                if text:
                    cells.append(text)
            if cells:
                yield "\t".join(cells)


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
        from lxml.etree import XMLSyntaxError
    except ImportError as error:
        raise ResumeExtractionError("DOCX extraction requires python-docx: install -r requirements-resume.txt.") from error

    try:
        # Passing our stream avoids misreporting an unreadable file as a bad package.
        with path.open("rb") as stream:
            document = Document(stream)
        headers, footers, seen_parts = [], [], set()
        for section in document.sections:
            variants = [(section.header, section.footer)]
            if section.different_first_page_header_footer:
                variants.append((section.first_page_header, section.first_page_footer))
            if document.settings.odd_and_even_pages_header_footer:
                variants.append((section.even_page_header, section.even_page_footer))
            for header, footer in variants:
                for container, target in ((header, headers), (footer, footers)):
                    if container.is_linked_to_previous:
                        continue
                    part_name = str(container.part.partname)
                    if part_name not in seen_parts:
                        seen_parts.add(part_name)
                        target.extend(_docx_blocks(container))
        return "\n".join([*headers, *_docx_blocks(document), *footers])
    except (BadZipFile, PackageNotFoundError, XMLSyntaxError, KeyError, TypeError, ValueError) as error:
        raise ResumeExtractionError("Could not read the DOCX; it may be damaged or not a Word document.") from error


def extract_resume_text(
    path: str | Path, *, password: str | None = None,
    ocr: bool = False, ocr_python: str | Path | None = None,
) -> str:
    """Read a PDF, DOCX or UTF-8 TXT file and return nonempty document text.

    With ocr=True, image-only PDF pages use Baidu Unlimited-OCR in an isolated
    subprocess. Without OCR they raise OCRRequiredError, including mixed PDFs.
    DOCX text includes body paragraphs, tables and active header/footer variants;
    floating text boxes and text embedded in images are not supported.
    Original files are never modified. Missing/unreadable files raise OSError.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise ResumeExtractionError(
            "Unsupported resume format. Use .pdf, .docx or UTF-8 .txt; convert legacy .doc files to .docx."
        )
    if not path.exists():
        raise FileNotFoundError(f"Resume file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Resume path is not a file: {path}")
    if suffix == ".pdf":
        text = _extract_pdf(path, password, ocr, ocr_python)
    elif suffix == ".docx":
        text = _extract_docx(path)
    else:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise ResumeExtractionError("The text file is not UTF-8; save it with UTF-8 encoding.") from error
    text = _clean(text)
    if not text:
        raise NoResumeTextError(
            "The document contains no extractable text. Image-based resumes need OCR or a text export."
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract resume text from PDF, DOCX or UTF-8 TXT.")
    parser.add_argument("resume", type=Path)
    parser.add_argument("--ocr", action="store_true", help="Use Baidu Unlimited-OCR for image-only PDF pages")
    parser.add_argument("--ocr-python", type=Path, help="Python executable in the separate OCR environment")
    args = parser.parse_args()
    try:
        text = extract_resume_text(args.resume, ocr=args.ocr, ocr_python=args.ocr_python)
    except (OSError, ResumeExtractionError, RuntimeError) as error:
        parser.exit(1, f"Extraction failed: {error}\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
