"""JobStreet crawling and local AI comparison in a PySide6 workspace."""

from __future__ import annotations

from collections.abc import Callable
import gc
from pathlib import Path
import sys

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)


from src.models.config import GEMMA_MODEL_ID, OLLAMA_MODEL_ID

DEFAULT_MODEL_ID = OLLAMA_MODEL_ID


class OperationWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            result = self.operation()
        except Exception as error:
            self.failed.emit(str(error) or type(error).__name__)
        else:
            self.succeeded.emit(result)


class AgentSession:
    """Own one lazy agent across serialized worker operations, including retries."""

    def __init__(self):
        self.agent = None
        self.model_id = None
        self.backend = None

    def assess(self, model_id: str, job_text: str, resume_text: str, *, backend: str = "ollama"):
        if backend not in {"ollama", "transformers"}:
            raise ValueError("Select a supported AI backend.")
        if self.agent is None or (self.model_id, self.backend) != (model_id, backend):
            self.release()
            if backend == "ollama":
                from src.models.ollama_agent import OllamaJobMatchAgent

                self.agent = OllamaJobMatchAgent(model_id)
            else:
                from src.models.model import JobMatchAgent

                self.agent = JobMatchAgent(model_id)
            self.model_id = model_id
            self.backend = backend
        return self.agent.assess_resume(job_text, resume_text)

    def release(self):
        """Release cached inference memory before a separate OCR process uses CUDA."""
        if self.agent is None:
            return
        if hasattr(self.agent, "unload"):
            self.agent.unload()
        self.agent = None
        self.model_id = None
        self.backend = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def info_label(text: str, name: str = "hint") -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


def format_assessment(report) -> str:
    """Present validated requirements with both job and resume evidence."""
    summary = report.job_summary
    lines = [report.fit.replace("_", " ").title(), report.reasoning, "",
             "JOB SUMMARY", summary.title or "Title not stated", summary.summary]
    if summary.responsibilities:
        lines += ["", "Responsibilities", *[f"• {item}" for item in summary.responsibilities]]
    if summary.skills:
        lines += ["", "Skills: " + ", ".join(summary.skills)]
    lines += ["", "REQUIREMENT EVIDENCE"]
    for match in sorted(report.requirements, key=lambda item: item.requirement_index):
        requirement = summary.requirements[match.requirement_index]
        lines += ["", f"{match.requirement_index + 1}. {requirement.requirement}",
                  f"Importance: {requirement.importance}",
                  f"Assessment: {match.status.replace('_', ' ')}",
                  f"Job evidence: {requirement.job_evidence}",
                  f"Resume evidence: {match.resume_evidence or 'Not mentioned in the resume'}",
                  match.explanation]
    if report.follow_up_questions:
        lines += ["", "FOLLOW-UP QUESTIONS", *[f"• {item}" for item in report.follow_up_questions]]
    return "\n".join(lines)


class JobWorkspace(QWidget):
    busy_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(self, resume_text: Callable[[], str], resume_path: Callable, parent=None):
        super().__init__(parent)
        self.resume_text = resume_text
        self.resume_path = resume_path
        self.worker: OperationWorker | None = None
        self.external_busy = False
        self.listings = []
        self.report_json = ""
        self.session = AgentSession()
        self._pending_error = None
        self._result_handler = None
        self._build_ui()
        self.refresh_controls()

    @property
    def busy(self) -> bool:
        return self.worker is not None or self.external_busy

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        search_card = QFrame()
        search_card.setObjectName("card")
        search_card.setMinimumWidth(280)
        search_layout = QVBoxLayout(search_card)
        search_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        search_layout.setContentsMargins(18, 18, 18, 18)
        search_layout.setSpacing(10)
        search_layout.addWidget(info_label("Find JobStreet jobs", "sectionTitle"))
        form = QFormLayout()
        self.title_input = QLineEdit("Data Analyst")
        self.title_input.setAccessibleName("Job search title")
        self.location_input = QLineEdit("Kuala Lumpur")
        self.location_input.setAccessibleName("Job search location")
        self.page_input = QSpinBox()
        self.page_input.setRange(1, 1000)
        form.addRow("Role", self.title_input)
        form.addRow("Location", self.location_input)
        form.addRow("Page", self.page_input)
        search_layout.addLayout(form)
        self.search_button = QPushButton("Search jobs")
        self.search_button.setObjectName("primary")
        self.search_button.clicked.connect(self.search_jobs)
        search_layout.addWidget(self.search_button)
        self.results_hint = info_label("Search one page, then select an ad to fetch its full description.")
        search_layout.addWidget(self.results_hint)
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(100)
        self.results_list.setAccessibleName("JobStreet search results")
        self.results_list.currentRowChanged.connect(self._selection_changed)
        search_layout.addWidget(self.results_list, 1)
        self.fetch_button = QPushButton("Load selected job")
        self.fetch_button.clicked.connect(self.fetch_selected)
        search_layout.addWidget(self.fetch_button)
        search_layout.addWidget(info_label("Or load a JobStreet Malaysia ad URL:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://my.jobstreet.com/job/…")
        self.url_input.setAccessibleName("JobStreet ad URL")
        search_layout.addWidget(self.url_input)
        self.fetch_url_button = QPushButton("Load URL")
        self.fetch_url_button.clicked.connect(self.fetch_url)
        search_layout.addWidget(self.fetch_url_button)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setWidget(search_card)
        splitter.addWidget(scroll)

        detail_card = QFrame()
        detail_card.setObjectName("card")
        detail_card.setMinimumWidth(360)
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        self.detail_tabs = QTabWidget()
        detail_layout.addWidget(self.detail_tabs, 1)
        job_page = QWidget()
        job_layout = QVBoxLayout(job_page)
        self.job_source = info_label("Paste a description or load a full job ad.")
        job_layout.addWidget(self.job_source)
        self.job_text = QPlainTextEdit()
        self.job_text.setMinimumHeight(180)
        self.job_text.setPlaceholderText("Paste the complete job description here if online retrieval is unavailable.")
        self.job_text.setAccessibleName("Job description for matching")
        job_layout.addWidget(self.job_text, 1)
        self.detail_tabs.addTab(job_page, "Job description")
        report_page = QWidget()
        report_layout = QVBoxLayout(report_page)
        self.fit_label = info_label("No assessment yet", "sectionTitle")
        report_layout.addWidget(self.fit_label)
        self.report_preview = QPlainTextEdit()
        self.report_preview.setMinimumHeight(180)
        self.report_preview.setReadOnly(True)
        self.report_preview.setPlaceholderText("AI results will show requirement-by-requirement evidence here.")
        self.report_preview.setAccessibleName("AI match assessment")
        report_layout.addWidget(self.report_preview, 1)
        export_layout = QHBoxLayout()
        self.copy_report_button = QPushButton("Copy report")
        self.copy_report_button.clicked.connect(self.copy_report)
        self.save_report_button = QPushButton("Save JSON…")
        self.save_report_button.clicked.connect(self.save_report)
        export_layout.addStretch()
        export_layout.addWidget(self.copy_report_button)
        export_layout.addWidget(self.save_report_button)
        report_layout.addLayout(export_layout)
        self.detail_tabs.addTab(report_page, "Match report")

        self.resume_hint = info_label("Extract a resume in the Resume tab to enable matching.")
        detail_layout.addWidget(self.resume_hint)
        self.model_input = QLineEdit(DEFAULT_MODEL_ID)
        self.model_input.setAccessibleName("AI model name or ID")
        self.backend_input = QComboBox()
        self.backend_input.addItem("Ollama (GGUF)", "ollama")
        self.backend_input.addItem("Transformers (Gemma)", "transformers")
        self.backend_input.setAccessibleName("AI inference backend")
        model_form = QFormLayout()
        model_form.addRow("Backend", self.backend_input)
        model_form.addRow("AI model", self.model_input)
        detail_layout.addLayout(model_form)
        self.model_hint = info_label(
            "Qwen3-8B Q4_K_M runs locally through Ollama with an 8K context. "
            "Start Ollama before matching. Review the advisory result and its evidence."
        )
        detail_layout.addWidget(self.model_hint)
        match_layout = QHBoxLayout()
        self.similarity_button = QPushButton("Text similarity")
        self.similarity_button.clicked.connect(self.compare_text)
        self.match_button = QPushButton("Run AI match")
        self.match_button.setObjectName("primary")
        self.match_button.clicked.connect(self.match_resume)
        match_layout.addWidget(self.similarity_button)
        match_layout.addWidget(self.match_button)
        detail_layout.addLayout(match_layout)
        self.similarity_label = info_label("Text similarity: —")
        detail_layout.addWidget(self.similarity_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        detail_layout.addWidget(self.progress)
        detail_scroll = QScrollArea()
        detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setMinimumWidth(380)
        detail_scroll.setWidget(detail_card)
        splitter.addWidget(detail_scroll)
        splitter.setSizes([320, 650])

        self.title_input.textChanged.connect(self.refresh_controls)
        self.url_input.textChanged.connect(self.refresh_controls)
        self.model_input.textChanged.connect(self.invalidate_results)
        self.backend_input.currentIndexChanged.connect(self._backend_changed)
        self.job_text.textChanged.connect(self._job_edited)

    def refresh_controls(self):
        busy = self.busy
        has_resume = bool(self.resume_text().strip())
        has_job = bool(self.job_text.toPlainText().strip())
        self.search_button.setEnabled(not busy and bool(self.title_input.text().strip()))
        self.fetch_button.setEnabled(not busy and self.results_list.currentRow() >= 0)
        self.fetch_url_button.setEnabled(not busy and bool(self.url_input.text().strip()))
        self.similarity_button.setEnabled(not busy and has_resume and has_job)
        self.match_button.setEnabled(not busy and has_resume and has_job and bool(self.model_input.text().strip()))
        for control in (self.title_input, self.location_input, self.page_input,
                        self.url_input, self.model_input, self.backend_input, self.results_list):
            control.setEnabled(not busy)
        self.job_text.setReadOnly(busy)
        for control in (self.copy_report_button, self.save_report_button):
            control.setEnabled(not busy and bool(self.report_json))
        self.resume_hint.setText(
            "Resume ready for comparison." if has_resume else "Extract a resume in the Resume tab to enable matching."
        )
        self.progress.setVisible(self.worker is not None)

    def set_external_busy(self, busy: bool):
        self.external_busy = busy
        self.refresh_controls()

    @Slot(int)
    def _backend_changed(self, index: int):
        ollama = self.backend_input.currentData() == "ollama"
        self.model_input.setText(OLLAMA_MODEL_ID if ollama else GEMMA_MODEL_ID)
        self.model_hint.setText(
            "Qwen3-8B Q4_K_M runs locally through Ollama with an 8K context. Start Ollama before matching. "
            "Review the advisory result and its evidence." if ollama else
            "Gemma requires requirements-model.txt and substantial memory. Its first use may download weights."
        )
        self.invalidate_results()

    @Slot()
    def invalidate_results(self):
        self.report_json = ""
        self.report_preview.clear()
        self.fit_label.setText("No assessment yet")
        self.similarity_label.setText("Text similarity: —")
        self.refresh_controls()

    @Slot()
    def _job_edited(self):
        self.job_source.setText("Current job description (editable)")
        self.invalidate_results()

    def _start(self, operation: Callable, handler: Callable, status: str, kind: str):
        if self.busy:
            return
        self._pending_error = None
        self._result_handler = handler
        self._operation_kind = kind
        self.worker = OperationWorker(operation, self)
        self.worker.succeeded.connect(self._show_result)
        self.worker.failed.connect(self._record_error)
        self.worker.finished.connect(self._finished)
        self.busy_changed.emit(True)
        self.refresh_controls()
        self.status_changed.emit(status)
        self.worker.start()

    @Slot(object)
    def _show_result(self, result):
        self._result_handler(result)

    @Slot(str)
    def _record_error(self, message: str):
        self._pending_error = message

    @Slot()
    def _finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self._result_handler = None
        self.busy_changed.emit(False)
        self.refresh_controls()
        error, self._pending_error = self._pending_error, None
        if error:
            if self._operation_kind == "search":
                self.results_hint.setText("Search failed. Try again or paste a job description.")
            elif self._operation_kind == "fetch":
                self.job_source.setText("Retrieval failed. Paste a job description to continue.")
            elif self._operation_kind == "match":
                self.fit_label.setText("AI assessment failed")
            elif self._operation_kind == "similarity":
                self.similarity_label.setText("Text similarity could not be calculated.")
            self.status_changed.emit("Operation failed: " + error)
            message = error
            if "403" in error or "blocked" in error.lower() or "JavaScript" in error:
                message += "\n\nYou can paste the job description to continue matching."
            QMessageBox.warning(self, "Could not complete operation", message)

    @Slot()
    def search_jobs(self):
        title, location, page = self.title_input.text().strip(), self.location_input.text().strip(), self.page_input.value()
        if self.busy or not title:
            return
        self.listings = []
        self.results_list.clear()
        self.results_hint.setText("Searching…")

        def search():
            from src.crawler.jobstreet import JobDescriptionSearch

            with JobDescriptionSearch(timeout=20) as crawler:
                return crawler.search_job(title, location, page=page)

        self._start(search, self._show_listings, "Searching JobStreet…", "search")

    def _show_listings(self, jobs):
        self.listings = jobs
        for job in jobs:
            item = QListWidgetItem(f"{job.title}\n{job.company or 'Company not stated'} · {job.location or 'Location not stated'}")
            item.setToolTip(job.url)
            self.results_list.addItem(item)
        noun = "job" if len(jobs) == 1 else "jobs"
        self.results_hint.setText(f"{len(jobs)} {noun} on this page." if jobs else "No jobs found. Try another role or location.")
        self.status_changed.emit(self.results_hint.text())
        if jobs:
            self.results_list.setCurrentRow(0)

    @Slot(int)
    def _selection_changed(self, row: int):
        if 0 <= row < len(self.listings):
            self.url_input.setText(self.listings[row].url)
            self.job_text.clear()
            self.job_source.setText(f"Selected: {self.listings[row].title}. Load the full ad to continue.")
        self.refresh_controls()

    @Slot()
    def fetch_selected(self):
        row = self.results_list.currentRow()
        if 0 <= row < len(self.listings):
            self._fetch(self.listings[row].url)

    @Slot()
    def fetch_url(self):
        self._fetch(self.url_input.text().strip())

    def _fetch(self, url: str):
        if self.busy or not url:
            return
        # Remove the previous ad so a failed fetch cannot leave it as the new job.
        self.job_text.clear()
        self.job_source.setText("Loading job description…")
        self.detail_tabs.setCurrentIndex(0)

        def fetch():
            from src.crawler.jobstreet import JobDescriptionSearch

            with JobDescriptionSearch(timeout=20) as crawler:
                return crawler.fetch_job(url)

        self._start(fetch, self._show_job, "Loading full job description…", "fetch")

    def _show_job(self, job):
        self.job_text.setPlainText(job.description or "")
        self.job_source.setText(f"{job.title} · {job.company or 'Company not stated'}\n{job.url}")
        self.status_changed.emit("Full job description loaded. Review it before matching.")

    @Slot()
    def compare_text(self):
        if self.busy:
            return
        resume, job = self.resume_text().strip(), self.job_text.toPlainText().strip()
        if not resume or not job:
            return
        self.similarity_label.setText("Calculating text similarity…")

        def compare():
            from src.nlp.matcher import calculate_similarity

            return calculate_similarity(resume, job)

        self._start(compare, self._show_similarity, "Calculating TF-IDF text similarity…", "similarity")

    def _show_similarity(self, score):
        self.similarity_label.setText(f"TF-IDF text similarity: {score:.1%} (text overlap, not a hiring probability)")
        self.status_changed.emit("Text similarity calculated.")

    @Slot()
    def match_resume(self):
        if self.busy:
            return
        resume, job, model_id = self.resume_text().strip(), self.job_text.toPlainText().strip(), self.model_input.text().strip()
        backend = self.backend_input.currentData()
        if not resume or not job or not model_id:
            return
        self.invalidate_results()
        self.fit_label.setText("Running AI assessment…")
        self.detail_tabs.setCurrentIndex(1)
        self._start(
            lambda: self.session.assess(model_id, job, resume, backend=backend), self._show_assessment,
            "Running local AI matching… The model may take a moment to load.",
            "match",
        )

    def _show_assessment(self, report):
        self.report_json = report.model_dump_json(indent=2)
        self.report_preview.setPlainText(format_assessment(report))
        self.fit_label.setText(report.fit.replace("_", " ").title())
        self.status_changed.emit("AI assessment complete. Review the evidence and follow-up questions.")

    @Slot()
    def copy_report(self):
        if self.report_json:
            QApplication.clipboard().setText(self.report_preview.toPlainText())
            self.status_changed.emit("Assessment copied to the clipboard.")

    @Slot()
    def save_report(self):
        if self.busy or not self.report_json:
            return
        resume_path = self.resume_path()
        destination = resume_path.with_name(resume_path.stem + "-match.json") if resume_path else Path("resume-match.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save AI assessment", str(destination), "JSON (*.json)")
        if not path:
            return
        try:
            target = Path(path).resolve()
            if resume_path and (target == resume_path or (target.exists() and target.samefile(resume_path))):
                QMessageBox.warning(self, "Choose another file", "Save to a different path to preserve your original resume.")
                return
            target.write_text(self.report_json, encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save assessment", str(error))
        else:
            self.status_changed.emit(f"Assessment saved to {target}")
