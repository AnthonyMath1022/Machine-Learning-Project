"""PySide6 desktop workspace for resumes, job crawling and AI matching.

Launch from the project root with ``python -m src.main``.
"""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QLayout, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

# Support both `python -m src.main` and `python src/main.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nlp.skill_extractor import extract_skills
from src.resume.parser import EncryptedPDFError, extract_resume_text
from src.ui.job_workspace import JobWorkspace


class ResumeWorker(QThread):
    """Keep document reading and optional OCR off the GUI thread."""

    extracted = Signal(str)
    failed = Signal(str, bool)

    def __init__(self, path: Path, *, ocr: bool, password: str | None, before_extract=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.ocr = ocr
        self.password = password
        self.before_extract = before_extract

    def run(self):
        try:
            if self.before_extract is not None:
                self.before_extract()
            text = extract_resume_text(self.path, ocr=self.ocr, password=self.password)
        except Exception as error:
            # Report backend/dependency errors to the desktop user as well.
            self.failed.emit(str(error), isinstance(error, EncryptedPDFError))
        else:
            self.extracted.emit(text)
        finally:
            self.password = None


STYLE = """
QMainWindow, QWidget#workspace { background: #f3f5f9; color: #18243a; }
QWidget { font-family: 'Segoe UI'; font-size: 10pt; color: #18243a; }
QLabel#title { font-size: 24pt; font-weight: 700; color: #172541; }
QLabel#subtitle, QLabel#hint { color: #53637b; }
QLabel#sectionTitle { font-size: 13pt; font-weight: 600; color: #172541; }
QLabel#fileName { font-weight: 600; }
QFrame#card { background: white; border: 1px solid #dbe2ed; border-radius: 12px; }
QPushButton { background: white; color: #243550; border: 1px solid #c7d2e2; min-height: 18px;
              border-radius: 6px; padding: 9px 14px; }
QPushButton:hover { background: #eef3fc; border-color: #7895ca; }
QPushButton:pressed { background: #dde8fa; }
QPushButton#primary { background: #285bd5; color: white; border-color: #285bd5; }
QPushButton#primary:hover { background: #204cb5; }
QPushButton:disabled, QPushButton#primary:disabled {
    background: #eef1f6; color: #929cab; border-color: #e0e5ed;
}
QPlainTextEdit { background: white; color: #18243a; border: 1px solid #dbe2ed;
                 border-radius: 6px; padding: 10px; selection-background-color: #285bd5; }
QProgressBar { border: 0; border-radius: 3px; background: #e8edf6; height: 5px; }
QProgressBar::chunk { background: #285bd5; border-radius: 3px; }
QSplitter::handle { background: transparent; }
QStatusBar { background: #e8edf5; color: #354762; }
QTabWidget::pane { border: 0; }
QTabBar::tab { padding: 10px 18px; background: #e8edf5; border-radius: 4px; }
QTabBar::tab:selected { background: #285bd5; color: white; }
QLineEdit, QSpinBox, QListWidget {
    background: white; color: #18243a; border: 1px solid #dbe2ed; border-radius: 5px; padding: 5px;
}
QListWidget::item { padding: 8px 4px; }
QListWidget::item:selected { background: #e1ebff; color: #18243a; }
"""


def label(text: str, name: str | None = None) -> QLabel:
    widget = QLabel(text)
    widget.setWordWrap(True)
    # File names and extracted skills must always be displayed literally.
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if name:
        widget.setObjectName(name)
    return widget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_path: Path | None = None
        self.worker: ResumeWorker | None = None
        self._pending_error: tuple[str, bool] | None = None
        self.setWindowTitle("Resume & Job Workspace")
        self.resize(1140, 860)
        self.setMinimumSize(780, 560)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._build_menu()
        self._refresh_controls()
        self.statusBar().showMessage("Ready — choose a resume to get started.")

    def _build_ui(self):
        workspace = QWidget()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(28, 24, 28, 18)
        layout.setSpacing(16)
        layout.addWidget(label("Resume & job workspace", "title"))
        layout.addWidget(label("Review your resume, find jobs and compare requirement evidence.", "subtitle"))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        resume_page = QWidget()
        layout = QVBoxLayout(resume_page)
        layout.setContentsMargins(0, 12, 0, 0)
        self.tabs.addTab(resume_page, "Resume")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        file_card = QFrame()
        file_card.setObjectName("card")
        file_card.setMinimumWidth(270)
        file_layout = QVBoxLayout(file_card)
        file_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        file_layout.setContentsMargins(20, 20, 20, 20)
        file_layout.setSpacing(14)
        file_layout.addWidget(label("1. Choose a resume", "sectionTitle"))
        file_layout.addWidget(label("PDF, Word (.docx) or UTF-8 text (.txt)", "hint"))
        self.file_name = label("No file selected", "fileName")
        self.file_path = label("Browse your computer to select a document.", "hint")
        self.file_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_layout.addWidget(self.file_name)
        file_layout.addWidget(self.file_path)
        self.browse_button = QPushButton("Browse files…")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        self.ocr_checkbox = QCheckBox("Use OCR for scanned PDF pages")
        self.ocr_checkbox.setToolTip("Requires the separate Baidu OCR environment described in README.md.")
        file_layout.addWidget(self.ocr_checkbox)
        file_layout.addWidget(label("OCR requires the optional setup. Its first run may download model weights.", "hint"))
        self.parse_button = QPushButton("Extract resume")
        self.parse_button.setObjectName("primary")
        self.parse_button.clicked.connect(lambda: self.parse_resume())
        file_layout.addWidget(self.parse_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        file_layout.addWidget(self.progress)
        self.open_button = QPushButton("Open original document")
        self.open_button.clicked.connect(self.open_original)
        file_layout.addWidget(self.open_button)
        self.clear_button = QPushButton("Clear selection")
        self.clear_button.clicked.connect(self.clear_selection)
        file_layout.addWidget(self.clear_button)
        file_layout.addStretch()
        file_layout.addWidget(label("Your original document is never modified.", "hint"))
        file_scroll = QScrollArea()
        file_scroll.setFrameShape(QFrame.Shape.NoFrame)
        file_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        file_scroll.setWidgetResizable(True)
        file_scroll.setMinimumWidth(290)
        file_scroll.setWidget(file_card)
        splitter.addWidget(file_scroll)

        text_card = QFrame()
        text_card.setObjectName("card")
        text_layout = QVBoxLayout(text_card)
        text_layout.setContentsMargins(20, 20, 20, 20)
        text_layout.setSpacing(12)
        text_layout.addWidget(label("2. Review extracted text", "sectionTitle"))
        text_layout.addWidget(label("Check reading order and completeness before using the text for matching.", "hint"))
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setPlaceholderText("Your extracted resume will appear here.")
        self.text_preview.setAccessibleName("Extracted resume text")
        text_layout.addWidget(self.text_preview, 1)
        self.text_stats = label("No text extracted yet", "hint")
        self.skills_label = label("Detected skills: —", "hint")
        self.skills_label.setToolTip("Keyword matches from src/nlp/skill_extractor.py; not a complete skills assessment.")
        text_layout.addWidget(self.text_stats)
        text_layout.addWidget(self.skills_label)
        actions = QHBoxLayout()
        self.copy_button = QPushButton("Copy text")
        self.copy_button.clicked.connect(self.copy_text)
        self.save_button = QPushButton("Save text…")
        self.save_button.clicked.connect(self.save_text)
        actions.addStretch()
        actions.addWidget(self.copy_button)
        actions.addWidget(self.save_button)
        text_layout.addLayout(actions)
        splitter.addWidget(text_card)
        splitter.setSizes([300, 680])
        self.jobs_panel = JobWorkspace(
            self.text_preview.toPlainText, lambda: self.selected_path, self,
        )
        self.jobs_panel.busy_changed.connect(lambda _: self._refresh_controls())
        self.jobs_panel.status_changed.connect(self.statusBar().showMessage)
        self.text_preview.textChanged.connect(self.jobs_panel.invalidate_results)
        self.tabs.addTab(self.jobs_panel, "Jobs & matching")
        self.setCentralWidget(workspace)

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")
        self.browse_action = QAction("&Open resume…", self)
        self.browse_action.setShortcut(QKeySequence.StandardKey.Open)
        self.browse_action.triggered.connect(self.browse_file)
        menu.addAction(self.browse_action)
        self.save_action = QAction("&Save extracted text…", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_text)
        menu.addAction(self.save_action)
        menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

    def _refresh_controls(self):
        busy = self.busy
        selected = self.selected_path is not None
        has_text = bool(self.text_preview.toPlainText())
        self.browse_button.setEnabled(not busy)
        self.browse_action.setEnabled(not busy)
        for button in (self.parse_button, self.open_button, self.clear_button):
            button.setEnabled(selected and not busy)
        self.ocr_checkbox.setEnabled(not busy and selected and self.selected_path.suffix.lower() == ".pdf")
        for control in (self.copy_button, self.save_button, self.save_action):
            control.setEnabled(has_text and not busy)
        self.parse_button.setText("Extracting…" if self.worker is not None else "Extract resume")
        self.progress.setVisible(self.worker is not None)
        self.jobs_panel.set_external_busy(self.worker is not None)

    @property
    def busy(self) -> bool:
        return self.worker is not None or self.jobs_panel.worker is not None

    def _clear_preview(self):
        self.text_preview.clear()
        self.text_stats.setText("No text extracted yet")
        self.skills_label.setText("Detected skills: —")

    @Slot()
    def browse_file(self):
        if self.busy:
            return
        directory = str(self.selected_path.parent) if self.selected_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a resume", directory,
            "Resume documents (*.pdf *.docx *.txt);;PDF (*.pdf);;Word (*.docx);;Text (*.txt)",
        )
        if path:
            self.select_file(path)

    def select_file(self, path: str | Path):
        """Select a document without starting expensive extraction or OCR."""
        if self.busy:
            return
        candidate = Path(path).resolve()
        if candidate.suffix.lower() not in {".pdf", ".docx", ".txt"} or not candidate.is_file():
            QMessageBox.warning(self, "Unsupported resume", "Choose an existing PDF, DOCX or TXT file.")
            return
        self.selected_path = candidate
        self.file_name.setText(candidate.name)
        self.file_path.setText(str(candidate))
        self.ocr_checkbox.setChecked(False)
        self._clear_preview()
        self._refresh_controls()
        self.statusBar().showMessage("Resume selected. Click Extract resume to read it.")

    def parse_resume(self, file_path: str | Path | None = None, *, password: str | None = None):
        if self.busy:
            return
        if file_path is not None:
            self.select_file(file_path)
            if self.selected_path != Path(file_path).resolve():
                return
        if self.selected_path is None:
            return
        self._clear_preview()
        self._pending_error = None
        self.worker = ResumeWorker(
            self.selected_path, ocr=self.ocr_checkbox.isChecked(), password=password, parent=self,
            before_extract=self.jobs_panel.session.release if self.ocr_checkbox.isChecked() else None,
        )
        self.worker.extracted.connect(self._show_text)
        self.worker.failed.connect(self._record_error)
        self.worker.finished.connect(self._extraction_finished)
        self._refresh_controls()
        self.statusBar().showMessage("Extracting resume… You can continue using the window while it runs.")
        self.worker.start()

    @Slot(str)
    def _show_text(self, text: str):
        self.text_preview.setPlainText(text)
        self.text_stats.setText(f"{len(text.split()):,} words · {len(text):,} characters")
        skills = extract_skills(text)
        self.skills_label.setText("Detected skills: " + (", ".join(skills) if skills else "No catalog matches"))
        self.statusBar().showMessage("Extraction complete. Review the text, then copy or save it.")

    @Slot(str, bool)
    def _record_error(self, message: str, password_required: bool):
        self._pending_error = (message, password_required)

    @Slot()
    def _extraction_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._refresh_controls()
        error, self._pending_error = self._pending_error, None
        if error is None:
            return
        message, password_required = error
        self.statusBar().showMessage("Extraction failed. " + message)
        if password_required:
            password, accepted = QInputDialog.getText(
                self, "Encrypted PDF", "Enter the PDF password:", QLineEdit.EchoMode.Password,
            )
            if accepted:
                self.parse_resume(password=password)
        else:
            QMessageBox.warning(self, "Could not extract resume", message)

    @Slot()
    def clear_selection(self):
        if self.busy:
            return
        self.selected_path = None
        self.file_name.setText("No file selected")
        self.file_path.setText("Browse your computer to select a document.")
        self.ocr_checkbox.setChecked(False)
        self._clear_preview()
        self._refresh_controls()
        self.statusBar().showMessage("Ready — choose a resume to get started.")

    @Slot()
    def open_original(self):
        if self.selected_path is not None:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.selected_path))):
                QMessageBox.warning(self, "Could not open file", "Your desktop could not open this document.")

    @Slot()
    def copy_text(self):
        text = self.text_preview.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Extracted text copied to the clipboard.")

    @Slot()
    def save_text(self):
        text = self.text_preview.toPlainText()
        if not text or self.selected_path is None:
            return
        destination = self.selected_path.with_name(self.selected_path.stem + "-extracted.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save extracted text", str(destination), "Text (*.txt)")
        if not path:
            return
        try:
            target = Path(path).resolve()
            if target == self.selected_path or (target.exists() and target.samefile(self.selected_path)):
                QMessageBox.warning(self, "Choose another file", "Save to a different path to preserve your original resume.")
                return
            target.write_text(text, encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save text", str(error))
        else:
            self.statusBar().showMessage(f"Saved extracted text to {target}")

    def closeEvent(self, event):
        # Never destroy a live QThread or interrupt the OCR subprocess mid-write.
        if self.busy:
            self.statusBar().showMessage("An operation is still running. Wait for it to finish before closing.")
            event.ignore()
        else:
            event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Resume & Job Workspace")
    app.setOrganizationName("Job Market Project")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
