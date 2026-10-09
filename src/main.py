import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn  
import requests
import sys
from bs4 import BeautifulSoup

from PySide6.QtCore import Qt
import PySide6
from PySide6.QtWidgets import (
	QApplication,
	QWidget,
	QMainWindow,
	QFileDialog,
	QLabel,
	QPushButton,
	QVBoxLayout,
	QMessageBox,
)

print(PySide6.__version__)
print(PySide6.__version__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resume Parser")
        self.setGeometry(100, 100, 400, 200)

        self.label = QLabel("Select a resume file to parse", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Browse", self)
        self.button.clicked.connect(self.browse_file)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Resume File", "", "PDF Files (*.pdf);;Word Files (*.docx);;All Files (*)")
        
        if file_path:
            self.label.setText(f"Selected File: {file_path}")

        msg = QMessageBox.information(self, "File Selected", f"You selected: {file_path},are you sure you want to parse this file?", QMessageBox.Yes | QMessageBox.No)
        result = msg.exec()
        if result == QMessageBox.Yes:
            self.parse_resume(file_path)
        else:
            self.clear_selection()
def main():
    """Launch the existing resume-selection interface."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
