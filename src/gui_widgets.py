from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class ModernButton(QPushButton):
    def __init__(self, text, parent=None, primary=False):
        super().__init__(text, parent)
        self.primary = primary
        self.setup_style()

    def setup_style(self):
        if self.primary:
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 13px;
                    font-weight: 600;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #0051D5;
                }
                QPushButton:pressed {
                    background-color: #003D99;
                }
                QPushButton:disabled {
                    background-color: #C7C7CC;
                    color: #8E8E93;
                }
            """
            )
        else:
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #F2F2F7;
                    color: #007AFF;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 13px;
                    font-weight: 600;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #E5E5EA;
                }
                QPushButton:pressed {
                    background-color: #D1D1D6;
                }
                QPushButton:disabled {
                    background-color: #F2F2F7;
                    color: #C7C7CC;
                }
            """
            )


class FileListItem(QFrame):
    remove_clicked = pyqtSignal(str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setup_ui()

    def setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 1px solid #E5E5EA;
                border-radius: 8px;
                padding: 6px;
                margin: 2px;
            }
            QFrame:hover {
                background-color: #F9F9F9;
                border-color: #007AFF;
            }
        """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        icon_label = QLabel("🎬")
        icon_label.setFont(QFont("Segoe UI Emoji", 16))
        icon_label.setFixedWidth(24)

        from pathlib import Path

        file_name = Path(self.file_path).name
        name_label = QLabel(file_name)
        name_label.setFont(QFont("Segoe UI", 10))
        name_label.setStyleSheet("color: #1C1C1E; background: transparent;")
        name_label.setWordWrap(False)
        name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        path_label = QLabel(self.file_path)
        path_label.setFont(QFont("Segoe UI", 8))
        path_label.setStyleSheet("color: #8E8E93; background: transparent;")
        path_label.setWordWrap(True)
        path_label.setMaximumHeight(30)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(name_label)
        text_layout.addWidget(path_label)

        remove_btn = QPushButton("X")
        remove_btn.setFixedSize(28, 28)
        remove_btn.setStyleSheet(
            """
                QPushButton {
                    background-color: transparent;
                    color: #8E8E93;
                    border: none;
                    border-radius: 14px;
                    font-size: 14px;
                    font-weight: bold;
                    qproperty-alignment: 'AlignCenter';
                    padding: 0;
                }
                QPushButton:hover {
                    background-color: #FF3B30;
                    color: white;
                }
            """
        )
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.file_path))  # type: ignore

        layout.addWidget(icon_label)
        layout.addLayout(text_layout, 1)
        layout.addWidget(remove_btn)

        self.setLayout(layout)


class SectionCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 1px solid #E5E5EA;
                border-radius: 10px;
            }
        """
        )

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(14, 12, 14, 12)

        title_label = QLabel(self.title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(
            "color: #1C1C1E; border: none; padding: 0; background: transparent;"
        )

        self.main_layout.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(8)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addLayout(self.content_layout)

        self.setLayout(self.main_layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class ModernProgressBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(
            """
            QFrame {
                background-color: #F2F2F7;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #1C1C1E; background: transparent;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #E5E5EA;
                text-align: center;
                font-size: 9px;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background-color: #007AFF;
            }
        """
        )

        self.details_label = QLabel("")
        self.details_label.setFont(QFont("Segoe UI", 8))
        self.details_label.setStyleSheet("color: #8E8E93; background: transparent;")
        self.details_label.setWordWrap(True)
        self.details_label.setMaximumHeight(40)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.details_label)

        self.setLayout(layout)

    def set_status(self, status, progress=None, details=""):
        self.status_label.setText(status)
        if progress is not None:
            self.progress_bar.setValue(progress)
        self.details_label.setText(details)


class LogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()

    def setup_style(self):
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 8))
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setStyleSheet(
            """
            QTextEdit {
                background-color: #1C1C1E;
                color: #F2F2F7;
                border: 1px solid #3A3A3C;
                border-radius: 6px;
                padding: 8px;
            }
        """
        )

    def log(self, message, level="INFO"):
        colors = {
            "INFO": "#007AFF",
            "SUCCESS": "#34C759",
            "WARNING": "#FF9500",
            "ERROR": "#FF3B30",
            "DEBUG": "#8E8E93",
        }
        color = colors.get(level, "#F2F2F7")

        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        html = f'<span style="color: #8E8E93;">[{timestamp}]</span> '
        html += f'<span style="color: {color}; font-weight: bold;">{level}</span>: '
        html += f'<span style="color: #F2F2F7;">{message}</span>'

        self.append(html)

        scrollbar = self.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
