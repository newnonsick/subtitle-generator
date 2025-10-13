import logging
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt as QtCore
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .models import Subtitle

logger = logging.getLogger(__name__)


class SubtitleEditorDialog(QDialog):
    def __init__(
        self,
        audio_file: str,
        subtitles: List[Subtitle],
        output_format: str = "srt",
        output_dir: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_file = audio_file
        self.subtitles = subtitles
        self.output_format = output_format
        self.output_dir = output_dir
        self.modified = False
        self.current_row = -1

        # Initialize media player
        self.media_player = QMediaPlayer()
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_file)))
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.stateChanged.connect(self.on_state_changed)

        self.setup_ui()
        self.load_subtitles()

    def setup_ui(self):
        self.setWindowTitle(f"Edit Subtitles - {Path(self.audio_file).name}")
        self.setMinimumSize(1000, 600)
        self.resize(1200, 700)

        self.setStyleSheet(
            """
            QDialog {
                background-color: #F5F5F7;
            }
            QLabel {
                color: #1C1C1E;
            }
            QPushButton {
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                background-color: #007AFF;
                color: white;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0051D5;
            }
            QPushButton:pressed {
                background-color: #003D99;
            }
            QPushButton:disabled {
                background-color: #D1D1D6;
                color: #8E8E93;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                gridline-color: #E5E5EA;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #007AFF;
                color: white;
            }
            QHeaderView::section {
                background-color: #F2F2F7;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #D1D1D6;
                font-weight: 600;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #D1D1D6;
                border-radius: 6px;
                padding: 8px;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        header = self.create_header()
        layout.addWidget(header)

        self.subtitle_table = QTableWidget()
        self.subtitle_table.setColumnCount(5)
        self.subtitle_table.setHorizontalHeaderLabels(
            ["#", "Start (s)", "End (s)", "Duration", "Text"]
        )
        self.subtitle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subtitle_table.setSelectionMode(QAbstractItemView.SingleSelection)

        self.subtitle_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            or QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.subtitle_table.itemChanged.connect(self.on_item_changed)
        self.subtitle_table.itemSelectionChanged.connect(self.on_selection_changed)

        header_view = self.subtitle_table.horizontalHeader()
        if header_view:
            header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header_view.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header_view.setSectionResizeMode(4, QHeaderView.Stretch)

        layout.addWidget(self.subtitle_table, 1)

        playback_panel = self.create_playback_panel()
        layout.addWidget(playback_panel)

        button_layout = self.create_action_buttons()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet(
            """
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)

        title = QLabel("Subtitle Editor")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))

        file_label = QLabel(f"Audio File: {Path(self.audio_file).name}")
        file_label.setFont(QFont("Segoe UI", 10))
        file_label.setStyleSheet("color: #8E8E93;")

        count_label = QLabel(f"Total Subtitles: {len(self.subtitles)}")
        count_label.setFont(QFont("Segoe UI", 10))
        count_label.setStyleSheet("color: #8E8E93;")

        layout.addWidget(title)
        layout.addWidget(file_label)
        layout.addWidget(count_label)

        header.setLayout(layout)
        return header

    def create_playback_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedHeight(100)
        panel.setStyleSheet(
            """
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        title = QLabel("Audio Playback Controls")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(title)

        self.time_label = QLabel(
            "Double-click any Text cell to edit. Select a row and click Play to hear audio."
        )
        self.time_label.setStyleSheet("color: #8E8E93;")
        self.time_label.setWordWrap(True)
        layout.addWidget(self.time_label)

        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play Selected Segment")
        self.play_btn.setFixedHeight(32)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_selected_segment)
        self.play_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #34C759;
                padding: 8px 16px;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2BA94A;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #228B3B;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #D1D1D6;
                color: #FFFFFF;
            }
        """
        )

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFixedHeight(32)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FF3B30;
                padding: 8px 16px;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #CC2F26;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #99221C;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #D1D1D6;
                color: #FFFFFF;
            }
        """
        )

        self.playback_position_label = QLabel("")
        self.playback_position_label.setStyleSheet("color: #8E8E93")
        self.playback_position_label.setMinimumWidth(150)

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.playback_position_label)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        panel.setLayout(layout)
        return panel

    def create_action_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Save Changes")
        self.save_btn.clicked.connect(self.save_changes)
        self.save_btn.setEnabled(False)

        self.cancel_btn = QPushButton("✖ Close")
        self.cancel_btn.clicked.connect(self.close_dialog)

        layout.addStretch()
        layout.addWidget(self.save_btn)
        layout.addWidget(self.cancel_btn)

        return layout

    def load_subtitles(self):
        self.subtitle_table.blockSignals(True)
        self.subtitle_table.setRowCount(len(self.subtitles))

        for row, subtitle in enumerate(self.subtitles):
            index_item = QTableWidgetItem(str(subtitle.index))
            index_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            index_item.setFlags(
                Qt.ItemFlag(index_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            )
            self.subtitle_table.setItem(row, 0, index_item)

            start_item = QTableWidgetItem(f"{subtitle.start:.2f}")
            start_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            start_item.setFlags(
                Qt.ItemFlag(start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            )
            self.subtitle_table.setItem(row, 1, start_item)

            end_item = QTableWidgetItem(f"{subtitle.end:.2f}")
            end_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            end_item.setFlags(
                Qt.ItemFlag(end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            )
            self.subtitle_table.setItem(row, 2, end_item)

            duration = subtitle.end - subtitle.start
            duration_item = QTableWidgetItem(f"{duration:.2f}s")
            duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            duration_item.setFlags(
                Qt.ItemFlag(duration_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            )
            self.subtitle_table.setItem(row, 3, duration_item)

            text_item = QTableWidgetItem(subtitle.text)
            self.subtitle_table.setItem(row, 4, text_item)

        self.subtitle_table.blockSignals(False)

    def on_item_changed(self, item: QTableWidgetItem):
        if item.column() == 4:
            self.modified = True
            self.save_btn.setEnabled(True)

    def on_selection_changed(self):
        selected_rows = self.subtitle_table.selectedItems()
        if not selected_rows:
            self.current_row = -1
            self.play_btn.setEnabled(False)
            return

        self.current_row = self.subtitle_table.currentRow()
        if 0 <= self.current_row < len(self.subtitles):
            subtitle = self.subtitles[self.current_row]

            duration = subtitle.end - subtitle.start
            self.time_label.setText(
                f"Selected: Subtitle #{subtitle.index} | "
                f"Time: {subtitle.start:.2f}s - {subtitle.end:.2f}s "
                f"(Duration: {duration:.2f}s)"
            )

            self.play_btn.setEnabled(True)

    def play_selected_segment(self):
        if self.current_row < 0:
            return

        subtitle = self.subtitles[self.current_row]

        start_ms = int(subtitle.start * 1000)
        end_ms = int(subtitle.end * 1000)

        logger.info(f"Playing segment: {subtitle.start:.2f}s - {subtitle.end:.2f}s")

        self.media_player.setPosition(start_ms)
        self.media_player.play()

        QTimer.singleShot(
            int((subtitle.end - subtitle.start) * 1000), self.stop_playback
        )

        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_playback(self):
        if self.media_player.state() != QMediaPlayer.State.StoppedState:
            self.media_player.stop()

        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.playback_position_label.setText("")

    def on_position_changed(self, position):
        if self.media_player.state() == QMediaPlayer.State.PlayingState:
            seconds = position / 1000.0
            self.playback_position_label.setText(f"Playing: {seconds:.2f}s")

    def on_state_changed(self, state):
        if state == QMediaPlayer.State.StoppedState:
            self.stop_btn.setEnabled(False)
            if self.current_row >= 0:
                self.play_btn.setEnabled(True)
            self.playback_position_label.setText("")

    def save_changes(self):
        if not self.modified:
            return

        for row in range(self.subtitle_table.rowCount()):
            text_item = self.subtitle_table.item(row, 4)
            if text_item and row < len(self.subtitles):
                self.subtitles[row].text = text_item.text()

        self.modified = False
        self.save_btn.setEnabled(False)

        export_success = self.export_subtitles()

        if export_success:
            QMessageBox.information(
                self,
                "Saved & Exported",
                f"✅ All subtitle changes have been saved and exported!\n\n"
                f"• Subtitles updated: {len(self.subtitles)}\n"
                f"• Format: {self.output_format}\n"
                f"• Location: {self.get_output_path()}\n\n"
                "You can continue editing or close this dialog.",
            )
        else:
            QMessageBox.warning(
                self,
                "Saved (Export Failed)",
                f"⚠️ Subtitles were saved but export failed.\n\n"
                f"Changes are preserved in memory but not written to file.\n"
                "You can try saving again or check the logs for errors.",
            )

    def get_output_path(self) -> str:
        if self.output_dir:
            output_base = Path(self.output_dir) / Path(self.audio_file).stem
        else:
            output_base = Path(self.audio_file).parent / Path(self.audio_file).stem

        if self.output_format == "all":
            return str(output_base) + " (all formats)"
        else:
            return str(output_base) + f".{self.output_format}"

    def export_subtitles(self) -> bool:
        try:
            from .config import Config
            from .subtitle_generator import SubtitleGenerator

            config = Config()
            generator = SubtitleGenerator(config)

            if self.output_dir:
                output_base = Path(self.output_dir) / Path(self.audio_file).stem
            else:
                output_base = Path(self.audio_file).parent / Path(self.audio_file).stem

            if self.output_format == "all":
                output_path = str(output_base)
                success = generator.export(
                    self.subtitles, output_path, None, format="all"
                )
            else:
                output_path = str(output_base) + f".{self.output_format}"
                success = generator.export(
                    self.subtitles, output_path, None, format=self.output_format
                )

            if success:
                logger.info(f"Successfully exported edited subtitles to: {output_path}")
            else:
                logger.error(f"Failed to export edited subtitles to: {output_path}")

            return success

        except Exception as e:
            logger.error(f"Error exporting subtitles: {e}")
            return False

    def close_dialog(self):
        if self.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )

            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                self.save_changes()

        self.stop_playback()

        self.accept()

    def get_subtitles(self) -> List[Subtitle]:
        return self.subtitles

    def closeEvent(self, event):
        self.stop_playback()
        event.accept()
