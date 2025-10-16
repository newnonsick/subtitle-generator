import logging
import sys
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .config import (
    AVAILABLE_COMPUTE_TYPES,
    AVAILABLE_DEVICES,
    AVAILABLE_MODELS,
    AVAILABLE_OUTPUT_FORMATS,
    DEFAULT_MIN_SILENCE,
    DEFAULT_SILENCE_THRESH,
    LANGUAGE_SUPPORTS,
    MAX_CHUNK_DURATION,
    SUPPORTED_FORMATS,
    TARGET_CHUNK_DURATION,
    Config,
)
from .gui_widgets import (
    FileListItem,
    LogConsole,
    ModernButton,
    ModernProgressBar,
    SectionCard,
)
from .subtitle_generator import SubtitleGenerator

logger = logging.getLogger(__name__)


class SubtitleWorker(QThread):
    progress = pyqtSignal(str, int, str)
    log_message = pyqtSignal(str, str)
    finished = pyqtSignal(bool, str)
    subtitles_generated = pyqtSignal(
        dict
    )

    def __init__(
        self,
        file_paths: List[str],
        config: Config,
        output_format: str,
        output_dir: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.config = config
        self.output_format = output_format
        self.output_dir = output_dir
        self.languages = languages
        self._is_running = True
        self.current_file_idx = 0
        self.total_files = 0
        self.current_chunk = 0
        self.total_chunks = 0
        self.current_language_idx = 0
        self.total_languages = len(languages) if languages else 1
        self.processed_subtitles = {}

    def calculate_progress(self) -> int:
        if self.total_files == 0:
            return 0

        total_work_units = self.total_files * self.total_languages
        completed_files = self.current_file_idx * self.total_languages
        completed_languages = self.current_language_idx

        base_progress = (completed_files + completed_languages) / total_work_units

        if self.total_chunks > 0:
            chunk_progress = self.current_chunk / self.total_chunks
            base_progress += chunk_progress / total_work_units

        return int(base_progress * 100)

    def run(self):
        try:
            self.log_message.emit("Initializing subtitle generator...", "INFO")
            self.progress.emit("Initializing...", 0, "Loading Whisper model...")

            generator = SubtitleGenerator(self.config)

            self.total_files = len(self.file_paths)

            is_multilingual = self.languages and len(self.languages) > 0

            for idx, file_path in enumerate(self.file_paths):
                if not self._is_running:
                    self.log_message.emit("Processing cancelled by user", "WARNING")
                    self.finished.emit(False, "Cancelled by user")
                    return

                self.current_file_idx = idx
                file_name = Path(file_path).name

                start_progress = int((idx / self.total_files) * 100)
                self.progress.emit(
                    f"Processing {idx + 1}/{self.total_files}",
                    start_progress,
                    f"Loading: {file_name}",
                )
                self.log_message.emit(f"Processing: {file_name}", "INFO")

                try:
                    if is_multilingual:
                        if not self.languages:
                            self.log_message.emit(
                                "No languages specified for multilingual generation",
                                "ERROR",
                            )
                            if self.total_files == 1:
                                self.finished.emit(False, "No languages specified")
                                return
                            continue
                        self.log_message.emit(
                            f"Generating subtitles in {len(self.languages)} language(s): {', '.join(self.languages)}",
                            "INFO",
                        )

                        results = self._generate_multilingual_with_progress(
                            generator, file_path, file_name
                        )

                        if not results:
                            self.log_message.emit(
                                f"✗ No subtitles generated for {file_name}", "WARNING"
                            )
                            if self.total_files == 1:
                                self.finished.emit(False, "No subtitles generated")
                                return
                            continue

                        if self.output_dir:
                            output_base = Path(self.output_dir) / Path(file_path).stem
                        else:
                            output_base = Path(file_path).parent / Path(file_path).stem

                        export_progress = int(((idx + 0.95) / self.total_files) * 100)
                        self.progress.emit(
                            f"Processing {idx + 1}/{self.total_files}",
                            export_progress,
                            f"Exporting all languages: {file_name}",
                        )

                        all_exported = True
                        for language, (subtitles, stats) in results.items():
                            lang_output_base = f"{output_base}.{language}"

                            if self.output_format == "all":
                                success = generator.export(
                                    subtitles, lang_output_base, stats, format="all"
                                )
                            else:
                                lang_output_path = (
                                    f"{lang_output_base}.{self.output_format}"
                                )
                                success = generator.export(
                                    subtitles,
                                    lang_output_path,
                                    stats,
                                    format=self.output_format,
                                )

                            if success:
                                self.log_message.emit(
                                    f"  ✓ Exported {language}: {lang_output_base}",
                                    "SUCCESS",
                                )
                            else:
                                self.log_message.emit(
                                    f"  ✗ Failed to export {language}", "ERROR"
                                )
                                all_exported = False

                        if all_exported:
                            complete_progress = int(
                                ((idx + 1) / self.total_files) * 100
                            )
                            self.progress.emit(
                                f"Processing {idx + 1}/{self.total_files}",
                                complete_progress,
                                f"Completed: {file_name}",
                            )
                            self.log_message.emit(
                                f"✓ Completed all languages: {file_name}", "SUCCESS"
                            )

                            self.processed_subtitles[file_path] = results
                    else:
                        self.log_message.emit(f"Analyzing audio...", "DEBUG")

                        subtitles, stats = self._generate_with_progress(
                            generator, file_path, file_name
                        )

                        if not subtitles:
                            self.log_message.emit(
                                f"✗ No subtitles generated for {file_name}", "WARNING"
                            )
                            if self.total_files == 1:
                                self.finished.emit(False, "No subtitles generated")
                                return
                            continue

                        export_progress = int(((idx + 0.95) / self.total_files) * 100)
                        self.progress.emit(
                            f"Processing {idx + 1}/{self.total_files}",
                            export_progress,
                            f"Exporting: {file_name}",
                        )

                        if self.output_dir:
                            output_base = Path(self.output_dir) / Path(file_path).stem
                        else:
                            output_base = Path(file_path).parent / Path(file_path).stem

                        if self.output_format == "all":
                            output_path = str(output_base)
                            success = generator.export(
                                subtitles, output_path, stats, format="all"
                            )
                        else:
                            output_path = str(output_base) + f".{self.output_format}"
                            success = generator.export(
                                subtitles, output_path, stats, format=self.output_format
                            )

                        if success:
                            complete_progress = int(
                                ((idx + 1) / self.total_files) * 100
                            )
                            self.progress.emit(
                                f"Processing {idx + 1}/{self.total_files}",
                                complete_progress,
                                f"Completed: {file_name}",
                            )

                            self.log_message.emit(
                                f"✓ Completed: {file_name}", "SUCCESS"
                            )
                            self.log_message.emit(f"  Output: {output_path}", "INFO")

                            if stats:
                                self.log_message.emit(
                                    f"  Stats: {stats.subtitles_generated} subtitles, "
                                    f"{stats.processing_time:.1f}s processing time, "
                                    f"{stats.speed_ratio():.1f}x speed",
                                    "DEBUG",
                                )

                            self.processed_subtitles[file_path] = subtitles
                        else:
                            self.log_message.emit(
                                f"✗ Failed to export {file_name}", "ERROR"
                            )

                except Exception as e:
                    self.log_message.emit(
                        f"✗ Error processing {file_name}: {str(e)}", "ERROR"
                    )
                    if self.total_files == 1:
                        self.finished.emit(False, f"Error: {str(e)}")
                        return

            self.progress.emit(
                "Complete!", 100, f"Successfully processed {self.total_files} file(s)"
            )

            if self.processed_subtitles:
                self.subtitles_generated.emit(self.processed_subtitles)

            lang_count = len(self.languages) if self.languages else 0
            lang_msg = (
                f" in {lang_count} language(s)"
                if is_multilingual and lang_count > 0
                else ""
            )
            self.finished.emit(
                True,
                f"Successfully generated subtitles{lang_msg} for {self.total_files} file(s)!",
            )

        except Exception as e:
            error_msg = f"Fatal error: {str(e)}"
            self.log_message.emit(error_msg, "ERROR")
            self.finished.emit(False, error_msg)

    def _generate_with_progress(self, generator, file_path, file_name):
        import time

        from .models import ProcessingStats

        start_time = time.time()

        audio_result = generator.audio_processor.load_audio(file_path)
        if audio_result is None:
            return [], ProcessingStats(0, 0, 0, 0)

        audio, sr = audio_result
        audio_duration = len(audio) / sr

        if generator.config.use_noise_reduction:
            self.progress.emit(
                f"Processing {self.current_file_idx + 1}/{self.total_files}",
                self.calculate_progress(),
                f"Reducing noise: {file_name}",
            )
            audio = generator.audio_processor.apply_noise_reduction(audio, sr)

        chunks = generator.audio_processor.detect_speech_chunks(
            audio,
            int(sr),
            generator.config.min_silence_len,
            generator.config.silence_thresh,
            generator.config.keep_silence,
        )

        if not chunks:
            return [], ProcessingStats(audio_duration, time.time() - start_time, 0, 0)

        if generator.config.merge_chunks:
            chunks = generator.audio_processor.merge_nearby_chunks(
                chunks, int(sr), generator.config.max_gap_ms
            )

        chunks = generator.audio_processor.split_long_chunks(
            chunks,
            audio,
            sr,
            max_duration=generator.config.max_chunk_duration,
            target_duration=generator.config.target_chunk_duration,
        )

        self.total_chunks = len(chunks)
        self.current_chunk = 0
        self.log_message.emit(
            f"Transcribing {self.total_chunks} audio chunks...", "INFO"
        )

        all_subtitles = []

        for chunk_idx, (start_sample, end_sample) in enumerate(chunks):
            if not self._is_running:
                return [], ProcessingStats(
                    audio_duration, time.time() - start_time, 0, 0
                )

            self.current_chunk = chunk_idx + 1
            progress = self.calculate_progress()

            self.progress.emit(
                f"Processing {self.current_file_idx + 1}/{self.total_files}",
                progress,
                f"Transcribing chunk {self.current_chunk}/{self.total_chunks}: {file_name}",
            )

            chunk_start_sec = start_sample / sr
            audio_chunk = audio[start_sample:end_sample]

            subtitles = generator.transcribe_chunk(
                audio_chunk, int(sr), chunk_start_sec, generator.config.language
            )
            all_subtitles.extend(subtitles)

        if all_subtitles and generator.config.fix_overlaps:
            self.progress.emit(
                f"Processing {self.current_file_idx + 1}/{self.total_files}",
                int(((self.current_file_idx + 0.9) / self.total_files) * 100),
                f"Fixing overlaps: {file_name}",
            )
            all_subtitles = generator._fix_overlapping_subtitles(
                all_subtitles, min_gap=generator.config.min_subtitle_gap
            )
            self.log_message.emit("Fixed overlapping timestamps", "DEBUG")

        for idx, sub in enumerate(all_subtitles, 1):
            sub.index = idx

        processing_time = time.time() - start_time
        stats = ProcessingStats(
            total_duration=audio_duration,
            processing_time=processing_time,
            chunks_processed=len(chunks),
            subtitles_generated=len(all_subtitles),
        )

        return all_subtitles, stats

    def _generate_multilingual_with_progress(self, generator, file_path, file_name):
        import time

        from .models import ProcessingStats

        if not self.languages:
            return {}

        start_time = time.time()

        audio, sr, chunks, audio_duration = generator._process_audio_to_chunks(
            file_path
        )

        if audio is None or sr is None or not chunks:
            self.log_message.emit("Failed to process audio", "ERROR")
            return {}

        if len(chunks) == 0:
            self.log_message.emit("No speech detected", "WARNING")
            return {}

        self.total_chunks = len(chunks)
        results = {}

        for lang_idx, language in enumerate(self.languages):
            if not self._is_running:
                return results

            self.current_language_idx = lang_idx
            self.current_chunk = 0

            self.log_message.emit(
                f"  Language {lang_idx + 1}/{len(self.languages)}: {language}", "INFO"
            )
            self.log_message.emit(
                f"  Transcribing {self.total_chunks} chunks for {language}...", "INFO"
            )

            all_subtitles = []
            lang_start_time = time.time()

            for chunk_idx, (start_sample, end_sample) in enumerate(chunks):
                if not self._is_running:
                    return results

                self.current_chunk = chunk_idx + 1
                progress = self.calculate_progress()

                self.progress.emit(
                    f"Processing {self.current_file_idx + 1}/{self.total_files}",
                    progress,
                    f"[{language}] Chunk {self.current_chunk}/{self.total_chunks}: {file_name}",
                )

                chunk_start_sec = start_sample / sr
                audio_chunk = audio[start_sample:end_sample]

                subtitles = generator.transcribe_chunk(
                    audio_chunk, int(sr), chunk_start_sec, language
                )
                all_subtitles.extend(subtitles)

            if all_subtitles and generator.config.fix_overlaps:
                all_subtitles = generator._fix_overlapping_subtitles(
                    all_subtitles, min_gap=generator.config.min_subtitle_gap
                )
                self.log_message.emit(
                    f"  Fixed overlapping timestamps for {language}", "DEBUG"
                )

            for idx, sub in enumerate(all_subtitles, 1):
                sub.index = idx

            lang_processing_time = time.time() - lang_start_time
            stats = ProcessingStats(
                total_duration=audio_duration,
                processing_time=lang_processing_time,
                chunks_processed=len(chunks),
                subtitles_generated=len(all_subtitles),
            )

            results[language] = (all_subtitles, stats)

            self.log_message.emit(
                f"  ✓ {language}: {len(all_subtitles)} subtitles in {lang_processing_time:.1f}s",
                "SUCCESS",
            )

        total_time = time.time() - start_time
        self.log_message.emit(
            f"Multilingual generation complete in {total_time:.1f}s", "INFO"
        )

        return results

    def stop(self):
        self._is_running = False


class DragDropFileWidget(QWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        event_mime = event.mimeData()
        if event_mime is None:
            event.ignore()
            return

        if event_mime.hasUrls():
            urls = event_mime.urls()
            has_files = any(url.isLocalFile() for url in urls)
            if has_files:
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        event_mime = event.mimeData()
        if event_mime is None:
            event.ignore()
            return

        if event_mime.hasUrls():
            file_paths = []
            for url in event_mime.urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    file_paths.append(file_path)

            if file_paths:
                self.files_dropped.emit(file_paths)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()


class SubtitleGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_paths = []
        self.worker = None
        self.processed_subtitles = {}
        self.setup_ui()
        self.setup_logging()

    def setup_ui(self):
        self.setWindowTitle("Advanced Subtitle Generator")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        app_font = QFont("Segoe UI", 9)
        self.setFont(app_font)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #F5F5F7;
            }
            QLabel {
                color: #1C1C1E;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px 8px;
                border: 1px solid #D1D1D6;
                border-radius: 6px;
                background-color: white;
                min-height: 24px;
                selection-background-color: #007AFF;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #007AFF;
                background-color: #FAFAFA;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #D1D1D6;
                border-bottom: 1px solid #D1D1D6;
                border-top-right-radius: 6px;
                background-color: #F2F2F7;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #E5E5EA;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background-color: #D1D1D6;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #8E8E93;
                width: 0;
                height: 0;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border-left: 1px solid #D1D1D6;
                border-bottom-right-radius: 6px;
                background-color: #F2F2F7;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #E5E5EA;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #D1D1D6;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #8E8E93;
                width: 0;
                height: 0;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #8E8E93;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #D1D1D6;
                background-color: white;
                selection-background-color: #007AFF;
                selection-color: white;
                outline: none;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 28px;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #E5F2FF;
                color: #1C1C1E;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #007AFF;
                color: white;
            }
            QCheckBox {
                spacing: 8px;
                color: #1C1C1E;
                padding: 4px;
            }
            QCheckBox:hover {
                background-color: rgba(0, 122, 255, 0.05);
                border-radius: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #C7C7CC;
                background-color: white;
            }
            QCheckBox::indicator:hover {
                border-color: #007AFF;
            }
            QCheckBox::indicator:checked {
                background-color: #007AFF;
                border-color: #007AFF;
                image: url(none);
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: #F2F2F7;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #C7C7CC;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #8E8E93;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        )

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        header = self.create_header()
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #D1D1D6;
            }
        """
        )

        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, 1)

        button_layout = self.create_action_buttons()
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)

    def create_header(self):
        header = QWidget()
        header.setFixedHeight(90)
        header.setStyleSheet(
            """
            QWidget {
                background-color: white;
                border-radius: 12px;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(4)

        title = QLabel("Advanced Subtitle Generator")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #1C1C1E;")

        subtitle = QLabel("AI-powered subtitle generation using Whisper")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #8E8E93;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        header.setLayout(layout)
        return header

    def create_control_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 8, 0)

        model_card = SectionCard("🤖 Model Configuration")

        model_layout = QHBoxLayout()
        model_label = QLabel("Model Size:")
        model_label.setFixedWidth(100)
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS)
        self.model_combo.setCurrentText("large-v3")
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)
        model_card.add_layout(model_layout)

        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setFixedWidth(100)
        self.device_combo = QComboBox()
        self.device_combo.addItems(AVAILABLE_DEVICES)
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo, 1)
        model_card.add_layout(device_layout)

        compute_layout = QHBoxLayout()
        compute_label = QLabel("Compute Type:")
        compute_label.setFixedWidth(100)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(AVAILABLE_COMPUTE_TYPES)
        self.compute_combo.setCurrentText("float16")
        self.compute_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        compute_layout.addWidget(compute_label)
        compute_layout.addWidget(self.compute_combo, 1)
        model_card.add_layout(compute_layout)

        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        lang_label.setFixedWidth(100)
        self.language_combo = QComboBox()
        language_items = [f"{code} ({name})" for code, name in LANGUAGE_SUPPORTS]
        language_items.insert(0, "Auto Detect")
        self.language_combo.addItems(language_items)
        self.language_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.language_combo.setMaxVisibleItems(12)
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.language_combo, 1)
        model_card.add_layout(lang_layout)

        multilingual_card = SectionCard("🌍 Multilingual Options")
        multilingual_card.setVisible(False)

        self.multilingual_check = QCheckBox("Enable Multilingual Generation")
        self.multilingual_check.setToolTip(
            "Generate subtitles in multiple languages at once"
        )
        self.multilingual_check.stateChanged.connect(self.toggle_multilingual_mode)
        model_card.add_widget(self.multilingual_check)

        multilang_inner_layout = QHBoxLayout()

        self.select_langs_btn = ModernButton("📝 Select Languages", primary=True)
        self.select_langs_btn.setMaximumWidth(175)
        self.select_langs_btn.clicked.connect(self.open_language_selector)
        multilang_inner_layout.addWidget(self.select_langs_btn)

        self.selected_langs_label = QLabel("⚠ No languages selected")
        self.selected_langs_label.setStyleSheet(
            """
            color: #8E8E93; 
            font-size: 10px;
            padding: 8px;
            background-color: #F9F9F9;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            """
        )
        self.selected_langs_label.setWordWrap(True)
        multilang_inner_layout.addWidget(self.selected_langs_label, 1)

        multilingual_card.add_layout(multilang_inner_layout)

        self.multilingual_card = multilingual_card

        self.selected_languages = []

        layout.addWidget(model_card)
        layout.addWidget(self.multilingual_card)

        output_card = SectionCard("💾 Output Configuration")

        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        format_label.setFixedWidth(100)
        self.format_combo = QComboBox()
        self.format_combo.addItems(AVAILABLE_OUTPUT_FORMATS)
        self.format_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo, 1)
        output_card.add_layout(format_layout)

        dir_layout = QHBoxLayout()
        dir_label = QLabel("Output Dir:")
        dir_label.setFixedWidth(100)
        self.output_dir_label = QLabel("(Same as input)")
        self.output_dir_label.setStyleSheet("color: #8E8E93; font-size: 9px;")
        self.output_dir_label.setWordWrap(True)
        dir_btn = ModernButton("Browse", primary=False)
        dir_btn.setMaximumWidth(80)
        dir_btn.clicked.connect(self.select_output_directory)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.output_dir_label, 1)
        dir_layout.addWidget(dir_btn)
        output_card.add_layout(dir_layout)

        layout.addWidget(output_card)

        audio_card = SectionCard("🎵 Audio Processing")

        self.noise_reduction_check = QCheckBox("Enable Noise Reduction")
        self.noise_reduction_check.setToolTip(
            "Apply noise reduction for cleaner transcriptions"
        )
        audio_card.add_widget(self.noise_reduction_check)

        self.word_timestamps_check = QCheckBox("Generate Word-Level Timestamps")
        self.word_timestamps_check.setToolTip(
            "Include word-level timing (JSON format only)"
        )
        audio_card.add_widget(self.word_timestamps_check)

        silence_layout = QHBoxLayout()
        silence_label = QLabel("Silence Threshold:")
        silence_label.setFixedWidth(100)
        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(-60, 0)
        self.silence_spin.setValue(DEFAULT_SILENCE_THRESH)
        self.silence_spin.setSuffix(" dBFS")
        self.silence_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        silence_layout.addWidget(silence_label)
        silence_layout.addWidget(self.silence_spin, 1)
        audio_card.add_layout(silence_layout)

        min_silence_layout = QHBoxLayout()
        min_silence_label = QLabel("Min Silence:")
        min_silence_label.setFixedWidth(100)
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 2000)
        self.min_silence_spin.setValue(DEFAULT_MIN_SILENCE)
        self.min_silence_spin.setSuffix(" ms")
        self.min_silence_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        min_silence_layout.addWidget(min_silence_label)
        min_silence_layout.addWidget(self.min_silence_spin, 1)
        audio_card.add_layout(min_silence_layout)

        max_chunk_layout = QHBoxLayout()
        max_chunk_label = QLabel("Max Chunk:")
        max_chunk_label.setFixedWidth(100)
        self.max_chunk_spin = QDoubleSpinBox()
        self.max_chunk_spin.setRange(1.0, 60.0)
        self.max_chunk_spin.setValue(MAX_CHUNK_DURATION)
        self.max_chunk_spin.setSingleStep(1.0)
        self.max_chunk_spin.setDecimals(1)
        self.max_chunk_spin.setSuffix(" sec")
        self.max_chunk_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.max_chunk_spin.setToolTip("Maximum duration of a single audio chunk")
        max_chunk_layout.addWidget(max_chunk_label)
        max_chunk_layout.addWidget(self.max_chunk_spin, 1)
        audio_card.add_layout(max_chunk_layout)

        target_chunk_layout = QHBoxLayout()
        target_chunk_label = QLabel("Target Chunk:")
        target_chunk_label.setFixedWidth(100)
        self.target_chunk_spin = QDoubleSpinBox()
        self.target_chunk_spin.setRange(1.0, 60.0)
        self.target_chunk_spin.setValue(TARGET_CHUNK_DURATION)
        self.target_chunk_spin.setSingleStep(1.0)
        self.target_chunk_spin.setDecimals(1)
        self.target_chunk_spin.setSuffix(" sec")
        self.target_chunk_spin.setButtonSymbols(
            QDoubleSpinBox.ButtonSymbols.UpDownArrows
        )
        self.target_chunk_spin.setToolTip("Target duration when splitting long chunks")
        target_chunk_layout.addWidget(target_chunk_label)
        target_chunk_layout.addWidget(self.target_chunk_spin, 1)
        audio_card.add_layout(target_chunk_layout)

        min_sub_gap_layout = QHBoxLayout()
        min_sub_gap_label = QLabel("Min Subtitle Gap:")
        min_sub_gap_label.setFixedWidth(100)
        self.min_sub_gap_spin = QDoubleSpinBox()
        self.min_sub_gap_spin.setRange(0.0, 5.0)
        self.min_sub_gap_spin.setValue(0.0)
        self.min_sub_gap_spin.setSingleStep(0.1)
        self.min_sub_gap_spin.setDecimals(1)
        self.min_sub_gap_spin.setSuffix(" sec")
        self.min_sub_gap_spin.setButtonSymbols(
            QDoubleSpinBox.ButtonSymbols.UpDownArrows
        )
        self.min_sub_gap_spin.setToolTip("Minimum gap between subtitles when fixing")
        min_sub_gap_layout.addWidget(min_sub_gap_label)
        min_sub_gap_layout.addWidget(self.min_sub_gap_spin, 1)
        audio_card.add_layout(min_sub_gap_layout)

        layout.addWidget(audio_card)

        advanced_card = SectionCard("⚙️ Advanced Options")

        self.fix_overlaps_check = QCheckBox("Fix Overlapping Timestamps")
        self.fix_overlaps_check.setChecked(True)
        self.fix_overlaps_check.setToolTip(
            "Automatically correct overlapping subtitle timings"
        )
        advanced_card.add_widget(self.fix_overlaps_check)

        self.merge_chunks_check = QCheckBox("Merge Audio Chunks")
        self.merge_chunks_check.setChecked(True)
        self.merge_chunks_check.setToolTip("Combine nearby chunks for better context")
        advanced_card.add_widget(self.merge_chunks_check)

        workers_layout = QHBoxLayout()
        workers_label = QLabel("CPU Workers:")
        workers_label.setFixedWidth(100)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(1)
        self.workers_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        workers_layout.addWidget(workers_label)
        workers_layout.addWidget(self.workers_spin, 1)
        advanced_card.add_layout(workers_layout)

        layout.addWidget(advanced_card)

        layout.addStretch()

        panel.setLayout(layout)
        scroll.setWidget(panel)

        return scroll

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(8, 0, 0, 0)

        file_card = SectionCard("📁 Input Files")

        add_btn = ModernButton("+ Add Files", primary=False)
        add_btn.setMaximumWidth(150)
        add_btn.clicked.connect(self.add_files)
        file_card.add_widget(add_btn)

        self.file_list_widget = DragDropFileWidget()
        self.file_list_widget.files_dropped.connect(self.handle_dropped_files)
        self.file_list_layout = QVBoxLayout()
        self.file_list_layout.setSpacing(4)
        self.file_list_layout.setContentsMargins(4, 4, 4, 4)

        self.drop_hint_label = QLabel("📥 Drag & drop files here")
        self.drop_hint_label.setMinimumHeight(98)
        self.drop_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_hint_label.setStyleSheet(
            """
            QLabel {
                color: #8E8E93;
                font-size: 12px;
                padding: 20px;
                background: transparent;
            }
        """
        )
        self.file_list_layout.addWidget(self.drop_hint_label)
        self.file_list_layout.addStretch()
        self.file_list_widget.setLayout(self.file_list_layout)

        file_scroll = QScrollArea()
        file_scroll.setWidgetResizable(True)
        file_scroll.setWidget(self.file_list_widget)
        file_scroll.setMinimumHeight(110)
        file_scroll.setMaximumHeight(110)
        file_scroll.setStyleSheet(
            """
            QScrollArea {
                background-color: #F9F9F9;
                border: 2px dashed #C7C7CC;
                border-radius: 8px;
            }
        """
        )

        file_card.add_widget(file_scroll)

        layout.addWidget(file_card)

        progress_card = SectionCard("📊 Progress")

        self.progress_widget = ModernProgressBar()
        progress_card.add_widget(self.progress_widget)

        layout.addWidget(progress_card)

        log_card = SectionCard("📋 Log")

        self.log_console = LogConsole()
        self.log_console.setMaximumHeight(300)
        log_card.add_widget(self.log_console)

        layout.addWidget(log_card)
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def create_action_buttons(self):
        layout = QHBoxLayout()
        layout.addStretch()

        self.clear_btn = ModernButton("Clear Files", primary=False)
        self.clear_btn.clicked.connect(self.clear_files)

        self.cancel_btn = ModernButton("Cancel", primary=False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)

        self.edit_btn = ModernButton("✏ Edit Subtitles", primary=False)
        self.edit_btn.clicked.connect(self.open_subtitle_editor)
        self.edit_btn.setEnabled(False)
        self.edit_btn.setVisible(False)

        self.generate_btn = ModernButton("Generate Subtitles", primary=True)
        self.generate_btn.clicked.connect(self.start_processing)

        layout.addWidget(self.clear_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.generate_btn)

        return layout

    def setup_logging(self):
        self.log_console.log("Subtitle Generator initialized", "INFO")
        self.log_console.log("Ready to process files", "SUCCESS")

    def add_files(self):
        file_filter = "Media Files ("
        for fmt in SUPPORTED_FORMATS:
            file_filter += f"*{fmt} "
        file_filter += ");;All Files (*.*)"

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Media Files", "", file_filter
        )

        if files:
            self._add_files_to_list(files)

    def handle_dropped_files(self, file_paths: List[str]):
        valid_files = []
        invalid_files = []

        for file_path in file_paths:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in SUPPORTED_FORMATS:
                valid_files.append(file_path)
            else:
                invalid_files.append(Path(file_path).name)

        if valid_files:
            self._add_files_to_list(valid_files)

        if invalid_files:
            self.log_console.log(
                f"Skipped {len(invalid_files)} unsupported file(s)", "WARNING"
            )

    def _add_files_to_list(self, files: List[str]):
        added_count = 0
        for file_path in files:
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
                self.add_file_item(file_path)
                added_count += 1

        if added_count > 0:
            self.log_console.log(f"Added {added_count} file(s)", "INFO")

    def add_file_item(self, file_path):
        if self.drop_hint_label.isVisible():
            self.drop_hint_label.hide()

        if self.file_list_layout.count() > 0:
            item = self.file_list_layout.takeAt(self.file_list_layout.count() - 1)
            if item:
                item.widget()

        file_item = FileListItem(file_path)
        file_item.remove_clicked.connect(self.remove_file)
        self.file_list_layout.addWidget(file_item)

        self.file_list_layout.addStretch()

    def remove_file(self, file_path):
        if file_path in self.file_paths:
            self.file_paths.remove(file_path)

            for i in range(self.file_list_layout.count()):
                item = self.file_list_layout.itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if (
                        isinstance(widget, FileListItem)
                        and widget.file_path == file_path
                    ):
                        widget.deleteLater()
                        break

            if len(self.file_paths) == 0:
                self.drop_hint_label.show()

            self.log_console.log(f"Removed: {Path(file_path).name}", "INFO")

    def clear_files(self):
        if not self.file_paths:
            return

        reply = QMessageBox.question(
            self,
            "Clear Files",
            f"Remove all {len(self.file_paths)} file(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.file_paths.clear()

            for i in reversed(range(self.file_list_layout.count())):
                item = self.file_list_layout.itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

            self.drop_hint_label.show()

            self.log_console.log("All files cleared", "INFO")

            self.file_list_layout.addStretch()
            self.log_console.log("Cleared all files", "INFO")

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )

        if directory:
            self.output_dir_label.setText(directory)
            self.log_console.log(f"Output directory: {directory}", "INFO")

    def toggle_multilingual_mode(self, state):
        is_multilingual = state == Qt.CheckState.Checked

        self.multilingual_card.setVisible(is_multilingual)
        self.language_combo.setEnabled(not is_multilingual)

        if is_multilingual:
            self.log_console.log("Multilingual mode enabled", "INFO")
            if not self.selected_languages:
                self.selected_langs_label.setText(
                    "⚠ No languages selected - click 'Select Languages' button"
                )
                self.selected_langs_label.setStyleSheet(
                    """
                    color: #FF9500; 
                    font-size: 12px;
                    padding: 8px;
                    background-color: #FFF3E0;
                    border: 1px solid #FFB74D;
                    border-radius: 4px;
                    """
                )
        else:
            self.log_console.log("Single language mode enabled", "INFO")
            self.selected_languages = []
            self.selected_langs_label.setText("⚠ No languages selected")
            self.selected_langs_label.setStyleSheet(
                """
                color: #8E8E93; 
                font-size: 10px;
                padding: 8px;
                background-color: #F9F9F9;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                """
            )

    def open_language_selector(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Languages for Multilingual Generation")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout()

        info_label = QLabel(
            "Select one or more languages for subtitle generation.\n"
            "The audio will be transcribed into each selected language."
        )
        info_label.setFont(QFont("Segoe UI", 10))
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        for code, name in LANGUAGE_SUPPORTS:
            item = QListWidgetItem(f"{code} - {name}")
            item.setData(Qt.ItemDataRole.UserRole, code)
            list_widget.addItem(item)

            if code in self.selected_languages:
                item.setSelected(True)

        layout.addWidget(list_widget)

        quick_select_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(list_widget.selectAll)
        quick_select_layout.addWidget(select_all_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(list_widget.clearSelection)
        quick_select_layout.addWidget(clear_all_btn)

        popular_btn = QPushButton("Popular (EN, TH)")

        def select_popular():
            list_widget.clearSelection()
            popular_codes = ["en", "th"]
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item is not None:
                    code = item.data(Qt.ItemDataRole.UserRole)
                    if code in popular_codes:
                        item.setSelected(True)

        popular_btn.clicked.connect(select_popular)
        quick_select_layout.addWidget(popular_btn)

        quick_select_layout.addStretch()
        layout.addLayout(quick_select_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = ModernButton("OK", primary=True)
        ok_btn.clicked.connect(dialog.accept)

        cancel_btn = ModernButton("Cancel", primary=False)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.DialogCode.Accepted:
            self.selected_languages = []
            for item in list_widget.selectedItems():
                code = item.data(Qt.ItemDataRole.UserRole)
                self.selected_languages.append(code)

            if self.selected_languages:
                count = len(self.selected_languages)
                if count <= 5:
                    langs_text = ", ".join(self.selected_languages)
                    display_text = f"✓ {count} language(s): {langs_text}"
                else:
                    first_three = ", ".join(self.selected_languages[:3])
                    display_text = f"✓ {count} language(s): {first_three}, ... and {count - 3} more"

                self.selected_langs_label.setText(display_text)
                self.selected_langs_label.setStyleSheet(
                    """
                    color: #34C759; 
                    font-size: 12px;
                    font-weight: 500;
                    padding: 8px;
                    background-color: #E8F5E9;
                    border: 1px solid #A5D6A7;
                    border-radius: 4px;
                    """
                )
                self.log_console.log(
                    f"Selected {count} language(s): {', '.join(self.selected_languages)}",
                    "INFO",
                )
            else:
                self.selected_langs_label.setText(
                    "⚠ No languages selected - click 'Select Languages'"
                )
                self.selected_langs_label.setStyleSheet(
                    """
                    color: #FF9500; 
                    font-size: 10px;
                    padding: 8px;
                    background-color: #FFF3E0;
                    border: 1px solid #FFB74D;
                    border-radius: 4px;
                    """
                )
                self.log_console.log("No languages selected", "WARNING")

    def get_config(self) -> Config:
        language = None
        lang_text = self.language_combo.currentText()
        if not lang_text.startswith("Auto"):
            language = lang_text.split("(")[0].strip()

        config = Config(
            model_size=self.model_combo.currentText(),
            device=self.device_combo.currentText(),
            compute_type=self.compute_combo.currentText(),
            language=language,
            use_noise_reduction=self.noise_reduction_check.isChecked(),
            word_timestamps=self.word_timestamps_check.isChecked(),
            silence_thresh=self.silence_spin.value(),
            min_silence_len=self.min_silence_spin.value(),
            fix_overlaps=self.fix_overlaps_check.isChecked(),
            merge_chunks=self.merge_chunks_check.isChecked(),
            num_workers=self.workers_spin.value(),
            max_chunk_duration=self.max_chunk_spin.value(),
            target_chunk_duration=self.target_chunk_spin.value(),
            min_subtitle_gap=self.min_sub_gap_spin.value(),
        )

        return config

    def start_processing(self):
        if not self.file_paths:
            QMessageBox.warning(
                self, "No Files", "Please add at least one media file to process."
            )
            return

        is_multilingual = self.multilingual_check.isChecked()
        languages_to_use = None

        if is_multilingual:
            if not self.selected_languages:
                QMessageBox.warning(
                    self,
                    "No Languages Selected",
                    "Please select at least one language for multilingual generation.\n\n"
                    "Click 'Select Languages...' to choose languages.",
                )
                return
            languages_to_use = self.selected_languages.copy()

        output_dir = None
        if self.output_dir_label.text() != "(Same as input)":
            output_dir = self.output_dir_label.text()

        output_format = self.format_combo.currentText()

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        config = self.get_config()

        self.worker = SubtitleWorker(
            self.file_paths,
            config,
            output_format,
            output_dir,
            languages=languages_to_use,
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.log_message.connect(self.add_log)
        self.worker.finished.connect(self.processing_finished)
        self.worker.subtitles_generated.connect(self.on_subtitles_generated)

        self.log_console.log("=" * 50, "INFO")
        self.log_console.log("Starting subtitle generation...", "INFO")
        self.log_console.log(f"Files to process: {len(self.file_paths)}", "INFO")
        self.log_console.log(f"Model: {config.model_size}", "INFO")
        self.log_console.log(f"Device: {config.device}", "INFO")
        if is_multilingual and languages_to_use:
            self.log_console.log(f"Languages: {', '.join(languages_to_use)}", "INFO")
        self.log_console.log(f"Output format: {output_format}", "INFO")

        self.worker.start()

    def cancel_processing(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Processing",
                "Are you sure you want to cancel the current operation?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.log_console.log("Cancelling processing...", "WARNING")

    def update_progress(self, status: str, percentage: int, details: str):
        self.progress_widget.set_status(status, percentage, details)

    def add_log(self, message: str, level: str):
        self.log_console.log(message, level)

    def on_subtitles_generated(self, processed_subtitles: dict):
        self.processed_subtitles = processed_subtitles
        file_count = len(processed_subtitles)
        self.log_console.log(
            f"Subtitles ready for editing: {file_count} file(s)", "INFO"
        )

    def processing_finished(self, success: bool, message: str):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            self.log_console.log(message, "SUCCESS")
            self.progress_widget.set_status("Ready", 0, "")

            if self.processed_subtitles:
                self.edit_btn.setEnabled(True)
                self.edit_btn.setVisible(True)
                self.log_console.log(
                    "Click 'Edit Subtitles' to review and modify", "INFO"
                )

            QMessageBox.information(self, "Success", message)
        else:
            self.log_console.log(message, "ERROR")
            self.progress_widget.set_status(
                "Cancelled" if "cancel" in message.lower() else "Ready",
                0,
                "",
            )
            QMessageBox.warning(self, "Error", message)

    def open_subtitle_editor(self):
        if not self.processed_subtitles:
            QMessageBox.warning(
                self, "No Subtitles", "Please generate subtitles first before editing."
            )
            return

        from .subtitle_editor import SubtitleEditorDialog

        try:
            output_format = self.format_combo.currentText()
            output_dir = None
            if self.output_dir_label.text() != "(Same as input)":
                output_dir = self.output_dir_label.text()

            if len(self.processed_subtitles) > 1:
                selected_file = self.show_file_selector()
                if not selected_file:
                    return
                files_to_edit = [
                    (selected_file, self.processed_subtitles[selected_file])
                ]
            else:
                files_to_edit = list(self.processed_subtitles.items())

            for file_path, subtitles_data in files_to_edit:
                if isinstance(subtitles_data, dict):
                    language = self.show_language_selector_for_edit(
                        subtitles_data, file_path
                    )
                    if not language:
                        continue

                    subtitles, stats = subtitles_data[language]

                    editor = SubtitleEditorDialog(
                        file_path, subtitles, output_format, output_dir, self
                    )
                    result = editor.exec_()

                    if result == editor.Accepted:
                        edited_subtitles = editor.get_subtitles()
                        self.processed_subtitles[file_path][language] = (
                            edited_subtitles,
                            stats,
                        )

                        self.log_console.log(
                            f"✓ Edited and exported subtitles [{language}] for: {Path(file_path).name}",
                            "SUCCESS",
                        )
                else:
                    editor = SubtitleEditorDialog(
                        file_path, subtitles_data, output_format, output_dir, self
                    )
                    result = editor.exec_()

                    if result == editor.Accepted:
                        edited_subtitles = editor.get_subtitles()
                        self.processed_subtitles[file_path] = edited_subtitles

                        # Log success
                        self.log_console.log(
                            f"✓ Edited and exported subtitles for: {Path(file_path).name}",
                            "SUCCESS",
                        )

        except Exception as e:
            logger.error(f"Error opening subtitle editor: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to open subtitle editor: {str(e)}"
            )

    def show_file_selector(self) -> Optional[str]:

        dialog = QDialog(self)
        dialog.setWindowTitle("Select File to Edit")
        dialog.setMinimumSize(500, 300)

        layout = QVBoxLayout()

        label = QLabel(
            f"Select a file to edit subtitles ({len(self.processed_subtitles)} files processed):"
        )
        label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(label)

        file_list = QListWidget()
        for file_path in self.processed_subtitles.keys():
            item = QListWidgetItem(Path(file_path).name)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            file_list.addItem(item)
        file_list.setCurrentRow(0)
        layout.addWidget(file_list)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = ModernButton("Select", primary=True)
        ok_btn.clicked.connect(dialog.accept)

        cancel_btn = ModernButton("Cancel", primary=False)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.DialogCode.Accepted:
            current_item = file_list.currentItem()
            if current_item:
                return current_item.data(Qt.ItemDataRole.UserRole)
        return None

    def show_language_selector_for_edit(
        self, multilingual_data: dict, file_path: str
    ) -> Optional[str]:
        """Show a dialog to select which language to edit."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Language to Edit")
        dialog.setMinimumSize(500, 350)

        layout = QVBoxLayout()

        file_name = Path(file_path).name
        file_label = QLabel(f"📄 File: {file_name}")
        file_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        file_label.setStyleSheet(
            """
            color: #1C1C1E;
            padding: 8px;
            background-color: #F2F2F7;
            border-radius: 6px;
            """
        )
        file_label.setWordWrap(True)
        layout.addWidget(file_label)

        label = QLabel(
            f"Select a language to edit ({len(multilingual_data)} languages available):"
        )
        label.setFont(QFont("Segoe UI", 10))
        label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(label)

        language_list = QListWidget()
        language_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #D1D1D6;
                border-radius: 6px;
                background-color: white;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: #E5F2FF;
            }
            QListWidget::item:selected {
                background-color: #007AFF;
                color: white;
            }
            """
        )

        for language, (subtitles, stats) in multilingual_data.items():
            subtitle_count = len(subtitles)
            item_text = f"🌐 {language.upper()} — {subtitle_count} subtitles, {stats.processing_time:.1f}s"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, language)
            language_list.addItem(item)
        language_list.setCurrentRow(0)
        layout.addWidget(language_list)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = ModernButton("Edit", primary=True)
        ok_btn.clicked.connect(dialog.accept)

        cancel_btn = ModernButton("Cancel", primary=False)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.DialogCode.Accepted:
            current_item = language_list.currentItem()
            if current_item:
                return current_item.data(Qt.ItemDataRole.UserRole)
        return None

    def export_edited_subtitles(self, file_path: str, subtitles: list):
        try:
            from .subtitle_generator import SubtitleGenerator

            config = self.get_config()
            generator = SubtitleGenerator(config)

            output_format = self.format_combo.currentText()

            if self.output_dir_label.text() != "(Same as input)":
                output_base = Path(self.output_dir_label.text()) / Path(file_path).stem
            else:
                output_base = Path(file_path).parent / Path(file_path).stem

            if output_format == "all":
                output_path = str(output_base)
                success = generator.export(subtitles, output_path, None, format="all")
            else:
                output_path = str(output_base) + f".{output_format}"
                success = generator.export(
                    subtitles, output_path, None, format=output_format
                )

            if success:
                self.log_console.log(
                    f"✓ Re-exported edited subtitles: {output_path}", "SUCCESS"
                )
                QMessageBox.information(
                    self,
                    "Success",
                    f"Edited subtitles exported successfully to:\n{output_path}",
                )
            else:
                self.log_console.log("✗ Failed to export edited subtitles", "ERROR")
                QMessageBox.warning(self, "Error", "Failed to export edited subtitles")

        except Exception as e:
            logger.error(f"Error exporting edited subtitles: {e}")
            self.log_console.log(f"✗ Export error: {str(e)}", "ERROR")
            QMessageBox.critical(
                self, "Error", f"Failed to export edited subtitles: {str(e)}"
            )


def main():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = SubtitleGeneratorGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
