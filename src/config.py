import os
import warnings
from typing import List

# ============================================================================
# Environment Setup
# ============================================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ============================================================================
# Audio Processing Constants
# ============================================================================

WHISPER_SAMPLE_RATE = 16000
"""Standard sample rate expected by Whisper models."""

SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
"""Supported audio file formats."""

SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov"]
"""Supported video file formats."""

SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
"""All supported media file formats."""

# ============================================================================
# Silence Detection Defaults
# ============================================================================

DEFAULT_SILENCE_THRESH = -40
"""Default silence threshold in dBFS (decibels relative to full scale)."""

DEFAULT_MIN_SILENCE = 700
"""Default minimum silence duration in milliseconds."""

DEFAULT_KEEP_SILENCE = 300
"""Default amount of silence to keep at chunk boundaries in milliseconds."""

DEFAULT_MAX_GAP_MS = 500
"""Default maximum gap between chunks for merging in milliseconds."""

# ============================================================================
# Chunk Processing Defaults
# ============================================================================

MAX_CHUNK_DURATION = 5.0
"""Maximum duration of a single audio chunk in seconds."""

TARGET_CHUNK_DURATION = 5.0
"""Target duration when splitting long chunks in seconds."""

MIN_CHUNK_DURATION = 0.1
"""Minimum duration of a valid audio chunk in seconds."""

# ============================================================================
# Whisper Model Configuration
# ============================================================================

AVAILABLE_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "turbo",
]
"""List of available Whisper model sizes."""

DEFAULT_MODEL = "large-v3"
"""Default Whisper model to use."""

AVAILABLE_DEVICES = ["auto", "cpu", "cuda"]
"""Available compute devices."""

AVAILABLE_COMPUTE_TYPES = ["float16", "float32", "int8", "int8_float16"]
"""Available compute types for model inference."""

# ============================================================================
# Output Format Configuration
# ============================================================================

AVAILABLE_OUTPUT_FORMATS = ["srt", "vtt", "json", "all"]
"""Available subtitle output formats."""

DEFAULT_OUTPUT_FORMAT = "srt"
"""Default output format for subtitles."""

# ============================================================================
# Configuration Class
# ============================================================================


class Config:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "float16",
        num_workers: int = 1,
        silence_thresh: int = DEFAULT_SILENCE_THRESH,
        min_silence_len: int = DEFAULT_MIN_SILENCE,
        keep_silence: int = DEFAULT_KEEP_SILENCE,
        max_gap_ms: int = DEFAULT_MAX_GAP_MS,
        max_chunk_duration: float = MAX_CHUNK_DURATION,
        target_chunk_duration: float = TARGET_CHUNK_DURATION,
        merge_chunks: bool = True,
        word_timestamps: bool = False,
        use_noise_reduction: bool = False,
        language: str | None = None,
        fix_overlaps: bool = True,
        min_subtitle_gap: float = 0.0,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_workers = num_workers

        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence
        self.max_gap_ms = max_gap_ms

        self.max_chunk_duration = max_chunk_duration
        self.target_chunk_duration = target_chunk_duration
        self.merge_chunks = merge_chunks

        self.word_timestamps = word_timestamps
        self.use_noise_reduction = use_noise_reduction

        self.language = language
        self.fix_overlaps = fix_overlaps
        self.min_subtitle_gap = min_subtitle_gap

    def validate(self) -> bool:
        if self.model_size not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model size: {self.model_size}. "
                f"Available: {', '.join(AVAILABLE_MODELS)}"
            )

        if self.device not in AVAILABLE_DEVICES:
            raise ValueError(
                f"Invalid device: {self.device}. "
                f"Available: {', '.join(AVAILABLE_DEVICES)}"
            )

        if self.compute_type not in AVAILABLE_COMPUTE_TYPES:
            raise ValueError(
                f"Invalid compute type: {self.compute_type}. "
                f"Available: {', '.join(AVAILABLE_COMPUTE_TYPES)}"
            )

        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        if self.max_chunk_duration <= 0:
            raise ValueError("max_chunk_duration must be positive")

        if self.target_chunk_duration <= 0:
            raise ValueError("target_chunk_duration must be positive")

        return True

    def to_dict(self) -> dict:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "num_workers": self.num_workers,
            "silence_thresh": self.silence_thresh,
            "min_silence_len": self.min_silence_len,
            "keep_silence": self.keep_silence,
            "max_gap_ms": self.max_gap_ms,
            "max_chunk_duration": self.max_chunk_duration,
            "target_chunk_duration": self.target_chunk_duration,
            "merge_chunks": self.merge_chunks,
            "word_timestamps": self.word_timestamps,
            "use_noise_reduction": self.use_noise_reduction,
            "language": self.language,
            "fix_overlaps": self.fix_overlaps,
            "min_subtitle_gap": self.min_subtitle_gap,
        }
