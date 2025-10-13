import datetime
import os
from pathlib import Path
from typing import Optional, Tuple

from .config import SUPPORTED_AUDIO_FORMATS, SUPPORTED_FORMATS, SUPPORTED_VIDEO_FORMATS

def format_timestamp_srt(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    secs = int(delta.total_seconds() % 60)
    milliseconds = int((delta.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    secs = int(delta.total_seconds() % 60)
    milliseconds = int((delta.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def detect_file_format(file_path: str) -> Optional[str]:
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_AUDIO_FORMATS:
        return "audio"
    elif ext in SUPPORTED_VIDEO_FORMATS:
        return "video"
    return None


def validate_file(file_path: str) -> Tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        return (
            False,
            f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    return True, "OK"


def get_output_path(
    input_path: str, output_path: Optional[str] = None, format: str = "srt"
) -> Path:
    if output_path:
        path = Path(output_path)
        if path.is_dir():
            base_name = Path(input_path).stem
            return path / f"{base_name}.{format}"
        return path

    input_file = Path(input_path)
    return input_file.parent / f"{input_file.stem}.{format}"


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0.0
