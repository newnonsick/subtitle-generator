"""
Advanced Subtitle Generator - A powerful AI-powered subtitle generation tool.

This package provides automatic subtitle generation from audio and video files
using OpenAI's Whisper model with advanced audio processing capabilities.
"""

__version__ = "2.0.0"
__author__ = "newnonsick"
__license__ = "MIT"

from .config import Config
from .models import ProcessingStats, Subtitle
from .subtitle_generator import SubtitleGenerator

__all__ = [
    "SubtitleGenerator",
    "Subtitle",
    "ProcessingStats",
    "Config",
]
