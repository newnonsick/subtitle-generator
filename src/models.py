from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class Subtitle:
    index: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    words: Optional[List[Dict]] = None

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"Subtitle({self.index}: {self.start:.2f}-{self.end:.2f}s, '{self.text[:30]}...')"


@dataclass
class ProcessingStats:
    total_duration: float
    processing_time: float
    chunks_processed: int
    subtitles_generated: int
    avg_confidence: Optional[float] = None
    detected_language: Optional[str] = None

    def speed_ratio(self) -> float:
        return (
            self.total_duration / self.processing_time
            if self.processing_time > 0
            else 0
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"ProcessingStats(duration={self.total_duration:.2f}s, "
            f"time={self.processing_time:.2f}s, "
            f"speed={self.speed_ratio():.2f}x, "
            f"subtitles={self.subtitles_generated})"
        )
