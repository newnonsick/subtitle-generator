import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from .models import ProcessingStats, Subtitle
from .utils import format_timestamp_srt, format_timestamp_vtt

logger = logging.getLogger(__name__)


class SubtitleExporter(ABC):
    @abstractmethod
    def export(
        self, subtitles: List[Subtitle], output_path: str, stats: ProcessingStats | None = None
    ) -> bool:
        pass


class SRTExporter(SubtitleExporter):
    def export(
        self,
        subtitles: List[Subtitle],
        output_path: str,
        stats: ProcessingStats | None = None,
    ) -> bool:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for sub in subtitles:
                    f.write(f"{sub.index}\n")
                    f.write(
                        f"{format_timestamp_srt(sub.start)} --> "
                        f"{format_timestamp_srt(sub.end)}\n"
                    )
                    f.write(f"{sub.text}\n\n")

            logger.info(f"💾 Saved SRT: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save SRT: {e}")
            return False


class VTTExporter(SubtitleExporter):
    def export(
        self,
        subtitles: List[Subtitle],
        output_path: str,
        stats: ProcessingStats | None = None,
    ) -> bool:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for sub in subtitles:
                    f.write(
                        f"{format_timestamp_vtt(sub.start)} --> "
                        f"{format_timestamp_vtt(sub.end)}\n"
                    )
                    f.write(f"{sub.text}\n\n")

            logger.info(f"💾 Saved VTT: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save VTT: {e}")
            return False


class JSONExporter(SubtitleExporter):
    def __init__(self, indent: int = 2, include_words: bool = True):
        self.indent = indent
        self.include_words = include_words

    def export(
        self,
        subtitles: List[Subtitle],
        output_path: str,
        stats: ProcessingStats | None = None,
    ) -> bool:
        try:
            data = {"metadata": stats.to_dict() if stats else {}, "subtitles": []}

            for sub in subtitles:
                sub_dict = sub.to_dict()
                if not self.include_words and "words" in sub_dict:
                    del sub_dict["words"]
                data["subtitles"].append(sub_dict)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=self.indent, ensure_ascii=False)

            logger.info(f"💾 Saved JSON: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save JSON: {e}")
            return False


class ExporterFactory:
    _exporters = {
        "srt": SRTExporter,
        "vtt": VTTExporter,
        "json": JSONExporter,
    }

    @classmethod
    def get_exporter(cls, format: str, **kwargs) -> SubtitleExporter:
        format = format.lower()
        if format not in cls._exporters:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(cls._exporters.keys())}"
            )

        return cls._exporters[format](**kwargs)

    @classmethod
    def export_all_formats(
        cls,
        subtitles: List[Subtitle],
        base_path: str,
        stats: ProcessingStats | None = None,
        formats: List[str] | None = None,
    ) -> dict:
        if formats is None:
            formats = list(cls._exporters.keys())

        base = Path(base_path)
        results = {}

        for fmt in formats:
            try:
                output_path = f"{base.parent / base.stem}.{fmt}"
                exporter = cls.get_exporter(fmt)
                results[fmt] = exporter.export(subtitles, output_path, stats)
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
                results[fmt] = False

        return results
