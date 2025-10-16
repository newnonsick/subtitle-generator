import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .audio_processor import AudioProcessor
from .config import WHISPER_SAMPLE_RATE, Config
from .exporters import ExporterFactory
from .models import ProcessingStats, Subtitle
from .utils import ensure_directory, validate_file

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    def __init__(self, config: Optional[Config] = None):
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required. Install with: pip install numpy")

        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required. "
                "Install with: pip install faster-whisper"
            )

        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required. " "Install with: pip install librosa soundfile"
            )

        self.config = config or Config()
        self.config.validate()

        self.audio_processor = AudioProcessor(
            use_noise_reduction=self.config.use_noise_reduction
        )

        self._initialize_model()

    def _initialize_model(self):
        logger.info(f"🚀 Initializing Whisper model: {self.config.model_size}")
        logger.info(
            f"   Device: {self.config.device}, " f"Compute: {self.config.compute_type}"
        )

        try:
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                num_workers=self.config.num_workers,
                download_root=None,
                local_files_only=False,
                cpu_threads=(
                    self.config.num_workers if self.config.device == "cpu" else 0
                ),
            )
            logger.info("✅ Model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            logger.error("   Ensure CTranslate2 backend is properly installed")
            raise RuntimeError(f"Failed to initialize Whisper model: {e}")

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sr: int,
        chunk_start_sec: float,
        language: Optional[str] = None,
    ) -> List[Subtitle]:
        subtitles = []

        try:
            if sr != WHISPER_SAMPLE_RATE:
                audio_chunk = librosa.resample(
                    audio_chunk,
                    orig_sr=sr,
                    target_sr=WHISPER_SAMPLE_RATE,
                    res_type="kaiser_fast",
                )
                sr = WHISPER_SAMPLE_RATE

            segments, info = self.model.transcribe(
                audio_chunk,
                language=language,
                beam_size=3,
                best_of=1,
                temperature=0.0,
                word_timestamps=self.config.word_timestamps,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            for segment in segments:
                absolute_start = chunk_start_sec + segment.start
                absolute_end = chunk_start_sec + segment.end

                words = None
                if (
                    self.config.word_timestamps
                    and hasattr(segment, "words")
                    and segment.words
                ):
                    words = [
                        {
                            "word": w.word,
                            "start": chunk_start_sec + w.start,
                            "end": chunk_start_sec + w.end,
                            "probability": w.probability,
                        }
                        for w in segment.words
                    ]

                confidence = getattr(segment, "avg_logprob", None)
                if confidence is not None:
                    confidence += 1

                subtitle = Subtitle(
                    index=0,
                    start=absolute_start,
                    end=absolute_end,
                    text=segment.text.strip(),
                    confidence=confidence,
                    words=words,
                )
                subtitles.append(subtitle)

        except Exception as e:
            logger.error(f"❌ Transcription failed for chunk: {e}")

        return subtitles

    def _fix_overlapping_subtitles(
        self, subtitles: List[Subtitle], min_gap: float = 0.05
    ) -> List[Subtitle]:
        """Fix overlapping subtitle timestamps with minimum gap."""
        if not subtitles:
            return subtitles

        subtitles.sort(key=lambda s: s.start)
        overlaps_found = 0
        adjustments_made = 0

        fixed_subtitles = []

        for i, subtitle in enumerate(subtitles):
            current_start = subtitle.start
            current_end = subtitle.end

            if i < len(subtitles) - 1:
                next_start = subtitles[i + 1].start

                if current_end >= next_start - min_gap:
                    overlaps_found += 1
                    old_end = current_end
                    current_end = next_start - min_gap
                    adjustments_made += 1

                    logger.debug(
                        f"Overlap detected at subtitle {i+1}: "
                        f"adjusted end from {old_end:.3f}s to {current_end:.3f}s"
                    )

                    min_duration = 0.1
                    if current_end - current_start < min_duration:
                        current_end = current_start + min_duration
                        logger.debug(
                            f"Enforced minimum duration for subtitle {i+1}: "
                            f"end time set to {current_end:.3f}s"
                        )

            fixed_subtitle = Subtitle(
                index=subtitle.index,
                start=current_start,
                end=current_end,
                text=subtitle.text,
                confidence=subtitle.confidence,
                words=subtitle.words,
            )
            fixed_subtitles.append(fixed_subtitle)

        for i in range(1, len(fixed_subtitles)):
            prev_end = fixed_subtitles[i - 1].end
            current_start = fixed_subtitles[i].start

            if current_start < prev_end + min_gap:
                old_start = current_start
                fixed_subtitles[i].start = prev_end + min_gap
                adjustments_made += 1

                logger.debug(
                    f"Start time adjustment for subtitle {i+1}: "
                    f"{old_start:.3f}s → {fixed_subtitles[i].start:.3f}s"
                )

                if fixed_subtitles[i].end <= fixed_subtitles[i].start:
                    fixed_subtitles[i].end = fixed_subtitles[i].start + 0.1
                    logger.debug(
                        f"End time extended to maintain minimum duration: "
                        f"{fixed_subtitles[i].end:.3f}s"
                    )

        if overlaps_found > 0:
            logger.info(
                f"Fixed {overlaps_found} overlapping subtitles "
                f"with {adjustments_made} total adjustments (min_gap={min_gap}s)"
            )
        else:
            logger.debug("No overlapping subtitles found")

        return fixed_subtitles

    def _process_audio_to_chunks(
        self, media_path: str
    ) -> Tuple[
        Optional[np.ndarray], Optional[float], Optional[List[Tuple[int, int]]], float
    ]:
        is_valid, msg = validate_file(media_path)
        if not is_valid:
            logger.error(f"❌ {msg}")
            return None, None, None, 0.0

        audio_result = self.audio_processor.load_audio(media_path)
        if audio_result is None:
            return None, None, None, 0.0

        audio, sr = audio_result
        audio_duration = len(audio) / sr

        if self.config.use_noise_reduction:
            audio = self.audio_processor.apply_noise_reduction(audio, sr)

        chunks = self.audio_processor.detect_speech_chunks(
            audio,
            int(sr),
            self.config.min_silence_len,
            self.config.silence_thresh,
            self.config.keep_silence,
        )

        if not chunks:
            logger.warning("⚠️  No speech detected in audio")
            return audio, sr, [], audio_duration

        if self.config.merge_chunks:
            chunks = self.audio_processor.merge_nearby_chunks(
                chunks, int(sr), self.config.max_gap_ms
            )

        chunks = self.audio_processor.split_long_chunks(
            chunks,
            audio,
            sr,
            max_duration=self.config.max_chunk_duration,
            target_duration=self.config.target_chunk_duration,
        )

        return audio, sr, chunks, audio_duration

    def generate(
        self,
        media_path: str,
        language: Optional[str] = None,
    ) -> Tuple[List[Subtitle], ProcessingStats]:
        start_time = time.time()

        if language is None:
            language = self.config.language

        audio, sr, chunks, audio_duration = self._process_audio_to_chunks(media_path)

        if audio is None or sr is None or not chunks:
            return [], ProcessingStats(0, time.time() - start_time, 0, 0)

        logger.info(
            f"🎬 Processing {len(chunks)} chunks for language: {language or 'auto-detect'}..."
        )
        all_subtitles = []

        chunk_iter = (
            tqdm(chunks, desc="Transcribing", disable=logger.level > logging.INFO)
            if TQDM_AVAILABLE
            else chunks
        )

        for start_sample, end_sample in chunk_iter:
            chunk_start_sec = start_sample / sr
            audio_chunk = audio[start_sample:end_sample]

            subtitles = self.transcribe_chunk(
                audio_chunk, int(sr), chunk_start_sec, language
            )
            all_subtitles.extend(subtitles)

        if all_subtitles and self.config.fix_overlaps:
            all_subtitles = self._fix_overlapping_subtitles(
                all_subtitles, min_gap=self.config.min_subtitle_gap
            )
            logger.info("🔧 Fixed overlapping subtitle timestamps")

        for idx, sub in enumerate(all_subtitles, 1):
            sub.index = idx

        processing_time = time.time() - start_time
        avg_confidence = None

        if all_subtitles and all_subtitles[0].confidence is not None:
            confidences = [s.confidence for s in all_subtitles if s.confidence]
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else None
            )

        stats = ProcessingStats(
            total_duration=audio_duration,
            processing_time=processing_time,
            chunks_processed=len(chunks),
            subtitles_generated=len(all_subtitles),
            avg_confidence=avg_confidence,
            detected_language=language,
        )

        logger.info(
            f"✅ Generated {len(all_subtitles)} subtitles " f"in {processing_time:.2f}s"
        )
        logger.info(f"   Speed: {stats.speed_ratio():.2f}x realtime")
        if avg_confidence:
            logger.info(f"   Avg confidence: {avg_confidence:.3f}")

        return all_subtitles, stats

    def generate_multilingual(
        self,
        media_path: str,
        languages: List[str],
    ) -> Dict[str, Tuple[List[Subtitle], ProcessingStats]]:
        if not languages:
            logger.error("❌ No languages specified for multilingual generation")
            return {}

        logger.info(
            f"🌍 Starting multilingual generation for {len(languages)} language(s): {', '.join(languages)}"
        )
        logger.info(f"   Synchronized mode: All languages will share the same timings")

        start_time = time.time()

        audio, sr, chunks, audio_duration = self._process_audio_to_chunks(media_path)

        if audio is None or sr is None or not chunks:
            logger.error("❌ Failed to process audio or no speech detected")
            return {lang: ([], ProcessingStats(0, 0, 0, 0)) for lang in languages}

        results = {}
        master_timings = None

        for lang_idx, language in enumerate(languages, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🌐 Language {lang_idx}/{len(languages)}: {language}")
            logger.info(f"{'='*60}")

            lang_start_time = time.time()

            if lang_idx == 1:
                logger.info(
                    f"🎬 Processing {len(chunks)} chunks for {language} (establishing master timings)..."
                )
                all_subtitles = []

                chunk_iter = (
                    tqdm(
                        chunks,
                        desc=f"Transcribing ({language})",
                        disable=logger.level > logging.INFO,
                    )
                    if TQDM_AVAILABLE
                    else chunks
                )

                for start_sample, end_sample in chunk_iter:
                    chunk_start_sec = start_sample / sr
                    audio_chunk = audio[start_sample:end_sample]

                    subtitles = self.transcribe_chunk(
                        audio_chunk, int(sr), chunk_start_sec, language
                    )
                    all_subtitles.extend(subtitles)

                if all_subtitles and self.config.fix_overlaps:
                    all_subtitles = self._fix_overlapping_subtitles(
                        all_subtitles, min_gap=self.config.min_subtitle_gap
                    )
                    logger.info(
                        f"🔧 Fixed overlapping subtitle timestamps for {language}"
                    )

                master_timings = [(sub.start, sub.end) for sub in all_subtitles]
                logger.info(
                    f"📌 Established {len(master_timings)} synchronized timing segments"
                )

            else:
                if master_timings is None:
                    logger.error("❌ Master timings not established")
                    continue

                logger.info(
                    f"🎬 Processing with {len(master_timings)} synchronized segments for {language}..."
                )
                all_subtitles = []

                segment_iter = (
                    tqdm(
                        master_timings,
                        desc=f"Transcribing ({language})",
                        disable=logger.level > logging.INFO,
                    )
                    if TQDM_AVAILABLE
                    else master_timings
                )

                for seg_idx, (seg_start, seg_end) in enumerate(segment_iter):
                    start_sample = int(seg_start * sr)
                    end_sample = int(seg_end * sr)

                    audio_segment = audio[start_sample:end_sample]

                    segment_subtitles = self.transcribe_chunk(
                        audio_segment, int(sr), seg_start, language
                    )

                    if segment_subtitles:
                        combined_text = " ".join(sub.text for sub in segment_subtitles)
                        avg_confidence = None
                        if segment_subtitles[0].confidence is not None:
                            confidences = [
                                s.confidence for s in segment_subtitles if s.confidence
                            ]
                            avg_confidence = (
                                sum(confidences) / len(confidences)
                                if confidences
                                else None
                            )

                        synced_subtitle = Subtitle(
                            index=seg_idx + 1,
                            start=seg_start,
                            end=seg_end,
                            text=combined_text.strip(),
                            confidence=avg_confidence,
                            words=None,
                        )
                        all_subtitles.append(synced_subtitle)
                    else:
                        synced_subtitle = Subtitle(
                            index=seg_idx + 1,
                            start=seg_start,
                            end=seg_end,
                            text="[No speech detected]",
                            confidence=None,
                            words=None,
                        )
                        all_subtitles.append(synced_subtitle)

                logger.info(
                    f"✅ Synchronized {len(all_subtitles)} subtitle segments for {language}"
                )

            for idx, sub in enumerate(all_subtitles, 1):
                sub.index = idx

            lang_processing_time = time.time() - lang_start_time
            avg_confidence = None

            if all_subtitles and all_subtitles[0].confidence is not None:
                confidences = [s.confidence for s in all_subtitles if s.confidence]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else None
                )

            stats = ProcessingStats(
                total_duration=audio_duration,
                processing_time=lang_processing_time,
                chunks_processed=(
                    len(chunks)
                    if lang_idx == 1
                    else (len(master_timings) if master_timings else 0)
                ),
                subtitles_generated=len(all_subtitles),
                avg_confidence=avg_confidence,
                detected_language=language,
            )

            logger.info(
                f"✅ Generated {len(all_subtitles)} subtitles for {language} "
                f"in {lang_processing_time:.2f}s"
            )
            logger.info(f"   Speed: {stats.speed_ratio():.2f}x realtime")
            if avg_confidence:
                logger.info(f"   Avg confidence: {avg_confidence:.3f}")

            results[language] = (all_subtitles, stats)

        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Multilingual generation complete!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Languages processed: {len(results)}/{len(languages)}")
        if master_timings:
            logger.info(
                f"   Synchronization: All languages use {len(master_timings)} matching time segments"
            )
        logger.info(f"{'='*60}\n")

        return results

    def export(
        self,
        subtitles: List[Subtitle],
        output_path: str,
        stats: Optional[ProcessingStats] = None,
        format: str = "srt",
    ) -> bool:
        if format == "all":
            results = ExporterFactory.export_all_formats(subtitles, output_path, stats)
            return all(results.values())
        else:
            exporter = ExporterFactory.get_exporter(format)
            return exporter.export(subtitles, output_path, stats)

    def export_multilingual(
        self,
        results: Dict[str, Tuple[List[Subtitle], ProcessingStats]],
        output_base_path: str,
        format: str = "srt",
    ) -> Dict[str, bool]:
        export_results = {}

        for language, (subtitles, stats) in results.items():
            if not subtitles:
                logger.warning(f"⚠️  No subtitles to export for language: {language}")
                export_results[language] = False
                continue

            lang_output_path = f"{output_base_path}.{language}"

            if format == "all":
                success = self.export(subtitles, lang_output_path, stats, format="all")
            else:
                lang_output_path += f".{format}"
                success = self.export(subtitles, lang_output_path, stats, format=format)

            export_results[language] = success

            if success:
                logger.info(f"💾 Exported {language} subtitles to: {lang_output_path}")
            else:
                logger.error(f"❌ Failed to export {language} subtitles")

        return export_results

    def batch_process(
        self,
        file_paths: List[str],
        output_dir: str,
        output_format: str = "srt",
        language: Optional[str] = None,
    ) -> Dict[str, bool]:
        ensure_directory(output_dir)
        results = {}

        logger.info(f"📦 Batch processing {len(file_paths)} files...")

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"File {i}/{len(file_paths)}: {Path(file_path).name}")
            logger.info(f"{'='*60}")

            try:
                subtitles, stats = self.generate(file_path, language)

                if subtitles:
                    base_name = Path(file_path).stem
                    output_path = str(Path(output_dir) / f"{base_name}.{output_format}")

                    success = self.export(
                        subtitles, output_path, stats, format=output_format
                    )
                    results[file_path] = success
                else:
                    logger.warning(f"⚠️  No subtitles generated for {file_path}")
                    results[file_path] = False

            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                results[file_path] = False

        success_count = sum(results.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Batch complete: {success_count}/{len(file_paths)} successful")
        logger.info(f"{'='*60}")

        return results
