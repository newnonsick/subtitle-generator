import logging
from typing import List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    import soundfile as sf

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import noisereduce as nr

    NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None
    NOISEREDUCE_AVAILABLE = False

from .config import (
    DEFAULT_KEEP_SILENCE,
    DEFAULT_MAX_GAP_MS,
    DEFAULT_MIN_SILENCE,
    DEFAULT_SILENCE_THRESH,
    MIN_CHUNK_DURATION,
    WHISPER_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, use_noise_reduction: bool = False):
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for audio processing")

        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa and soundfile are required for audio processing")

        self.use_noise_reduction = use_noise_reduction and NOISEREDUCE_AVAILABLE

        if use_noise_reduction and not NOISEREDUCE_AVAILABLE:
            logger.warning("Noise reduction requested but noisereduce not installed")

    def load_audio(self, file_path: str) -> Optional[Tuple[np.ndarray, float]]:
        try:
            logger.info(f"📁 Loading: {file_path}")

            audio, sr = librosa.load(
                file_path,
                sr=WHISPER_SAMPLE_RATE,
                mono=True,
                res_type="kaiser_fast",
            )

            duration = len(audio) / sr
            logger.info(f"✅ Loaded: {duration:.2f}s, {sr}Hz, mono")

            return audio, float(sr)
        except Exception as e:
            logger.error(f"❌ Failed to load audio: {e}")
            return None

    def apply_noise_reduction(
        self, audio: np.ndarray, sr: float, prop_decrease: float = 0.8
    ) -> np.ndarray:
        if not self.use_noise_reduction or not NOISEREDUCE_AVAILABLE:
            return audio

        try:
            if nr is not None:
                reduced = nr.reduce_noise(
                    y=audio,
                    sr=int(sr),
                    stationary=True,
                    prop_decrease=prop_decrease,
                )
                logger.info("🔇 Applied noise reduction")
                return reduced
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")

        return audio

    def detect_speech_chunks(
        self,
        audio: np.ndarray,
        sr: int,
        min_silence_len: int = DEFAULT_MIN_SILENCE,
        silence_thresh: int = DEFAULT_SILENCE_THRESH,
        keep_silence: int = DEFAULT_KEEP_SILENCE,
    ) -> List[Tuple[int, int]]:
        logger.info(
            f"🔍 Detecting speech chunks (silence_thresh={silence_thresh}dBFS)..."
        )

        min_silence_samples = int(min_silence_len * sr / 1000)
        keep_silence_samples = int(keep_silence * sr / 1000)

        frame_length = int(sr * 0.025)
        hop_length = int(sr * 0.010)

        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        energy_db = librosa.amplitude_to_db(energy, ref=np.max)

        is_speech = energy_db > silence_thresh

        frame_to_sample = lambda f: f * hop_length

        chunks = []
        in_speech = False
        start_frame = 0
        silence_counter = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    start_frame = i
                    in_speech = True
                silence_counter = 0
            else:
                if in_speech:
                    silence_counter += 1
                    if silence_counter * hop_length >= min_silence_samples:
                        end_frame = i - silence_counter
                        start_sample = max(
                            0, frame_to_sample(start_frame) - keep_silence_samples
                        )
                        end_sample = min(
                            len(audio),
                            frame_to_sample(end_frame) + keep_silence_samples,
                        )

                        if end_sample - start_sample > sr * MIN_CHUNK_DURATION:
                            chunks.append((int(start_sample), int(end_sample)))

                        in_speech = False
                        silence_counter = 0

        if in_speech:
            end_sample = min(
                len(audio), frame_to_sample(len(is_speech)) + keep_silence_samples
            )
            start_sample = max(0, frame_to_sample(start_frame) - keep_silence_samples)
            if end_sample - start_sample > sr * MIN_CHUNK_DURATION:
                chunks.append((int(start_sample), int(end_sample)))

        if not chunks:
            logger.warning("⚠️  No speech detected")
            return []

        logger.info(f"✅ Found {len(chunks)} speech chunks")
        return chunks

    def merge_nearby_chunks(
        self,
        chunks: List[Tuple[int, int]],
        sr: int,
        max_gap_ms: int = DEFAULT_MAX_GAP_MS,
    ) -> List[Tuple[int, int]]:
        if not chunks:
            return []

        max_gap_samples = int(max_gap_ms * sr / 1000)
        merged = [chunks[0]]

        for start, end in chunks[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap_samples:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        if len(merged) < len(chunks):
            logger.info(f"🔗 Merged {len(chunks)} chunks → {len(merged)} chunks")

        return merged

    def split_long_chunks(
        self,
        chunks: List[Tuple[int, int]],
        audio: np.ndarray,
        sr: float,
        max_duration: float = 30.0,
        target_duration: float = 20.0,
    ) -> List[Tuple[int, int]]:
        final_chunks = []
        split_count = 0

        for start, end in chunks:
            duration = (end - start) / sr

            if duration <= max_duration:
                final_chunks.append((start, end))
            else:
                split_count += 1
                logger.debug(f"   Splitting {duration:.1f}s chunk at silent parts")

                chunk_audio = audio[start:end]

                frame_length = int(sr * 0.025)
                hop_length = int(sr * 0.010)

                energy = librosa.feature.rms(
                    y=chunk_audio, frame_length=frame_length, hop_length=hop_length
                )[0]

                energy_db = librosa.amplitude_to_db(energy, ref=np.max(energy))

                silence_threshold = -40
                is_silent = energy_db < silence_threshold

                min_silence_frames = int(0.2 * sr / hop_length)

                silent_regions = []
                in_silence = False
                silence_start = 0

                for i, silent in enumerate(is_silent):
                    if silent:
                        if not in_silence:
                            silence_start = i
                            in_silence = True
                    else:
                        if in_silence:
                            silence_length = i - silence_start
                            if silence_length >= min_silence_frames:
                                silence_start_sample = start + (
                                    silence_start * hop_length
                                )
                                silence_end_sample = start + (i * hop_length)
                                split_point = (
                                    silence_start_sample + silence_end_sample
                                ) // 2
                                silent_regions.append(split_point)
                            in_silence = False

                if (
                    in_silence
                    and (len(is_silent) - silence_start) >= min_silence_frames
                ):
                    silence_start_sample = start + (silence_start * hop_length)
                    silence_end_sample = end
                    split_point = (silence_start_sample + silence_end_sample) // 2
                    silent_regions.append(split_point)

                if not silent_regions:
                    logger.debug(
                        f"   No clear silence found, using low-energy splitting"
                    )
                    target_samples = int(target_duration * sr)
                    current = start

                    while current < end:
                        if end - current <= target_samples:
                            final_chunks.append((current, end))
                            break

                        ideal_split = current + target_samples
                        search_window = int(sr * 2)
                        search_start = max(
                            current + int(sr * 5), ideal_split - search_window
                        )
                        search_end = min(end, ideal_split + search_window)

                        frame_start = max(0, (search_start - start) // hop_length)
                        frame_end = min(len(energy), (search_end - start) // hop_length)

                        if frame_end > frame_start:
                            local_energies = energy[frame_start:frame_end]
                            if len(local_energies) > 0:
                                min_idx = np.argmin(local_energies)
                                split_pos = start + (
                                    (frame_start + min_idx) * hop_length
                                )
                            else:
                                split_pos = ideal_split
                        else:
                            split_pos = ideal_split

                        final_chunks.append((int(current), int(split_pos)))
                        current = split_pos

                    continue

                num_segments = int(np.ceil(duration / target_duration))
                target_samples = int(target_duration * sr)
                current_start = start

                for seg_idx in range(num_segments - 1):
                    ideal_split = current_start + target_samples

                    min_next_split = current_start + int(5 * sr)

                    valid_silences = [
                        s
                        for s in silent_regions
                        if min_next_split <= s < end and s > current_start
                    ]

                    if valid_silences:
                        split_pos = min(
                            valid_silences, key=lambda x: abs(x - ideal_split)
                        )
                        final_chunks.append((int(current_start), int(split_pos)))
                        current_start = split_pos
                        silent_regions = [s for s in silent_regions if s != split_pos]
                    else:
                        continue

                if current_start < end:
                    final_chunks.append((int(current_start), int(end)))

        if split_count > 0:
            logger.info(
                f"✂️  Split {split_count} long chunks at silent parts → {len(final_chunks)} total chunks"
            )

        return final_chunks
