import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .config import (
    AVAILABLE_COMPUTE_TYPES,
    AVAILABLE_DEVICES,
    AVAILABLE_MODELS,
    AVAILABLE_OUTPUT_FORMATS,
    DEFAULT_MIN_SILENCE,
    DEFAULT_SILENCE_THRESH,
    MAX_CHUNK_DURATION,
    TARGET_CHUNK_DURATION,
    Config,
)
from .subtitle_generator import SubtitleGenerator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False):
    level = logging.INFO

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-subtitle",
        description="🎬 Advanced Subtitle Generator with Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  gen-subtitle video.mp4
  
  # Specify language and model
  gen-subtitle video.mp4 --language en --model base
  
  # Enable noise reduction
  gen-subtitle noisy_audio.mp3 --noise-reduction
  
  # GPU acceleration with high quality
  gen-subtitle video.mp4 --model large-v3 --device cuda --compute-type float16
  
  # Word-level timestamps in JSON format
  gen-subtitle video.mp4 --word-timestamps --format json
  
  # Batch processing
  gen-subtitle *.mp4 --output-dir ./subtitles
  
  # Export all formats
  gen-subtitle video.mp4 --format all
        """,
    )

    # ========================================================================
    # Input/Output Arguments
    # ========================================================================

    parser.add_argument(
        "media_files",
        type=str,
        nargs="+",
        help="Path(s) to media file(s) (audio or video)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (for single file) or directory (for batch)",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=AVAILABLE_OUTPUT_FORMATS,
        default="srt",
        help="Output format (default: srt)",
    )

    # ========================================================================
    # Model Configuration
    # ========================================================================

    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        "-m",
        "--model",
        type=str,
        default="large-v3",
        choices=AVAILABLE_MODELS,
        help="Whisper model size (default: large-v3)",
    )

    model_group.add_argument(
        "-l",
        "--language",
        type=str,
        help="Language code (en, es, fr, etc.) or None for auto-detect",
    )

    model_group.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        choices=AVAILABLE_DEVICES,
        help="Device to use (default: auto)",
    )

    model_group.add_argument(
        "-c",
        "--compute-type",
        type=str,
        default="float16",
        choices=AVAILABLE_COMPUTE_TYPES,
        help="Compute type (default: float16)",
    )

    model_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    # ========================================================================
    # Audio Processing Arguments
    # ========================================================================

    audio_group = parser.add_argument_group("Audio Processing")

    audio_group.add_argument(
        "--silence-thresh",
        type=int,
        default=DEFAULT_SILENCE_THRESH,
        help=f"Silence threshold in dBFS (default: {DEFAULT_SILENCE_THRESH})",
    )

    audio_group.add_argument(
        "--min-silence",
        type=int,
        default=DEFAULT_MIN_SILENCE,
        help=f"Minimum silence length in ms (default: {DEFAULT_MIN_SILENCE})",
    )

    audio_group.add_argument(
        "--noise-reduction",
        action="store_true",
        help="Apply noise reduction (requires noisereduce package)",
    )

    audio_group.add_argument(
        "--max-chunk-duration",
        type=float,
        default=MAX_CHUNK_DURATION,
        help=f"Maximum chunk duration in seconds (default: {MAX_CHUNK_DURATION})",
    )

    audio_group.add_argument(
        "--target-chunk-duration",
        type=float,
        default=TARGET_CHUNK_DURATION,
        help=f"Target chunk duration when splitting (default: {TARGET_CHUNK_DURATION})",
    )

    # ========================================================================
    # Feature Arguments
    # ========================================================================

    feature_group = parser.add_argument_group("Features")

    feature_group.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Generate word-level timestamps (JSON format only)",
    )

    feature_group.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable chunk merging (process all chunks separately)",
    )

    feature_group.add_argument(
        "--no-fix-overlaps",
        action="store_true",
        help="Disable automatic fixing of overlapping subtitle timestamps",
    )

    feature_group.add_argument(
        "--min-subtitle-gap",
        type=float,
        default=0,
        help="Minimum gap between subtitles in seconds (default: 0)",
    )

    # ========================================================================
    # Logging Arguments
    # ========================================================================

    logging_group = parser.add_argument_group("Logging")

    logging_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    logging_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output except errors",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    if args.noise_reduction:
        try:
            import noisereduce
        except ImportError:
            logger.error("❌ Noise reduction requires 'noisereduce' package")
            logger.error("   Install with: pip install noisereduce")
            return False

    if args.word_timestamps and args.format not in ["json", "all"]:
        logger.warning("⚠️  Word timestamps only available in JSON format")

    return True


def process_single_file(
    generator: SubtitleGenerator,
    media_path: str,
    args: argparse.Namespace,
) -> bool:
    subtitles, stats = generator.generate(media_path, language=args.language)

    if not subtitles:
        logger.error("❌ No subtitles generated")
        return False

    if args.output:
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
    else:
        output_base = Path(media_path).stem
        output_dir = Path(media_path).parent

    if args.format == "all":
        output_path = str(output_dir / output_base)
        success = generator.export(subtitles, output_path, stats, format="all")
    else:
        output_path = str(output_dir / f"{output_base}.{args.format}")
        success = generator.export(subtitles, output_path, stats, format=args.format)

    return success


def process_batch(
    generator: SubtitleGenerator,
    file_paths: List[str],
    args: argparse.Namespace,
) -> bool:
    output_dir = args.output or "./subtitles"

    results = generator.batch_process(
        file_paths,
        output_dir,
        output_format=args.format,
        language=args.language,
    )

    return all(results.values())


def main(argv: List[str] | None = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    if not validate_arguments(args):
        return 1

    logger.info(f"{'='*60}")
    logger.info("🎬 Advanced Subtitle Generator")
    logger.info(f"{'='*60}\n")

    try:
        config = Config(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            num_workers=args.workers,
            silence_thresh=args.silence_thresh,
            min_silence_len=args.min_silence,
            max_chunk_duration=args.max_chunk_duration,
            target_chunk_duration=args.target_chunk_duration,
            merge_chunks=not args.no_merge,
            word_timestamps=args.word_timestamps,
            use_noise_reduction=args.noise_reduction,
            language=args.language,
            fix_overlaps=not args.no_fix_overlaps,
            min_subtitle_gap=args.min_subtitle_gap,
        )

        generator = SubtitleGenerator(config)

        is_batch = len(args.media_files) > 1

        if is_batch:
            success = process_batch(generator, args.media_files, args)
        else:
            success = process_single_file(generator, args.media_files[0], args)

        if success:
            logger.info(f"{'='*60}")
            logger.info("✨ All done!")
            logger.info(f"{'='*60}")
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        logger.error("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
