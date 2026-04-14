"""
Stage 01 — Audio Input
Load audio files (MP3, WAV, FLAC, OGG) using librosa.
Converts to mono and resamples to the configured sample rate.
"""

import librosa
import numpy as np
from pathlib import Path

from pipeline.config import SAMPLE_RATE, MONO
from pipeline.utils import stage, get_logger

log = get_logger("Stage 01")


@stage(1, "Audio Input")
def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (audio_array, sample_rate).

    Supports: MP3, WAV, FLAC, OGG
    - Converts to mono if MONO=True in config
    - Resamples to SAMPLE_RATE (default 44100 Hz)

    Parameters
    ----------
    file_path : str or Path
        Path to the input audio file.

    Returns
    -------
    tuple of (np.ndarray, int)
        The audio waveform as a 1-D float32 array and the sample rate.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    supported = {".mp3", ".mpeg", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    if file_path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported format '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )

    log.info(f"   Loading: {file_path.name}")

    # librosa.load handles format detection, resampling, and mono conversion
    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=MONO,
    )

    duration = len(audio) / sr
    log.info(f"   Duration: {duration:.1f}s | Samples: {len(audio):,} | SR: {sr} Hz")

    return audio, sr
