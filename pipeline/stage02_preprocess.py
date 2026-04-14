"""
Stage 02 — Preprocessing
Normalise amplitude, trim silence, apply high-pass filter,
and perform beat/onset detection using librosa.
(aubio is optional — used if available, otherwise pure-librosa fallback.)
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from pipeline.config import (
    SAMPLE_RATE, HOP_LENGTH, HIGHPASS_FREQ, TRIM_TOP_DB, OUTPUT_DIR,
)
from pipeline.utils import stage, get_logger, save_wav

log = get_logger("Stage 02")

# ── Try importing aubio (optional on Windows) ───────────────────────────
try:
    import aubio
    HAS_AUBIO = True
    log.info("   aubio available — using for onset/beat detection")
except ImportError:
    HAS_AUBIO = False
    log.info("   aubio not found — falling back to librosa for onset/beat detection")


def _highpass_filter(audio: np.ndarray, sr: int, cutoff: float = HIGHPASS_FREQ) -> np.ndarray:
    """Apply a 4th-order Butterworth high-pass filter."""
    sos = butter(4, cutoff, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _detect_onsets_aubio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Onset detection using aubio."""
    hop = HOP_LENGTH
    onset_detector = aubio.onset("default", 2048, hop, sr)
    onsets = []
    for i in range(0, len(audio) - hop, hop):
        frame = audio[i : i + hop].astype(np.float32)
        if onset_detector(frame):
            onsets.append(onset_detector.get_last())
    return np.array(onsets) / sr  # convert to seconds


def _detect_beats_aubio(audio: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Beat tracking using aubio. Returns (tempo_bpm, beat_times_seconds)."""
    hop = HOP_LENGTH
    tempo_detector = aubio.tempo("default", 2048, hop, sr)
    beats = []
    for i in range(0, len(audio) - hop, hop):
        frame = audio[i : i + hop].astype(np.float32)
        if tempo_detector(frame):
            beats.append(tempo_detector.get_last())
    bpm = tempo_detector.get_bpm()
    return bpm, np.array(beats) / sr


@stage(2, "Preprocessing")
def preprocess(audio: np.ndarray, sr: int) -> dict:
    """
    Preprocess the raw audio signal.

    Steps:
      1. Normalise amplitude to [-1, 1]
      2. Trim leading/trailing silence
      3. High-pass filter (remove <30 Hz rumble)
      4. Beat tracking (tempo + beat positions)
      5. Onset detection

    Returns
    -------
    dict with keys:
        audio      — preprocessed waveform (np.ndarray)
        sr         — sample rate
        tempo      — estimated BPM (float)
        beats      — beat times in seconds (np.ndarray)
        onsets     — onset times in seconds (np.ndarray)
    """

    # 1. Normalise
    audio = librosa.util.normalize(audio)
    log.info("   [OK] Normalised amplitude")

    # 2. Trim silence
    audio, trim_idx = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    trimmed_dur = len(audio) / sr
    log.info(f"   [OK] Trimmed silence → {trimmed_dur:.1f}s remaining")

    # 3. High-pass filter
    audio = _highpass_filter(audio, sr)
    log.info(f"   [OK] High-pass filter at {HIGHPASS_FREQ} Hz")

    # 4. Beat tracking
    if HAS_AUBIO:
        tempo, beats = _detect_beats_aubio(audio, sr)
    else:
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=HOP_LENGTH)
        # librosa may return tempo as ndarray; extract scalar
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    log.info(f"   [OK] Tempo: {tempo:.1f} BPM | Beats detected: {len(beats)}")

    # 5. Onset detection
    if HAS_AUBIO:
        onsets = _detect_onsets_aubio(audio, sr)
    else:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=HOP_LENGTH)
        onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    log.info(f"   [OK] Onsets detected: {len(onsets)}")

    # Save preprocessed audio
    out_path = OUTPUT_DIR / "01_preprocessed.wav"
    save_wav(audio, out_path, sr)

    return {
        "audio": audio,
        "sr": sr,
        "tempo": float(tempo),
        "beats": beats,
        "onsets": onsets,
    }
