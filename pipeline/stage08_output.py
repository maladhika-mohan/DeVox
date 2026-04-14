"""
Stage 08 — Final Audio Output
Mix, normalise, and export the final instrumental in WAV and MP3
using pydub + ffmpeg.
"""

import numpy as np
from pathlib import Path
from pydub import AudioSegment

# Point pydub to ffmpeg if not on PATH (common on Windows winget installs)
import shutil
if not shutil.which("ffmpeg"):
    import glob
    _ffmpeg_candidates = glob.glob(
        str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/**/ffmpeg.exe"),
        recursive=True,
    )
    if _ffmpeg_candidates:
        AudioSegment.converter = _ffmpeg_candidates[0]

from pipeline.config import (
    SAMPLE_RATE, OUTPUT_DIR,
    FADE_IN_MS, FADE_OUT_MS,
    OUTPUT_FORMAT_WAV, OUTPUT_FORMAT_MP3,
)
from pipeline.utils import stage, get_logger, save_wav

log = get_logger("Stage 08")


def _numpy_to_audiosegment(audio: np.ndarray, sr: int) -> AudioSegment:
    """Convert a float32 numpy array to a pydub AudioSegment."""
    # Convert float32 [-1, 1] → int16 [-32768, 32767]
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,       # 16-bit
        frame_rate=sr,
        channels=1,           # mono
    )


@stage(8, "Final Audio Output (pydub + ffmpeg)")
def export_final(
    instrumental: np.ndarray,
    sr: int,
    synthesized: np.ndarray | None = None,
) -> dict:
    """
    Produce the final output files.

    Steps:
      1. Normalise the instrumental waveform
      2. Apply fade-in / fade-out
      3. Export as WAV (lossless) + MP3 (portable)

    Parameters
    ----------
    instrumental : np.ndarray
        The vocal-removed instrumental from Stage 05.
    sr : int
        Sample rate.
    synthesized : np.ndarray or None
        Optional FluidSynth-rendered version from Stage 07.

    Returns
    -------
    dict with output file paths.
    """
    outputs = {}

    # ── Normalise ────────────────────────────────────────────────────────
    max_val = np.abs(instrumental).max()
    if max_val > 0:
        instrumental = instrumental / max_val * 0.95
    log.info("   [OK] Normalised instrumental")

    # ── Convert to pydub AudioSegment ────────────────────────────────────
    segment = _numpy_to_audiosegment(instrumental, sr)

    # ── Apply fades ──────────────────────────────────────────────────────
    segment = segment.fade_in(FADE_IN_MS).fade_out(FADE_OUT_MS)
    log.info(f"   [OK] Applied fade-in ({FADE_IN_MS}ms) + fade-out ({FADE_OUT_MS}ms)")

    # ── Export WAV ───────────────────────────────────────────────────────
    wav_path = OUTPUT_DIR / f"final_instrumental.{OUTPUT_FORMAT_WAV}"
    segment.export(str(wav_path), format=OUTPUT_FORMAT_WAV)
    outputs["wav"] = wav_path
    log.info(f"   [OK] Exported: {wav_path.name} ({wav_path.stat().st_size / 1024:.0f} KB)")

    # ── Export MP3 (requires ffmpeg) ─────────────────────────────────────
    mp3_path = OUTPUT_DIR / f"final_instrumental.{OUTPUT_FORMAT_MP3}"
    try:
        segment.export(str(mp3_path), format=OUTPUT_FORMAT_MP3, bitrate="192k")
        outputs["mp3"] = mp3_path
        log.info(f"   [OK] Exported: {mp3_path.name} ({mp3_path.stat().st_size / 1024:.0f} KB)")
    except (FileNotFoundError, OSError):
        log.warning("   [!] ffmpeg not found — skipping MP3 export (WAV still available)")

    # ── Optionally export synthesized version ────────────────────────────
    if synthesized is not None:
        max_val = np.abs(synthesized).max()
        if max_val > 0:
            synthesized = synthesized / max_val * 0.95
        synth_segment = _numpy_to_audiosegment(synthesized, sr)
        synth_segment = synth_segment.fade_in(FADE_IN_MS).fade_out(FADE_OUT_MS)

        synth_wav = OUTPUT_DIR / "final_synthesized.wav"
        synth_segment.export(str(synth_wav), format="wav")
        outputs["synthesized_wav"] = synth_wav
        log.info(f"   [OK] Exported synthesized: {synth_wav.name}")

    return outputs
