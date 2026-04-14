"""
Stage 06 — MIDI Generation
Convert instrumental audio to polyphonic MIDI using Basic Pitch (Spotify).
Basic Pitch uses ONNX runtime for inference.
"""

import numpy as np
from pathlib import Path

from pipeline.config import (
    SAMPLE_RATE, OUTPUT_DIR,
    MIDI_ONSET_THRESHOLD, MIDI_FRAME_THRESHOLD, MIDI_MIN_NOTE_LENGTH,
)
from pipeline.utils import stage, get_logger, save_wav

log = get_logger("Stage 06")


@stage(6, "MIDI Generation (Basic Pitch)")
def generate_midi(audio: np.ndarray, sr: int) -> Path:
    """
    Convert audio to a MIDI file using Basic Pitch.

    Basic Pitch's predict() requires a file path, so we save the
    audio to a temporary WAV first, then run inference.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (typically the instrumental from Stage 05).
    sr : int
        Sample rate.

    Returns
    -------
    Path
        Path to the generated .mid file.
    """
    from basic_pitch.inference import predict

    # Basic Pitch requires a file path — save instrumental to a temp WAV
    temp_wav = OUTPUT_DIR / "06_temp_input.wav"
    save_wav(audio, temp_wav, sr)

    log.info("   Running Basic Pitch inference (ONNX backend)...")

    # predict() returns: (model_output_dict, midi_data, note_events_list)
    model_output, midi_data, note_events = predict(
        audio_path=str(temp_wav),
        onset_threshold=MIDI_ONSET_THRESHOLD,
        frame_threshold=MIDI_FRAME_THRESHOLD,
        minimum_note_length=MIDI_MIN_NOTE_LENGTH,
    )

    # Save MIDI file
    midi_path = OUTPUT_DIR / "06_instrumental.mid"
    midi_data.write(str(midi_path))

    # Log stats
    n_notes = len(note_events)
    if n_notes > 0:
        # note_events is a list of tuples: (start_time, end_time, midi_pitch, amplitude, pitch_bends)
        pitches = [n[2] for n in note_events]
        log.info(f"   [OK] Notes detected: {n_notes}")
        log.info(f"   [OK] MIDI pitch range: {min(pitches)} – {max(pitches)}")
    else:
        log.info("   [!] No notes detected — input may be too noisy or percussive")

    log.info(f"   Saved MIDI → {midi_path.name}")

    # Clean up temp file
    try:
        temp_wav.unlink()
    except OSError:
        pass

    return midi_path
