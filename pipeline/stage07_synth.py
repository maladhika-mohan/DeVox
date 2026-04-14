"""
Stage 07 — Instrument Synthesis
Render MIDI to audio using FluidSynth with a SoundFont (.sf2).
This stage is OPTIONAL — it is skipped if FluidSynth or a SoundFont is not available.
"""

import numpy as np
from pathlib import Path

from pipeline.config import (
    OUTPUT_DIR, DEFAULT_SOUNDFONT, SYNTH_SAMPLE_RATE, SYNTH_GAIN,
)
from pipeline.utils import stage, get_logger, save_wav

log = get_logger("Stage 07")

# ── Check FluidSynth availability ────────────────────────────────────────
HAS_FLUIDSYNTH = False
try:
    import fluidsynth
    HAS_FLUIDSYNTH = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if FluidSynth and a SoundFont are available."""
    if not HAS_FLUIDSYNTH:
        log.warning("   pyfluidsynth not installed — skipping synthesis")
        return False
    if not DEFAULT_SOUNDFONT.exists():
        log.warning(f"   SoundFont not found: {DEFAULT_SOUNDFONT}")
        log.warning("   Run setup_soundfont.py to download one, or skip this stage")
        return False
    return True


@stage(7, "Instrument Synthesis (FluidSynth)")
def synthesize_midi(midi_path: Path) -> np.ndarray | None:
    """
    Render a MIDI file to WAV using FluidSynth.

    Parameters
    ----------
    midi_path : Path
        Path to the .mid file from Stage 06.

    Returns
    -------
    np.ndarray or None
        Synthesized audio waveform, or None if FluidSynth unavailable.
    """
    if not is_available():
        log.info("   [>>]  Skipping FluidSynth synthesis (not available)")
        return None

    log.info(f"   SoundFont: {DEFAULT_SOUNDFONT.name}")

    # Initialize FluidSynth
    fs = fluidsynth.Synth(gain=SYNTH_GAIN, samplerate=float(SYNTH_SAMPLE_RATE))
    sfid = fs.sfload(str(DEFAULT_SOUNDFONT))
    fs.program_select(0, sfid, 0, 0)

    # Read MIDI using pretty_midi (bundled with basic-pitch)
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Synthesize using pretty_midi's built-in FluidSynth integration
    audio = midi.fluidsynth(fs=float(SYNTH_SAMPLE_RATE), sf2_path=str(DEFAULT_SOUNDFONT))

    # Normalise
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = (audio / max_val * 0.95).astype(np.float32)

    # Save
    out_path = OUTPUT_DIR / "07_synthesized.wav"
    save_wav(audio, out_path, SYNTH_SAMPLE_RATE)

    fs.delete()
    return audio
