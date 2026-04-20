"""
Instrument Replay — Re-synthesize MIDI in a chosen instrument voice.

Uses pretty_midi to rewrite all MIDI tracks to a single instrument program,
then renders via FluidSynth (if available) or pretty_midi's built-in synth.
"""

import numpy as np
from pathlib import Path

from pipeline.config import OUTPUT_DIR, DEFAULT_SOUNDFONT, SYNTH_SAMPLE_RATE
from pipeline.utils import get_logger, save_wav

log = get_logger("Instrument Replay")

# General MIDI program numbers (0-indexed)
INSTRUMENT_MAP = {
    "Piano": 0,
    "Bright Piano": 1,
    "Electric Piano": 4,
    "Harpsichord": 6,
    "Guitar (Nylon)": 24,
    "Guitar (Steel)": 25,
    "Electric Guitar (Clean)": 27,
    "Electric Guitar (Distortion)": 30,
    "Acoustic Bass": 32,
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Harp": 46,
    "Strings Ensemble": 48,
    "Trumpet": 56,
    "Trombone": 57,
    "French Horn": 60,
    "Saxophone (Alto)": 65,
    "Saxophone (Tenor)": 66,
    "Clarinet": 71,
    "Flute": 73,
    "Recorder": 74,
    "Pan Flute": 75,
    "Sitar": 104,
    "Banjo": 105,
    "Shamisen": 106,
    "Kalimba": 108,
    "Bagpipe": 109,
    "Fiddle": 110,
    "Synth Lead (Square)": 80,
    "Synth Lead (Sawtooth)": 81,
    "Synth Pad (Warm)": 89,
    "Organ": 19,
    "Harmonica": 22,
    "Veena": 104,       # Closest GM match — Sitar
    "Tabla": 116,       # Closest GM match — Taiko Drum (percussion)
}


def get_instrument_names() -> list[str]:
    """Return sorted list of available instrument names."""
    return sorted(INSTRUMENT_MAP.keys())


def replay_as_instrument(midi_path: str | Path, instrument_name: str) -> Path | None:
    """
    Re-synthesize a MIDI file using the specified instrument.

    Parameters
    ----------
    midi_path : str or Path
        Path to the source MIDI file (from Stage 06).
    instrument_name : str
        Name of the target instrument (must be a key in INSTRUMENT_MAP).

    Returns
    -------
    Path or None
        Path to the rendered WAV file, or None on failure.
    """
    import pretty_midi

    midi_path = Path(midi_path)
    if not midi_path.exists():
        log.error(f"MIDI file not found: {midi_path}")
        return None

    program = INSTRUMENT_MAP.get(instrument_name)
    if program is None:
        log.error(f"Unknown instrument: {instrument_name}")
        return None

    log.info(f"Replaying as {instrument_name} (GM program {program})...")

    # Load original MIDI
    original = pretty_midi.PrettyMIDI(str(midi_path))

    # Create new MIDI with all notes on the chosen instrument
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=original.estimate_tempo())

    # Determine if this is a percussion instrument
    is_drum = program >= 112  # GM drums/percussion range

    new_instrument = pretty_midi.Instrument(
        program=program,
        is_drum=is_drum,
        name=instrument_name,
    )

    # Collect all notes from all tracks
    for inst in original.instruments:
        for note in inst.notes:
            new_instrument.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            ))

    # Sort notes by start time
    new_instrument.notes.sort(key=lambda n: n.start)
    new_midi.instruments.append(new_instrument)

    # Save the re-instrumented MIDI
    safe_name = instrument_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_midi = OUTPUT_DIR / f"replay_{safe_name}.mid"
    new_midi.write(str(out_midi))
    log.info(f"Saved re-instrumented MIDI: {out_midi.name}")

    # Synthesize to audio
    audio = None

    # Try FluidSynth with SoundFont first
    if DEFAULT_SOUNDFONT.exists():
        try:
            log.info(f"Synthesizing with SoundFont: {DEFAULT_SOUNDFONT.name}")
            audio = new_midi.fluidsynth(
                fs=float(SYNTH_SAMPLE_RATE),
                sf2_path=str(DEFAULT_SOUNDFONT),
            )
        except Exception as e:
            log.warning(f"FluidSynth synthesis failed: {e}")
            audio = None

    # Fallback: pretty_midi's built-in sine-wave synth
    if audio is None:
        log.info("Using built-in synthesizer (sine wave approximation)")
        audio = new_midi.synthesize(fs=float(SYNTH_SAMPLE_RATE))

    if audio is None or len(audio) == 0:
        log.error("Synthesis produced no audio")
        return None

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = (audio / max_val * 0.95).astype(np.float32)

    # Save WAV
    out_wav = OUTPUT_DIR / f"replay_{safe_name}.wav"
    save_wav(audio, out_wav, SYNTH_SAMPLE_RATE)
    log.info(f"Saved: {out_wav.name} ({len(audio) / SYNTH_SAMPLE_RATE:.1f}s)")

    return out_wav
