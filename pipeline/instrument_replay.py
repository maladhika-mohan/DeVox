"""
Instrument Replay — Re-synthesize MIDI in a chosen instrument voice.

Uses pretty_midi to rewrite all MIDI tracks to a single instrument program,
then renders via FluidSynth with a high-quality SoundFont.
Applies sustain pedal, velocity humanization, and audio post-processing
for more natural-sounding output.
"""

import numpy as np
from pathlib import Path

from pipeline.config import OUTPUT_DIR, DEFAULT_SOUNDFONT, SYNTH_SAMPLE_RATE
from pipeline.utils import get_logger, save_wav

log = get_logger("Instrument Replay")

# General MIDI program numbers (0-indexed)
# Grouped by instrument family for better UX
INSTRUMENT_MAP = {
    # Keyboards
    "Piano": 0,
    "Bright Piano": 1,
    "Electric Grand Piano": 2,
    "Honky-tonk Piano": 3,
    "Electric Piano": 4,
    "Harpsichord": 6,
    "Clavinet": 7,
    "Organ": 19,
    "Accordion": 21,
    "Harmonica": 22,
    # Guitars
    "Guitar (Nylon)": 24,
    "Guitar (Steel)": 25,
    "Guitar (Jazz)": 26,
    "Electric Guitar (Clean)": 27,
    "Electric Guitar (Muted)": 28,
    "Electric Guitar (Overdrive)": 29,
    "Electric Guitar (Distortion)": 30,
    # Bass
    "Acoustic Bass": 32,
    "Electric Bass (Finger)": 33,
    "Electric Bass (Slap)": 36,
    "Synth Bass": 38,
    # Strings
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Contrabass": 43,
    "Harp": 46,
    "Strings Ensemble": 48,
    "Synth Strings": 50,
    "Fiddle": 110,
    # Brass
    "Trumpet": 56,
    "Trombone": 57,
    "Tuba": 58,
    "French Horn": 60,
    "Brass Section": 61,
    # Woodwinds
    "Soprano Sax": 64,
    "Saxophone (Alto)": 65,
    "Saxophone (Tenor)": 66,
    "Baritone Sax": 67,
    "Oboe": 68,
    "English Horn": 69,
    "Bassoon": 70,
    "Clarinet": 71,
    "Piccolo": 72,
    "Flute": 73,
    "Recorder": 74,
    "Pan Flute": 75,
    # Ethnic / World
    "Sitar": 104,
    "Banjo": 105,
    "Shamisen": 106,
    "Koto": 107,
    "Kalimba": 108,
    "Bagpipe": 109,
    "Veena": 104,        # Closest GM match — Sitar family
    "Shehnai": 111,      # GM Shanai
    # Synth
    "Synth Lead (Square)": 80,
    "Synth Lead (Sawtooth)": 81,
    "Synth Pad (Warm)": 89,
    "Synth Pad (Choir)": 91,
    # Percussion (note: these use channel 10 in GM)
    "Tabla": 116,
    "Steel Drums": 114,
    "Marimba": 12,
    "Xylophone": 13,
    "Vibraphone": 11,
    "Celesta": 8,
    "Music Box": 10,
    "Tubular Bells": 14,
}


def get_instrument_names() -> list[str]:
    """Return sorted list of available instrument names."""
    return sorted(INSTRUMENT_MAP.keys())


def _humanize_midi(instrument, rng=None):
    """
    Add subtle timing and velocity variations to make MIDI sound more natural.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    for note in instrument.notes:
        # Slight timing jitter (±10ms) — makes it feel less robotic
        jitter = rng.normal(0, 0.005)
        note.start = max(0, note.start + jitter)
        note.end = max(note.start + 0.02, note.end + jitter)

        # Velocity variation (±8) — adds dynamics
        vel_jitter = int(rng.normal(0, 4))
        note.velocity = max(30, min(127, note.velocity + vel_jitter))


def _add_sustain_pedal(instrument, pedal_threshold=0.25):
    """
    Add sustain pedal (CC64) events for sustained instruments like piano/strings.
    Groups notes into phrases and adds pedal hold during each phrase.
    """
    import pretty_midi

    if not instrument.notes:
        return

    notes = sorted(instrument.notes, key=lambda n: n.start)
    pedal_events = []
    phrase_start = notes[0].start
    phrase_end = notes[0].end

    for note in notes[1:]:
        gap = note.start - phrase_end
        if gap < pedal_threshold:
            # Continue the phrase
            phrase_end = max(phrase_end, note.end)
        else:
            # End current phrase pedal, start new one
            pedal_events.append(
                pretty_midi.ControlChange(number=64, value=127, time=phrase_start)
            )
            pedal_events.append(
                pretty_midi.ControlChange(number=64, value=0, time=phrase_end + 0.05)
            )
            phrase_start = note.start
            phrase_end = note.end

    # Final phrase
    pedal_events.append(
        pretty_midi.ControlChange(number=64, value=127, time=phrase_start)
    )
    pedal_events.append(
        pretty_midi.ControlChange(number=64, value=0, time=phrase_end + 0.05)
    )

    instrument.control_changes.extend(pedal_events)


def _post_process_audio(audio, sr):
    """
    Apply light post-processing: soft compression, gentle reverb via convolution,
    and fade in/out for a polished sound.
    """
    # Soft knee compression — tame peaks, bring up quiet parts
    threshold = 0.6
    ratio = 3.0
    mask = np.abs(audio) > threshold
    compressed = audio.copy()
    compressed[mask] = np.sign(audio[mask]) * (
        threshold + (np.abs(audio[mask]) - threshold) / ratio
    )
    audio = compressed

    # Simple reverb via feedback delay (lightweight, no extra deps)
    delay_samples = int(sr * 0.03)   # 30ms early reflection
    decay = 0.15
    reverbed = audio.copy()
    if len(audio) > delay_samples:
        reverbed[delay_samples:] += audio[:-delay_samples] * decay
    # Second tap
    delay2 = int(sr * 0.07)
    if len(audio) > delay2:
        reverbed[delay2:] += audio[:-delay2] * (decay * 0.5)

    audio = reverbed

    # Fade in/out
    fade_in = int(sr * 0.01)   # 10ms
    fade_out = int(sr * 0.05)  # 50ms
    if len(audio) > fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in)
    if len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out)

    return audio


# Instruments that benefit from sustain pedal
_SUSTAIN_INSTRUMENTS = {
    "Piano", "Bright Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano", "Harpsichord", "Clavinet", "Organ",
    "Harp", "Strings Ensemble", "Synth Strings", "Celesta",
    "Music Box", "Vibraphone", "Marimba", "Tubular Bells",
    "Guitar (Nylon)", "Guitar (Steel)", "Guitar (Jazz)",
}


def replay_as_instrument(midi_path: str | Path, instrument_name: str) -> Path | None:
    """
    Re-synthesize a MIDI file using the specified instrument with
    humanization, sustain pedal, and audio post-processing.

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
    tempo = original.estimate_tempo()
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo if tempo > 0 else 120)

    # Determine if this is a percussion instrument
    is_drum = program >= 112

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

    # Humanize — add subtle timing/velocity variations
    _humanize_midi(new_instrument)

    # Add sustain pedal for applicable instruments
    if instrument_name in _SUSTAIN_INSTRUMENTS:
        _add_sustain_pedal(new_instrument)
        log.info("   Added sustain pedal")

    # Add expression (CC11) — slight volume swell for realism
    if not is_drum:
        new_instrument.control_changes.append(
            pretty_midi.ControlChange(number=11, value=110, time=0)
        )
        # Reverb send (CC91)
        new_instrument.control_changes.append(
            pretty_midi.ControlChange(number=91, value=80, time=0)
        )

    new_midi.instruments.append(new_instrument)

    # Save the re-instrumented MIDI
    safe_name = instrument_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_midi = OUTPUT_DIR / f"replay_{safe_name}.mid"
    new_midi.write(str(out_midi))
    log.info(f"Saved re-instrumented MIDI: {out_midi.name}")

    # Synthesize to audio
    audio = None

    # Try FluidSynth with SoundFont
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

    # Fallback: pretty_midi's built-in synth
    if audio is None:
        log.info("Using built-in synthesizer (sine wave approximation)")
        audio = new_midi.synthesize(fs=float(SYNTH_SAMPLE_RATE))

    if audio is None or len(audio) == 0:
        log.error("Synthesis produced no audio")
        return None

    # Post-process: compression, reverb, fades
    audio = _post_process_audio(audio, SYNTH_SAMPLE_RATE)

    # Final normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = (audio / max_val * 0.95).astype(np.float32)

    # Save WAV
    out_wav = OUTPUT_DIR / f"replay_{safe_name}.wav"
    save_wav(audio, out_wav, SYNTH_SAMPLE_RATE)
    log.info(f"Saved: {out_wav.name} ({len(audio) / SYNTH_SAMPLE_RATE:.1f}s)")

    return out_wav
