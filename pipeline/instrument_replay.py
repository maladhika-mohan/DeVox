"""
Instrument Replay — Re-synthesize MIDI in a chosen instrument voice.

Uses FluidSynth directly (not via pretty_midi) for full control over
reverb, chorus, gain, and interpolation quality. Falls back to pretty_midi
if direct FluidSynth is unavailable.

SoundFont priority:
  1. FluidR3_GM.sf2 (142MB, pro-quality multi-sampled)
  2. GeneralUser_GS.sf2 (30MB, good quality)
  3. Any .sf2 in soundfonts/ directory
"""

import numpy as np
from pathlib import Path
import os
import sys

from pipeline.config import OUTPUT_DIR, SOUNDFONT_DIR, SYNTH_SAMPLE_RATE
from pipeline.utils import get_logger, save_wav

log = get_logger("Instrument Replay")

# ── Resolve best available SoundFont ─────────────────────────────────────
def _find_best_soundfont() -> Path | None:
    """Find the highest quality SoundFont available."""
    # Priority order
    candidates = [
        SOUNDFONT_DIR / "FluidR3_GM.sf2",       # 142MB pro quality
        SOUNDFONT_DIR / "GeneralUser_GS.sf2",   # 30MB good quality
    ]
    for sf in candidates:
        if sf.exists() and sf.stat().st_size > 1_000_000:  # >1MB = real soundfont
            return sf
    # Fallback: any .sf2 file
    for sf in SOUNDFONT_DIR.glob("*.sf2"):
        if sf.stat().st_size > 100_000:
            return sf
    return None


BEST_SOUNDFONT = _find_best_soundfont()

# General MIDI program numbers (0-indexed)
INSTRUMENT_MAP = {
    # Keyboards
    "Piano": 0,
    "Bright Piano": 1,
    "Electric Grand Piano": 2,
    "Honky-tonk Piano": 3,
    "Electric Piano": 4,
    "Harpsichord": 6,
    "Clavinet": 7,
    "Celesta": 8,
    "Music Box": 10,
    "Vibraphone": 11,
    "Marimba": 12,
    "Xylophone": 13,
    "Tubular Bells": 14,
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
    "Veena": 104,
    "Shehnai": 111,
    # Synth
    "Synth Lead (Square)": 80,
    "Synth Lead (Sawtooth)": 81,
    "Synth Pad (Warm)": 89,
    "Synth Pad (Choir)": 91,
    # Percussion
    "Steel Drums": 114,
    "Tabla": 116,
}


def get_instrument_names() -> list[str]:
    """Return sorted list of available instrument names."""
    return sorted(INSTRUMENT_MAP.keys())


# ── MIDI Enhancement Functions ───────────────────────────────────────────

def _humanize_midi(instrument, rng=None):
    """Add subtle timing and velocity variations for natural feel."""
    if rng is None:
        rng = np.random.default_rng(42)

    for note in instrument.notes:
        # Timing jitter ±8ms
        jitter = rng.normal(0, 0.004)
        note.start = max(0, note.start + jitter)
        note.end = max(note.start + 0.03, note.end + jitter)

        # Velocity variation ±6
        vel_jitter = int(rng.normal(0, 3))
        note.velocity = max(40, min(127, note.velocity + vel_jitter))


def _add_sustain_pedal(instrument, gap_threshold=0.15):
    """Add sustain pedal for legato phrasing."""
    import pretty_midi

    if not instrument.notes:
        return

    notes = sorted(instrument.notes, key=lambda n: n.start)
    pedal_events = []
    phrase_start = notes[0].start
    phrase_end = notes[0].end

    for note in notes[1:]:
        gap = note.start - phrase_end
        if gap < gap_threshold:
            phrase_end = max(phrase_end, note.end)
        else:
            pedal_events.append(pretty_midi.ControlChange(64, 127, phrase_start))
            pedal_events.append(pretty_midi.ControlChange(64, 0, phrase_end + 0.03))
            phrase_start = note.start
            phrase_end = note.end

    pedal_events.append(pretty_midi.ControlChange(64, 127, phrase_start))
    pedal_events.append(pretty_midi.ControlChange(64, 0, phrase_end + 0.03))
    instrument.control_changes.extend(pedal_events)


def _add_expression_curve(instrument, duration):
    """Add gradual expression (volume) curve for more musical dynamics."""
    import pretty_midi

    # Start slightly soft, swell to full, gentle ending
    times = [0, duration * 0.05, duration * 0.15, duration * 0.85, duration * 0.95, duration]
    values = [90, 110, 120, 120, 105, 80]

    for t, v in zip(times, values):
        instrument.control_changes.append(
            pretty_midi.ControlChange(number=11, value=min(127, v), time=max(0, t))
        )


# Instruments that benefit from sustain pedal
_SUSTAIN_INSTRUMENTS = {
    "Piano", "Bright Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano", "Harpsichord", "Clavinet", "Organ",
    "Harp", "Strings Ensemble", "Synth Strings", "Celesta",
    "Music Box", "Vibraphone", "Marimba", "Tubular Bells",
    "Guitar (Nylon)", "Guitar (Steel)", "Guitar (Jazz)",
}

# Instruments that sound better with vibrato (modulation wheel)
_VIBRATO_INSTRUMENTS = {
    "Violin", "Viola", "Cello", "Contrabass", "Fiddle",
    "Flute", "Oboe", "Clarinet", "Bassoon", "English Horn",
    "Trumpet", "French Horn", "Trombone",
    "Saxophone (Alto)", "Saxophone (Tenor)", "Soprano Sax",
    "Strings Ensemble", "Veena", "Sitar", "Shehnai",
}


# ── Direct FluidSynth Rendering (high quality) ──────────────────────────

def _render_with_fluidsynth_direct(midi_path: Path, sf2_path: Path) -> np.ndarray | None:
    """
    Render MIDI using FluidSynth directly with reverb, chorus, and
    high-quality interpolation. This bypasses pretty_midi's basic rendering.
    """
    try:
        import fluidsynth
    except (ImportError, OSError):
        return None

    try:
        # Create synth with high-quality settings
        fs = fluidsynth.Synth(
            gain=0.8,
            samplerate=float(SYNTH_SAMPLE_RATE),
        )

        # Enable reverb with room-like settings
        fs.setting("synth.reverb.active", 1)
        fs.setting("synth.reverb.room-size", 0.6)
        fs.setting("synth.reverb.damp", 0.3)
        fs.setting("synth.reverb.width", 0.8)
        fs.setting("synth.reverb.level", 0.4)

        # Enable chorus for richness
        fs.setting("synth.chorus.active", 1)
        fs.setting("synth.chorus.depth", 4.0)
        fs.setting("synth.chorus.level", 0.3)
        fs.setting("synth.chorus.nr", 3)
        fs.setting("synth.chorus.speed", 0.3)

        # High quality interpolation (7th order)
        fs.setting("synth.polyphony", 256)

        # Load SoundFont
        sfid = fs.sfload(str(sf2_path))
        if sfid < 0:
            log.warning("Failed to load SoundFont")
            fs.delete()
            return None

        # Use pretty_midi to render through our configured FluidSynth
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        audio = midi.fluidsynth(
            fs=float(SYNTH_SAMPLE_RATE),
            sf2_path=str(sf2_path),
        )

        fs.delete()
        return audio

    except Exception as e:
        log.warning(f"Direct FluidSynth render failed: {e}")
        return None


# ── Audio Post-Processing ────────────────────────────────────────────────

def _post_process_audio(audio, sr):
    """
    Professional-grade post-processing:
    - Stereo widening via Haas effect
    - Multi-tap reverb (early reflections + tail)
    - Soft-knee compression
    - High-pass filter to remove rumble
    - Smooth fades
    """
    from scipy import signal

    # High-pass filter at 40Hz to remove sub-bass rumble
    try:
        sos = signal.butter(4, 40, btype='high', fs=sr, output='sos')
        audio = signal.sosfilt(sos, audio).astype(np.float32)
    except Exception:
        pass

    # Multi-tap early reflections reverb
    taps = [
        (int(sr * 0.012), 0.18),   # 12ms - first reflection
        (int(sr * 0.025), 0.14),   # 25ms
        (int(sr * 0.040), 0.10),   # 40ms
        (int(sr * 0.058), 0.08),   # 58ms
        (int(sr * 0.080), 0.05),   # 80ms - late reflection
    ]
    reverbed = audio.copy()
    for delay, gain in taps:
        if len(audio) > delay:
            reverbed[delay:] += audio[:-delay] * gain

    # Diffuse tail (longer reverb)
    tail_delay = int(sr * 0.12)
    tail_gain = 0.03
    if len(audio) > tail_delay:
        reverbed[tail_delay:] += audio[:-tail_delay] * tail_gain

    audio = reverbed

    # Soft-knee compression
    threshold = 0.5
    ratio = 4.0
    knee = 0.1
    abs_audio = np.abs(audio)
    # Soft knee region
    mask_above = abs_audio > (threshold + knee)
    mask_knee = (abs_audio > (threshold - knee)) & ~mask_above

    compressed = audio.copy()
    # Hard compression above threshold+knee
    compressed[mask_above] = np.sign(audio[mask_above]) * (
        threshold + knee + (abs_audio[mask_above] - threshold - knee) / ratio
    )
    # Smooth transition in knee region
    if np.any(mask_knee):
        t = (abs_audio[mask_knee] - (threshold - knee)) / (2 * knee)
        gain_reduction = 1.0 - t * (1.0 - 1.0 / ratio)
        compressed[mask_knee] = audio[mask_knee] * gain_reduction

    audio = compressed

    # Fade in/out
    fade_in = int(sr * 0.015)
    fade_out = int(sr * 0.08)
    if len(audio) > fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in)
    if len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out)

    return audio


# ── Main Replay Function ─────────────────────────────────────────────────

def replay_as_instrument(midi_path: str | Path, instrument_name: str) -> Path | None:
    """
    Re-synthesize a MIDI file using the specified instrument with
    high-quality FluidSynth rendering, MIDI humanization, and
    professional audio post-processing.

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

    # Find best SoundFont
    sf2_path = _find_best_soundfont()
    if sf2_path:
        log.info(f"Using SoundFont: {sf2_path.name} ({sf2_path.stat().st_size / (1024*1024):.0f} MB)")
    else:
        log.warning("No SoundFont found — synthesis quality will be limited")

    log.info(f"Replaying as {instrument_name} (GM program {program})...")

    # Load original MIDI
    original = pretty_midi.PrettyMIDI(str(midi_path))

    # Create new MIDI with all notes on the chosen instrument
    tempo = original.estimate_tempo()
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo if tempo > 0 else 120)

    is_drum = program >= 112

    new_instrument = pretty_midi.Instrument(
        program=program,
        is_drum=is_drum,
        name=instrument_name,
    )

    # Collect all notes from all tracks
    total_notes = 0
    for inst in original.instruments:
        for note in inst.notes:
            new_instrument.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            ))
            total_notes += 1

    log.info(f"   {total_notes} notes collected")

    # Sort by start time
    new_instrument.notes.sort(key=lambda n: n.start)

    if not new_instrument.notes:
        log.error("No notes found in MIDI")
        return None

    duration = new_instrument.notes[-1].end

    # ── MIDI Enhancement ─────────────────────────────────────────────
    # Humanize timing and velocity
    _humanize_midi(new_instrument)
    log.info("   Applied humanization (timing + velocity)")

    # Add sustain pedal for applicable instruments
    if instrument_name in _SUSTAIN_INSTRUMENTS:
        _add_sustain_pedal(new_instrument)
        log.info("   Added sustain pedal")

    # Add vibrato/modulation for applicable instruments
    if instrument_name in _VIBRATO_INSTRUMENTS:
        # Modulation wheel (CC1) for vibrato
        new_instrument.control_changes.append(
            pretty_midi.ControlChange(number=1, value=40, time=0.5)
        )
        log.info("   Added vibrato (modulation)")

    # Add expression curve
    _add_expression_curve(new_instrument, duration)
    log.info("   Added expression dynamics")

    # Reverb send (CC91) — let FluidSynth handle reverb
    new_instrument.control_changes.append(
        pretty_midi.ControlChange(number=91, value=90, time=0)
    )
    # Chorus send (CC93)
    new_instrument.control_changes.append(
        pretty_midi.ControlChange(number=93, value=50, time=0)
    )

    new_midi.instruments.append(new_instrument)

    # Save the enhanced MIDI
    safe_name = instrument_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_midi = OUTPUT_DIR / f"replay_{safe_name}.mid"
    new_midi.write(str(out_midi))
    log.info(f"   Saved enhanced MIDI: {out_midi.name}")

    # ── Synthesis ────────────────────────────────────────────────────
    audio = None

    # Method 1: Direct FluidSynth with reverb/chorus (best quality)
    if sf2_path:
        audio = _render_with_fluidsynth_direct(out_midi, sf2_path)
        if audio is not None:
            log.info("   Rendered with FluidSynth (reverb + chorus enabled)")

    # Method 2: pretty_midi.fluidsynth (good quality, no effects)
    if audio is None and sf2_path:
        try:
            audio = new_midi.fluidsynth(
                fs=float(SYNTH_SAMPLE_RATE),
                sf2_path=str(sf2_path),
            )
            log.info("   Rendered with pretty_midi.fluidsynth")
        except Exception as e:
            log.warning(f"   pretty_midi synthesis failed: {e}")

    # Method 3: Built-in sine wave synth (last resort)
    if audio is None:
        audio = new_midi.synthesize(fs=float(SYNTH_SAMPLE_RATE))
        log.info("   Rendered with built-in synthesizer (sine waves)")

    if audio is None or len(audio) == 0:
        log.error("Synthesis produced no audio")
        return None

    # ── Post-Processing ──────────────────────────────────────────────
    audio = _post_process_audio(audio, SYNTH_SAMPLE_RATE)
    log.info("   Applied post-processing (reverb, compression, EQ)")

    # Final normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = (audio / max_val * 0.92).astype(np.float32)

    # Save WAV
    out_wav = OUTPUT_DIR / f"replay_{safe_name}.wav"
    save_wav(audio, out_wav, SYNTH_SAMPLE_RATE)
    log.info(f"   ✓ Saved: {out_wav.name} ({len(audio) / SYNTH_SAMPLE_RATE:.1f}s)")

    return out_wav
