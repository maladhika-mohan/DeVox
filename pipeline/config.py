"""
Central configuration for the DeVox pipeline.
All tuneable parameters live here so stages stay clean.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
SOUNDFONT_DIR = PROJECT_ROOT / "soundfonts"
DEFAULT_SOUNDFONT = SOUNDFONT_DIR / "GeneralUser_GS.sf2"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SOUNDFONT_DIR.mkdir(exist_ok=True)

# ── Audio defaults ───────────────────────────────────────────────────────
SAMPLE_RATE = 44100          # Hz — standard CD quality
MONO = True                  # Convert to mono for processing
HOP_LENGTH = 512             # librosa hop length
N_FFT = 2048                 # FFT window size
N_MFCC = 13                  # Number of MFCC coefficients

# ── Preprocessing ────────────────────────────────────────────────────────
HIGHPASS_FREQ = 30           # Hz — remove sub-bass rumble below this
TRIM_TOP_DB = 25             # dB threshold for silence trimming

# ── Pitch detection (CREPE / torchcrepe) ─────────────────────────────────
CREPE_MODEL = "full"         # "tiny", "small", "medium", "large", "full"
CREPE_STEP_SIZE = 10         # ms per frame
CREPE_CONFIDENCE_THRESHOLD = 0.5

# ── Source separation (Demucs) ───────────────────────────────────────────
DEMUCS_MODEL = "htdemucs"     # Hybrid transformer (faster than htdemucs_ft, still good quality)
DEMUCS_DEVICE = None          # None = auto-detect (cuda if available, else cpu)
DEMUCS_SHIFTS = 0             # 0 = no random shifts (fastest), increase for quality (max 5)
DEMUCS_OVERLAP = 0.1          # Lower overlap = faster processing

# ── MIDI generation (Basic Pitch) ────────────────────────────────────────
MIDI_ONSET_THRESHOLD = 0.5
MIDI_FRAME_THRESHOLD = 0.3
MIDI_MIN_NOTE_LENGTH = 58     # ms

# ── FluidSynth ───────────────────────────────────────────────────────────
SYNTH_SAMPLE_RATE = 44100
SYNTH_GAIN = 1.0

# ── Output ───────────────────────────────────────────────────────────────
OUTPUT_FORMAT_WAV = "wav"
OUTPUT_FORMAT_MP3 = "mp3"
FADE_IN_MS = 100              # milliseconds
FADE_OUT_MS = 500             # milliseconds
