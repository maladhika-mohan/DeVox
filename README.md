# 🎵 DeVox — AI-Powered Vocal Removal & Instrumental Extraction

DeVox is an 8-stage audio processing pipeline that removes vocals from music and extracts individual instrument stems using state-of-the-art AI models. It ships with a Streamlit web UI for easy drag-and-drop usage.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## What It Does

- Removes vocals from any audio file (MP3, WAV, FLAC, OGG, M4A)
- Separates audio into 4 stems: **drums**, **bass**, **other instruments**, **vocals**
- Generates a combined instrumental mix
- Optionally transcribes audio to MIDI
- Optionally synthesizes MIDI back to audio via FluidSynth
- Exports final output as WAV and MP3

## Pipeline Stages

| Stage | Name | Tool | Description |
|-------|------|------|-------------|
| 01 | Audio Input | librosa | Load and decode audio files |
| 02 | Preprocessing | librosa | Normalize, trim silence, high-pass filter, detect tempo/beats |
| 03 | Feature Extraction | librosa | MFCCs, chroma, spectral centroid, RMS, ZCR |
| 04 | Pitch Detection | CREPE / torchcrepe | Neural pitch tracking (optional, slow on CPU) |
| 05 | Vocal Removal | Demucs v4 (htdemucs) | AI source separation into 4 stems |
| 06 | MIDI Generation | Basic Pitch | Audio-to-MIDI transcription (optional) |
| 07 | Instrument Synthesis | FluidSynth | Render MIDI to audio with SoundFonts (optional) |
| 08 | Final Output | pydub + ffmpeg | Normalize, fade, export WAV + MP3 |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/maladhika-mohan/DeVox.git
cd DeVox
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser, upload an audio file, and click "Remove Vocals & Generate Instrumental".

### CLI Usage

You can also run the pipeline from the command line:

```bash
python run_pipeline.py --input input/my_song.mp3
```

Options:
```
--input, -i        Path to input audio file (required)
--output-dir, -o   Output directory (default: ./output)
--skip-features    Skip Stage 03 (feature extraction)
--skip-pitch       Skip Stage 04 (pitch detection — slow on CPU)
--skip-midi        Skip Stage 06 (MIDI generation)
--skip-synth       Skip Stage 07 (FluidSynth synthesis)
--device           Force cpu or cuda
```

## Optional: FluidSynth Setup (Stage 07)

Stage 07 requires FluidSynth and a SoundFont file. This stage is optional and will be skipped automatically if not configured.

**Windows:**
1. Download [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) and extract to `C:\tools\fluidsynth\`
2. Install the Python binding: `pip install pyfluidsynth`
3. Place a `.sf2` SoundFont file in the `soundfonts/` directory

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth libfluidsynth-dev

# macOS
brew install fluid-synth

pip install pyfluidsynth
```

## Optional: ffmpeg (for MP3 export)

MP3 export in Stage 08 requires ffmpeg. WAV export works without it.

```bash
# Windows
winget install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Project Structure

```
DeVox/
├── app.py                    # Streamlit web UI
├── run_pipeline.py           # CLI entry point
├── pipeline/
│   ├── config.py             # All tuneable parameters
│   ├── stage01_input.py      # Audio loading
│   ├── stage02_preprocess.py # Normalization, filtering, tempo detection
│   ├── stage03_features.py   # Feature extraction (MFCCs, chroma, etc.)
│   ├── stage04_pitch.py      # CREPE pitch detection
│   ├── stage05_separation.py # Demucs vocal separation
│   ├── stage06_midi.py       # Basic Pitch MIDI transcription
│   ├── stage07_synth.py      # FluidSynth MIDI rendering
│   ├── stage08_output.py     # Final export (WAV + MP3)
│   └── utils.py              # Shared utilities
├── input/                    # Place input audio files here
├── output/                   # Generated output files
├── soundfonts/               # SoundFont files for FluidSynth
├── requirements.txt
├── packages.txt              # System packages for Streamlit Cloud
└── .python-version           # Python version for Streamlit Cloud
```

## Configuration

All pipeline parameters are in `pipeline/config.py`:

- **Demucs model**: `htdemucs` (default, good balance of speed/quality) or `htdemucs_ft` (best quality, slower)
- **CREPE model**: `tiny` / `small` / `medium` / `large` / `full`
- **Sample rate**: 44100 Hz (CD quality)
- **Shifts/Overlap**: Tune for speed vs. quality tradeoff in source separation

## Streamlit Cloud Deployment

The repo is configured for Streamlit Cloud deployment:
- `.python-version` pins Python 3.11
- `packages.txt` installs system dependencies (ffmpeg, libsndfile1)
- `requirements.txt` uses CPU-only PyTorch for smaller footprint

## Tech Stack

- **Demucs v4** — Meta's hybrid transformer model for source separation
- **CREPE / torchcrepe** — Neural network pitch detection
- **Basic Pitch** — Spotify's audio-to-MIDI model
- **librosa** — Audio analysis and feature extraction
- **FluidSynth** — MIDI synthesis with SoundFonts
- **Streamlit** — Web UI framework
- **pydub + ffmpeg** — Audio format conversion
