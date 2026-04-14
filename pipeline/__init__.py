# DeVox Pipeline — Audio-to-Instrumental Conversion
"""
8-stage pipeline:
  01 Audio Input        (librosa)
  02 Preprocessing      (librosa + aubio)
  03 Feature Extraction (librosa)
  04 Pitch Detection    (torchcrepe / CREPE)
  05 Source Separation  (Demucs v4 — htdemucs_ft)
  06 MIDI Generation    (Basic Pitch / Spotify)
  07 Instrument Synth   (FluidSynth)
  08 Final Output       (pydub + ffmpeg)
"""

__version__ = "0.1.0"
