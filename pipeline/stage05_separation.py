"""
Stage 05 — Source Separation (Vocal Removal)
Uses Demucs v4 with the htdemucs_ft model to separate audio into
4 stems: vocals, drums, bass, other.
Discards vocals and mixes the remaining 3 stems → instrumental.
"""

import numpy as np
import torch
import torchaudio
from pathlib import Path

from pipeline.config import (
    SAMPLE_RATE, OUTPUT_DIR,
    DEMUCS_MODEL, DEMUCS_DEVICE, DEMUCS_SHIFTS, DEMUCS_OVERLAP,
)
from pipeline.utils import stage, get_logger, save_wav

log = get_logger("Stage 05")


@stage(5, "Source Separation — Vocal Removal (Demucs v4)")
def separate_vocals(audio: np.ndarray, sr: int) -> dict:
    """
    Separate audio into stems using Demucs v4 (htdemucs_ft).

    Produces:
      - Individual stems: vocals, drums, bass, other
      - Combined instrumental (drums + bass + other)

    Parameters
    ----------
    audio : np.ndarray
        Mono or stereo audio waveform (float32).
    sr : int
        Sample rate of the input audio.

    Returns
    -------
    dict with keys:
        vocals         — vocal stem (np.ndarray)
        drums          — drum stem (np.ndarray)
        bass           — bass stem (np.ndarray)
        other          — other instruments stem (np.ndarray)
        instrumental   — combined instrumental (drums + bass + other)
        sr             — sample rate
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # ── Determine device ─────────────────────────────────────────────────
    if DEMUCS_DEVICE:
        device = torch.device(DEMUCS_DEVICE)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"   Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    log.info(f"   Loading model: {DEMUCS_MODEL}")
    model = get_model(DEMUCS_MODEL)
    model = model.to(device)
    model.eval()

    # ── Prepare audio tensor ─────────────────────────────────────────────
    # Demucs expects (batch, channels, samples) at the model's sample rate
    model_sr = model.samplerate

    # Convert mono to stereo if needed (Demucs expects stereo)
    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio])  # (2, samples)
    else:
        audio_stereo = audio

    wav = torch.tensor(audio_stereo, dtype=torch.float32).unsqueeze(0)  # (1, 2, samples)

    # Resample to model's expected sample rate if different
    if sr != model_sr:
        log.info(f"   Resampling {sr} Hz → {model_sr} Hz for model")
        wav = torchaudio.transforms.Resample(sr, model_sr)(wav)

    wav = wav.to(device)

    # ── Apply model ──────────────────────────────────────────────────────
    log.info(f"   Running separation (shifts={DEMUCS_SHIFTS}, overlap={DEMUCS_OVERLAP})...")
    log.info(f"   Audio length: {wav.shape[-1] / model_sr:.1f}s — this may take a while on CPU")

    with torch.no_grad():
        sources = apply_model(
            model, wav,
            shifts=DEMUCS_SHIFTS,
            overlap=DEMUCS_OVERLAP,
            device=device,
            split=True,           # split into segments to reduce memory
            progress=True,        # show progress bar
        )
    # sources shape: (1, n_sources, 2, samples)

    # ── Extract stems ────────────────────────────────────────────────────
    source_names = model.sources  # typically ['drums', 'bass', 'other', 'vocals']
    stems = {}
    for i, name in enumerate(source_names):
        stem = sources[0, i].cpu().numpy()  # (2, samples)
        # Convert to mono for consistency
        stem_mono = stem.mean(axis=0)
        stems[name] = stem_mono
        log.info(f"   [OK] Extracted: {name} ({len(stem_mono) / model_sr:.1f}s)")

    # ── Create instrumental mix (everything except vocals) ───────────────
    instrumental_parts = [stems[name] for name in source_names if name != "vocals"]
    instrumental = sum(instrumental_parts)
    # Normalise to prevent clipping
    max_val = np.abs(instrumental).max()
    if max_val > 0:
        instrumental = instrumental / max_val * 0.95

    stems["instrumental"] = instrumental

    # ── Resample back to pipeline sample rate if needed ──────────────────
    if model_sr != sr:
        log.info(f"   Resampling stems {model_sr} Hz → {sr} Hz")
        for key in stems:
            t = torch.tensor(stems[key], dtype=torch.float32).unsqueeze(0)
            t = torchaudio.transforms.Resample(model_sr, sr)(t)
            stems[key] = t.squeeze().numpy()

    stems["sr"] = sr

    # ── Save stems ───────────────────────────────────────────────────────
    for name in source_names:
        save_wav(stems[name], OUTPUT_DIR / f"05_stem_{name}.wav", sr)

    save_wav(stems["instrumental"], OUTPUT_DIR / "05_instrumental.wav", sr)
    log.info("   [*] Instrumental mix saved (vocals removed)")

    return stems
