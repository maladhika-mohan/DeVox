"""
Stage 03 — Feature Extraction
Extract MFCCs, chromagram, spectral centroid, RMS energy,
and zero-crossing rate using librosa.
Saves a JSON summary and optional diagnostic plots.
"""

import json
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")       # non-interactive backend
import matplotlib.pyplot as plt

from pipeline.config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MFCC, OUTPUT_DIR,
)
from pipeline.utils import stage, get_logger

log = get_logger("Stage 03")


def _to_serializable(obj):
    """Convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


@stage(3, "Feature Extraction")
def extract_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract audio features for diagnostics.

    Features extracted:
      - MFCCs (13 coefficients) — timbral texture
      - Chromagram — pitch class distribution
      - Spectral centroid — brightness
      - RMS energy — loudness envelope
      - Zero-crossing rate — noisiness/percussiveness

    Returns
    -------
    dict with feature arrays and summary statistics.
    """

    # ── MFCCs ────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    log.info(f"   [OK] MFCCs: shape {mfccs.shape}")

    # ── Chromagram ───────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    log.info(f"   [OK] Chromagram: shape {chroma.shape}")

    # ── Spectral centroid ────────────────────────────────────────────────
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    log.info(f"   [OK] Spectral centroid: mean {spectral_centroid.mean():.0f} Hz")

    # ── RMS energy ───────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    log.info(f"   [OK] RMS energy: mean {rms.mean():.4f}")

    # ── Zero-crossing rate ───────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
    log.info(f"   [OK] Zero-crossing rate: mean {zcr.mean():.4f}")

    # ── Summary dict ─────────────────────────────────────────────────────
    features = {
        "mfccs": mfccs,
        "chroma": chroma,
        "spectral_centroid": spectral_centroid,
        "rms": rms,
        "zcr": zcr,
        "summary": {
            "mfcc_mean": mfccs.mean(axis=1).tolist(),
            "spectral_centroid_mean": float(spectral_centroid.mean()),
            "rms_mean": float(rms.mean()),
            "rms_max": float(rms.max()),
            "zcr_mean": float(zcr.mean()),
            "duration_seconds": len(audio) / sr,
        },
    }

    # ── Save JSON summary ────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "03_features_summary.json"
    with open(summary_path, "w") as f:
        json.dump(features["summary"], f, indent=2, default=_to_serializable)
    log.info(f"   Saved feature summary → {summary_path.name}")

    # ── Save diagnostic plots ────────────────────────────────────────────
    _plot_features(features, sr)

    return features


def _plot_features(features: dict, sr: int):
    """Generate and save a combined diagnostic plot."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("DeVox — Feature Extraction Report", fontsize=14, fontweight="bold")

    frames = np.arange(features["mfccs"].shape[1])
    time_axis = librosa.frames_to_time(frames, sr=sr, hop_length=512)

    # MFCCs
    img = axes[0].imshow(features["mfccs"], aspect="auto", origin="lower",
                          cmap="magma", extent=[0, time_axis[-1], 0, N_MFCC])
    axes[0].set_ylabel("MFCC Coeff")
    axes[0].set_title("MFCCs")
    fig.colorbar(img, ax=axes[0], label="Amplitude")

    # Chromagram
    img2 = axes[1].imshow(features["chroma"], aspect="auto", origin="lower",
                           cmap="coolwarm", extent=[0, time_axis[-1], 0, 12])
    axes[1].set_ylabel("Pitch Class")
    axes[1].set_title("Chromagram")
    axes[1].set_yticks(np.arange(0.5, 12, 1))
    axes[1].set_yticklabels(["C", "C#", "D", "D#", "E", "F",
                              "F#", "G", "G#", "A", "A#", "B"])
    fig.colorbar(img2, ax=axes[1], label="Energy")

    # Spectral centroid + RMS
    ax2_twin = axes[2].twinx()
    axes[2].plot(time_axis, features["spectral_centroid"], color="#FF6B6B",
                  alpha=0.8, linewidth=0.8, label="Spectral Centroid")
    ax2_twin.plot(time_axis, features["rms"], color="#4ECDC4",
                   alpha=0.8, linewidth=0.8, label="RMS Energy")
    axes[2].set_ylabel("Centroid (Hz)", color="#FF6B6B")
    ax2_twin.set_ylabel("RMS", color="#4ECDC4")
    axes[2].set_title("Spectral Centroid & RMS Energy")
    axes[2].legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    # Zero-crossing rate
    axes[3].plot(time_axis, features["zcr"], color="#45B7D1", linewidth=0.8)
    axes[3].set_ylabel("ZCR")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Zero-Crossing Rate")
    axes[3].fill_between(time_axis, features["zcr"], alpha=0.2, color="#45B7D1")

    plot_path = OUTPUT_DIR / "03_features_plot.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    log.info(f"   Saved feature plot → {plot_path.name}")
