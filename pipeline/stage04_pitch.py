"""
Stage 04 — Pitch Detection
Monophonic pitch estimation using CREPE (torchcrepe).
Outputs Hz frequency + confidence per 10 ms frame.
Saves pitch contour CSV and plot.
"""

import numpy as np
import torch
import torchcrepe
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import (
    SAMPLE_RATE, OUTPUT_DIR,
    CREPE_MODEL, CREPE_STEP_SIZE, CREPE_CONFIDENCE_THRESHOLD,
)
from pipeline.utils import stage, get_logger

log = get_logger("Stage 04")


@stage(4, "Pitch Detection (CREPE)")
def detect_pitch(audio: np.ndarray, sr: int) -> dict:
    """
    Detect pitch frame-by-frame using torchcrepe.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio waveform.
    sr : int
        Sample rate.

    Returns
    -------
    dict with keys:
        time       — time stamps in seconds (np.ndarray)
        frequency  — estimated pitch in Hz per frame (np.ndarray)
        confidence — confidence [0, 1] per frame (np.ndarray)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"   Device: {device}")

    # torchcrepe expects a batched tensor of shape (batch, samples)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    # Run CREPE
    frequency, confidence = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=int(sr * CREPE_STEP_SIZE / 1000),  # step_size in samples
        fmin=50,
        fmax=2000,
        model=CREPE_MODEL,
        device=device,
        batch_size=512,
        return_periodicity=True,
    )

    # Move to numpy
    frequency = frequency.squeeze().cpu().numpy()
    confidence = confidence.squeeze().cpu().numpy()

    # Time axis
    n_frames = len(frequency)
    time_axis = np.arange(n_frames) * CREPE_STEP_SIZE / 1000.0

    # Filter low-confidence frames
    frequency_filtered = frequency.copy()
    frequency_filtered[confidence < CREPE_CONFIDENCE_THRESHOLD] = 0.0

    log.info(f"   Frames: {n_frames} | "
             f"Confident: {(confidence >= CREPE_CONFIDENCE_THRESHOLD).sum()} / {n_frames}")

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "04_pitch_contour.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "frequency_hz", "confidence"])
        for t, freq, conf in zip(time_axis, frequency, confidence):
            writer.writerow([f"{t:.3f}", f"{freq:.2f}", f"{conf:.4f}"])
    log.info(f"   Saved pitch data → {csv_path.name}")

    # ── Plot ─────────────────────────────────────────────────────────────
    _plot_pitch(time_axis, frequency_filtered, confidence)

    return {
        "time": time_axis,
        "frequency": frequency_filtered,
        "confidence": confidence,
    }


def _plot_pitch(time_axis: np.ndarray, frequency: np.ndarray, confidence: np.ndarray):
    """Save a pitch contour plot with confidence heat-mapping."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                     gridspec_kw={"height_ratios": [3, 1]},
                                     constrained_layout=True)
    fig.suptitle("DeVox — Pitch Contour (CREPE)", fontsize=14, fontweight="bold")

    # Pitch contour (scatter with confidence as colour)
    voiced = frequency > 0
    sc = ax1.scatter(time_axis[voiced], frequency[voiced],
                      c=confidence[voiced], cmap="viridis",
                      s=2, alpha=0.7, vmin=0, vmax=1)
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Detected Pitch")
    ax1.set_ylim(0, min(frequency[voiced].max() * 1.2, 2000) if voiced.any() else 2000)
    fig.colorbar(sc, ax=ax1, label="Confidence")

    # Confidence over time
    ax2.fill_between(time_axis, confidence, alpha=0.4, color="#4ECDC4")
    ax2.plot(time_axis, confidence, color="#4ECDC4", linewidth=0.5)
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color="#FF6B6B", linestyle="--", alpha=0.5, label="Threshold")
    ax2.legend()

    plot_path = OUTPUT_DIR / "04_pitch_plot.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    log.info(f"   Saved pitch plot → {plot_path.name}")
