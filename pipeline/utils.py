"""
Shared utility helpers for the DeVox pipeline.
Logging, timing, and common I/O operations.
"""

import time
import logging
import functools
import numpy as np
import soundfile as sf
from pathlib import Path

from pipeline.config import SAMPLE_RATE, OUTPUT_DIR

# ── Logging setup ────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Stage decorator ──────────────────────────────────────────────────────
def stage(number: int, title: str):
    """Decorator that logs stage entry/exit and elapsed time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger(f"Stage {number:02d}")
            log.info(f"[>]  {title}")
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            log.info(f"[OK]  {title} completed in {elapsed:.1f}s")
            return result
        return wrapper
    return decorator


# ── Audio I/O helpers ────────────────────────────────────────────────────
def save_wav(audio: np.ndarray, path: Path, sr: int = SAMPLE_RATE):
    """Save a numpy array as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    log = get_logger("utils")
    log.info(f"   Saved {path.name}  ({audio.shape[-1] / sr:.1f}s, {sr} Hz)")


def ensure_output_dir() -> Path:
    """Make sure the output directory exists and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
