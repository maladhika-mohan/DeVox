"""
Saamai API — Backend for the mobile app.
Receives audio + instrument name, returns synthesized WAV.
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.config import OUTPUT_DIR, SAMPLE_RATE
from pipeline.instrument_replay import get_instrument_names, replay_as_instrument, INSTRUMENT_MAP
from pipeline.utils import get_logger

log = get_logger("Saamai API")

app = FastAPI(
    title="Saamai API",
    description="Voice-driven instrument conversion API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "Saamai", "version": "1.0.0"}


@app.get("/instruments")
def list_instruments():
    """Return available instruments."""
    return {"instruments": get_instrument_names()}


@app.post("/convert")
async def convert_audio(
    audio: UploadFile = File(...),
    instrument: str = Form("Piano"),
):
    """
    Convert uploaded audio (humming/singing) to the specified instrument.

    Pipeline: Audio → Preprocess → Basic Pitch (MIDI) → Instrument Replay → WAV
    """
    if instrument not in INSTRUMENT_MAP:
        raise HTTPException(400, f"Unknown instrument: {instrument}. Use /instruments to list options.")

    log.info(f"Converting to {instrument}: {audio.filename} ({audio.content_type})")
    t_start = time.time()

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    tmp_input = Path(tempfile.mktemp(suffix=suffix))
    with open(tmp_input, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        # Stage 1: Load audio
        from pipeline.stage01_input import load_audio
        audio_data, sr = load_audio(str(tmp_input))
        duration = len(audio_data) / sr
        log.info(f"   Loaded: {duration:.1f}s at {sr}Hz")

        # Stage 2: Light preprocessing (normalize + trim)
        from pipeline.stage02_preprocess import preprocess
        prep = preprocess(audio_data, sr)
        audio_clean = prep["audio"]

        # Stage 3: Audio → MIDI (Basic Pitch)
        from pipeline.stage06_midi import generate_midi
        midi_path = generate_midi(audio_clean, sr)
        log.info(f"   MIDI generated: {midi_path}")

        # Stage 4: Instrument Replay
        wav_path = replay_as_instrument(str(midi_path), instrument)
        if wav_path is None or not wav_path.exists():
            raise HTTPException(500, "Synthesis failed")

        elapsed = time.time() - t_start
        log.info(f"   Done in {elapsed:.1f}s → {wav_path.name}")

        return FileResponse(
            path=str(wav_path),
            media_type="audio/wav",
            filename=f"saamai_{instrument.lower().replace(' ', '_')}.wav",
            headers={"X-Processing-Time": f"{elapsed:.2f}"},
        )

    finally:
        tmp_input.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
