# Saamai — Mobile App

Voice-driven instrument conversion. Hum a melody, get it back as Piano, Violin, Flute, or 70+ instruments.

## Architecture

```
Flutter App (mobile) ←→ FastAPI Backend (your PC/server)
     │                         │
     ├─ Record audio           ├─ Receive audio
     ├─ Voice commands (STT)   ├─ Basic Pitch → MIDI
     ├─ Instrument picker      ├─ FluidSynth → WAV
     └─ Play result            └─ Return WAV
```

## Setup

### 1. Start the Backend

```bash
cd mobile/backend
pip install -r requirements.txt
python main.py
```

This starts the API at `http://localhost:8000`. Test it:
```
GET  http://localhost:8000/instruments
POST http://localhost:8000/convert  (multipart: audio file + instrument name)
```

### 2. Run the Flutter App

```bash
cd mobile/saamai_app
flutter pub get
flutter run
```

### 3. Configure API URL

Edit `lib/services/api_service.dart`:
- Android emulator → `http://10.0.2.2:8000`
- Physical device → `http://<your-pc-ip>:8000`
- Production → your deployed server URL

## Features

- 🎤 **Record** — Tap mic to record humming/singing
- 🎵 **Convert** — Transform to any of 70+ instruments
- 🗣️ **Voice Commands** — Say "violin" or "play this in piano"
- 🔊 **Playback** — Listen to the result immediately
- 📱 **Offline-ready UI** — Works with local network backend

## Voice Commands

The app recognizes instrument names from speech:
- "Play this in violin"
- "Guitar version"
- "Convert to flute"
- "Sitar"
- "Piano"

## Dependencies

### Flutter Packages
- `record` — Audio recording
- `audioplayers` — Audio playback
- `speech_to_text` — Voice command recognition
- `http` — API communication
- `permission_handler` — Mic/speech permissions
- `google_fonts` — Typography
- `path_provider` — File storage

### Backend
- FastAPI + uvicorn
- All DeVox pipeline dependencies (Basic Pitch, FluidSynth, etc.)

## Future: Fully On-Device

To eliminate the backend dependency:
1. Replace Basic Pitch with ONNX Runtime Mobile (~50MB model)
2. Embed FluidSynth C library via FFI (~3MB + 30MB SoundFont)
3. Total app size: ~100MB, fully offline, 3-5 sec processing
