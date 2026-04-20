"""
DeVox -- Streamlit Interface
Upload audio -> get instrumental output (vocals removed).
"""

import streamlit as st
import numpy as np
import tempfile
import time
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import OUTPUT_DIR, SAMPLE_RATE
from pipeline.utils import get_logger

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeVox - Audio to Instrumental",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ─────────────────────────────────────────────
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "results" not in st.session_state:
    st.session_state.results = None
if "total_elapsed" not in st.session_state:
    st.session_state.total_elapsed = 0
if "replay_wav_path" not in st.session_state:
    st.session_state.replay_wav_path = None
if "replay_instrument" not in st.session_state:
    st.session_state.replay_instrument = None

# ── Custom CSS — Light Theme ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}
.main .block-container { padding-top: 2rem; max-width: 1100px; }

.hero-container {
    text-align: center; padding: 2.5rem 1.5rem 2rem; margin-bottom: 2rem;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    border-radius: 20px; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
}
.hero-title { font-size: 3rem; font-weight: 900; color: #fff; margin-bottom: 0.3rem; letter-spacing: -1px; }
.hero-subtitle { font-size: 1.1rem; color: rgba(255,255,255,0.85); font-weight: 400; margin-top: 0; }

.stage-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 0.6rem; transition: all 0.2s ease;
}
.stage-card:hover { border-color: #cbd5e1; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.stage-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #7c3aed; }
.stage-title { font-size: 1rem; font-weight: 600; color: #1e293b; margin-top: 0.15rem; }
.stage-status { font-size: 0.85rem; color: #64748b; margin-top: 0.3rem; }
.stage-done { border-left: 4px solid #10b981; }
.stage-running { border-left: 4px solid #f59e0b; animation: pulse-border 1.5s infinite; }
.stage-pending { border-left: 4px solid #cbd5e1; opacity: 0.5; }
@keyframes pulse-border { 0%,100%{border-left-color:#f59e0b;} 50%{border-left-color:rgba(245,158,11,0.4);} }

.result-card { background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; }

.upload-zone {
    background: #fff; border: 2px dashed #cbd5e1; border-radius: 16px;
    padding: 2.5rem; text-align: center; transition: border-color 0.3s ease;
}
.upload-zone:hover { border-color: #8b5cf6; }

.stat-box { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.1rem 1rem; text-align: center; }
.stat-value { font-size: 1.4rem; font-weight: 800; color: #6366f1; }
.stat-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.35rem; }

section[data-testid="stSidebar"] > div { background: #fff !important; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h5,
section[data-testid="stSidebar"] span { color: #334155 !important; }

.stMarkdown h3, .stMarkdown h5 { color: #1e293b !important; }
.stMarkdown p { color: #334155 !important; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">DeVox</div>
        <div class="hero-subtitle">AI-Powered Vocal Removal &amp; Instrumental Extraction</div>
    </div>
    """, unsafe_allow_html=True)


def render_stage_card(number, title, tool, status="pending", detail=""):
    status_icons = {"done": "&#10003;", "running": "&#9654;", "pending": "&#9679;"}
    icon = status_icons.get(status, "&#9679;")
    css_class = f"stage-{status}"
    st.markdown(f"""
    <div class="stage-card {css_class}">
        <div class="stage-label">Stage {number:02d} &mdash; {tool}</div>
        <div class="stage-title">{icon} {title}</div>
        {"<div class='stage-status'>" + detail + "</div>" if detail else ""}
    </div>
    """, unsafe_allow_html=True)


STAGE_DEFS = [
    (1, "Audio Input", "librosa"),
    (2, "Preprocessing", "librosa"),
    (3, "Feature Extraction", "librosa"),
    (4, "Pitch Detection", "CREPE / torchcrepe"),
    (5, "Vocal Removal", "Demucs v4"),
    (6, "MIDI Generation", "Basic Pitch"),
    (7, "Instrument Synthesis", "FluidSynth"),
    (8, "Final Output", "pydub + ffmpeg"),
]


def run_pipeline_with_progress(audio_path: str, skip_pitch: bool, skip_midi: bool, skip_synth: bool):
    """Run the pipeline with live Streamlit progress updates."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    stage_statuses = {i: "pending" for i in range(1, 9)}
    stage_details = {i: "" for i in range(1, 9)}
    total_stages = 8

    progress_bar = st.progress(0, text="Initializing pipeline...")
    stage_placeholder = st.empty()

    def update_ui(current_stage, pct_override=None):
        if pct_override is not None:
            pct = pct_override
        else:
            done = sum(1 for s in stage_statuses.values() if s == "done")
            pct = done / total_stages
        progress_bar.progress(pct, text=f"Stage {current_stage:02d} in progress...")
        with stage_placeholder.container():
            for num, title, tool in STAGE_DEFS:
                render_stage_card(num, title, tool, stage_statuses[num], stage_details[num])

    # ── Stage 01 ─────────────────────────────────────────────────────────
    stage_statuses[1] = "running"
    update_ui(1, 0.02)
    from pipeline.stage01_input import load_audio
    t0 = time.time()
    audio, sr = load_audio(audio_path)
    elapsed = time.time() - t0
    duration = len(audio) / sr
    stage_statuses[1] = "done"
    stage_details[1] = f"Loaded {duration:.1f}s audio at {sr} Hz ({elapsed:.1f}s)"
    results["duration"] = duration
    results["sr"] = sr
    update_ui(1)

    # ── Stage 02 ─────────────────────────────────────────────────────────
    stage_statuses[2] = "running"
    update_ui(2)
    from pipeline.stage02_preprocess import preprocess
    t0 = time.time()
    prep = preprocess(audio, sr)
    elapsed = time.time() - t0
    audio_clean = prep["audio"]
    stage_statuses[2] = "done"
    stage_details[2] = f"Tempo: {prep['tempo']:.0f} BPM | {len(prep['onsets'])} onsets ({elapsed:.1f}s)"
    results["tempo"] = prep["tempo"]
    results["beats"] = len(prep["beats"])
    results["onsets"] = len(prep["onsets"])
    update_ui(2)

    # ── Stage 03 ─────────────────────────────────────────────────────────
    stage_statuses[3] = "running"
    update_ui(3)
    from pipeline.stage03_features import extract_features
    t0 = time.time()
    features = extract_features(audio_clean, sr)
    elapsed = time.time() - t0
    stage_statuses[3] = "done"
    stage_details[3] = f"MFCCs, chroma, spectral centroid, RMS, ZCR ({elapsed:.1f}s)"
    results["features"] = features["summary"]
    update_ui(3)

    # ── Stage 04 ─────────────────────────────────────────────────────────
    if not skip_pitch:
        stage_statuses[4] = "running"
        update_ui(4)
        from pipeline.stage04_pitch import detect_pitch
        t0 = time.time()
        pitch_data = detect_pitch(audio_clean, sr)
        elapsed = time.time() - t0
        stage_statuses[4] = "done"
        confident = (pitch_data["confidence"] >= 0.5).sum()
        stage_details[4] = f"{confident}/{len(pitch_data['confidence'])} confident frames ({elapsed:.1f}s)"
    else:
        stage_statuses[4] = "done"
        stage_details[4] = "Skipped"
    update_ui(4)

    # ── Stage 05 (CORE) ─────────────────────────────────────────────────
    stage_statuses[5] = "running"
    update_ui(5)
    from pipeline.stage05_separation import separate_vocals
    t0 = time.time()
    stems = separate_vocals(audio_clean, sr)
    elapsed = time.time() - t0
    instrumental = stems["instrumental"]
    stage_statuses[5] = "done"
    stage_details[5] = f"4 stems separated, vocals removed ({elapsed:.1f}s)"
    update_ui(5)

    # ── Stage 06 ─────────────────────────────────────────────────────────
    if not skip_midi:
        stage_statuses[6] = "running"
        update_ui(6)
        from pipeline.stage06_midi import generate_midi
        t0 = time.time()
        midi_path = generate_midi(instrumental, sr)
        elapsed = time.time() - t0
        stage_statuses[6] = "done"
        stage_details[6] = f"MIDI transcription complete ({elapsed:.1f}s)"
        results["midi_path"] = str(midi_path)
    else:
        midi_path = None
        stage_statuses[6] = "done"
        stage_details[6] = "Skipped"
    update_ui(6)

    # ── Stage 07 ─────────────────────────────────────────────────────────
    synthesized = None
    if not skip_synth and midi_path is not None:
        stage_statuses[7] = "running"
        update_ui(7)
        from pipeline.stage07_synth import synthesize_midi, is_available
        if is_available():
            t0 = time.time()
            synthesized = synthesize_midi(midi_path)
            elapsed = time.time() - t0
            stage_statuses[7] = "done"
            stage_details[7] = f"Synthesized via FluidSynth ({elapsed:.1f}s)"
        else:
            stage_statuses[7] = "done"
            stage_details[7] = "Skipped (FluidSynth not available)"
    else:
        stage_statuses[7] = "done"
        stage_details[7] = "Skipped"
    update_ui(7)

    # ── Stage 08 ─────────────────────────────────────────────────────────
    stage_statuses[8] = "running"
    update_ui(8)
    from pipeline.stage08_output import export_final
    t0 = time.time()
    outputs = export_final(instrumental, sr, synthesized=synthesized)
    elapsed = time.time() - t0
    stage_statuses[8] = "done"
    stage_details[8] = f"WAV + MP3 exported ({elapsed:.1f}s)"
    results["outputs"] = {k: str(v) for k, v in outputs.items()}
    update_ui(8, 1.0)

    progress_bar.progress(1.0, text="Pipeline complete!")
    return results


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 2rem; font-weight: 900; color: #6366f1;">DeVox</span>
        <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.25rem;">v0.1.0 &bull; 8-Stage Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Pipeline Controls")

    skip_pitch = st.checkbox("Skip Pitch Detection", value=True,
                             help="CREPE pitch detection can be slow on CPU (~60s)")
    skip_midi = st.checkbox("Skip MIDI Generation", value=False,
                            help="Skip audio-to-MIDI conversion")
    skip_synth = st.checkbox("Skip FluidSynth", value=True,
                             help="Skip MIDI synthesis (requires SoundFont)")

    st.markdown("---")
    st.markdown("##### Pipeline Stages")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #334155; line-height: 2;">
    <b>01</b> Audio Input <span style="color:#64748b">librosa</span><br>
    <b>02</b> Preprocessing <span style="color:#64748b">librosa+aubio</span><br>
    <b>03</b> Feature Extraction <span style="color:#64748b">librosa</span><br>
    <b>04</b> Pitch Detection <span style="color:#64748b">CREPE</span><br>
    <b>05</b> Vocal Removal <span style="color:#64748b">Demucs v4</span><br>
    <b>06</b> MIDI Generation <span style="color:#64748b">Basic Pitch</span><br>
    <b>07</b> Instrument Synth <span style="color:#64748b">FluidSynth</span><br>
    <b>08</b> Final Output <span style="color:#64748b">pydub+ffmpeg</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #94a3b8; text-align: center;">
        Built with Demucs v4 &bull; CREPE &bull; Basic Pitch<br>
        All processing runs locally on your machine
    </div>
    """, unsafe_allow_html=True)

    # Reset button to process a new file
    if st.session_state.pipeline_done:
        st.markdown("---")
        if st.button("🔄 Process New File", use_container_width=True):
            st.session_state.pipeline_done = False
            st.session_state.results = None
            st.session_state.replay_wav_path = None
            st.session_state.replay_instrument = None
            st.rerun()


# ── Helper: show results (used both after pipeline and on rerun) ─────────
def show_results(results, total_elapsed):
    st.markdown("---")
    st.markdown("### Results")

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 1.3rem; font-weight: 700; color: #059669; margin-bottom: 0.5rem;">
            ✓ Pipeline Complete
        </div>
        <div style="color: #065f46; font-size: 0.95rem;">
            Processed in <b>{total_elapsed:.1f}s</b>
            &bull; Tempo: <b>{results.get('tempo', 0):.0f} BPM</b>
            &bull; Duration: <b>{results.get('duration', 0):.1f}s</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Audio player for result
    st.markdown("##### Instrumental Output (Vocals Removed)")
    wav_path = OUTPUT_DIR / "final_instrumental.wav"
    mp3_path = OUTPUT_DIR / "final_instrumental.mp3"

    if wav_path.exists():
        st.audio(str(wav_path), format="audio/wav")

    # Pre-read file bytes so download buttons don't cause issues
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        if wav_path.exists():
            wav_bytes = wav_path.read_bytes()
            st.download_button(
                "Download WAV (Lossless)", data=wav_bytes,
                file_name="instrumental.wav", mime="audio/wav",
                use_container_width=True, key="dl_wav",
            )
    with dl_col2:
        if mp3_path.exists():
            mp3_bytes = mp3_path.read_bytes()
            st.download_button(
                "Download MP3", data=mp3_bytes,
                file_name="instrumental.mp3", mime="audio/mpeg",
                use_container_width=True, key="dl_mp3",
            )
    with dl_col3:
        midi_file = OUTPUT_DIR / "06_instrumental.mid"
        if midi_file.exists():
            midi_bytes = midi_file.read_bytes()
            st.download_button(
                "Download MIDI", data=midi_bytes,
                file_name="instrumental.mid", mime="audio/midi",
                use_container_width=True, key="dl_midi",
            )

    # Separated stems
    st.markdown("---")
    st.markdown("##### Separated Stems")
    stem_names = ["drums", "bass", "other", "vocals"]
    stem_cols = st.columns(4)
    for i, name in enumerate(stem_names):
        stem_path = OUTPUT_DIR / f"05_stem_{name}.wav"
        if stem_path.exists():
            with stem_cols[i]:
                st.markdown(f"**{name.capitalize()}**")
                st.audio(str(stem_path), format="audio/wav")
                stem_bytes = stem_path.read_bytes()
                st.download_button(
                    f"Download {name}", data=stem_bytes,
                    file_name=f"stem_{name}.wav", mime="audio/wav",
                    use_container_width=True, key=f"dl_stem_{name}",
                )

    # Diagnostic plots
    st.markdown("---")
    st.markdown("##### Diagnostic Visualizations")
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        features_plot = OUTPUT_DIR / "03_features_plot.png"
        if features_plot.exists():
            st.image(str(features_plot), caption="Feature Extraction Report", use_container_width=True)
    with plot_col2:
        pitch_plot = OUTPUT_DIR / "04_pitch_plot.png"
        if pitch_plot.exists():
            st.image(str(pitch_plot), caption="Pitch Contour (CREPE)", use_container_width=True)

    # Feature summary
    if "features" in results:
        with st.expander("Feature Summary (JSON)"):
            st.json(results["features"])

    # ── Instrument Replay Section ────────────────────────────────────────
    midi_file = OUTPUT_DIR / "06_instrumental.mid"
    if midi_file.exists():
        st.markdown("---")
        st.markdown("### 🎹 Replay as Different Instrument")
        st.markdown(
            '<div style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">'
            "Choose an instrument and we'll re-synthesize the MIDI transcription in that sound."
            "</div>",
            unsafe_allow_html=True,
        )

        from pipeline.instrument_replay import get_instrument_names, replay_as_instrument

        instruments = get_instrument_names()

        col_inst, col_btn = st.columns([3, 1])
        with col_inst:
            selected = st.selectbox(
                "Choose an instrument",
                instruments,
                index=instruments.index("Piano"),
                label_visibility="collapsed",
            )
        with col_btn:
            replay_clicked = st.button("🎵 Replay", type="primary", use_container_width=True)

        if replay_clicked:
            with st.spinner(f"Synthesizing as {selected}..."):
                wav_out = replay_as_instrument(str(midi_file), selected)
            if wav_out and wav_out.exists():
                st.session_state.replay_wav_path = str(wav_out)
                st.session_state.replay_instrument = selected

        # Show replay result (persists across reruns)
        if st.session_state.replay_wav_path and Path(st.session_state.replay_wav_path).exists():
            replay_path = Path(st.session_state.replay_wav_path)
            st.markdown(f"##### 🔊 {st.session_state.replay_instrument} Version")
            st.audio(str(replay_path), format="audio/wav")

            replay_bytes = replay_path.read_bytes()
            safe = st.session_state.replay_instrument.replace(" ", "_").lower()
            st.download_button(
                f"Download {st.session_state.replay_instrument} WAV",
                data=replay_bytes,
                file_name=f"replay_{safe}.wav",
                mime="audio/wav",
                use_container_width=True,
                key="dl_replay",
            )


# ── Main Content ─────────────────────────────────────────────────────────
render_hero()

# If pipeline already completed, show results directly
if st.session_state.pipeline_done and st.session_state.results is not None:
    st.markdown("### Upload Your Audio")
    st.info("Pipeline already completed. Use the sidebar '🔄 Process New File' button to start over.")
    show_results(st.session_state.results, st.session_state.total_elapsed)

elif not st.session_state.pipeline_done:
    # Upload section
    st.markdown("### Upload Your Audio")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse for an audio file",
        type=["mp3", "mpeg", "wav", "flac", "ogg", "m4a", "aac", "wma"],
        help="Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{uploaded_file.name.split('.')[-1].upper()}</div>
                <div class="stat-label">Format</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{file_size_mb:.1f} MB</div>
                <div class="stat-label">File Size</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value" style="font-size:1.1rem;">{uploaded_file.name[:25]}</div>
                <div class="stat-label">Filename</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("##### Original Audio Preview")
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        st.markdown("---")

        if st.button("Remove Vocals & Generate Instrumental", type="primary", use_container_width=True):
            input_dir = OUTPUT_DIR.parent / "input"
            input_dir.mkdir(exist_ok=True)

            filename = uploaded_file.name
            ext_map = {".mpeg": ".mp3", ".mpga": ".mp3"}
            stem = Path(filename).stem
            ext = Path(filename).suffix.lower()
            if ext in ext_map:
                filename = stem + ext_map[ext]

            temp_path = input_dir / filename
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.markdown("### Processing Pipeline")
            total_start = time.time()

            try:
                results = run_pipeline_with_progress(
                    str(temp_path),
                    skip_pitch=skip_pitch,
                    skip_midi=skip_midi,
                    skip_synth=skip_synth,
                )
                total_elapsed = time.time() - total_start

                # Save to session state so results persist across reruns
                st.session_state.pipeline_done = True
                st.session_state.results = results
                st.session_state.total_elapsed = total_elapsed

                show_results(results, total_elapsed)

            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.exception(e)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except:
                    pass

    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎵</div>
            <div style="font-size: 1.1rem; color: #1e293b; font-weight: 600; margin-bottom: 0.3rem;">
                Upload an audio file to get started
            </div>
            <div style="font-size: 0.85rem; color: #64748b;">
                Supports MP3, WAV, FLAC, OGG, M4A &bull; Processing runs entirely on your machine
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### How It Works")
        how_cols = st.columns(4)
        steps = [
            ("Upload", "Drop your audio file (song, humming, recording)"),
            ("Separate", "Demucs v4 AI splits audio into drums, bass, other, vocals"),
            ("Remove Vocals", "Vocal stem is discarded, instruments are remixed"),
            ("Download", "Get your instrumental as WAV or MP3"),
        ]
        for i, (title, desc) in enumerate(steps):
            with how_cols[i]:
                st.markdown(f"""
                <div class="stat-box" style="min-height: 130px;">
                    <div class="stat-value" style="font-size: 1.2rem;">{i+1}</div>
                    <div style="font-weight: 600; color: #1e293b; margin: 0.4rem 0 0.2rem; font-size: 0.95rem;">{title}</div>
                    <div style="font-size: 0.78rem; color: #64748b;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
