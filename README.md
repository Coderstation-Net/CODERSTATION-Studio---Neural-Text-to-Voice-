# CODERSTATION Studio — Neural Text-to-Voice  

![Python](https://img.shields.io/badge/Python-3.10%2B-062E52?style=flat-square&logo=python&logoColor=white)
![Kokoro ONNX](https://img.shields.io/badge/Engine-Kokoro%20v1.0%20ONNX-FF5C0A?style=flat-square)
![CUDA](https://img.shields.io/badge/Acceleration-NVIDIA%20CUDA%20%2F%20CPU-10B981?style=flat-square)
![FastAPI](https://img.shields.io/badge/UI-FastAPI%20%2B%20Glassmorphic%20Web%20Studio-38BDF8?style=flat-square)

**CODERSTATION Studio** is a local, GPU-accelerated Text-to-Voice and Voice Cloning workstation powered by **Kokoro v1.0 ONNX**. It pairs studio-grade expressive speech synthesis with a glassmorphic web interface, a 60 FPS 3D holographic dot-matrix avatar, and real-time hardware telemetry.

---
![Uploading image.png…]()

## Key Features

- **Precision Punctuation Pauses** — Calibrated silence and prosodic phrasing for commas (`,`), periods (`.`), ellipses (`...`), and em dashes (`—`).
- **Voice Cloning (Mic & Reference Audio)** — Record via the built-in guided teleprompter or upload `.wav` / `.mp3` audio, choose **Male** or **Female**, and automatically extract F0 pitch, 129-bin cepstral vocal-tract formants, speaking speed, and intensity.
- **Expressive Speech Feelings** — Global and inline `[feeling]` tags (`[neutral]`, `[happy]`, `[excited]`, `[dramatic]`, `[sad]`, `[whisper]`, `[angry]`, `[calm]`, `[fearful]`) with strict speaker identity lock.
- **3D Holographic Avatar & Live Telemetry** — 60 FPS pure-dot 3D avatar with audio-reactive vocal pulses, in-avatar synthesis progress animation, and transparent live **GPU / RAM / CPU** charts.
- **Manual-Only Export** — Synthesizes cleanly in memory and saves to lossless **24 kHz `.wav`** or compressed **`.mp3`** only when requested.

---

## Quick Start

### 1. Prerequisites
- Python **3.10+**
- `kokoro-v1.0.onnx` and `voices-v1.0.bin` placed in the project root
- *(Optional)* NVIDIA GPU with CUDA/cuDNN for hardware acceleration (falls back to CPU automatically)

### 2. Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Web Studio
```bash
run_coderstation.bat
```
Or start manually:
```bash
python coderstation.py --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000** in your browser.

### 4. CLI Usage (Optional)
```bash
python coderstation.py --text "[dramatic] Hello, world... ready for launch." --voice af_heart --output outputs/demo.wav
```

---

## Project Structure

```text
Coderstation -Voice-Clone/
├── coderstation.py        # FastAPI server, Kokoro ONNX engine, voice cloning & DSP
├── run_coderstation.bat   # One-click Windows launcher
├── requirements.txt       # Python dependencies
├── static/
│   ├── index.html         # Glassmorphic 2-column studio UI & modal settings
│   ├── styles.css         # Unified glassmorphism, comboboxes & scrollbars
│   └── app.js             # 3D holographic avatar, audio sync, mic recorder & telemetry
├── cloned_voices/         # Saved cloned voice profiles & reference WAVs
└── outputs/               # Manually exported WAV / MP3 files
```

---

## Credits

Built with [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx), FastAPI, SciPy, and HTML5 Canvas.
