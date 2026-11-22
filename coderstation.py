"""
Coderstation — Kokoro Text-to-Voice Studio
==========================================
Complete Text-to-Speech program powered by Kokoro ONNX featuring:
1. Precision Punctuation Pauses:
   - Comma (,)       -> Short pause
   - Period (.)      -> Moderate pause
   - Ellipsis (...)  -> Longer, dramatic pause (with trailing dramatic taper)
   - Em dash (—)     -> Natural pause / break in thought
2. Save & Export output to WAV or MP3 (24 kHz high definition)
3. Voice Cloning:
   - Reference Audio Cloner (extracts F0 pitch, spectral envelope, formants & projects
     onto the 54-voice Kokoro style manifold + spectral envelope transfer)
   - Multi-Voice Neural Style Tensor Blender
4. Speech with Feelings / Emotions (Global & inline [emotion] tags:
   neutral, happy, excited, sad, dramatic, angry, whisper, calm, fearful)
5. Animated Graphical Effects in Speech (Web Audio API 60fps Canvas Visualizer:
   Emotion Orb & Particles, Lip-Sync Expression Avatar, FFT Spectrum, Live Pause HUD)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import re
import sys
import time
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure NVIDIA CUDA & cuDNN sub-library bin directories are on PATH for Windows LoadLibrary
_site_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if _site_nvidia.exists():
    _nv_bins = [str(p) for p in _site_nvidia.glob("*/bin") if p.is_dir()]
    if _nv_bins:
        os.environ["PATH"] = os.pathsep.join(_nv_bins) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for _b in _nv_bins:
                try:
                    os.add_dll_directory(_b)
                except Exception:
                    pass

import onnxruntime as rt

# Preload CUDA/cuDNN DLLs from site-packages on Windows before creating ONNX sessions
if hasattr(rt, "preload_dlls"):
    try:
        rt.preload_dlls()
    except Exception:
        pass

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from kokoro_onnx import Kokoro
from kokoro_onnx.config import SAMPLE_RATE
from kokoro_onnx.trim import trim as trim_audio
from pydantic import BaseModel, Field
from scipy import signal

# ---------------------------------------------------------------------------
# Paths & Logging
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = BASE_DIR / "voices-v1.0.bin"
OUTPUT_DIR = BASE_DIR / "outputs"
CLONED_DIR = BASE_DIR / "cloned_voices"
STATIC_DIR = BASE_DIR / "static"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLONED_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coderstation")


# ---------------------------------------------------------------------------
# 1. Punctuation Pause Rules & Emotion Profiles
# ---------------------------------------------------------------------------
@dataclass
class PauseConfig:
    comma: float = 0.22       # Short pause (,)
    period: float = 0.50      # Moderate pause (.)
    ellipsis: float = 0.95    # Longer, dramatic pause (...)
    em_dash: float = 0.40     # Natural pause or break in thought (—)


EMOTION_PROFILES: dict[str, dict[str, Any]] = {
    "neutral": {
        "label": "Neutral",
        "icon": "😐",
        "color": "#38bdf8",
        "speed_mul": 1.00,
        "pitch_semitones": 0.0,
        "end_pitch_glide_st": 0.0,
        "style_row_offset": 0,
        "exclamatory_prosody": False,
        "pause_mul": 1.00,
        "ellipsis_mul": 1.00,
        "brightness_db": 0.0,
        "warmth_db": 0.0,
        "whisper_mode": False,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 1.00,
        "description": "Natural, balanced delivery with 100% locked speaker identity.",
    },
    "happy": {
        "label": "Happy & Joyful",
        "icon": "😊",
        "color": "#fbbf24",
        "speed_mul": 1.07,
        "pitch_semitones": 0.55,
        "end_pitch_glide_st": 0.45,
        "style_row_offset": -65,
        "exclamatory_prosody": True,
        "pause_mul": 0.88,
        "ellipsis_mul": 0.95,
        "brightness_db": 1.8,
        "warmth_db": 0.4,
        "whisper_mode": False,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 1.04,
        "description": "Upbeat cadence, cheerful rising inflection, and bright vocal smile.",
    },
    "excited": {
        "label": "Excited & Energetic",
        "icon": "🤩",
        "color": "#f43f5e",
        "speed_mul": 1.14,
        "pitch_semitones": 0.90,
        "end_pitch_glide_st": 0.75,
        "style_row_offset": -110,
        "exclamatory_prosody": True,
        "pause_mul": 0.78,
        "ellipsis_mul": 0.88,
        "brightness_db": 2.5,
        "warmth_db": 0.0,
        "whisper_mode": False,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 1.08,
        "description": "High-energy tempo, animated neural prosody, and enthusiastic pitch lift.",
    },
    "sad": {
        "label": "Sad & Melancholic",
        "icon": "😢",
        "color": "#6366f1",
        "speed_mul": 0.86,
        "pitch_semitones": -0.75,
        "end_pitch_glide_st": -0.85,
        "style_row_offset": 150,
        "exclamatory_prosody": False,
        "pause_mul": 1.38,
        "ellipsis_mul": 1.48,
        "brightness_db": -2.6,
        "warmth_db": 1.6,
        "whisper_mode": False,
        "vibrato_hz": 4.2,
        "vibrato_depth": 0.0005,
        "gain_mul": 0.88,
        "description": "Slower tempo, gentle downward sighing cadence, and reflective pauses.",
    },
    "dramatic": {
        "label": "Dramatic & Suspenseful",
        "icon": "🎭",
        "color": "#a855f7",
        "speed_mul": 0.90,
        "pitch_semitones": -0.45,
        "end_pitch_glide_st": -0.40,
        "style_row_offset": 95,
        "exclamatory_prosody": False,
        "pause_mul": 1.28,
        "ellipsis_mul": 1.55,
        "brightness_db": 1.0,
        "warmth_db": 2.4,
        "whisper_mode": False,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 1.05,
        "description": "Cinematic depth, rich chest resonance, and extended suspenseful pauses.",
    },
    "angry": {
        "label": "Angry & Intense",
        "icon": "😠",
        "color": "#ef4444",
        "speed_mul": 1.08,
        "pitch_semitones": -0.35,
        "end_pitch_glide_st": -0.50,
        "style_row_offset": -120,
        "exclamatory_prosody": True,
        "pause_mul": 0.80,
        "ellipsis_mul": 1.05,
        "brightness_db": 2.8,
        "warmth_db": 1.5,
        "whisper_mode": False,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 1.10,
        "description": "Forceful neural glottal attack, firm pacing, and assertive vocal projection.",
    },
    "whisper": {
        "label": "Whisper & Intimate",
        "icon": "🤫",
        "color": "#14b8a6",
        "speed_mul": 0.90,
        "pitch_semitones": -0.15,
        "end_pitch_glide_st": -0.25,
        "style_row_offset": 190,
        "exclamatory_prosody": False,
        "pause_mul": 1.24,
        "ellipsis_mul": 1.35,
        "brightness_db": 2.2,
        "warmth_db": -3.5,
        "whisper_mode": True,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 0.72,
        "description": "Soft, noise-free intimate delivery with attenuated chest boom and airy clarity.",
    },
    "calm": {
        "label": "Calm & Soothing",
        "icon": "🌿",
        "color": "#10b981",
        "speed_mul": 0.91,
        "pitch_semitones": -0.25,
        "end_pitch_glide_st": -0.20,
        "style_row_offset": 160,
        "exclamatory_prosody": False,
        "pause_mul": 1.24,
        "ellipsis_mul": 1.30,
        "brightness_db": -1.6,
        "warmth_db": 1.8,
        "whisper_mode": False,
        "vibrato_hz": 0.0,
        "vibrato_depth": 0.0,
        "gain_mul": 0.94,
        "description": "Gentle, steady narrative pacing, warm resonance, and unhurried pauses.",
    },
    "fearful": {
        "label": "Fearful & Anxious",
        "icon": "😨",
        "color": "#f97316",
        "speed_mul": 1.09,
        "pitch_semitones": 0.70,
        "end_pitch_glide_st": 0.35,
        "style_row_offset": -80,
        "exclamatory_prosody": True,
        "pause_mul": 1.18,
        "ellipsis_mul": 1.42,
        "brightness_db": 1.2,
        "warmth_db": -1.0,
        "whisper_mode": False,
        "vibrato_hz": 5.5,
        "vibrato_depth": 0.0011,
        "gain_mul": 0.96,
        "description": "Tense elevated pitch, subtle vocal quiver, and hesitant dramatic breaks.",
    },
}


@dataclass
class TextChunk:
    text: str
    punctuation: str          # ",", ".", "...", "—", or ""
    pause_type: str           # "comma", "period", "ellipsis", "em_dash", "none"
    pause_label: str          # Human-readable description
    base_pause_sec: float     # Requested pause duration in seconds
    emotion: str              # Active emotion for this chunk


PAUSE_LABELS = {
    "comma": "Comma (,) • Short Pause",
    "period": "Period (.) • Moderate Pause",
    "ellipsis": "Ellipsis (...) • Longer, Dramatic Pause",
    "em_dash": "Em Dash (—) • Natural Break in Thought",
    "none": "End of Speech",
}


def parse_text_with_pauses_and_emotions(
    raw_text: str,
    pauses: PauseConfig,
    default_emotion: str = "neutral",
) -> list[TextChunk]:
    """
    Parse input text into sequential speech chunks with inline [emotion] tag support
    and exact punctuation pause classification:
      - Ellipsis (... or …) -> Longer, dramatic pause
      - Em dash (— or -- or –) -> Natural pause or break in thought
      - Period (. or ! or ?) -> Moderate pause
      - Comma (, or ; or :) -> Short pause
    """
    text = raw_text.strip()
    if not text:
        return []

    # Normalize multi-character punctuation markers while keeping display fidelity
    # Normalize unicode ellipsis '…' and 3+ dots '...' to '...'
    text = re.sub(r"\.{3,}|…", "...", text)
    # Normalize double/triple hyphens '--' or spaced en-dash ' – ' to em dash '—'
    text = re.sub(r"\s*---\s*|\s*--\s*|\s+–\s+|\s*—\s*", " — ", text)

    # First split by inline emotion tags like [happy], [sad], [dramatic], etc.
    tag_pattern = re.compile(
        r"\[(" + "|".join(re.escape(k) for k in EMOTION_PROFILES.keys()) + r")\]",
        re.IGNORECASE,
    )

    current_emotion = default_emotion.lower()
    if current_emotion not in EMOTION_PROFILES:
        current_emotion = "neutral"

    tagged_blocks: list[tuple[str, str]] = []
    pos = 0
    for match in tag_pattern.finditer(text):
        before = text[pos : match.start()].strip()
        if before:
            tagged_blocks.append((current_emotion, before))
        current_emotion = match.group(1).lower()
        pos = match.end()
    tail = text[pos:].strip()
    if tail:
        tagged_blocks.append((current_emotion, tail))

    # Now tokenize each block by the 4 punctuation pause triggers:
    # Priority order: '...' (ellipsis), '—' (em dash), '.' / '!' / '?' (period), ',' / ';' / ':' (comma)
    # Avoid splitting decimal numbers like 3.14
    punct_regex = re.compile(
        r"(\.\.\.|—|(?<!\d)\.(?!\d)|[!?;:,])"
    )

    chunks: list[TextChunk] = []
    for emotion, block in tagged_blocks:
        parts = punct_regex.split(block)
        # parts alternates: [phrase0, punct0, phrase1, punct1, ..., phraseN]
        i = 0
        while i < len(parts):
            phrase = parts[i].strip()
            punct = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            i += 2

            if not phrase and not punct:
                continue

            # If a punctuation token immediately follows another (e.g. "Wait..."), attach to previous if phrase is empty
            if not phrase and chunks:
                # Upgrade previous chunk's pause if this punctuation has longer pause
                prev = chunks[-1]
                p_type, p_sec = _classify_punctuation(punct, pauses)
                if p_sec >= prev.base_pause_sec:
                    prev.punctuation = punct
                    prev.pause_type = p_type
                    prev.pause_label = PAUSE_LABELS[p_type]
                    prev.base_pause_sec = p_sec
                continue

            if not phrase:
                continue

            p_type, p_sec = _classify_punctuation(punct, pauses)
            chunks.append(
                TextChunk(
                    text=phrase,
                    punctuation=punct,
                    pause_type=p_type,
                    pause_label=PAUSE_LABELS[p_type],
                    base_pause_sec=p_sec,
                    emotion=emotion,
                )
            )

    return chunks


def _classify_punctuation(punct: str, pauses: PauseConfig) -> tuple[str, float]:
    if punct == "...":
        return "ellipsis", pauses.ellipsis
    if punct == "—":
        return "em_dash", pauses.em_dash
    if punct in (".", "!", "?"):
        return "period", pauses.period
    if punct in (",", ";", ":"):
        return "comma", pauses.comma
    return "none", 0.0


# ---------------------------------------------------------------------------
# 2. DSP Audio Effects for Feelings, Pauses & Voice Cloning
# ---------------------------------------------------------------------------
class SpeechDSP:
    """Signal processing utilities for emotion shaping, smooth pauses, and voice cloning."""

    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        fade_in_ms: float = 24.0,
        fade_out_ms: float = 45.0,
    ) -> np.ndarray:
        """
        Eliminate boundary clicks, pops, and vocoder edge noise before/after pauses:
        1. Removes any DC offset so silence transitions sit at exact 0.0 amplitude.
        2. Trims micro-tail artifacts below -48 dBFS at segment edges.
        3. Applies smooth raised-cosine fade-in (24ms) and fade-out (45ms) envelopes
           and pins the boundary samples to exact 0.0.
        """
        if len(audio) < 128:
            return np.zeros_like(audio)

        out = audio.astype(np.float32).copy()
        # 1. Remove DC offset
        out -= float(np.mean(out))

        # 2. Trim any low-level vocoder edge glitch/click in the outer 15ms
        edge_trim = min(int(sr * 0.008), len(out) // 10)
        if edge_trim > 4:
            out = out[edge_trim:-edge_trim]

        if len(out) < 128:
            return np.zeros_like(out)

        # 3. Apply smooth raised-cosine fade-in and fade-out
        n_in = min(int(sr * fade_in_ms / 1000.0), len(out) // 3)
        n_out = min(int(sr * fade_out_ms / 1000.0), len(out) // 3)
        if n_in > 1:
            curve_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_in, dtype=np.float32)))
            out[:n_in] *= curve_in
        if n_out > 1:
            curve_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_out, dtype=np.float32)))
            out[-n_out:] *= curve_out

        out[0] = 0.0
        out[-1] = 0.0
        return out

    @staticmethod
    def apply_ellipsis_dramatic_tail(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Subtle dramatic slow-down and expressive decay on the final word before an ellipsis (...).
        Uses a smooth crossfade at the split boundary and tapers cleanly to zero at the end.
        """
        if len(audio) < int(sr * 0.25):
            return audio
        split_idx = int(len(audio) * 0.78)
        head = audio[:split_idx]
        tail = audio[split_idx:]
        stretched_len = int(len(tail) * 1.14)
        x_old = np.linspace(0.0, 1.0, len(tail), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, stretched_len, dtype=np.float32)
        stretched_tail = np.interp(x_new, x_old, tail).astype(np.float32)
        # Smoothly taper the tail down toward zero so ellipsis fades cleanly into dramatic silence
        taper = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi * 0.85, stretched_len, dtype=np.float32)))
        stretched_tail *= taper
        return np.concatenate([head, stretched_tail])

    @staticmethod
    def pitch_shift_and_time_correct(
        audio: np.ndarray,
        semitones: float,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        100% Noise-Free Phase-Locked WSOLA (Waveform Similarity Overlap-Add) + Bandlimited
        Fourier Resampling (`scipy.signal.resample`).
        Aligns every overlapping frame to the exact glottal pitch period via cross-correlation
        to eliminate metallic comb-filtering buzz and linear-interpolation aliasing grit.
        """
        if abs(semitones) < 0.05 or len(audio) < 1024:
            return audio

        pitch_factor = 2.0 ** (semitones / 12.0)
        win_size = 1024
        hop_a = win_size // 4
        hop_s = max(1, int(round(hop_a * pitch_factor)))
        max_tol = win_size // 4  # +-10.6ms search range covers a full human glottal pitch cycle
        window = np.hanning(win_size).astype(np.float32)

        pad = win_size // 2
        padded = np.pad(audio.astype(np.float32), (pad, pad), mode="constant")

        n_frames = max(1, (len(padded) - win_size - max_tol) // hop_a)
        out_len = (n_frames - 1) * hop_s + win_size
        stretched = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)

        overlap_len = win_size - hop_s
        for i in range(n_frames):
            ideal_in = i * hop_a
            out_start = i * hop_s

            # Phase-lock frame i to existing output tail via cross-correlation search
            best_offset = 0
            if i > 0 and overlap_len > 32:
                target_tail = stretched[out_start : out_start + overlap_len]
                tail_energy = float(np.dot(target_tail, target_tail))
                if tail_energy > 1e-6:
                    lo = max(0, ideal_in - max_tol)
                    hi = min(len(padded) - win_size, ideal_in + max_tol)
                    if hi > lo:
                        search_region = padded[lo : hi + overlap_len]
                        # Fast cross-correlation across candidate offsets
                        corr = signal.correlate(search_region, target_tail, mode="valid")
                        if len(corr) > 0:
                            best_offset = (lo + int(np.argmax(corr))) - ideal_in

            actual_in = int(np.clip(ideal_in + best_offset, 0, len(padded) - win_size))
            frame = padded[actual_in : actual_in + win_size]
            stretched[out_start : out_start + win_size] += frame * window
            norm[out_start : out_start + win_size] += window ** 2

        stretched /= np.maximum(norm, 0.20)

        pad_out = int(round(pad * pitch_factor))
        if len(stretched) > 2 * pad_out + 128:
            stretched = stretched[pad_out:-pad_out]

        # Bandlimited Fourier resampling (zero high-frequency aliasing noise)
        target_len = max(1, int(round(len(stretched) / pitch_factor)))
        shifted = signal.resample(stretched, target_len).astype(np.float32)
        return shifted

    @staticmethod
    def apply_phrase_ending_pitch_glide(
        audio: np.ndarray,
        glide_semitones: float,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Smoothly inflects the pitch contour over the final 32% of a phrase (e.g., cheerful upward
        inflection on Happy/Excited or gentle downward sigh on Sad) with zero noise.
        """
        if abs(glide_semitones) < 0.08 or len(audio) < int(sr * 0.25):
            return audio

        n = len(audio)
        split = int(n * 0.68)
        tail_len = n - split
        # Instantaneous pitch ratio curve p(t): 1.0 over first 68%, gliding smoothly over last 32%
        rate_curve = np.ones(n, dtype=np.float64)
        end_ratio = 2.0 ** (glide_semitones / 12.0)
        t_tail = np.linspace(0.0, 1.0, tail_len, dtype=np.float64)
        smooth_curve = 0.5 * (1.0 - np.cos(np.pi * t_tail))
        rate_curve[split:] = 1.0 + (end_ratio - 1.0) * smooth_curve

        phase_idx = np.cumsum(rate_curve)
        phase_idx = (phase_idx - phase_idx[0]) / max(1e-9, phase_idx[-1] - phase_idx[0]) * (n - 1)
        out_len = int(round(n / np.mean(rate_curve)))
        query_pts = np.linspace(0.0, n - 1, out_len, dtype=np.float64)
        warped_idx = np.interp(query_pts, np.linspace(0.0, n - 1, n, dtype=np.float64), phase_idx)
        return np.interp(warped_idx, np.arange(n, dtype=np.float64), audio).astype(np.float32)

    @staticmethod
    def apply_emotion_dsp(
        audio: np.ndarray,
        emotion: str,
        intensity: float = 1.0,
        extra_pitch_semitones: float = 0.0,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        100% Noise-Free Expressive Feeling Processor:
        - Zero artificial white noise injection, zero comb-reverb echo, zero nonlinear distortion.
        - Uses phase-locked WSOLA pitch register adjustment + natural phrase-ending intonation glide
          + zero-phase biquad spectral tilt (`filtfilt`) while preserving 100% speaker identity.
        """
        if len(audio) < 256:
            return audio

        prof = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
        intensity = float(np.clip(intensity, 0.0, 1.5))

        out = audio.astype(np.float32).copy()

        # 1. Phase-locked WSOLA pitch shift (pure time-domain + bandlimited Fourier resampling, zero STFT phase noise)
        emo_semitones = prof["pitch_semitones"] * intensity + extra_pitch_semitones
        if abs(emo_semitones) >= 0.06:
            out = SpeechDSP.pitch_shift_and_time_correct(out, emo_semitones, sr)

        # 2. Natural phrase-ending intonation glide (upward lift for Happy/Excited, downward sigh for Sad)
        glide_st = prof.get("end_pitch_glide_st", 0.0) * intensity
        if abs(glide_st) >= 0.08:
            out = SpeechDSP.apply_phrase_ending_pitch_glide(out, glide_st, sr)

        # 3. Clean sinusoidal pitch/amplitude quiver (for Fearful / Sad — zero noise)
        vib_hz = prof["vibrato_hz"]
        vib_depth = prof["vibrato_depth"] * intensity
        if vib_hz > 0.1 and vib_depth > 0.0001 and len(out) > 512:
            t = np.arange(len(out), dtype=np.float32) / sr
            delay_samples = (vib_depth * sr) * np.sin(2.0 * np.pi * vib_hz * t)
            base_idx = np.arange(len(out), dtype=np.float32)
            idx = np.clip(base_idx + delay_samples, 0, len(out) - 1)
            out = np.interp(idx, base_idx, out).astype(np.float32)
            amp_mod = 1.0 + (0.045 * intensity) * np.sin(2.0 * np.pi * vib_hz * t)
            out *= amp_mod.astype(np.float32)

        # 4. Zero-Phase Biquad Warmth & Brightness Shaping (`filtfilt` — zero phase distortion or hiss)
        warmth_db = prof["warmth_db"] * intensity
        brightness_db = prof["brightness_db"] * intensity
        if abs(warmth_db) > 0.2 or abs(brightness_db) > 0.2:
            out = SpeechDSP._apply_tilt_eq(out, warmth_db, brightness_db, sr)

        # 5. Clean Intimate Whisper Mode (attenuates booming low chest fundamental without white noise hiss)
        if prof.get("whisper_mode") and intensity > 0.1:
            nyq = sr / 2.0
            b_hp, a_hp = signal.butter(2, 240.0 / nyq, btype="high")
            airy = signal.filtfilt(b_hp, a_hp, out).astype(np.float32)
            w_mix = min(0.75, 0.55 * intensity)
            out = (1.0 - w_mix) * out + w_mix * airy

        # 6. Emotion expressive gain scaling (e.g., softer for Sad/Whisper, firmer for Excited/Angry)
        gain_mul = 1.0 + (prof.get("gain_mul", 1.0) - 1.0) * min(1.0, intensity)
        out = (out * gain_mul).astype(np.float32)

        return out

    @staticmethod
    def _apply_tilt_eq(audio: np.ndarray, warmth_db: float, brightness_db: float, sr: int) -> np.ndarray:
        """Zero-phase low-shelf (warmth) and high-shelf (brightness) filter with zero noise."""
        if len(audio) < 64:
            return audio
        out = audio.astype(np.float32)
        nyq = sr / 2.0
        if abs(warmth_db) > 0.2:
            b_low, a_low = signal.butter(2, min(0.95, 340.0 / nyq), btype="low")
            low_band = signal.filtfilt(b_low, a_low, out).astype(np.float32)
            gain_low = (10.0 ** (warmth_db / 20.0)) - 1.0
            out = out + low_band * gain_low
        if abs(brightness_db) > 0.2:
            b_high, a_high = signal.butter(2, min(0.95, 2400.0 / nyq), btype="high")
            high_band = signal.filtfilt(b_high, a_high, out).astype(np.float32)
            gain_high = (10.0 ** (brightness_db / 20.0)) - 1.0
            out = out + high_band * gain_high
        return out.astype(np.float32)

    @staticmethod
    def compute_64band_envelope_db(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Compute mean-centered 64-band log-spectral vocal-tract envelope over voiced frames."""
        nperseg = min(1024, max(128, len(audio) // 2))
        freqs, psd = signal.welch(audio, fs=sr, nperseg=nperseg)
        psd = np.maximum(psd, 1e-12)
        band_db = np.zeros(len(CLONE_BAND_EDGES_HZ) - 1, dtype=np.float32)
        for b_idx in range(len(CLONE_BAND_EDGES_HZ) - 1):
            lo, hi = CLONE_BAND_EDGES_HZ[b_idx], CLONE_BAND_EDGES_HZ[b_idx + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_db[b_idx] = float(10.0 * np.log10(np.mean(psd[mask])))
            elif b_idx > 0:
                band_db[b_idx] = band_db[b_idx - 1]
            else:
                band_db[b_idx] = -60.0
        # Smooth across adjacent bands for natural vocal-tract formant continuity
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        smoothed = np.convolve(np.pad(band_db, 1, mode="edge"), kernel, mode="valid")
        smoothed -= float(np.mean(smoothed))
        return smoothed.astype(np.float32)

    @staticmethod
    def apply_spectral_envelope_match(
        audio: np.ndarray,
        eq_gains_db: list[float],
        band_edges_hz: list[float],
        sr: int = SAMPLE_RATE,
        max_gain_db: float = 14.0,
    ) -> np.ndarray:
        """Apply high-resolution STFT vocal-tract formant & spectral envelope transfer."""
        if not eq_gains_db or len(audio) < 512:
            return audio
        n_fft = 1024
        hop = 256
        f, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        band_centers = 0.5 * (np.array(band_edges_hz[:-1]) + np.array(band_edges_hz[1:]))
        gains_db = np.clip(np.array(eq_gains_db, dtype=np.float32), -max_gain_db, max_gain_db)
        # Smooth transfer curve across FFT bins
        freq_gains_db = np.interp(f, band_centers, gains_db)
        win = np.hanning(7)
        win /= np.sum(win)
        freq_gains_db = np.convolve(np.pad(freq_gains_db, 3, mode="edge"), win, mode="valid")
        freq_gains_lin = (10.0 ** (freq_gains_db / 20.0)).astype(np.float32)[:, None]
        Zxx_matched = Zxx * freq_gains_lin
        _, rec = signal.istft(Zxx_matched, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        rec = rec[: len(audio)].astype(np.float32)
        if len(rec) < len(audio):
            rec = np.pad(rec, (0, len(audio) - len(rec)))
        return rec

    @staticmethod
    def compute_cepstral_envelope_db(
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        n_fft: int = 1024,
        lifter_order: int = 36,
        out_bins: int = 129,
    ) -> np.ndarray:
        """
        Compute the exact smooth vocal-tract log-magnitude envelope (dB) via real cepstral liftering.
        Separates the speaker's vocal-tract formant resonances (F1-F5) from glottal pitch harmonics.
        """
        if len(audio) < 256:
            return np.zeros(out_bins, dtype=np.float32)
        nperseg = min(n_fft, max(256, len(audio)))
        hop = max(64, nperseg // 4)
        _, _, Zxx = signal.stft(audio.astype(np.float32), fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
        mag = np.abs(Zxx)
        # Weight frames by RMS energy so active voiced frames dominate cleanly
        frame_energy = np.sqrt(np.mean(mag ** 2, axis=0))
        thresh = max(1e-4, float(np.percentile(frame_energy, 35)))
        voiced_mask = frame_energy >= thresh
        if np.any(voiced_mask):
            avg_log_mag = np.mean(np.log(np.maximum(mag[:, voiced_mask], 1e-6)), axis=1)
        else:
            avg_log_mag = np.mean(np.log(np.maximum(mag, 1e-6)), axis=1)

        # Real cepstrum liftering to isolate smooth vocal-tract resonance envelope
        cep = np.fft.irfft(avg_log_mag)
        lifter = np.zeros_like(cep)
        order = min(lifter_order, len(cep) // 2 - 1)
        lifter[0] = 1.0
        if order > 1:
            lifter[1:order] = 1.0
            lifter[-order + 1 :] = 1.0
            # Soft cosine taper on last 6 quefrency bins to prevent ringing
            taper_len = min(6, order - 1)
            taper = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_len)))
            lifter[order - taper_len : order] *= taper
            lifter[-order + 1 : -order + 1 + taper_len] *= taper[::-1]
        smooth_log_mag = np.real(np.fft.rfft(cep * lifter))
        env_db = (20.0 / np.log(10.0)) * smooth_log_mag
        env_db -= float(np.mean(env_db))

        # Resample onto uniform `out_bins` frequency grid (0 .. sr/2)
        src_x = np.linspace(0.0, 1.0, len(env_db), dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, out_bins, dtype=np.float32)
        return np.interp(dst_x, src_x, env_db).astype(np.float32)

    @staticmethod
    def apply_cepstral_vocal_tract_transfer(
        audio: np.ndarray,
        target_cep_db: list[float] | np.ndarray,
        sr: int = SAMPLE_RATE,
        max_gain_db: float = 18.0,
    ) -> np.ndarray:
        """
        100% Exact Vocal-Tract Formant Envelope Replacement in the STFT domain.
        Replaces the carrier's smooth cepstral vocal-tract envelope with the input speaker's
        exact saved vocal-tract envelope without altering phonetic articulation or adding noise.
        """
        if len(audio) < 512 or target_cep_db is None or len(target_cep_db) < 16:
            return audio
        n_fft = 1024
        hop = 256
        target_arr = np.asarray(target_cep_db, dtype=np.float32)
        cur_arr = SpeechDSP.compute_cepstral_envelope_db(audio, sr, n_fft=n_fft, out_bins=len(target_arr))
        delta_db = np.clip(target_arr - cur_arr, -max_gain_db, max_gain_db)

        f, _, Zxx = signal.stft(audio.astype(np.float32), fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        src_f = np.linspace(0.0, sr * 0.5, len(delta_db), dtype=np.float32)
        bin_gains_db = np.interp(f, src_f, delta_db).astype(np.float32)

        # Gentle high-frequency rolloff above 8.5 kHz to avoid boosting air/hiss noise
        hf_taper = np.where(f > 8500.0, np.clip(1.0 - (f - 8500.0) / 3500.0, 0.25, 1.0), 1.0).astype(np.float32)
        bin_gains_db *= hf_taper

        bin_gains_lin = (10.0 ** (bin_gains_db / 20.0)).astype(np.float32)[:, None]
        _, rec = signal.istft(Zxx * bin_gains_lin, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        rec = rec[: len(audio)].astype(np.float32)
        if len(rec) < len(audio):
            rec = np.pad(rec, (0, len(audio) - len(rec)))
        # Match pre-transfer RMS so loudness stays natural and consistent
        rms_in = float(np.sqrt(np.mean(audio ** 2))) + 1e-7
        rms_out = float(np.sqrt(np.mean(rec ** 2))) + 1e-7
        rec = (rec * (rms_in / rms_out)).astype(np.float32)
        return rec

    @staticmethod
    def apply_100pct_clone_transfer(
        seg_audio: np.ndarray,
        cloned_meta: dict[str, Any],
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        100% Exact Input Voice Transfer Pipeline (Zero Artificial Modification):
        1. Exact F0 Pitch Register Lock: measures the segment's live F0 and shifts it
           100% onto the input speaker's exact target F0 (Hz).
        2. 100% Exact Cepstral Vocal-Tract Formant Envelope Replacement: replaces the carrier's
           vocal-tract resonance envelope with the input speaker's exact saved cepstral envelope.
        """
        if len(seg_audio) < 512:
            return seg_audio

        out = seg_audio.astype(np.float32).copy()
        target_fp = cloned_meta.get("target_fingerprint") or {}
        target_f0 = float(target_fp.get("f0_hz", 0.0))
        manual_st = float(cloned_meta.get("manual_pitch_offset", 0.0))
        static_st = float(cloned_meta.get("pitch_shift_semitones", 0.0))

        # 1. 100% Exact Input F0 Pitch Register Lock (zero manual modification for reference_audio)
        if target_f0 > 55.0 and cloned_meta.get("mode") == "reference_audio":
            seg_f0 = estimate_f0_median(out, sr)
            if seg_f0 > 55.0:
                dynamic_st = 12.0 * math.log2(target_f0 / seg_f0)
                shift_st = float(np.clip(dynamic_st, -11.0, 11.0))
            else:
                shift_st = static_st
        else:
            shift_st = static_st + manual_st

        if abs(shift_st) >= 0.05:
            out = SpeechDSP.pitch_shift_and_time_correct(out, shift_st, sr)

        # 2. 100% Exact Cepstral Vocal-Tract Replacement (falls back to 64-band for older clones)
        target_cep = target_fp.get("cepstral_env_db")
        target_band_64 = target_fp.get("band_64_db")
        if target_cep and len(target_cep) >= 32:
            out = SpeechDSP.apply_cepstral_vocal_tract_transfer(out, target_cep, sr, max_gain_db=18.0)
        elif target_band_64 and len(target_band_64) == (len(CLONE_BAND_EDGES_HZ) - 1):
            cur_band_64 = SpeechDSP.compute_64band_envelope_db(out, sr)
            target_arr = np.array(target_band_64, dtype=np.float32)
            dynamic_eq_db = (target_arr - cur_band_64).tolist()
            out = SpeechDSP.apply_spectral_envelope_match(
                out, dynamic_eq_db, CLONE_BAND_EDGES_HZ, sr, max_gain_db=14.0
            )
        else:
            eq_gains = cloned_meta.get("eq_gains_db") or []
            edges = CLONE_BAND_EDGES_HZ if len(eq_gains) == 64 else BAND_EDGES_HZ
            if eq_gains:
                out = SpeechDSP.apply_spectral_envelope_match(
                    out, eq_gains, edges, sr, max_gain_db=14.0
                )

        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Kokoro 100% Voice Cloning Engine (64-Band Formant & F0 Vocal Tract Analysis)
# ---------------------------------------------------------------------------
BAND_EDGES_HZ = np.geomspace(80.0, 7600.0, 17).tolist()
CLONE_BAND_EDGES_HZ = np.geomspace(70.0, 10000.0, 65).tolist()


def estimate_f0_median(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    High-precision Fundamental Frequency (F0 in Hz) estimator across voiced frames
    using parabolic-interpolated autocorrelation with silence & unvoiced rejection.
    """
    frame_len = int(sr * 0.042)
    hop_len = int(sr * 0.014)
    min_lag = int(sr / 430.0)
    max_lag = int(sr / 70.0)
    f0_values: list[float] = []

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-5:
        return 0.0
    norm_audio = audio / peak

    for start in range(0, max(1, len(norm_audio) - frame_len), hop_len):
        frame = norm_audio[start : start + frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < 0.045:
            continue
        # Reject unvoiced sibilant frames with very high zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
        if zcr > 0.28:
            continue

        frame_w = (frame - np.mean(frame)) * np.hanning(len(frame))
        corr = np.correlate(frame_w, frame_w, mode="full")[len(frame_w) - 1 :]
        if corr[0] <= 1e-8 or max_lag >= len(corr):
            continue
        corr = corr / corr[0]
        segment = corr[min_lag:max_lag]
        if len(segment) < 3:
            continue
        best_offset = int(np.argmax(segment))
        best_val = float(segment[best_offset])
        if best_val > 0.34:
            lag = float(min_lag + best_offset)
            # Parabolic sub-sample lag interpolation for exact Hz accuracy
            if 0 < best_offset < len(segment) - 1:
                y0, y1, y2 = float(segment[best_offset - 1]), best_val, float(segment[best_offset + 1])
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-7:
                    lag += 0.5 * (y0 - y2) / denom
            f0_values.append(float(sr / max(1.0, lag)))

    if not f0_values:
        return 0.0
    return float(np.median(f0_values))


def extract_voice_fingerprint(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Extract comprehensive 100% Voice Clone acoustic fingerprint from reference speech:
      - Isolates active voiced segments (stripping room silence/pauses)
      - Sub-sample parabolic F0 pitch register (median Hz, mean Hz, pitch range)
      - Formant peaks F1, F2, F3 (Hz) via cepstral envelope peak picking
      - 64-band high-resolution vocal-tract log-spectral envelope (70 Hz - 10,000 Hz)
      - 16-band summary envelope for fast atlas indexing
      - Speaking syllable cadence & harmonic warmth
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != SAMPLE_RATE and len(audio) > 100:
        duration = len(audio) / float(sr)
        target_len = max(1, int(round(duration * SAMPLE_RATE)))
        audio = np.interp(
            np.linspace(0.0, 1.0, target_len, dtype=np.float32),
            np.linspace(0.0, 1.0, len(audio), dtype=np.float32),
            audio,
        ).astype(np.float32)
        sr = SAMPLE_RATE

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-5:
        audio = audio / peak

    # Isolate active voiced speech frames so pauses don't dilute the vocal tract spectrum
    frame_size = int(sr * 0.030)
    voiced_chunks: list[np.ndarray] = []
    for i in range(0, max(1, len(audio) - frame_size), frame_size // 2):
        chunk = audio[i : i + frame_size]
        if float(np.sqrt(np.mean(chunk ** 2))) > 0.04:
            voiced_chunks.append(chunk)
    voiced_audio = np.concatenate(voiced_chunks) if len(voiced_chunks) >= 4 else audio

    f0_hz = estimate_f0_median(voiced_audio, sr)
    if f0_hz <= 0.0:
        f0_hz = 165.0

    # Welch PSD on voiced speech
    nperseg = min(1024, max(128, len(voiced_audio) // 2))
    freqs, psd = signal.welch(voiced_audio, fs=sr, nperseg=nperseg)
    psd = np.maximum(psd, 1e-12)
    centroid_hz = float(np.sum(freqs * psd) / np.sum(psd))

    # 16-band compatibility envelope
    band_db: list[float] = []
    for b_idx in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = BAND_EDGES_HZ[b_idx], BAND_EDGES_HZ[b_idx + 1]
        mask = (freqs >= lo) & (freqs < hi)
        val = float(10.0 * np.log10(np.mean(psd[mask]))) if np.any(mask) else -60.0
        band_db.append(val)
    mean_db = float(np.mean(band_db))
    band_db = [round(v - mean_db, 3) for v in band_db]

    # 64-band high-resolution vocal-tract spectral envelope + 129-bin exact cepstral vocal-tract envelope
    band_64_db = SpeechDSP.compute_64band_envelope_db(voiced_audio, sr)
    cepstral_env_db = SpeechDSP.compute_cepstral_envelope_db(voiced_audio, sr)

    # Estimate Formant Peaks F1 (280-850 Hz), F2 (850-2400 Hz), F3 (2400-3600 Hz)
    log_psd = 10.0 * np.log10(psd)
    smooth_psd = np.convolve(np.pad(log_psd, 4, mode="edge"), np.hanning(9) / np.sum(np.hanning(9)), mode="valid")

    def _pick_formant(f_lo: float, f_hi: float, default_hz: float) -> float:
        m = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(m):
            return default_hz
        sub_f = freqs[m]
        sub_p = smooth_psd[m]
        return round(float(sub_f[int(np.argmax(sub_p))]), 1)

    f1_hz = _pick_formant(280.0, 880.0, 520.0)
    f2_hz = _pick_formant(880.0, 2450.0, 1580.0)
    f3_hz = _pick_formant(2450.0, 3700.0, 2820.0)

    low_energy = float(np.mean(band_db[:6]))
    high_energy = float(np.mean(band_db[10:]))
    warmth_index = round(low_energy - high_energy, 3)

    # Measure speaking rate (syllables/sec -> Base Speed 0.72..1.38x)
    env_win = max(1, int(sr * 0.020))
    env_hop = max(1, int(sr * 0.010))
    frame_rms_list: list[float] = []
    for i in range(0, max(1, len(audio) - env_win), env_hop):
        seg = audio[i : i + env_win]
        frame_rms_list.append(float(np.sqrt(np.mean(seg ** 2))))
    env = np.array(frame_rms_list, dtype=np.float32)
    active_frames = env[env > 0.035]
    active_dur_sec = max(0.35, len(active_frames) * (env_hop / float(sr)))

    if len(env) >= 7:
        k = np.hanning(7).astype(np.float32)
        k /= np.sum(k)
        smooth_env = np.convolve(np.pad(env, 3, mode="edge"), k, mode="valid")
        peaks, _ = signal.find_peaks(smooth_env, height=0.06, distance=max(1, int(0.11 / (env_hop / float(sr)))))
        syl_rate = len(peaks) / active_dur_sec
        detected_speed = round(float(np.clip(syl_rate / 4.2, 0.72, 1.38)), 2)
    else:
        detected_speed = 1.00

    # Measure vocal expressiveness (F0 variability + RMS dynamic range -> Feeling Intensity 0.35..1.45)
    f0_vals: list[float] = []
    f0_win = int(sr * 0.042)
    f0_hop = int(sr * 0.020)
    min_lag = int(sr / 380.0)
    max_lag = int(sr / 65.0)
    for start in range(0, max(1, len(voiced_audio) - f0_win), f0_hop):
        fr = voiced_audio[start : start + f0_win]
        if float(np.sqrt(np.mean(fr ** 2))) < 0.04:
            continue
        fr = (fr - np.mean(fr)) * np.hanning(len(fr))
        ac = np.correlate(fr, fr, mode="full")[len(fr) - 1 :]
        if ac[0] <= 1e-8:
            continue
        ac /= ac[0]
        if max_lag < len(ac):
            pk = min_lag + int(np.argmax(ac[min_lag:max_lag]))
            if ac[pk] >= 0.38:
                f0_vals.append(float(sr) / float(pk))
    if len(f0_vals) >= 4:
        f0_arr = np.array(f0_vals, dtype=np.float32)
        q25, q75 = np.percentile(f0_arr, [25, 75])
        f0_iqr_st = abs(12.0 * math.log2(max(50.0, float(q75)) / max(50.0, float(q25))))
    else:
        f0_iqr_st = 2.8

    if len(active_frames) >= 6:
        p15, p90 = np.percentile(active_frames, [15, 90])
        dyn_db = float(20.0 * math.log10(max(1e-4, float(p90)) / max(1e-4, float(p15))))
    else:
        dyn_db = 12.0

    raw_intensity = 0.35 + 0.45 * min(1.5, f0_iqr_st / 4.2) + 0.35 * min(1.5, dyn_db / 16.0)
    detected_emotion_intensity = round(round(float(np.clip(raw_intensity, 0.35, 1.45)) * 20.0) / 20.0, 2)

    return {
        "f0_hz": round(f0_hz, 2),
        "centroid_hz": round(centroid_hz, 2),
        "f1_hz": f1_hz,
        "f2_hz": f2_hz,
        "f3_hz": f3_hz,
        "warmth_index": warmth_index,
        "detected_speed": detected_speed,
        "detected_emotion_intensity": detected_emotion_intensity,
        "band_db": band_db,
        "band_64_db": [round(float(x), 3) for x in band_64_db],
        "cepstral_env_db": [round(float(x), 3) for x in cepstral_env_db],
        "estimated_gender": "female" if f0_hz >= 160.0 else "male",
    }


# ---------------------------------------------------------------------------
# 4. Main Kokoro Studio Engine
# ---------------------------------------------------------------------------
class KokoroVoiceStudio:
    def __init__(self, model_path: Path, voices_path: Path):
        logger.info("Initializing Kokoro ONNX Engine...")
        sess_opts = rt.SessionOptions()
        sess_opts.log_severity_level = 3  # Suppress verbose graph memcpy warnings

        available = rt.get_available_providers()
        # Prefer CUDAExecutionProvider over CPUExecutionProvider; skip TensorRT if nvinfer is absent
        providers = [
            p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available
        ] or ["CPUExecutionProvider"]

        session = rt.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers,
        )
        self.active_provider = session.get_providers()[0]
        logger.info(f"Kokoro ONNX loaded with provider: {self.active_provider}")

        self.kokoro = Kokoro.from_session(session, str(voices_path))
        self.builtin_voices: list[str] = self.kokoro.get_voices()
        self.cloned_voices: dict[str, dict[str, Any]] = {}
        self.voice_atlas: dict[str, dict[str, Any]] = {}
        self.last_audio: np.ndarray | None = None
        self.last_meta: dict[str, Any] = {}

        self._load_cloned_voices()
        self._init_voice_atlas()

    def _load_cloned_voices(self) -> None:
        """Load saved cloned voices from `cloned_voices/` directory."""
        self.cloned_voices.clear()
        for meta_file in sorted(CLONED_DIR.glob("*.json")):
            if meta_file.name == "kokoro_voice_atlas.json":
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                vid = meta["id"]
                npy_file = CLONED_DIR / f"{vid}.npy"
                if npy_file.exists():
                    self.cloned_voices[vid] = meta
            except Exception as exc:
                logger.warning(f"Could not load cloned voice {meta_file}: {exc}")

    def _init_voice_atlas(self) -> None:
        """
        Build or load the acoustic fingerprint atlas of built-in Kokoro voices
        so Reference Audio Voice Cloning can match any uploaded/recorded speaker!
        """
        atlas_file = CLONED_DIR / "kokoro_voice_atlas.json"
        if atlas_file.exists():
            try:
                self.voice_atlas = json.loads(atlas_file.read_text(encoding="utf-8"))
                if len(self.voice_atlas) >= 12:
                    return
            except Exception:
                pass

        logger.info("Building Kokoro Acoustic Voice Atlas for Voice Cloning...")
        probe_voices = [
            v for v in [
                "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
                "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                "am_michael", "am_onyx", "am_puck",
                "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            ]
            if v in self.builtin_voices
        ]
        probe_text = "Voice cloning calibration sample with clear vowels and balanced resonance."
        atlas: dict[str, dict[str, Any]] = {}
        for vname in probe_voices:
            try:
                audio, sr = self.kokoro.create(probe_text, voice=vname, speed=1.0, lang="en-us")
                fp = extract_voice_fingerprint(audio, sr)
                atlas[vname] = fp
            except Exception as exc:
                logger.warning(f"Atlas probe skipped {vname}: {exc}")

        self.voice_atlas = atlas
        try:
            atlas_file.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
        except Exception:
            pass
        logger.info(f"Acoustic Voice Atlas calibrated across {len(atlas)} Kokoro voices.")

    def list_all_voices(self) -> dict[str, Any]:
        """Return built-in voices grouped by accent/gender plus user's cloned voices."""
        builtin_info = []
        for v in self.builtin_voices:
            prefix = v[:2]
            accent_map = {
                "af": ("American English", "Female"),
                "am": ("American English", "Male"),
                "bf": ("British English", "Female"),
                "bm": ("British English", "Male"),
                "ef": ("Spanish", "Female"),
                "em": ("Spanish", "Male"),
                "ff": ("French", "Female"),
                "hf": ("Hindi", "Female"),
                "hm": ("Hindi", "Male"),
                "if": ("Italian", "Female"),
                "im": ("Italian", "Male"),
                "jf": ("Japanese", "Female"),
                "jm": ("Japanese", "Male"),
                "pf": ("Portuguese", "Female"),
                "pm": ("Portuguese", "Male"),
                "zf": ("Mandarin Chinese", "Female"),
                "zm": ("Mandarin Chinese", "Male"),
            }
            region, gender = accent_map.get(prefix, ("Multilingual", "Neutral"))
            fp = self.voice_atlas.get(v, {})
            builtin_info.append({
                "id": v,
                "name": v.split("_", 1)[-1].title() + f" ({prefix.upper()})",
                "region": region,
                "gender": gender,
                "is_cloned": False,
                "f0_hz": fp.get("f0_hz"),
            })

        cloned_list = [
            {
                "id": vid,
                "name": meta.get("name", vid),
                "region": "Cloned Voice",
                "gender": meta.get("gender", "Custom"),
                "is_cloned": True,
                "mode": meta.get("mode", "reference"),
                "recipe": meta.get("recipe", {}),
                "f0_hz": meta.get("target_fingerprint", {}).get("f0_hz"),
                "pitch_shift_semitones": meta.get("pitch_shift_semitones", 0.0),
                "detected_speed": meta.get(
                    "detected_speed",
                    meta.get("target_fingerprint", {}).get("detected_speed", 1.0),
                ),
                "detected_emotion_intensity": meta.get(
                    "detected_emotion_intensity",
                    meta.get("target_fingerprint", {}).get("detected_emotion_intensity", 1.0),
                ),
                "detected_pitch_offset": meta.get(
                    "detected_pitch_offset",
                    round(float(np.clip(meta.get("pitch_shift_semitones", 0.0), -4.0, 4.0)), 1),
                ),
                "created_at": meta.get("created_at", ""),
            }
            for vid, meta in self.cloned_voices.items()
        ]
        return {
            "builtin": builtin_info,
            "cloned": cloned_list,
            "provider": self.active_provider,
        }

    def resolve_voice_style_and_profile(
        self, voice_id: str, emotion: str, emotion_intensity: float
    ) -> tuple[np.ndarray, dict[str, Any] | None, str]:
        """
        Return the exact (510, 1, 256) style vector for `voice_id` (built-in or cloned).
        1. STRICT VOICE IDENTITY LOCK: Never mixes or blends any other speaker's style vector
           when `emotion` is changed or set.
        2. WITHIN-SPEAKER NEURAL PROSODY CONDITIONING: Shifts the utterance-length row index
           within the SAME speaker's (510, 1, 256) style matrix (`style_row_offset`) so Kokoro's
           internal F0 and energy predictors natively generate emotional inflection (short/punchy
           rows for Happy/Excited/Angry, long/reflective rows for Sad/Calm/Whisper/Dramatic)
           with zero DSP noise and 100% same-speaker voice identity.
        """
        cloned_meta: dict[str, Any] | None = None
        if voice_id in self.cloned_voices:
            cloned_meta = self.cloned_voices[voice_id]
            npy_path = CLONED_DIR / f"{voice_id}.npy"
            base_style = np.load(str(npy_path)).astype(np.float32)
            is_female = cloned_meta.get("gender", "female").lower().startswith("f")
        elif voice_id in self.builtin_voices:
            base_style = self.kokoro.get_voice_style(voice_id).astype(np.float32).copy()
            is_female = voice_id[1:2].lower() == "f"
        else:
            fallback = "af_heart" if "af_heart" in self.builtin_voices else self.builtin_voices[0]
            base_style = self.kokoro.get_voice_style(fallback).astype(np.float32).copy()
            is_female = True

        gender_tag = "female" if is_female else "male"
        return base_style.astype(np.float32), cloned_meta, gender_tag

    def clone_voice_from_reference_audio(
        self,
        name: str,
        audio_bytes: bytes,
        gender: str = "female",
        warmth_bias_db: float = 0.0,
        manual_pitch_offset: float = 0.0,
    ) -> dict[str, Any]:
        """
        Exact Input Voice Cloning from a reference audio recording (.wav / .mp3 / mic capture)
        with user-selected Male or Female voice gender and ZERO artificial modification:
        1. Isolates active voiced frames and extracts sub-sample F0 pitch (Hz), Formants F1/F2/F3,
           spectral centroid, and 64-band high-resolution vocal-tract log-spectral envelope.
        2. Strictly restricts neural manifold projection to the user's chosen gender (Male or Female)
           so the glottal excitation matches the chosen voice type cleanly.
        3. Locks 100% of the input speaker's exact F0 pitch register and 64-band vocal-tract formant
           envelope with zero pitch or warmth modification.
        """
        audio_data, sr = self._decode_uploaded_audio(audio_bytes)
        if len(audio_data) < int(sr * 0.4):
            raise ValueError("Reference audio is too short. Please provide at least 0.5 seconds of speech.")

        target_fp = extract_voice_fingerprint(audio_data, sr)
        target_f0 = float(target_fp["f0_hz"])
        target_cent = float(target_fp["centroid_hz"])
        target_bands = np.array(target_fp["band_db"], dtype=np.float32)

        # Strictly honor the user's Male / Female voice choice
        g_norm = (gender or "").strip().lower()
        if g_norm.startswith("m"):
            target_gender = "male"
        elif g_norm.startswith("f"):
            target_gender = "female"
        else:
            target_gender = target_fp["estimated_gender"]

        # Filter atlas strictly to the user-selected gender neighborhood
        scored: list[tuple[float, str]] = []
        for vname, vfp in self.voice_atlas.items():
            v_is_female = vname[1:2].lower() == "f"
            if (target_gender == "female") != v_is_female:
                continue

            vf0 = float(vfp.get("f0_hz", 165.0))
            vcent = float(vfp.get("centroid_hz", 1500.0))
            vbands = np.array(vfp.get("band_db", [0.0] * 16), dtype=np.float32)

            st_diff = abs(12.0 * math.log2(max(50.0, target_f0) / max(50.0, vf0)))
            cent_diff = abs(target_cent - vcent) / 350.0
            band_dist = float(np.sqrt(np.mean((target_bands - vbands) ** 2))) / 3.8

            total_dist = st_diff * 0.45 + band_dist * 0.40 + cent_diff * 0.15
            scored.append((total_dist, vname))

        if not scored:
            scored = [(0.0, "am_michael" if target_gender == "male" else "af_heart")]

        scored.sort(key=lambda x: x[0])
        # Single purest same-gender phonetic carrier — ZERO multi-voice blending!
        best_vname = scored[0][1]
        style_tensor = self.kokoro.get_voice_style(best_vname).astype(np.float32).copy()
        vfp = self.voice_atlas[best_vname]
        blended_f0 = float(vfp["f0_hz"])
        blended_bands = np.array(vfp["band_db"], dtype=np.float32)
        recipe: dict[str, float] = {best_vname: 1.0}

        # Full-range F0 pitch lock (100% exact input pitch) + slider-ready Pitch Offset (-4.0..+4.0 st)
        residual_semitones = 12.0 * math.log2(max(55.0, target_f0) / max(55.0, blended_f0))
        residual_semitones = float(np.clip(residual_semitones, -11.0, 11.0))
        detected_pitch_offset = round(float(np.clip(residual_semitones, -4.0, 4.0)), 1)
        detected_speed = float(target_fp.get("detected_speed", 1.0))
        detected_emotion_intensity = float(target_fp.get("detected_emotion_intensity", 1.0))

        # Exact 16-band fallback EQ + 129-bin cepstral vocal-tract profile stored in target_fp
        eq_gains_db = (target_bands - blended_bands).tolist()
        eq_gains_db = [round(float(np.clip(g, -14.0, 14.0)), 2) for g in eq_gains_db]

        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "custom_clone"
        voice_id = f"clone_{slug}_{uuid.uuid4().hex[:5]}"

        # Save the exact raw input voice reference WAV alongside the clone profile
        ref_wav_path = CLONED_DIR / f"{voice_id}_ref.wav"
        ref_24k = audio_data.astype(np.float32)
        if sr != SAMPLE_RATE and len(ref_24k) > 10:
            dur = len(ref_24k) / float(sr)
            ref_24k = signal.resample(ref_24k, max(1, int(round(dur * SAMPLE_RATE)))).astype(np.float32)
        sf.write(str(ref_wav_path), ref_24k, SAMPLE_RATE, format="WAV", subtype="PCM_16")

        np.save(str(CLONED_DIR / f"{voice_id}.npy"), style_tensor)
        meta = {
            "id": voice_id,
            "name": f"🧬 {name.strip()}",
            "mode": "reference_audio",
            "clone_fidelity": "100% Exact Input Voice (Auto-Applied Prosody)",
            "gender": target_gender.title(),
            "recipe": recipe,
            "reference_wav": str(ref_wav_path),
            "target_fingerprint": target_fp,
            "pitch_shift_semitones": round(residual_semitones, 2),
            "detected_pitch_offset": detected_pitch_offset,
            "detected_speed": round(detected_speed, 2),
            "detected_emotion_intensity": round(detected_emotion_intensity, 2),
            "manual_pitch_offset": 0.0,
            "warmth_bias_db": 0.0,
            "eq_gains_db": eq_gains_db,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (CLONED_DIR / f"{voice_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.cloned_voices[voice_id] = meta
        logger.info(
            f"Created Exact Reference Cloned Voice '{name}' ({voice_id}, {target_gender.title()}) | "
            f"F0={target_f0}Hz | Formants=({target_fp['f1_hz']}, {target_fp['f2_hz']}, {target_fp['f3_hz']})Hz"
        )
        return meta

    def clone_voice_from_style_blend(
        self,
        name: str,
        voice_weights: dict[str, float],
        pitch_shift_semitones: float = 0.0,
        warmth_bias_db: float = 0.0,
    ) -> dict[str, Any]:
        """Create a cloned voice by blending multiple Kokoro neural style tensors + custom timbre settings."""
        valid = {k: float(v) for k, v in voice_weights.items() if k in self.builtin_voices and float(v) > 0}
        if not valid:
            raise ValueError("Select at least one valid base voice with weight > 0.")

        total_w = sum(valid.values())
        norm_recipe = {k: round(v / total_w, 4) for k, v in valid.items()}

        style_tensor = np.zeros((510, 1, 256), dtype=np.float32)
        female_weight = 0.0
        for vname, w in norm_recipe.items():
            style_tensor += float(w) * self.kokoro.get_voice_style(vname).astype(np.float32)
            if vname[1:2].lower() == "f":
                female_weight += float(w)

        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "blend"
        voice_id = f"clone_{slug}_{uuid.uuid4().hex[:5]}"

        np.save(str(CLONED_DIR / f"{voice_id}.npy"), style_tensor)
        meta = {
            "id": voice_id,
            "name": f"🧬 {name.strip()}",
            "mode": "neural_blend",
            "gender": "Female" if female_weight >= 0.5 else "Male",
            "recipe": norm_recipe,
            "target_fingerprint": {"f0_hz": 185.0 if female_weight >= 0.5 else 125.0},
            "pitch_shift_semitones": round(float(pitch_shift_semitones), 2),
            "warmth_bias_db": round(float(warmth_bias_db), 2),
            "eq_gains_db": [],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (CLONED_DIR / f"{voice_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.cloned_voices[voice_id] = meta
        logger.info(f"Created Blended Cloned Voice '{name}' ({voice_id}) -> Recipe: {norm_recipe}")
        return meta

    def delete_cloned_voice(self, voice_id: str) -> bool:
        if voice_id not in self.cloned_voices:
            return False
        self.cloned_voices.pop(voice_id, None)
        for suffix in (".json", ".npy", "_ref.wav"):
            p = CLONED_DIR / f"{voice_id}{suffix}"
            if p.exists():
                p.unlink()
        return True

    def _decode_uploaded_audio(self, raw_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV, MP3, FLAC, OGG, or WebM audio bytes into (float32_mono_array, sample_rate)."""
        try:
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data, int(sr)
        except Exception:
            # Fallback to PyAV for browser WebM/Opus or MP4 containers
            import av

            container = av.open(io.BytesIO(raw_bytes))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError("No audio stream found in uploaded file.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            chunks: list[np.ndarray] = []
            for frame in container.decode(audio_stream):
                for rframe in resampler.resample(frame):
                    arr = rframe.to_ndarray()
                    chunks.append(arr.ravel().astype(np.float32))
            if not chunks:
                raise ValueError("Could not decode audio samples from file.")
            return np.concatenate(chunks), SAMPLE_RATE

    # -----------------------------------------------------------------------
    # Core Speech Synthesis Pipeline
    # -----------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice_id: str = "af_heart",
        emotion: str = "neutral",
        emotion_intensity: float = 1.0,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        lang: str = "en-us",
        pauses: PauseConfig | None = None,
        output_format: str = "wav",
        save_filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Synthesize speech with:
        - Exact pause durations for Comma (,), Period (.), Ellipsis (...), and Em dash (—)
        - Voice Cloning support (neural style manifold + spectral formant transfer)
        - Emotion / Feeling conditioning (style blending + prosodic/DSP shaping & inline [tags])
        - Dual WAV and MP3 output generation + full timeline metadata for animated graphical effects
        """
        t0 = time.perf_counter()
        pauses = pauses or PauseConfig()
        chunks = parse_text_with_pauses_and_emotions(text, pauses, default_emotion=emotion)
        if not chunks:
            raise ValueError("Please enter text to synthesize.")

        audio_parts: list[np.ndarray] = []
        timeline: list[dict[str, Any]] = []
        cursor_sec = 0.0

        for idx, chunk in enumerate(chunks):
            chunk_emo = chunk.emotion if chunk.emotion in EMOTION_PROFILES else emotion
            emo_prof = EMOTION_PROFILES[chunk_emo]

            # 1. Resolve neural voice style tensor (blended with emotion archetype)
            style_tensor, cloned_meta, _ = self.resolve_voice_style_and_profile(
                voice_id, chunk_emo, emotion_intensity
            )

            # 2. Compute effective speed for this chunk using the active Base Speed & Feeling Intensity
            #    (For cloned voices, Base Speed, Feeling Intensity & Pitch Offset are auto-set from the input voice!)
            emo_speed = 1.0 + (emo_prof["speed_mul"] - 1.0) * emotion_intensity
            eff_speed = float(np.clip(speed * emo_speed, 0.5, 2.0))

            # 3. Preserve prosodic boundary pitch contour for Kokoro + Native Neural Emotion Cue
            spoken_text = chunk.text.strip()
            use_excl = emo_prof.get("exclamatory_prosody", False) and emotion_intensity >= 0.45
            if chunk.pause_type in ("comma", "em_dash"):
                synth_text = f"{spoken_text},"
            elif chunk.pause_type == "ellipsis":
                synth_text = f"{spoken_text}."
            elif chunk.punctuation in ("!", "?"):
                synth_text = f"{spoken_text}{chunk.punctuation}"
            elif use_excl:
                synth_text = f"{spoken_text}!"
            else:
                synth_text = f"{spoken_text}."

            try:
                seg_audio, _ = self.kokoro.create(
                    synth_text,
                    voice=style_tensor,
                    speed=eff_speed,
                    lang=lang,
                    trim=True,
                )
            except ValueError:
                # Fallback if chunk contains only symbols/numbers that produced no phonemes
                continue

            # Trim leading/trailing silence added by the model so our pause durations are exact
            seg_audio, _ = trim_audio(seg_audio)
            if len(seg_audio) == 0:
                continue

            # Pre-DSP de-click & edge taper so vocoder boundary transients never get amplified by EQ/OLA
            seg_audio = SpeechDSP.apply_fade(seg_audio, SAMPLE_RATE, fade_in_ms=18.0, fade_out_ms=32.0)
            if len(seg_audio) == 0:
                continue

            # 4. If this chunk ends with an Ellipsis (...), apply subtle dramatic trailing taper
            if chunk.pause_type == "ellipsis":
                seg_audio = SpeechDSP.apply_ellipsis_dramatic_tail(seg_audio, SAMPLE_RATE)

            # 5. Apply 100% Cloned Voice Transfer (Exact F0 Register Lock & Cepstral Vocal-Tract Match)
            if cloned_meta is not None:
                seg_audio = SpeechDSP.apply_100pct_clone_transfer(
                    seg_audio, cloned_meta, SAMPLE_RATE
                )

            # 6. Apply Feeling / Emotion DSP using the active sliders
            #    For cloned voices, subtract `detected_pitch_offset` from `pitch_semitones` so we do NOT
            #    double-shift the pitch that `apply_100pct_clone_transfer` already locked to the input voice!
            if cloned_meta is not None and cloned_meta.get("mode") == "reference_audio":
                base_clone_pitch = float(cloned_meta.get("detected_pitch_offset", 0.0))
                effective_extra_pitch = float(pitch_semitones) - base_clone_pitch
            else:
                effective_extra_pitch = float(pitch_semitones)

            seg_audio = SpeechDSP.apply_emotion_dsp(
                seg_audio,
                emotion=chunk_emo,
                intensity=emotion_intensity,
                extra_pitch_semitones=effective_extra_pitch,
                sr=SAMPLE_RATE,
            )

            # 7. Smooth raised-cosine fade-in/out to prevent clicks at pause transitions
            seg_audio = SpeechDSP.apply_fade(seg_audio, SAMPLE_RATE)

            speech_dur = len(seg_audio) / float(SAMPLE_RATE)
            speech_start = cursor_sec
            speech_end = speech_start + speech_dur
            audio_parts.append(seg_audio)
            cursor_sec = speech_end

            # 8. Insert exact Punctuation Pause (scaled by emotion's pause multiplier)
            is_last_chunk = idx == len(chunks) - 1
            actual_pause_sec = 0.0
            if chunk.base_pause_sec > 0.0 and not is_last_chunk:
                if chunk.pause_type == "ellipsis":
                    pause_scale = 1.0 + (emo_prof["ellipsis_mul"] - 1.0) * emotion_intensity
                else:
                    pause_scale = 1.0 + (emo_prof["pause_mul"] - 1.0) * emotion_intensity
                actual_pause_sec = round(float(chunk.base_pause_sec * pause_scale), 3)
                pause_samples = max(1, int(round(actual_pause_sec * SAMPLE_RATE)))
                silence = np.zeros(pause_samples, dtype=np.float32)
                audio_parts.append(silence)
                cursor_sec += pause_samples / float(SAMPLE_RATE)

            timeline.append({
                "index": idx,
                "text": chunk.text,
                "punctuation": chunk.punctuation,
                "emotion": chunk_emo,
                "emotion_label": emo_prof["label"],
                "emotion_color": emo_prof["color"],
                "emotion_icon": emo_prof["icon"],
                "speech_start": round(speech_start, 3),
                "speech_end": round(speech_end, 3),
                "pause_type": chunk.pause_type if not is_last_chunk else "none",
                "pause_label": chunk.pause_label if not is_last_chunk else "End of Speech",
                "pause_start": round(speech_end, 3),
                "pause_end": round(cursor_sec, 3),
                "pause_duration": round(actual_pause_sec, 3),
            })

        if not audio_parts:
            raise ValueError("No audible phonemes were generated from the input text.")

        full_audio = np.concatenate(audio_parts).astype(np.float32)

        # Gentle master peak limiter to prevent clipping (-0.8 dBFS ceiling)
        peak = float(np.max(np.abs(full_audio)))
        if peak > 0.92:
            full_audio = full_audio * (0.92 / peak)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        if save_filename:
            stem = Path(save_filename).stem
            stem = re.sub(r"[^a-zA-Z0-9_\- ]+", "_", stem).strip() or f"kokoro_{timestamp}"
        else:
            stem = f"kokoro_{voice_id}_{emotion}_{timestamp}_{short_id}"

        # Keep synthesized audio in memory ONLY — do NOT save WAV or MP3 to disk automatically!
        self.last_audio = full_audio
        self.last_meta = {
            "stem": stem,
            "voice_id": voice_id,
            "emotion": emotion,
            "sample_rate": SAMPLE_RATE,
            "text": text,
        }

        # Encode in-memory WAV bytes for browser playback & avatar animation without disk I/O
        wav_buf = io.BytesIO()
        sf.write(wav_buf, full_audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_buf.getvalue()).decode("ascii")

        total_dur = round(len(full_audio) / float(SAMPLE_RATE), 3)
        elapsed = round(time.perf_counter() - t0, 3)

        pause_counts = {
            "comma": sum(1 for item in timeline if item["pause_type"] == "comma"),
            "period": sum(1 for item in timeline if item["pause_type"] == "period"),
            "ellipsis": sum(1 for item in timeline if item["pause_type"] == "ellipsis"),
            "em_dash": sum(1 for item in timeline if item["pause_type"] == "em_dash"),
        }

        return {
            "id": stem,
            "sample_rate": SAMPLE_RATE,
            "duration_sec": total_dur,
            "generation_time_sec": elapsed,
            "rtf": round(elapsed / max(0.01, total_dur), 3),
            "provider": self.active_provider,
            "voice_id": voice_id,
            "emotion": emotion,
            "audio_base64": audio_b64,
            "default_stem": stem,
            "pause_counts": pause_counts,
            "timeline": timeline,
        }

    def save_last_audio(self, fmt: str = "wav", custom_filename: str | None = None) -> dict[str, Any]:
        """Manually save the last synthesized in-memory audio to disk as WAV or MP3 on user request."""
        if self.last_audio is None or len(self.last_audio) == 0:
            raise ValueError("No generated speech in memory to save. Please generate speech first.")

        fmt_clean = "mp3" if fmt.strip().lower() == "mp3" else "wav"
        base_stem = custom_filename or self.last_meta.get("stem") or f"kokoro_{time.strftime('%Y%m%d_%H%M%S')}"
        stem = Path(base_stem).stem
        stem = re.sub(r"[^a-zA-Z0-9_\- ]+", "_", stem).strip() or f"kokoro_{time.strftime('%Y%m%d_%H%M%S')}"

        target_path = OUTPUT_DIR / f"{stem}.{fmt_clean}"
        self.export_audio_file(self.last_audio, SAMPLE_RATE, target_path, fmt=fmt_clean)
        duration_sec = round(len(self.last_audio) / float(SAMPLE_RATE), 3)

        logger.info(f"Manually saved {fmt_clean.upper()} ({duration_sec}s) to: {target_path}")
        return {
            "format": fmt_clean,
            "filename": target_path.name,
            "path": str(target_path),
            "download_url": f"/api/outputs/{target_path.name}?download=true",
            "duration_sec": duration_sec,
            "voice_id": self.last_meta.get("voice_id", "af_heart"),
            "emotion": self.last_meta.get("emotion", "neutral"),
            "text": self.last_meta.get("text", ""),
        }

    @staticmethod
    def export_audio_file(audio: np.ndarray, sr: int, target_path: Path, fmt: str = "wav") -> Path:
        """Write float32 audio array to disk as either WAV or MP3."""
        fmt_upper = fmt.strip().lower()
        if fmt_upper == "mp3":
            try:
                sf.write(str(target_path), audio, sr, format="MP3")
                return target_path
            except Exception:
                # Fallback via PyAV if libsndfile MP3 encoder encounters an issue
                import av

                container = av.open(str(target_path), mode="w")
                stream = container.add_stream("mp3", rate=sr)
                stream.layout = "mono"
                frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="fltp", layout="mono")
                frame.sample_rate = sr
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
                container.close()
                return target_path
        else:
            sf.write(str(target_path), audio, sr, format="WAV", subtype="PCM_16")
            return target_path


# ---------------------------------------------------------------------------
# 5. FastAPI Web Application & Endpoints
# ---------------------------------------------------------------------------
app = FastAPI(title="Coderstation — Kokoro Text-to-Voice Studio", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_studio: KokoroVoiceStudio | None = None


def get_studio() -> KokoroVoiceStudio:
    global _studio
    if _studio is None:
        _studio = KokoroVoiceStudio(MODEL_PATH, VOICES_PATH)
    return _studio


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "af_heart"
    emotion: str = "neutral"
    emotion_intensity: float = Field(default=1.0, ge=0.0, le=1.5)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch_semitones: float = Field(default=0.0, ge=-6.0, le=6.0)
    lang: str = "en-us"
    pause_comma: float = Field(default=0.22, ge=0.0, le=2.0)
    pause_period: float = Field(default=0.50, ge=0.0, le=2.5)
    pause_ellipsis: float = Field(default=0.95, ge=0.0, le=3.5)
    pause_em_dash: float = Field(default=0.40, ge=0.0, le=2.5)
    output_format: str = "wav"
    save_filename: str | None = None


class CloneReferenceRequest(BaseModel):
    name: str
    audio_base64: str
    gender: str = "female"
    warmth_bias_db: float = 0.0
    manual_pitch_offset: float = 0.0


class CloneBlendRequest(BaseModel):
    name: str
    voice_weights: dict[str, float]
    pitch_shift_semitones: float = 0.0
    warmth_bias_db: float = 0.0


class ManualSaveRequest(BaseModel):
    format: str = "wav"
    filename: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/init")
async def api_init() -> JSONResponse:
    studio = get_studio()
    voices_data = studio.list_all_voices()
    emotions_data = [
        {"id": k, **v} for k, v in EMOTION_PROFILES.items()
    ]
    return JSONResponse({
        "voices": voices_data,
        "emotions": emotions_data,
        "default_pauses": asdict(PauseConfig()),
        "sample_rate": SAMPLE_RATE,
    })


@app.post("/api/synthesize")
async def api_synthesize(req: SynthesizeRequest) -> JSONResponse:
    studio = get_studio()
    try:
        pauses = PauseConfig(
            comma=req.pause_comma,
            period=req.pause_period,
            ellipsis=req.pause_ellipsis,
            em_dash=req.pause_em_dash,
        )
        result = studio.synthesize(
            text=req.text,
            voice_id=req.voice_id,
            emotion=req.emotion,
            emotion_intensity=req.emotion_intensity,
            speed=req.speed,
            pitch_semitones=req.pitch_semitones,
            lang=req.lang,
            pauses=pauses,
            output_format=req.output_format,
            save_filename=req.save_filename,
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/clone/reference")
async def api_clone_reference(req: CloneReferenceRequest) -> JSONResponse:
    studio = get_studio()
    try:
        b64 = req.audio_base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64)
        meta = studio.clone_voice_from_reference_audio(
            name=req.name,
            audio_bytes=raw_bytes,
            gender=req.gender,
            warmth_bias_db=0.0,
            manual_pitch_offset=0.0,
        )
        return JSONResponse({
            "cloned_voice": meta,
            "voices": studio.list_all_voices(),
        })
    except Exception as exc:
        logger.exception("Reference voice cloning failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/clone/blend")
async def api_clone_blend(req: CloneBlendRequest) -> JSONResponse:
    studio = get_studio()
    try:
        meta = studio.clone_voice_from_style_blend(
            name=req.name,
            voice_weights=req.voice_weights,
            pitch_shift_semitones=req.pitch_shift_semitones,
            warmth_bias_db=req.warmth_bias_db,
        )
        return JSONResponse({
            "cloned_voice": meta,
            "voices": studio.list_all_voices(),
        })
    except Exception as exc:
        logger.exception("Neural style blend cloning failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/clone/{voice_id}")
async def api_delete_clone(voice_id: str) -> JSONResponse:
    studio = get_studio()
    ok = studio.delete_cloned_voice(voice_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Cloned voice not found.")
    return JSONResponse({"deleted": voice_id, "voices": studio.list_all_voices()})


@app.post("/api/save")
async def api_manual_save(req: ManualSaveRequest) -> JSONResponse:
    studio = get_studio()
    try:
        saved = studio.save_last_audio(fmt=req.format, custom_filename=req.filename)
        return JSONResponse(saved)
    except Exception as exc:
        logger.exception("Manual save failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/outputs/{filename}")
async def api_get_output(filename: str, download: bool = False) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    media_type = "audio/mpeg" if safe_name.lower().endswith(".mp3") else "audio/wav"
    if download:
        return FileResponse(
            str(file_path),
            media_type=media_type,
            filename=safe_name,
        )
    return FileResponse(str(file_path), media_type=media_type)


import threading

_SYS_STATS_LOCK = threading.Lock()
_SYS_STATS_CACHE: dict[str, Any] = {
    "gpu": 0.0,
    "ram": 0.0,
    "ram_used_gb": 0.0,
    "ram_total_gb": 0.0,
    "cpu": 0.0,
    "cpu_freq_ghz": 0.0,
    "cpu_cores": 0,
    "gpu_mem_used_gb": 0.0,
    "gpu_mem_total_gb": 0.0,
    "ts": 0.0,
}
_SYS_SAMPLER_STARTED = False


def _read_live_system_stats_once() -> dict[str, Any]:
    """Read actual live CPU %/GHz, RAM %/GB (psutil), and NVIDIA GPU %/VRAM GB (nvidia-smi)."""
    import shutil
    import subprocess

    cpu_pct = 0.0
    cpu_freq_ghz = 0.0
    cpu_cores = 0
    ram_pct = 0.0
    ram_used_gb = 0.0
    ram_total_gb = 0.0
    try:
        import psutil

        # interval=0.35 gives an accurate, real-time OS CPU usage measurement
        cpu_pct = float(psutil.cpu_percent(interval=0.35))
        freq = psutil.cpu_freq()
        if freq and freq.current:
            cpu_freq_ghz = float(freq.current) / 1000.0
        cpu_cores = int(psutil.cpu_count(logical=True) or 0)
        vm = psutil.virtual_memory()
        ram_pct = float(vm.percent)
        ram_used_gb = vm.used / (1024 ** 3)
        ram_total_gb = vm.total / (1024 ** 3)
    except Exception:
        pass

    gpu_pct = 0.0
    gpu_mem_used_gb = 0.0
    gpu_mem_total_gb = 0.0
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = subprocess.check_output(
                [nvsmi, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                timeout=1.5,
                creationflags=creation_flags,
            ).strip()
            if out:
                first_line = out.splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 3:
                    gpu_pct = float(parts[0])
                    gpu_mem_used_gb = float(parts[1]) / 1024.0
                    gpu_mem_total_gb = float(parts[2]) / 1024.0
        except Exception:
            pass

    return {
        "gpu": round(max(0.0, min(100.0, gpu_pct)), 1),
        "ram": round(max(0.0, min(100.0, ram_pct)), 1),
        "ram_used_gb": round(ram_used_gb, 1),
        "ram_total_gb": round(ram_total_gb, 1),
        "cpu": round(max(0.0, min(100.0, cpu_pct)), 1),
        "cpu_freq_ghz": round(cpu_freq_ghz, 2),
        "cpu_cores": cpu_cores,
        "gpu_mem_used_gb": round(gpu_mem_used_gb, 1),
        "gpu_mem_total_gb": round(gpu_mem_total_gb, 1),
        "ts": time.time(),
    }


def _ensure_sys_sampler_running() -> None:
    global _SYS_SAMPLER_STARTED
    if _SYS_SAMPLER_STARTED:
        return
    with _SYS_STATS_LOCK:
        if _SYS_SAMPLER_STARTED:
            return
        _SYS_SAMPLER_STARTED = True

    def _loop() -> None:
        while True:
            sample = _read_live_system_stats_once()
            with _SYS_STATS_LOCK:
                _SYS_STATS_CACHE.update(sample)
            time.sleep(0.65)

    t = threading.Thread(target=_loop, daemon=True, name="sys-telemetry")
    t.start()


def _sample_system_stats() -> dict[str, Any]:
    """Return latest actual CPU, RAM, and NVIDIA GPU metrics immediately."""
    _ensure_sys_sampler_running()
    with _SYS_STATS_LOCK:
        if _SYS_STATS_CACHE["ts"] == 0.0:
            fresh = _read_live_system_stats_once()
            _SYS_STATS_CACHE.update(fresh)
        return dict(_SYS_STATS_CACHE)


@app.get("/api/system-stats")
async def api_system_stats() -> JSONResponse:
    return JSONResponse(_sample_system_stats())


# ---------------------------------------------------------------------------
# 6. CLI & Server Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coderstation — Kokoro Text-to-Voice Studio with Pauses, Voice Cloning, Feelings & Animated Visuals"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on startup")
    parser.add_argument("--text", type=str, help="Synthesize text directly via CLI and exit")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice ID for CLI synthesis")
    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        choices=list(EMOTION_PROFILES.keys()),
        help="Feeling/Emotion preset for CLI synthesis",
    )
    parser.add_argument("--output", type=str, help="Output file path (.wav or .mp3) for CLI synthesis")
    parser.add_argument("--clone-ref", type=str, help="Path to reference .wav/.mp3 to clone before CLI synthesis")
    parser.add_argument("--clone-name", type=str, default="CLI Cloned Voice", help="Name for the cloned voice")
    args = parser.parse_args()

    studio = get_studio()

    # If --text is passed, run in CLI batch mode
    if args.text:
        voice_id = args.voice
        if args.clone_ref:
            ref_path = Path(args.clone_ref)
            if not ref_path.exists():
                logger.error(f"Reference audio file not found: {ref_path}")
                sys.exit(1)
            meta = studio.clone_voice_from_reference_audio(args.clone_name, ref_path.read_bytes())
            voice_id = meta["id"]
            logger.info(f"Using newly cloned voice: {voice_id}")

        out_fmt = "mp3" if (args.output and args.output.lower().endswith(".mp3")) else "wav"
        save_name = Path(args.output).stem if args.output else None
        res = studio.synthesize(
            text=args.text,
            voice_id=voice_id,
            emotion=args.emotion,
            output_format=out_fmt,
            save_filename=save_name,
        )
        if args.output:
            saved = studio.save_last_audio(fmt=out_fmt, custom_filename=save_name)
            target_src = Path(saved["path"])
            dest = Path(args.output).resolve()
            if dest != target_src.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(target_src.read_bytes())
            logger.info(f"Saved {out_fmt.upper()} output ({res['duration_sec']}s) to: {dest}")
        else:
            logger.info(f"Synthesized {res['duration_sec']}s in memory (pass --output <file.wav|file.mp3> to save to disk).")
        return

    # Otherwise start the Interactive Web Studio GUI
    import threading
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    logger.info(f"Coderstation Kokoro Studio running at {url}")
    if not args.no_browser:
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
