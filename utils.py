"""
utils.py - Video/Audio/Text preprocessing utilities
Optimised for speed: sparse frame sampling, early-exit, parallel-safe design.
"""

import os
import re
import subprocess
import shutil
import threading
import numpy as np
import cv2

# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────
def preprocess_resume(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)   # keep printable ASCII
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# Audio Extraction  (ffmpeg – system or bundled)
# ─────────────────────────────────────────────
def _find_ffmpeg() -> str | None:
    """Search for ffmpeg: system PATH first, then common project locations."""
    # 1. System PATH
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    # 2. Common bundle locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "ffmpeg.exe"),
        os.path.join(os.path.dirname(__file__), "ffmpeg"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Users\udhay\OneDrive\Desktop\resume_analyzer_project\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def extract_audio_from_video(video_path: str, audio_path: str) -> tuple[bool, str]:
    """
    Extract mono 16 kHz WAV from video using ffmpeg.
    Returns (success: bool, error_message: str).
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False, (
            "ffmpeg not found. Install it: https://ffmpeg.org/download.html "
            "and add to PATH, or place ffmpeg.exe in the project folder."
        )

    cmd = [
        ffmpeg, "-y",                    # overwrite without prompt
        "-i", video_path,
        "-vn",                           # no video
        "-ac", "1",                      # mono
        "-ar", "16000",                  # 16 kHz – faster librosa load
        "-q:a", "0",
        audio_path,
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=120
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "ffmpeg error").strip()[-500:]
        print(f"[ffmpeg] Error: {msg}")
        return False, msg
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out (video too large)."
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Video Processing  (optimised – sparse + fast)
# ─────────────────────────────────────────────
# Thread-local FaceAnalyzer so we don't reload models per request
_tl = threading.local()

def _get_face_analyzer():
    if not hasattr(_tl, "fa"):
        from models import FaceAnalyzer
        _tl.fa = FaceAnalyzer()
    return _tl.fa


def process_video(
    video_path: str,
    max_seconds: int = 120,
    sample_every_n_frames: int = 45,   # ~1.5s at 30 fps
) -> tuple[bool, list, str, dict]:
    """
    Fast video processing:
    - Only process first `max_seconds` seconds
    - Sample 1 frame every `sample_every_n_frames`
    - Returns (fraud_flag, emotions_list, fraud_msg, stats_dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, ["neutral"], "Could not open video file.", {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, int(max_seconds * fps))

    fa = _get_face_analyzer()
    emotions        = []
    gaze_deviations = 0
    no_face_frames  = 0
    frames_sampled  = 0
    frame_idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        frame_idx += 1

        if frame_idx % sample_every_n_frames != 0:
            continue

        frames_sampled += 1
        # Resize for speed (half resolution)
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        emotion = fa.detect_emotion(small)
        if emotion:
            emotions.append(emotion)
        else:
            no_face_frames += 1

        if fa.gaze_deviation(small):
            gaze_deviations += 1

    cap.release()

    stats = {
        "frames_sampled": frames_sampled,
        "gaze_deviations": gaze_deviations,
        "no_face_frames": no_face_frames,
    }
    print(f"[Video] Sampled {frames_sampled} frames | gaze_dev={gaze_deviations} | no_face={no_face_frames}")

    # ── Fraud Rules ────────────────────────────────────────────────────────
    fraud_flag = False
    fraud_reasons = []

    if frames_sampled > 0:
        gaze_ratio    = gaze_deviations / frames_sampled
        no_face_ratio = no_face_frames  / frames_sampled

        if gaze_ratio > 0.55:
            fraud_reasons.append(f"high gaze deviation ({gaze_ratio:.0%} of frames)")
        if no_face_ratio > 0.60:
            fraud_reasons.append(f"face absent in {no_face_ratio:.0%} of frames (possible impersonation)")

    if fraud_reasons:
        fraud_flag = True
        fraud_msg  = "Suspicious behavior: " + "; ".join(fraud_reasons) + "."
    else:
        fraud_msg = "No fraud indicators detected."

    if not emotions:
        emotions = ["neutral"]

    return fraud_flag, emotions, fraud_msg, stats