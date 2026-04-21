"""
utils.py - Video/Audio/Text preprocessing utilities
Lightweight version - no torch/FaceAnalyzer dependency
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
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# Audio Extraction
# ─────────────────────────────────────────────
def _find_ffmpeg() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
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
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False, (
            "ffmpeg not found. Install it and add to PATH, "
            "or place ffmpeg.exe in the project folder."
        )
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-q:a", "0",
        audio_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or "ffmpeg error").strip()[-400:]
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out."
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Lightweight Face Analyzer (OpenCV only, no torch)
# ─────────────────────────────────────────────
_tl = threading.local()

def _get_face_analyzer():
    """Pure OpenCV face analyzer — no torch, no FaceAnalyzer import."""
    if not hasattr(_tl, "fa"):
        class _LightFA:
            def __init__(self):
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )

            def detect_emotion(self, frame) -> str:
                # Lightweight heuristic: brightness in face region
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                if len(faces) == 0:
                    return "neutral"
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi        = gray[y:y+h, x:x+w]
                brightness = float(np.mean(roi))
                variance   = float(np.var(roi))
                if brightness > 140 and variance > 800:
                    return "happy"
                elif brightness < 80:
                    return "sad"
                return "neutral"

            def gaze_deviation(self, frame, threshold_ratio=0.28) -> bool:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                if len(faces) == 0:
                    return False
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_cx    = x + w // 2
                frame_cx   = frame.shape[1] // 2
                return abs(face_cx - frame_cx) > frame.shape[1] * threshold_ratio

        _tl.fa = _LightFA()
    return _tl.fa


# ─────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────
def process_video(
    video_path: str,
    max_seconds: int = 120,
    sample_every_n_frames: int = 45,
) -> tuple[bool, list, str, dict]:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, ["neutral"], "Could not open video file.", {}

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames  = min(total_frames, int(max_seconds * fps))

    fa              = _get_face_analyzer()
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
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        emotion = fa.detect_emotion(small)
        emotions.append(emotion)

        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        ).detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        if len(faces) == 0:
            no_face_frames += 1
        elif fa.gaze_deviation(small):
            gaze_deviations += 1

    cap.release()

    stats = {
        "frames_sampled": frames_sampled,
        "gaze_deviations": gaze_deviations,
        "no_face_frames": no_face_frames,
    }
    print(f"[Video] Sampled {frames_sampled} frames | gaze={gaze_deviations} | no_face={no_face_frames}")

    fraud_flag    = False
    fraud_reasons = []

    if frames_sampled > 0:
        if (gaze_deviations / frames_sampled) > 0.55:
            fraud_reasons.append(f"high gaze deviation ({gaze_deviations}/{frames_sampled} frames)")
        if (no_face_frames / frames_sampled) > 0.60:
            fraud_reasons.append(f"face absent in {no_face_frames}/{frames_sampled} frames")

    if fraud_reasons:
        fraud_flag = True
        fraud_msg  = "Suspicious behavior: " + "; ".join(fraud_reasons) + "."
    else:
        fraud_msg = "No fraud indicators detected."

    if not emotions:
        emotions = ["neutral"]

    return fraud_flag, emotions, fraud_msg, stats
