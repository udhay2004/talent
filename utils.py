"""
utils.py - SPEED-OPTIMISED video/audio/text preprocessing
Target: < 10 seconds total processing on free cloud tier
"""

import os
import re
import subprocess
import shutil
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
# Audio Extraction — FAST settings
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
        return False, "ffmpeg not found. Install it and add to PATH."

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vn",           # no video stream
        "-ac", "1",      # mono
        "-ar", "8000",   # 8 kHz — much faster, enough for sentiment heuristic
        "-t", "30",      # ONLY first 30 seconds of audio
        "-q:a", "9",     # lowest quality = fastest encode
        audio_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or "ffmpeg error").strip()[-300:]
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out."
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Module-level cascade (loaded once, reused)
# ─────────────────────────────────────────────
_CASCADE = None

def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


# ─────────────────────────────────────────────
# Ultra-Fast Video Processing
# ─────────────────────────────────────────────
def process_video(
    video_path: str,
    max_seconds: int = 30,           # only analyse first 30s
    sample_every_n_frames: int = 90, # 1 frame every ~3s at 30fps → ~10 frames total
) -> tuple[bool, list, str, dict]:
    """
    Speed profile on a 60s video:
    - Reads only 30s worth of frames
    - Skips 89 of every 90 frames
    - Resizes to 320px wide before any processing
    - Total target: under 3 seconds on CPU
    """
    import time
    t0 = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, ["neutral"], "Could not open video file.", {}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames   = min(total_frames, int(max_seconds * fps))

    # Jump directly to sample frames using seek (much faster than reading every frame)
    sample_points = list(range(0, max_frames, sample_every_n_frames))

    cascade         = _get_cascade()
    emotions        = []
    gaze_deviations = 0
    no_face_frames  = 0
    frames_sampled  = 0

    for frame_pos in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        frames_sampled += 1

        # Aggressive resize to 320px wide
        h, w = frame.shape[:2]
        scale = min(1.0, 320 / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_NEAREST)

        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            no_face_frames += 1
            emotions.append("neutral")
            continue

        x, y, w_f, h_f = max(faces, key=lambda f: f[2] * f[3])

        # Fast brightness/variance heuristic for emotion
        roi        = gray[y:y+h_f, x:x+w_f]
        brightness = float(np.mean(roi))
        variance   = float(np.var(roi))
        if brightness > 135 and variance > 700:
            emotions.append("happy")
        elif brightness < 75:
            emotions.append("sad")
        else:
            emotions.append("neutral")

        # Gaze check
        face_cx  = x + w_f // 2
        frame_cx = small.shape[1] // 2
        if abs(face_cx - frame_cx) > small.shape[1] * 0.30:
            gaze_deviations += 1

    cap.release()
    elapsed = round(time.time() - t0, 2)
    print(f"[Video] {frames_sampled} frames in {elapsed}s | gaze={gaze_deviations} | no_face={no_face_frames}")

    stats = {
        "frames_sampled": frames_sampled,
        "gaze_deviations": gaze_deviations,
        "no_face_frames": no_face_frames,
        "process_time_s": elapsed,
    }

    fraud_reasons = []
    if frames_sampled > 0:
        if (gaze_deviations / frames_sampled) > 0.55:
            fraud_reasons.append(f"high gaze deviation ({gaze_deviations}/{frames_sampled} frames)")
        if (no_face_frames / frames_sampled) > 0.65:
            fraud_reasons.append(f"face absent in {no_face_frames}/{frames_sampled} frames")

    fraud_flag = bool(fraud_reasons)
    fraud_msg  = (
        "Suspicious behavior: " + "; ".join(fraud_reasons) + "."
        if fraud_reasons else "No fraud indicators detected."
    )

    return fraud_flag, emotions or ["neutral"], fraud_msg, stats
