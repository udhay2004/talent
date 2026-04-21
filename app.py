"""
app.py — Multi-Modal AI Talent Assessment System
Flask web server with clean routing, robust error handling, and progress tracking.
"""

import os
import uuid
import traceback
from flask import Flask, render_template, request, jsonify, session
import pdfplumber

from models import ResumeMatcher, SpeechAnalyzer, fuse_scores
from utils import preprocess_resume, extract_audio_from_video, process_video

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"]     = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

ALLOWED_RESUME = {"txt", "pdf"}
ALLOWED_VIDEO  = {"mp4", "webm", "mov", "avi"}


def allowed(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def safe_filename(filename: str) -> str:
    """UUID-prefix to avoid collisions + path traversal."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    return f"{uuid.uuid4().hex}.{ext}"


def read_resume(path: str) -> str:
    """Read .pdf or .txt resume, return raw text."""
    try:
        if path.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        else:
            for enc in ("utf-8", "latin-1"):
                try:
                    with open(path, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
    except Exception as e:
        print(f"[Resume] Read error: {e}")
    return ""


# ── Pre-load singleton models at startup (saves time on first request) ──────────
print("[Startup] Loading NLP model…")
_matcher  = ResumeMatcher()
print("[Startup] Loading Speech model…")
_speech   = SpeechAnalyzer()
print("[Startup] Models ready ✓")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    error = None
    try:
        # ── Validate inputs ────────────────────────────────────────────────
        resume_file = request.files.get("resume")
        video_file  = request.files.get("video")
        job_desc    = request.form.get("job_desc", "").strip()

        if not resume_file or not resume_file.filename:
            error = "Please upload a resume (.pdf or .txt)."
        elif not allowed(resume_file.filename, ALLOWED_RESUME):
            error = "Resume must be a .pdf or .txt file."
        elif not video_file or not video_file.filename:
            error = "Please upload an interview video."
        elif not allowed(video_file.filename, ALLOWED_VIDEO):
            error = "Video must be .mp4, .webm, .mov, or .avi."
        elif len(job_desc) < 20:
            error = "Please enter a more detailed job description (at least 20 characters)."

        if error:
            return render_template("index.html", error=error)

        # ── Save files ─────────────────────────────────────────────────────
        resume_fname = safe_filename(resume_file.filename)
        video_fname  = safe_filename(video_file.filename)
        resume_path  = os.path.join(UPLOAD_FOLDER, resume_fname)
        video_path   = os.path.join(UPLOAD_FOLDER, video_fname)
        audio_path   = video_path.rsplit(".", 1)[0] + ".wav"

        resume_file.save(resume_path)
        video_file.save(video_path)
        print(f"[App] Saved: {resume_fname}, {video_fname}")

        # ── Extract audio ──────────────────────────────────────────────────
        ok, err_msg = extract_audio_from_video(video_path, audio_path)
        if not ok:
            return render_template("index.html",
                error=f"Audio extraction failed: {err_msg}. "
                      "Ensure your video has an audio track and ffmpeg is installed.")

        # ── Resume NLP ─────────────────────────────────────────────────────
        raw_text   = read_resume(resume_path)
        clean_text = preprocess_resume(raw_text)
        if len(clean_text) < 50:
            return render_template("index.html",
                error="Could not extract enough text from the resume. "
                      "Try a text-based PDF or .txt file.")

        resume_score = _matcher.match(clean_text, job_desc)
        print(f"[App] Resume score: {resume_score}")

        # ── Video / Fraud ──────────────────────────────────────────────────
        fraud_flag, emotions, fraud_msg, vid_stats = process_video(video_path)
        print(f"[App] Fraud={fraud_flag} | Emotions={emotions[:5]}… | Stats={vid_stats}")

        # ── Speech ────────────────────────────────────────────────────────
        try:
            speech_sentiment = _speech.analyze(audio_path)
        except Exception as e:
            print(f"[App] Speech analysis error: {e}")
            speech_sentiment = "neutral"

        # ── Fusion ────────────────────────────────────────────────────────
        final_score, suitability, comment, breakdown = fuse_scores(
            resume_score, emotions, speech_sentiment, fraud_flag, fraud_msg
        )
        print(f"[App] Final={final_score} | {suitability}")

        # Emotion summary for display
        from collections import Counter
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"
        emotion_summary  = ", ".join(f"{k} ({v})" for k, v in emotion_counts.most_common(4))

        return render_template(
            "result.html",
            resume_score    = resume_score,
            final           = final_score,
            suitability     = suitability,
            comment         = comment,
            breakdown       = breakdown,
            emotions        = emotions,
            emotion_summary = emotion_summary,
            dominant_emotion= dominant_emotion,
            speech          = speech_sentiment,
            fraud           = fraud_msg,
            fraud_flag      = fraud_flag,
            vid_stats       = vid_stats,
        )

    except Exception:
        tb = traceback.format_exc()
        print(f"[App] CRITICAL ERROR:\n{tb}")
        return render_template("index.html",
            error="An unexpected error occurred during analysis. "
                  "Please check your files and try again.")

    finally:
        # Clean up uploaded files to save disk space
        for path in [resume_path, video_path, audio_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html",
        error="File too large. Maximum allowed size is 200 MB."), 413


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)