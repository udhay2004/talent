"""
models.py - Memory-efficient version for free cloud deployment
Uses TF-IDF + cosine similarity instead of sentence-transformers (no torch needed)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── NLP: Resume Matching (TF-IDF, ~5MB RAM vs 400MB for torch) ───
class ResumeMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english",
            sublinear_tf=True,
        )

    def match(self, resume_text: str, job_description: str) -> float:
        if not resume_text.strip() or not job_description.strip():
            return 50.0
        try:
            corpus = [resume_text, job_description]
            tfidf  = self.vectorizer.fit_transform(corpus)
            sim    = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            # Scale: raw TF-IDF sim ~0.1-0.6 → boost to useful 0-100 range
            score  = float(min(96.0, max(30.0, sim * 100 * 2.1)))
            return round(score, 1)
        except Exception as e:
            print(f"[ResumeMatcher] Error: {e}")
            return 55.0


# ── Speech: Librosa heuristic (no LSTM/torch needed) ─────────────
class SpeechAnalyzer:
    def analyze(self, audio_path: str) -> str:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=30, mono=True)
            if len(y) < 100:
                return "neutral"
            rms     = float(np.sqrt(np.mean(y ** 2)))
            zcr     = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            tempo, _= librosa.beat.beat_track(y=y, sr=sr)
            tempo   = float(tempo) if hasattr(tempo, '__float__') else 100.0
            # Heuristic scoring
            score = 0
            if rms > 0.025:  score += 1
            if rms > 0.05:   score += 1
            if zcr < 0.12:   score += 1
            if 80 < tempo < 160: score += 1
            if score >= 3:   return "positive"
            if score <= 1:   return "negative"
            return "neutral"
        except Exception as e:
            print(f"[SpeechAnalyzer] Error: {e}")
            return "neutral"


# ── Fusion ────────────────────────────────────────────────────────
_SENTIMENT_W = {"positive": 1.0, "neutral": 0.55, "negative": 0.15}
_EMOTION_W   = {"happy": 1.0, "surprise": 0.8, "neutral": 0.65,
                "sad": 0.4,  "fear": 0.3,  "angry": 0.2, "disgust": 0.1}

def fuse_scores(resume_score, emotions, speech_sentiment, fraud_flag, fraud_msg=""):
    if fraud_flag:
        return (35.0, "Not Suitable",
                f"Fraud indicators detected — {fraud_msg}. Flagged for manual review.",
                {"resume": resume_score, "emotion": 0, "speech": 0})

    raw_emotion   = np.mean([_EMOTION_W.get(e, 0.5) for e in emotions]) if emotions else 0.5
    emotion_score = float(np.clip(raw_emotion * 100, 20, 100))
    speech_score  = _SENTIMENT_W.get(speech_sentiment, 0.55) * 100

    final = round(float(np.clip(
        0.50 * resume_score + 0.25 * emotion_score + 0.25 * speech_score, 30, 98
    )), 1)

    breakdown = {
        "resume":  round(resume_score, 1),
        "emotion": round(emotion_score, 1),
        "speech":  round(speech_score, 1),
    }

    if final >= 82:
        suit    = "Highly Suitable"
        comment = "Exceptional candidate — strong technical alignment and positive communication."
    elif final >= 68:
        suit    = "Suitable"
        comment = "Strong candidate — good technical match with solid communication skills."
    elif final >= 52:
        suit    = "Borderline"
        comment = "Average candidate — technical skills present but communication needs improvement."
    else:
        suit    = "Not Suitable"
        comment = "Significant gaps in technical alignment or communication effectiveness."

    return final, suit, comment, breakdown
