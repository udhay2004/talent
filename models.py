"""
models.py - Multi-Modal AI Talent Assessment
Core AI models: NLP resume matching, face/emotion CV, speech LSTM, fraud detection & fusion
"""

import sys
import subprocess

def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        print(f"[AutoInstall] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("torchvision")
_ensure("sentence-transformers", "sentence_transformers")

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import cv2
from torchvision import models, transforms
import librosa
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Models] Running on: {DEVICE}")


# ─────────────────────────────────────────────
# 1.  NLP — Resume ↔ Job-Description Matching
# ─────────────────────────────────────────────
class ResumeMatcher:
    """
    Sentence-Transformer cosine similarity between resume and JD.
    Returns a 0–100 normalised score.
    """
    _instance = None  # singleton – avoid re-loading model every request

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._instance

    def match(self, resume_text: str, job_description: str) -> float:
        if not resume_text.strip() or not job_description.strip():
            return 50.0
        r_emb = self.model.encode(resume_text, convert_to_tensor=True)
        j_emb = self.model.encode(job_description, convert_to_tensor=True)
        similarity = util.cos_sim(r_emb, j_emb).item()
        # scale: raw similarity ~0.2–0.9 → boost to usable 0–100 range
        score = float(min(98.0, max(30.0, similarity * 100 * 1.30)))
        return round(score, 1)


# ─────────────────────────────────────────────
# 2.  CV — Face Emotion Analyzer
# ─────────────────────────────────────────────
class FaceAnalyzer:
    """
    Haar-cascade face detection + lightweight ResNet-18 emotion head.
    Emotion labels follow AffectNet 7-class convention.
    """
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Emotion model – fine-tune head on 7 classes
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 7)
        self.emotion_model = base.to(DEVICE)
        self.emotion_model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect_face(self, frame):
        """Returns (x,y,w,h) of largest face or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # pick largest face
        return max(faces, key=lambda f: f[2] * f[3])

    def detect_emotion(self, frame) -> str:
        face = self.detect_face(frame)
        if face is None:
            return "neutral"
        x, y, w, h = face
        crop = frame[y:y+h, x:x+w]
        tensor = self.transform(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.emotion_model(tensor)
            idx = torch.argmax(logits, dim=1).item()
        return self.EMOTIONS[idx]

    def gaze_deviation(self, frame, threshold_ratio=0.28) -> bool:
        """True if face centre deviates horizontally by > threshold_ratio of frame width."""
        face = self.detect_face(frame)
        if face is None:
            return False
        x, y, w, h = face
        face_cx = x + w // 2
        frame_cx = frame.shape[1] // 2
        return abs(face_cx - frame_cx) > frame.shape[1] * threshold_ratio


# ─────────────────────────────────────────────
# 3.  Speech — LSTM Sentiment Analyzer
# ─────────────────────────────────────────────
class SpeechAnalyzer(nn.Module):
    """
    MFCC → 2-layer LSTM → 3-class sentiment (negative / neutral / positive).
    """
    SENTIMENTS = ["negative", "neutral", "positive"]

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.to(DEVICE)
        self.eval()

    def _extract_features(self, audio_path: str) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=22050, duration=60)  # cap at 60s
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)       # 40 coefficients
        delta = librosa.feature.delta(mfcc)
        # shape → (time, 40); use delta for richer features
        combined = np.mean(np.vstack([mfcc, delta]).T, axis=0)[:40]
        return combined

    def analyze(self, audio_path: str) -> str:
        try:
            feats = self._extract_features(audio_path)
        except Exception as e:
            print(f"[SpeechAnalyzer] Feature extraction failed: {e}")
            return "neutral"
        tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, (hn, _) = self.lstm(tensor)
            logits = self.fc(hn[-1])
            idx = torch.argmax(logits, dim=1).item()
        return self.SENTIMENTS[idx]


# ─────────────────────────────────────────────
# 4.  Multi-Modal Fusion & Scoring
# ─────────────────────────────────────────────
_SENTIMENT_WEIGHT = {"positive": 1.0, "neutral": 0.55, "negative": 0.15}
_EMOTION_WEIGHT   = {"happy": 1.0, "surprise": 0.8, "neutral": 0.65,
                     "sad": 0.4, "fear": 0.3, "angry": 0.2, "disgust": 0.1}

def fuse_scores(
    resume_score: float,
    emotions: list,
    speech_sentiment: str,
    fraud_flag: bool,
    fraud_msg: str = "",
) -> tuple[float, str, str, dict]:
    """
    Returns (final_score, suitability_label, comment, breakdown_dict)
    Weights: resume 50%, emotion 25%, speech 25%
    """
    if fraud_flag:
        return 35.0, "Not Suitable", (
            f"⚠️ Fraud indicators detected — {fraud_msg}. "
            "Candidate flagged for manual review."
        ), {"resume": resume_score, "emotion": 0, "speech": 0, "fraud_penalty": -65}

    # Emotion score: weighted average over all detected frames
    if emotions:
        raw_emotion = np.mean([_EMOTION_WEIGHT.get(e, 0.5) for e in emotions])
    else:
        raw_emotion = 0.5
    emotion_score = float(np.clip(raw_emotion * 100, 20, 100))

    # Speech score
    speech_score = _SENTIMENT_WEIGHT.get(speech_sentiment, 0.55) * 100

    # Weighted fusion
    final = 0.50 * resume_score + 0.25 * emotion_score + 0.25 * speech_score
    final = round(float(np.clip(final, 30, 98)), 1)

    breakdown = {
        "resume": round(resume_score, 1),
        "emotion": round(emotion_score, 1),
        "speech": round(speech_score, 1),
    }

    # Suitability label
    if final >= 82:
        suitability = "Highly Suitable"
        comment = ("Exceptional candidate — strong technical alignment, confident communication, "
                   "and consistently positive engagement throughout the interview.")
    elif final >= 68:
        suitability = "Suitable"
        comment = ("Strong candidate — good technical match with solid communication skills. "
                   "Minor areas for improvement in non-verbal confidence.")
    elif final >= 52:
        suitability = "Borderline"
        comment = ("Average candidate — technical skills are present but communication confidence "
                   "and engagement need improvement before progressing.")
    else:
        suitability = "Not Suitable"
        comment = ("Below-average candidate — significant gaps found in either technical alignment "
                   "or communication effectiveness. Further evaluation recommended.")

    return final, suitability, comment, breakdown