# TalentMind — Multi-Modal AI Talent Assessment System
### Major Project · Dept. of AI & ML · Guru Tegh Bahadur Institute of Technology

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg  
**Windows** — download from https://ffmpeg.org/download.html  
- Extract and place `ffmpeg.exe` in the project root folder, **OR**
- Add ffmpeg to your system PATH (recommended)

**Mac** — `brew install ffmpeg`  
**Linux** — `sudo apt install ffmpeg`

### 3. Run the app
```bash
python app.py
```
Open http://localhost:5000 in your browser.

---

## Project Structure
```
ai_talent_scout/
├── app.py           ← Flask web server (main entry point)
├── models.py        ← AI models: NLP, CV, Speech, Fusion
├── utils.py         ← Video/audio/text preprocessing
├── requirements.txt
├── ffmpeg.exe       ← Place here if not on system PATH (Windows)
├── uploads/         ← Temp upload folder (auto-created)
└── templates/
    ├── index.html   ← Upload form
    └── result.html  ← Assessment report
```

---

## System Architecture
| Module | Model | Purpose |
|--------|-------|---------|
| NLP | Sentence-Transformers (all-MiniLM-L6-v2) | Resume ↔ Job-Description semantic matching |
| Computer Vision | ResNet-18 + Haar Cascade | Face detection, emotion recognition, gaze analysis |
| Speech | 2-layer LSTM on MFCC features | Sentiment classification (positive/neutral/negative) |
| Fraud Detection | Rule-based on CV signals | Gaze deviation, face absence ratio |
| Fusion | Weighted ensemble | Resume 50% + Emotion 25% + Speech 25% |

---

## Key Optimisations Made
1. **Singleton model loading** — NLP and Speech models load once at startup, not per request
2. **Sparse video sampling** — 1 frame every 45 frames (~1.5s) instead of every frame
3. **Half-resolution processing** — frames resized 50% before CV inference
4. **16 kHz mono audio** — faster librosa loading vs original 22 kHz stereo
5. **UUID filenames** — prevents collisions and path-traversal attacks
6. **Auto file cleanup** — uploaded files deleted after processing
7. **Threaded server** — `threaded=True` allows concurrent requests
8. **ffmpeg auto-discovery** — checks PATH first, then common locations

---

## IEEE References (as cited in the paper)
See the project report for full bibliography in IEEE format.
