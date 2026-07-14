# Python Face Detection Project

A face detection and recognition system with hand gesture controls, built with OpenCV, DeepFace, MediaPipe, and Streamlit.

## Features
- Detect faces from your webcam in real time.
- Detect faces in a single image and save the result.
- Face recognition for known people, with unrecognized faces labeled as Unknown.
- Hand gesture recognition including Open Palm, Thumbs Up, Thumbs Down, and Closed Fist.
- Gesture-driven PPT control (slide up/down and close action) in the live web interface.
- Streamlit-based web UI for all features.

## Project Structure

```
├── backend/                    # All ML and detection logic
│   ├── __init__.py
│   ├── face_detection/         # Face detection & recognition modules
│   │   ├── detector.py         # Haar cascade face detector
│   │   ├── recognizer.py       # DeepFace-based face recognizer
│   │   └── trainer.py          # Training script for face encodings
│   ├── hand_gesture/           # Hand gesture modules
│   │   ├── detector.py         # Skin-color hand detector
│   │   ├── recognizer.py       # KNN gesture classifier
│   │   ├── mp_recognizer.py    # MediaPipe gesture recognizer
│   │   ├── trainer.py          # Gesture model trainer
│   │   └── voice_file_opener.py # Voice-based file opener
│   ├── webcam_detect.py        # CLI webcam detection
│   ├── image_detect.py         # CLI image detection
│   ├── onboard.py              # Face onboarding (capture photos)
│   ├── gesture_onboard.py      # Gesture onboarding (capture samples)
│   ├── train.py                # Face training entry point
│   ├── train_gesture.py        # Gesture training entry point
│   ├── bootstrap_internet_gesture_data.py
│   └── import_palm_images.py
├── frontend/                   # Streamlit web application
│   └── app.py                  # Main Streamlit entry point
├── tests/                      # Test files
├── models/                     # Trained model files
├── known_faces/                # Labeled face images for training
├── gesture_data/               # Gesture training data
└── requirements.txt
```

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

### Web Interface (Streamlit)

```powershell
streamlit run frontend/app.py
```

This launches the full web UI with webcam, gesture controls, and file upload tabs.

### Frontend + Backend Locally

Run both services together on Windows:

```powershell
.\start-services.ps1
```

Endpoints:
- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- Backend health check: `http://localhost:8000/health`

The backend API accepts image uploads at `POST /detect` using multipart form-data
with the file field name `file`.

### CLI: Webcam detection

```powershell
python -m backend.webcam_detect --camera 0
```

Press `q` to quit.

### CLI: Image detection

```powershell
python -m backend.image_detect path\to\image.jpg --output outputs\result.jpg
```

## Notes
- Webcam usage depends on local camera permissions.
- Haar cascades are fast and lightweight, but may be less accurate than deep learning models.

## Face Recognition

Train and recognize custom faces:

```bash
# Capture 20 samples of each person
python -m backend.onboard --name "Alice" --webcam --photos 20
python -m backend.onboard --name "Bob" --webcam --photos 20

# Train embeddings
python -m backend.train

# Recognize in real-time
python -m backend.webcam_detect --recognize
```

See [Recognition Pipeline](#recognition-pipeline) below for architecture.

## Hand Gesture Recognition

Train custom hand gestures and trigger actions on recognition:

```bash
# Capture 20 samples of each gesture
python -m backend.gesture_onboard --gesture "palm" --samples 20
python -m backend.gesture_onboard --gesture "thumbs_up" --samples 20
python -m backend.gesture_onboard --gesture "thumbs_down" --samples 20
python -m backend.gesture_onboard --gesture "closed_fist" --samples 20

# Train gesture classifier
python -m backend.train_gesture

# Recognize gestures in real-time
python -m backend.webcam_detect --gesture

# Combine face + gesture recognition
python -m backend.webcam_detect --recognize --gesture
```

Supported gesture names: Open Palm, Thumbs Up, Thumbs Down, Closed Fist.

Naming note: runtime labels can appear as `open_palm`, `thumb_up`/`thumbs_up`,
`thumb_down`/`thumbs_down`, and `closed_fist` depending on model output.

PPT control note: in the Streamlit web app (`frontend/app.py`), Thumbs Up
and Thumbs Down map to slide navigation, and Closed Fist maps to close action.

For details see [GESTURE_GUIDE.md](GESTURE_GUIDE.md).

## Detection Algorithm

This project uses OpenCV's pre-trained `haarcascade_frontalface_default.xml`
(Viola-Jones Haar cascade) for frontal face detection.

Pipeline used in `FaceDetector.detect`:
1. Convert each input frame from BGR to grayscale.
2. Apply `CascadeClassifier.detectMultiScale(...)` to scan the image at
	multiple scales and positions.
3. Keep detections that satisfy neighborhood agreement (`minNeighbors`) and
	minimum face size (`minSize=(30, 30)`).
4. Return bounding boxes `(x, y, w, h)` and draw rectangles for visualization.

Current defaults:
- `scaleFactor=1.1`: finer scale steps (better recall, slightly slower).
- `minNeighbors=5`: balanced filtering of noisy detections.

Complexity and trade-offs:
- Runtime is roughly proportional to frame area and number of scales checked.
- Lower `scaleFactor` or `minNeighbors` can increase detections but may add
  false positives or latency.
- The detector is optimized for frontal faces under reasonable lighting.

## Recognition Pipeline

**Face Recognition Architecture:**
1. **Detection**: Haar cascade finds candidate face regions
2. **Validation**: DeepFace with `enforce_detection=True` rejects non-face crops
3. **Embedding**: Facenet512 generates 512-dim embeddings (rotation/lighting robust)
4. **Matching**: Cosine similarity to known embeddings with configurable threshold
5. **Optimization**: Skip-frame recognition (every N frames) + label remapping

**Gesture Recognition Architecture:**
1. **Detection**: MediaPipe extracts 21 hand landmarks (3D keypoints)
2. **Features**: Wrist-relative distances (20 features) — rotation-invariant
3. **Classification**: KNeighborsClassifier (k=3) trained on captured samples
4. **Confidence**: 1/(1+avg_distance) normalized to [0, 1]
