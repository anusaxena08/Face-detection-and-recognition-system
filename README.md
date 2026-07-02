# Python Face Detection Project

A simple starter project for face detection using OpenCV and Haar cascades.

## Features
- Detect faces from your webcam in real time.
- Detect faces in a single image and save the result.
- Lightweight setup, pure Python.
- Face recognition for known people, with unrecognized faces labeled as Unknown.
- Hand gesture recognition including Open Palm, Thumbs Up, Thumbs Down, and Closed Fist.
- Gesture-driven PPT control (slide up/down and close action) in the live web interface.

## Project Structure

- `src/webcam_detect.py`: real-time webcam face detection.
- `src/image_detect.py`: image file face detection.
- `src/face_detection/detector.py`: reusable detector logic.

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

### Webcam detection

```powershell
$env:PYTHONPATH = "src"
python src/webcam_detect.py --camera 0
```

Press `q` to quit.

### Image detection

```powershell
$env:PYTHONPATH = "src"
python src/image_detect.py path\to\image.jpg --output outputs\result.jpg
```

## Notes
- Webcam usage depends on local camera permissions.
- Haar cascades are fast and lightweight, but may be less accurate than deep learning models.

## Face Recognition

Train and recognize custom faces:

```bash
# Capture 20 samples of each person
python src/onboard.py --name "Alice" --webcam --photos 20
python src/onboard.py --name "Bob" --webcam --photos 20

# Train embeddings
python src/train.py

# Recognize in real-time
python src/webcam_detect.py --recognize
```

See [README details](#recognition-pipeline) below for architecture.

## Hand Gesture Recognition

Train custom hand gestures and trigger actions on recognition:

```bash
# Capture 20 samples of each gesture
python src/gesture_onboard.py --gesture "palm" --samples 20
python src/gesture_onboard.py --gesture "thumbs_up" --samples 20
python src/gesture_onboard.py --gesture "thumbs_down" --samples 20
python src/gesture_onboard.py --gesture "closed_fist" --samples 20

# Train gesture classifier
python src/train_gesture.py

# Recognize gestures in real-time
python src/webcam_detect.py --gesture

# Combine face + gesture recognition
python src/webcam_detect.py --recognize --gesture
```

Supported gesture names: Open Palm, Thumbs Up, Thumbs Down, Closed Fist.

Naming note: runtime labels can appear as `open_palm`, `thumb_up`/`thumbs_up`,
`thumb_down`/`thumbs_down`, and `closed_fist` depending on model output.

PPT control note: in the live web experience (`src/web_website.py`), Thumbs Up
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
