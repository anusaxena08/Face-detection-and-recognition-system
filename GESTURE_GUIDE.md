# Hand Gesture Recognition System

## Overview

This system enables training and recognizing custom hand gestures from your webcam. It uses MediaPipe for hand landmark detection and scikit-learn's KNN classifier for gesture recognition.

**Key Features:**
- Custom gesture training directly from your webcam
- Real-time gesture recognition at up to 30 FPS
- Integration with face recognition for unified detection pipeline
- Configurable gesture thresholds and camera settings

## Quick Start

### 1. Capture Training Samples

Capture training samples for a gesture (e.g., "palm"):

```bash
python src/gesture_onboard.py --gesture "palm" --samples 20
```

This captures 20 samples of your palm gesture:
- Press **SPACE** to capture a sample
- Press **q** to quit
- Samples are saved to `gesture_data/palm/`

Repeat for other gestures:

```bash
python src/gesture_onboard.py --gesture "thumbs_up" --samples 20
python src/gesture_onboard.py --gesture "fist" --samples 20
```

### 2. Train the Model

After capturing samples for all desired gestures:

```bash
python src/train_gesture.py
```

This trains a KNN classifier on all captured samples and saves it to `models/gesture_model.pkl`.

### 3. Recognize Gestures in Real-Time

Run live gesture recognition on your webcam:

```bash
python src/webcam_detect.py --gesture
```

To combine face AND gesture recognition:

```bash
python src/webcam_detect.py --recognize --gesture
```

Press **q** or **Esc** to quit.

## Architecture

### gesture_onboard.py

Interactive CLI for capturing training samples:
- Detects hands with MediaPipe
- Displays hand landmarks in real-time
- Saves wrist-to-keypoint distance features
- Generates `.npy` files in `gesture_data/<gesture_name>/`

**Arguments:**
- `--gesture NAME`: Gesture name (required)
- `--samples N`: Number of samples (default: 20)
- `--camera INDEX`: Camera index (default: 0)
- `--gesture-data-dir PATH`: Output directory (default: `gesture_data`)

### hand_gesture/detector.py

**HandDetector class** — wraps MediaPipe hand detection:

```python
detector = HandDetector(confidence=0.7)
landmarks = detector.detect(frame_bgr)  # List of (21, 3) arrays
features = detector.extract_features(landmarks[0])  # (20,) distances
```

**Methods:**
- `detect(frame_bgr)` → List of hand landmark arrays (21 keypoints × 3 coords)
- `extract_features(landmarks)` → Feature vector (20 wrist-relative distances)

### hand_gesture/trainer.py

**GestureTrainer class** — builds KNN classifier:

```python
trainer = GestureTrainer(n_neighbors=3)
trainer.train(gesture_data_dir="gesture_data", output_path="models/gesture_model.pkl")
```

Reads gesture samples from `gesture_data/<gesture>/*.npy` and trains a KNeighborsClassifier.

**Algorithm:**
1. Load all `.npy` files from gesture directories
2. Stack into training matrix X and create labels y
3. Fit KNeighborsClassifier (k=3 by default)
4. Save model dict with classifier and class names

### hand_gesture/recognizer.py

**GestureRecognizer class** — recognizes gestures in real-time:

```python
recognizer = GestureRecognizer(model_path="models/gesture_model.pkl")
gestures = recognizer.recognize(frame_bgr)  # List of (name, confidence)
# Returns: [("palm", 0.92), ("thumbs_up", 0.78)]
```

**Methods:**
- `recognize(frame_bgr)` → List of (gesture_name, confidence) tuples
- `draw_gestures(frame, gestures)` → Annotated frame with gesture labels

**Confidence Calculation:**
```
confidence = 1.0 / (1.0 + avg_distance_to_neighbors)
```

Ranges [0, 1] — higher = more confident.

### train_gesture.py

CLI entry point for training:

```bash
python src/train_gesture.py [--gesture-data-dir DIR] [--output PATH]
```

## Feature Extraction

Hand gestures are encoded as **20 features** (distances from wrist to each other keypoint):

```
Wrist (landmark 0) — reference point
Features = [distance_to_thumb, distance_to_index, ..., distance_to_pinky_tip]
```

This encoding is **rotation-invariant** (gesture orientation doesn't matter) but **scale-sensitive** (smaller/larger hands have different distances).

## Troubleshooting

### "Cannot open camera"

Ensure your camera index is correct:

```bash
python src/webcam_detect.py --camera 1  # Try camera 1 instead of 0
```

### "ModuleNotFoundError: No module named 'mediapipe'"

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Gestures not detected

1. **No samples captured?** Run gesture_onboard.py first
2. **Poor samples?** Retrain with more diverse poses
3. **Threshold too high?** Confidence defaults to [0, 1]; very low (< 0.5) may indicate poor training

Capture more samples and retrain:

```bash
python src/gesture_onboard.py --gesture "palm" --samples 30
python src/train_gesture.py
```

### Gesture confidence always low

This usually means hand poses don't match training data:
- Vary lighting, distance, and hand orientation while capturing
- Use 30-50 samples per gesture for better coverage

## Advanced: Action Triggering

To trigger actions on recognized gestures (e.g., open a presentation), extend `hand_gesture/recognizer.py`:

```python
def trigger_action(gesture_name: str) -> None:
    if gesture_name == "palm":
        subprocess.Popen(["powershell", "-c", "Start-Process 'presentation.pptx'"])
    elif gesture_name == "thumbs_up":
        print("Thumbs up!")
```

Call this in your main loop when `confidence > 0.8`.

## Performance Notes

- **Real-time speed:** ~30 FPS on CPU (MediaPipe is optimized for mobile)
- **Memory:** ~200 MB (MediaPipe models)
- **Hand detection:** Fast (< 5ms per frame)
- **Classification:** Instant (KNN with k=3 on 20 features)

## Example Workflow

```bash
# 1. Capture three gestures
python src/gesture_onboard.py --gesture "palm" --samples 25
python src/gesture_onboard.py --gesture "fist" --samples 25
python src/gesture_onboard.py --gesture "peace" --samples 25

# 2. Train
python src/train_gesture.py

# 3. Run with faces + gestures
python src/webcam_detect.py --recognize --gesture

# Output:
# - Face bounding boxes + names (if --recognize)
# - Gesture labels + confidence (if --gesture)
```

## References

- [MediaPipe Hands](https://github.com/google-mediapipe/mediapipe/wiki/Hands)
- [scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
