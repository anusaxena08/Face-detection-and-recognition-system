# Python Face Detection Project

A simple starter project for face detection using OpenCV and Haar cascades.

## Features
- Detect faces from your webcam in real time.
- Detect faces in a single image and save the result.
- Lightweight setup, pure Python.

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
