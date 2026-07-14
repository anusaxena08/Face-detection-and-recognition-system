from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from backend.face_detection.detector import FaceDetector
from backend.face_detection.recognizer import FaceRecognizer


app = FastAPI(title="Face Detection Backend", version="1.0.0")

_detector: FaceDetector | None = None
_recognizer: FaceRecognizer | None = None


def _get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(scale_factor=1.08, min_neighbors=5)
    return _detector


def _get_recognizer() -> FaceRecognizer | None:
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    try:
        _recognizer = FaceRecognizer(encodings_path="models/encodings.pkl", threshold=0.6, skip_frames=1)
    except Exception:
        _recognizer = None
    return _recognizer


def _decode_upload(content: bytes) -> np.ndarray:
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image upload.")
    return image


@app.get("/health")
def health() -> dict[str, Any]:
    recognizer = _get_recognizer()
    return {
        "status": "ok",
        "recognition_enabled": recognizer is not None,
    }


@app.post("/detect")
async def detect_faces(
    file: UploadFile = File(...),
    strict: bool = Query(False, description="Use stricter face filtering for still images."),
    recognize: bool = Query(True, description="Try to match faces against models/encodings.pkl."),
) -> dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = _decode_upload(content)
    detector = _get_detector()
    faces = detector.detect(image, strict=strict)

    payload: dict[str, Any] = {
        "face_count": len(faces),
        "faces": [
            {"x": x, "y": y, "width": w, "height": h}
            for x, y, w, h in faces
        ],
    }

    recognizer = _get_recognizer() if recognize else None
    if recognizer is None:
        payload["matches"] = []
        payload["recognition_enabled"] = False
        return payload

    matches = recognizer.recognize(image, faces)
    payload["recognition_enabled"] = True
    payload["matches"] = [
        {
            "name": name,
            "score": score,
            "box": {"x": x, "y": y, "width": w, "height": h},
        }
        for (x, y, w, h), name, score in matches
    ]
    return payload