from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2


"""Face detection utilities built on OpenCV's Haar cascade pipeline.

Algorithm summary:
1. Convert the input frame from BGR to grayscale.
2. Run a pre-trained Viola-Jones Haar cascade classifier with a multi-scale
   sliding-window search (detectMultiScale).
3. Merge and filter candidate windows using minNeighbors and return final
   face bounding boxes as (x, y, w, h) integer tuples.

Why this works:
- The cascade is trained on Haar-like intensity features and uses boosted
  weak learners organized in stages, which allows quick rejection of
  non-face regions.
- Multi-scale scanning supports faces appearing at different sizes.
"""


Rect = Tuple[int, int, int, int]


class FaceDetector:
    """OpenCV Haar cascade face detector.

    Parameters:
    - scale_factor: image pyramid reduction factor between scales.
      Smaller values increase accuracy/recall but are slower.
    - min_neighbors: minimum nearby detections required to keep a result.
      Higher values reduce false positives but may miss small/partial faces.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(str(cascade_path))
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

    def detect(self, frame) -> List[Rect]:
        """Detect faces in a BGR frame and return (x, y, w, h) boxes.

        Internal steps:
        - Convert frame to grayscale to match cascade training input.
        - Run detectMultiScale to scan over location and scale.
        - Enforce minSize=(30, 30) to ignore tiny unstable detections.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(30, 30),
        )
        return [tuple(map(int, face)) for face in faces]


def draw_faces(frame, faces: List[Rect]):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 120), 2)
    return frame
