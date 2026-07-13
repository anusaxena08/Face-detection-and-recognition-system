"""MediaPipe-based hand gesture recognizer using the pre-trained gesture model.

Uses mediapipe.tasks.python.vision.GestureRecognizer which ships a fully trained
model supporting: Open_Palm, Closed_Fist, Pointing_Up, Thumb_Down, Thumb_Up,
Victory, ILoveYou.

No custom training required — the .task model file is loaded at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL_PATH = "models/gesture_recognizer.task"

# MediaPipe hand skeleton connections (indices into the 21-landmark set).
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm cross-bar
]


class GestureResult(NamedTuple):
    gesture_name: str   # e.g. "Open_Palm", "Thumb_Up"
    score: float        # confidence 0-1
    handedness: str     # "Left" or "Right"
    landmarks: list     # list of NormalizedLandmark (x, y, z)


class MpGestureRecognizer:
    """Wraps MediaPipe's pre-trained GestureRecognizer task.

    Parameters
    ----------
    model_path:
        Path to the gesture_recognizer.task file.
    num_hands:
        Maximum number of hands to detect simultaneously.
    min_detection_confidence:
        Minimum confidence for hand detection and gesture classification.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 4,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe gesture model not found: {model_path}\n"
                "Download with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task "
                "-OutFile models/gesture_recognizer.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    def recognize(self, frame_bgr: np.ndarray) -> List[GestureResult]:
        """Run gesture recognition on a BGR frame.

        Parameters
        ----------
        frame_bgr:
            Input frame in BGR format (as returned by OpenCV).

        Returns
        -------
        list of GestureResult
            One entry per detected hand. Empty when no hands are visible.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize(mp_image)

        gestures: List[GestureResult] = []
        if not result.gestures:
            return gestures

        for i, hand_gestures in enumerate(result.gestures):
            if not hand_gestures:
                continue

            top = hand_gestures[0]
            name: str = top.category_name
            score: float = float(top.score)

            # Skip "None" category and low-confidence results.
            if not name or name.lower() == "none" or score < 0.5:
                continue

            handedness = (
                result.handedness[i][0].category_name
                if result.handedness and i < len(result.handedness)
                else "Unknown"
            )
            landmarks = (
                result.hand_landmarks[i]
                if result.hand_landmarks and i < len(result.hand_landmarks)
                else []
            )
            gestures.append(GestureResult(name, score, handedness, landmarks))

        return gestures

    def close(self) -> None:
        """Release the underlying MediaPipe recognizer."""
        try:
            self._recognizer.close()
        except Exception:
            pass


def draw_mp_gestures(frame_bgr: np.ndarray, gestures: List[GestureResult]) -> np.ndarray:
    """Draw hand skeleton and gesture label for each detected hand.

    Parameters
    ----------
    frame_bgr:
        Frame to annotate in-place (BGR).
    gestures:
        Results from MpGestureRecognizer.recognize().

    Returns
    -------
    Annotated frame (same object as input).
    """
    fh, fw = frame_bgr.shape[:2]

    for gr in gestures:
        if not gr.landmarks:
            continue

        # Convert normalised coordinates to pixel positions.
        pts = [
            (int(lm.x * fw), int(lm.y * fh))
            for lm in gr.landmarks
        ]

        # Draw skeleton connections.
        for a, b in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 255), 2)

        # Draw landmark dots.
        for px, py in pts:
            cv2.circle(frame_bgr, (px, py), 4, (0, 255, 255), -1)

        # Draw label above the wrist (landmark 0).
        label = f"{gr.gesture_name}  {gr.score:.0%}"
        wx, wy = pts[0] if pts else (10, 30)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx = max(0, min(wx, fw - tw - 12))
        ly = max(th + 12, wy - 15)
        cv2.rectangle(frame_bgr, (lx, ly - th - 8), (lx + tw + 10, ly + 4),
                      (0, 180, 220), -1)
        cv2.putText(frame_bgr, label, (lx + 5, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return frame_bgr
