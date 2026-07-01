"""Face recognizer — matches detected face regions against saved encodings.

Loads the encoding database produced by trainer.py and assigns names to
detected face bounding boxes by comparing DeepFace embeddings via cosine
similarity.

Algorithm:
1. Crop each detected face region from the frame.
2. Compute a face embedding for the crop using DeepFace (same model used
   during training).
3. Compute cosine similarity between the crop embedding and every stored
   embedding.
4. If the best match exceeds the similarity threshold, label the face with
   that person's name; otherwise label it "Unknown".
5. Return list of (rect, name, similarity_score) tuples.

Performance note:
   DeepFace.represent() is CPU-intensive. For real-time webcam usage,
   recognition runs every `skip_frames` frames; detection-only runs on the
   other frames to keep the display smooth.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from face_detection.detector import Rect

# Keep OpenCV on CPU to avoid OpenCL upload/resource failures during inference.
cv2.ocl.setUseOpenCL(False)


Match = Tuple[Rect, str, float]  # (x, y, w, h), name, similarity


class FaceRecognizer:
    """Matches face regions against a pre-trained encoding database.

    Parameters
    ----------
    encodings_path:
        Path to the pickle file produced by trainer.train().
    threshold:
        Cosine similarity threshold in [0, 1]. Matches below this are
        labelled "Unknown". Higher = stricter. Default 0.6 works well with
        Facenet512.
    skip_frames:
        Run recognition every N frames. Other frames reuse the last result
        for speed. Default 5.
    """

    def __init__(
        self,
        encodings_path: str | Path = "models/encodings.pkl",
        threshold: float = 0.6,
        skip_frames: int = 5,
    ) -> None:
        encodings_path = Path(encodings_path)
        if not encodings_path.exists():
            raise FileNotFoundError(
                f"Encodings file not found: {encodings_path}\n"
                "Run training first:  python src/train.py"
            )

        with open(encodings_path, "rb") as f:
            data = pickle.load(f)

        self._model_name: str = data.get("model", "Facenet512")
        self._db: dict = data.get("encodings", {})
        self._threshold = threshold
        self._skip_frames = skip_frames
        self._frame_count = 0
        self._last_matches: List[Match] = []

        # Pre-normalise stored embeddings for fast cosine similarity
        self._flat: List[Tuple[str, np.ndarray]] = []
        for name, embeddings in self._db.items():
            for emb in embeddings:
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                self._flat.append((name, vec))

        if not self._flat:
            raise ValueError(
                "Encodings file contains no embeddings. "
                "Re-run training with at least one person's photos."
            )

        # Warm up the DeepFace model now so the first real frame isn't slow
        self._deepface = self._warmup_model()

    # ------------------------------------------------------------------
    # Model warm-up
    # ------------------------------------------------------------------

    def _warmup_model(self):
        """Load DeepFace model into memory once at startup."""
        from deepface import DeepFace as _DF
        try:
            blank = np.zeros((160, 160, 3), dtype=np.uint8)
            _DF.represent(img_path=blank, model_name=self._model_name, enforce_detection=False)
        except Exception:
            pass
        return _DF

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, frame, faces: List[Rect]) -> List[Match]:
        """Return (rect, name, score) for every detected face in frame.

        Runs full recognition every `skip_frames` frames; on skipped frames
        the labels from the last recognition are remapped to the current
        face positions by nearest-centre distance so labels track movement.
        """
        self._frame_count += 1

        if not faces:
            self._last_matches = []
            return []

        if self._frame_count % self._skip_frames != 0 and self._last_matches:
            return self._remap_labels(faces, self._last_matches)

        matches: List[Match] = []
        for rect in faces:
            x, y, w, h = rect
            fh, fw = frame.shape[:2]
            pad_x = max(8, int(w * 0.15))
            pad_y = max(8, int(h * 0.15))
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
            crop = frame[y0:y1, x0:x1]

            if crop.size == 0:
                continue

            name, score, is_face = self._match_crop(_preprocess(crop))
            if not is_face:
                continue
            matches.append((rect, name, score))

        self._last_matches = matches
        return matches

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remap_labels(self, faces: List[Rect], last: List[Match]) -> List[Match]:
        """Assign labels from last recognition to current face rects.

        Matches each current rect to the nearest previous rect by centre
        distance so labels stay attached to the correct person as they move.
        """
        result = []
        for rect in faces:
            cx = rect[0] + rect[2] // 2
            cy = rect[1] + rect[3] // 2
            nearest = min(
                last,
                key=lambda m: abs((m[0][0] + m[0][2] // 2) - cx)
                + abs((m[0][1] + m[0][3] // 2) - cy),
            )

            (px, py, pw, ph), pname, pscore = nearest
            nearest_cx = px + pw // 2
            nearest_cy = py + ph // 2
            center_dist = abs(nearest_cx - cx) + abs(nearest_cy - cy)

            # Reject remap if box moved too far or scale changed too much.
            max_move = int(max(rect[2], rect[3], pw, ph) * 0.9)
            size_ratio_w = rect[2] / float(max(1, pw))
            size_ratio_h = rect[3] / float(max(1, ph))
            size_ok = 0.5 <= size_ratio_w <= 2.0 and 0.5 <= size_ratio_h <= 2.0

            if center_dist <= max_move and size_ok:
                result.append((rect, pname, pscore))
        return result

    def _is_human_face_crop(self, crop: np.ndarray) -> bool:
        """Return False if the crop does not contain enough human skin tone.

        Uses a lower threshold than the detector (20 %) because the crop is
        tightly bounded to the face region and has higher skin pixel density.
        Monkey/animal crops consistently fail this check.
        """
        if crop.size == 0:
            return False
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([0, 128, 70], dtype=np.uint8),
            np.array([255, 185, 140], dtype=np.uint8),
        )
        total = crop.shape[0] * crop.shape[1]
        ratio = float(np.count_nonzero(skin_mask)) / max(1, total)
        return ratio >= 0.20

    def _match_crop(self, crop: np.ndarray) -> Tuple[str, float, bool]:
        """Embed a validated face crop and find the closest stored encoding."""
        # Fast species pre-check: reject crops that lack human skin tone.
        # This stops animal crops that slip past the detector from reaching DeepFace.
        if not self._is_human_face_crop(crop):
            return "", 0.0, False

        try:
            result = self._deepface.represent(
                img_path=crop,
                model_name=self._model_name,
                enforce_detection=True,
                detector_backend="opencv",
            )
            raw = np.array(result[0]["embedding"], dtype=np.float32)
        except Exception:
            # Retry strict validation once on a normalized crop to avoid
            # "second attempt" misses without allowing non-face objects.
            try:
                retry = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                retry = _preprocess(retry)
                result = self._deepface.represent(
                    img_path=retry,
                    model_name=self._model_name,
                    enforce_detection=True,
                    detector_backend="opencv",
                )
                raw = np.array(result[0]["embedding"], dtype=np.float32)
            except Exception:
                return "", 0.0, False

        norm = np.linalg.norm(raw)
        if norm == 0:
            return "", 0.0, False
        query = raw / norm

        # Cosine similarity against all stored embeddings
        best_name, best_score = "Unknown", 0.0
        for name, stored in self._flat:
            score = float(np.dot(query, stored))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self._threshold:
            return "Unknown", best_score, True

        # Convert name back from safe_name format (underscores -> spaces)
        return best_name.replace("_", " "), best_score, True



# ---------------------------------------------------------------------------
# Shared preprocessing (mirrors trainer._preprocess for consistency)
# ---------------------------------------------------------------------------

def _preprocess(bgr_img: np.ndarray) -> np.ndarray:
    """Apply CLAHE lighting normalisation - must match trainer preprocessing."""
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def draw_recognition(frame, matches: List[Match]):
    """Draw bounding boxes and name labels on frame."""
    for (x, y, w, h), name, score in matches:
        color = (0, 200, 120) if name != "Unknown" else (0, 60, 200)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = name if name == "Unknown" else f"{name} ({score:.2f})"
        cv2.putText(
            frame,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame
