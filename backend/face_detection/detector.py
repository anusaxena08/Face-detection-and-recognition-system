from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

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
        if not hasattr(cv2, "CascadeClassifier"):
            version = getattr(cv2, "__version__", "unknown")
            cv2_path = getattr(cv2, "__file__", "unknown")
            raise RuntimeError(
                "The active OpenCV build does not provide CascadeClassifier. "
                "This project requires a stable 4.x opencv-python/opencv-python-headless build. "
                f"Detected cv2 version={version} from {cv2_path}. "
                "Remove conflicting OpenCV 5 packages and reinstall requirements."
            )

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(str(cascade_path))
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        # Secondary cascade — human eyes are highly distinctive vs. animal faces.
        eye_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
        self._eye_classifier = cv2.CascadeClassifier(str(eye_path))

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

    def detect(self, frame, strict: bool = False) -> List[Rect]:
        """Detect faces in a BGR frame and return (x, y, w, h) boxes.

        Parameters
        ----------
        strict:
            When True (still images / uploads) uses 2× min_neighbors and skips
            the dim-light fallback pass, preventing over-detection in group photos.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]

        # Scale min face size to image resolution.
        min_face_px = max(40, int(min(fh, fw) * 0.04))

        # Strict mode doubles the evidence requirement per detection window.
        neighbors = self._min_neighbors * 2 if strict else self._min_neighbors

        # Primary pass: CLAHE helps lift facial contrast in dim lighting.
        enhanced = self._enhance_gray(gray)
        faces = self._classifier.detectMultiScale(
            enhanced,
            scaleFactor=self._scale_factor,
            minNeighbors=neighbors,
            minSize=(min_face_px, min_face_px),
        )

        if len(faces) == 0 and not strict:
            # Fallback ONLY for genuinely dark real-time frames, never still images.
            if float(gray.mean()) < 110:
                boosted = self._boost_low_light(enhanced)
                faces = self._classifier.detectMultiScale(
                    boosted,
                    scaleFactor=self._scale_factor,
                    minNeighbors=max(4, neighbors - 2),
                    minSize=(min_face_px, min_face_px),
                )

        faces_list = [tuple(map(int, face)) for face in faces]
        faces_list = self._filter_face_candidates(faces_list, frame.shape[:2])
        faces_list = self._verify_human_faces(faces_list, enhanced, frame)
        faces_list = self._suppress_duplicate_faces(faces_list)
        # Trim the Haar background padding so boxes sit tight on the actual face.
        return [self._trim_face_box(x, y, w, h, fh, fw) for x, y, w, h in faces_list]

    def _trim_face_box(self, x: int, y: int, w: int, h: int, fh: int, fw: int) -> Rect:
        """Remove the background border that Haar detection windows add.

        Haar windows include ~7 % extra forehead, ~5 % sides, ~3 % chin.
        Trimming those makes the drawn box sit tightly on the face.
        """
        trim_top  = int(h * 0.07)
        trim_side = int(w * 0.05)
        trim_bot  = int(h * 0.03)
        nx = min(fw - 1, x + trim_side)
        ny = min(fh - 1, y + trim_top)
        nw = max(20, min(w - 2 * trim_side, fw - nx))
        nh = max(20, min(h - trim_top - trim_bot, fh - ny))
        return (nx, ny, nw, nh)

    def _verify_human_faces(
        self, faces: List[Rect], gray_enhanced: np.ndarray, frame_bgr: np.ndarray
    ) -> List[Rect]:
        """Keep only boxes that contain human eyes or strong human skin tone.

        Animals (monkeys, apes, dogs) are rejected because:
        - haarcascade_eye.xml was trained on human eyes and rarely fires on
          animal eye regions framed at human-face proportions.
        - Human skin occupies a narrow YCrCb band absent from most animal fur.
        """
        verified: List[Rect] = []
        fh, fw = frame_bgr.shape[:2]

        for x, y, w, h in faces:
            has_eye = False

            if not self._eye_classifier.empty():
                # Search upper 65 % of the box — where human eyes sit.
                eye_y2 = min(fh, y + int(h * 0.65))
                roi_gray = gray_enhanced[max(0, y):eye_y2, max(0, x):min(fw, x + w)]
                if roi_gray.size > 0:
                    eyes = self._eye_classifier.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(max(8, w // 8), max(8, w // 8)),
                    )
                    has_eye = len(eyes) >= 1

            if has_eye:
                # Eye cascade fired → almost certainly a human face.
                verified.append((x, y, w, h))
            elif self._has_human_skin(frame_bgr, x, y, w, h, fh, fw):
                # No eye hit but strong skin signal (closed eyes / dim light).
                verified.append((x, y, w, h))
            # else: animal or object — silently dropped.

        return verified

    def _has_human_skin(
        self,
        frame_bgr: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        fh: int,
        fw: int,
    ) -> bool:
        """Return True when the face ROI contains enough human-skin-coloured pixels.

        YCrCb range [0,128,70]→[255,185,140] covers all human ethnicities while
        excluding the grey/brown tones typical of animal fur.
        """
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([0, 128, 70], dtype=np.uint8),
            np.array([255, 185, 140], dtype=np.uint8),
        )
        total = roi.shape[0] * roi.shape[1]
        ratio = float(np.count_nonzero(skin_mask)) / max(1, total)
        # Human face boxes typically have 28-90 % skin-coloured pixels.
        return ratio >= 0.28

    def _filter_face_candidates(self, faces: List[Rect], frame_hw: Tuple[int, int]) -> List[Rect]:
        """Drop unlikely boxes to reduce false positives on non-face objects."""
        if not faces:
            return []

        fh, fw = frame_hw
        min_area = max(900, int(0.002 * fw * fh))
        filtered: List[Rect] = []

        for x, y, w, h in faces:
            if w <= 0 or h <= 0:
                continue
            area = w * h
            if area < min_area:
                continue

            ratio = w / float(max(1, h))
            if ratio < 0.65 or ratio > 1.55:
                continue

            filtered.append((x, y, w, h))

        return filtered

    def _suppress_duplicate_faces(self, faces: List[Rect]) -> List[Rect]:
        """Collapse overlapping/contained windows that point to the same face."""
        if not faces:
            return []

        # Keep larger, stronger windows first.
        sorted_faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        kept: List[Rect] = []

        for candidate in sorted_faces:
            if any(self._is_duplicate_face(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)

        return kept

    def _is_duplicate_face(self, a: Rect, b: Rect) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        a_x2, a_y2 = ax + aw, ay + ah
        b_x2, b_y2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return False

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        iou = inter_area / float(max(1, union))

        # One box mostly contained in another box.
        contain_a = inter_area / float(max(1, area_a))
        contain_b = inter_area / float(max(1, area_b))

        return iou >= 0.25 or contain_a >= 0.75 or contain_b >= 0.75

    def _enhance_gray(self, gray):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (3, 3), 0)

    def _boost_low_light(self, gray):
        mean_intensity = float(gray.mean())
        if mean_intensity >= 110:
            return gray

        gamma = 0.75 if mean_intensity >= 80 else 0.6
        normalized = gray.astype("float32") / 255.0
        boosted = np.power(normalized, gamma)
        boosted = np.clip(boosted * 255.0, 0, 255).astype("uint8")
        return boosted


def draw_faces(frame, faces: List[Rect]):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 120), 2)
    return frame
