"""Hand gesture recognizer for trained hand contours."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from backend.hand_gesture.detector import HandDetector, HandLandmarks

GestureMatch = Tuple[np.ndarray, str, float]


class GestureRecognizer:
    """Classifies hand gestures using a trained KNN model."""

    def __init__(self, model_path: str | Path = "models/gesture_model.pkl") -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Gesture model not found: {model_path}\n"
                "Train first: python src/train_gesture.py"
            )

        with open(model_path, "rb") as file_obj:
            data = pickle.load(file_obj)

        self.model = data["model"]
        self.classes = data["classes"]
        self.distance_thresholds = data.get("distance_thresholds", {})
        self.global_threshold = data.get("global_threshold", 0.35)
        self.one_class_profile = data.get("one_class_profile")
        self.min_confidence = 0.6
        self.detector = HandDetector()

    def recognize_contours(
        self,
        frame_bgr: np.ndarray,
        contours: List[HandLandmarks],
        face_rects=None,
    ) -> List[GestureMatch]:
        """Return only accepted palm detections for the supplied contours."""
        matches: List[GestureMatch] = []

        for contour in contours:
            if face_rects and self._overlaps_face(contour, face_rects):
                continue

            if not self._passes_hand_shape_gate(contour, frame_bgr):
                continue

            features = self.detector.extract_features(contour).reshape(1, -1)

            # Palm-only mode: reject anything outside learned palm feature manifold.
            if self.one_class_profile is not None:
                if not self._passes_one_class_gate(features[0]):
                    continue

            distances, _ = self.model.kneighbors(features)
            predicted_class = int(self.model.predict(features)[0])
            gesture_name = self.classes[predicted_class]

            avg_distance = float(np.mean(distances[0]))
            max_distance = float(self.distance_thresholds.get(gesture_name, self.global_threshold))
            if avg_distance > max_distance:
                continue

            confidence = max(0.0, 1.0 - (avg_distance / max_distance))

            # Keep only stable palm detections.
            if gesture_name.lower() != "palm" or confidence < self.min_confidence:
                continue

            matches.append((contour, gesture_name, confidence))

        return matches

    def _passes_one_class_gate(self, feature_vec: np.ndarray) -> bool:
        """Reject non-palm samples in palm-only training mode."""
        centroid = np.array(self.one_class_profile["centroid"], dtype=np.float32)
        max_distance = float(self.one_class_profile["max_distance"])
        distance = float(np.linalg.norm(feature_vec - centroid))
        return distance <= max_distance

    def _passes_hand_shape_gate(self, contour: np.ndarray, frame_bgr: np.ndarray) -> bool:
        """Reject non-hand contours before classification.

        This keeps detector recall high while filtering obvious non-hand objects.
        """
        contour_i = contour.astype(np.int32).reshape(-1, 1, 2)
        area = cv2.contourArea(contour_i)
        if area <= 0:
            return False

        frame_h, frame_w = frame_bgr.shape[:2]
        area_ratio = area / float(frame_h * frame_w)
        if area_ratio < 0.012 or area_ratio > 0.40:
            return False

        x, y, w, h = cv2.boundingRect(contour_i)
        if h == 0:
            return False
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.25 or aspect_ratio > 2.6:
            return False

        extent = area / float(max(1, w * h))
        if extent < 0.2 or extent > 0.9:
            return False

        hull = cv2.convexHull(contour_i)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False
        solidity = area / hull_area
        if solidity < 0.35 or solidity > 0.92:
            return False

        perimeter = cv2.arcLength(contour_i, True)
        if perimeter < 55:
            return False

        # Validate that this contour region is skin-like and not bright white glare.
        if not self._passes_skin_gate(contour_i, frame_bgr):
            return False

        # Open palms have visible finger valleys (convexity defects).
        # Require at least 2 deep defects for a reliable open palm.
        if len(contour_i) >= 5:
            hull_idx = cv2.convexHull(contour_i, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 4:
                return False
            defects = cv2.convexityDefects(contour_i, hull_idx)
            if defects is None:
                return False
            min_depth = max(1000.0, perimeter * 1.5)
            deep_defects = int(np.sum(defects[:, 0, 3] > min_depth))
            if deep_defects < 2:
                return False

        return True

    def _passes_skin_gate(self, contour_i: np.ndarray, frame_bgr: np.ndarray) -> bool:
        x, y, w, h = cv2.boundingRect(contour_i)
        if w <= 0 or h <= 0:
            return False

        roi = frame_bgr[y : y + h, x : x + w]
        if roi.size == 0:
            return False

        # Contour mask in ROI coordinates.
        local_contour = contour_i.copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [local_contour], -1, 255, thickness=-1)

        # HSV skin mask (broad).
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_skin_1 = cv2.inRange(
            hsv,
            np.array([0, 20, 50], dtype=np.uint8),
            np.array([25, 200, 255], dtype=np.uint8),
        )
        hsv_skin_2 = cv2.inRange(
            hsv,
            np.array([160, 20, 50], dtype=np.uint8),
            np.array([179, 200, 255], dtype=np.uint8),
        )
        hsv_skin = cv2.bitwise_or(hsv_skin_1, hsv_skin_2)

        # YCrCb skin mask — widened to cover darker skin tones.
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        ycrcb_skin = cv2.inRange(
            ycrcb,
            np.array([0, 130, 70], dtype=np.uint8),
            np.array([255, 180, 135], dtype=np.uint8),
        )

        skin_mask = cv2.bitwise_and(cv2.bitwise_or(hsv_skin, ycrcb_skin), contour_mask)

        contour_area = float(np.count_nonzero(contour_mask))
        if contour_area < 1:
            return False

        skin_ratio = float(np.count_nonzero(skin_mask)) / contour_area
        if skin_ratio < 0.30:
            return False

        # HSV channels used by both sanity checks and glare rejection.
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Additional color sanity checks for skin inside contour.
        contour_pixels = contour_mask > 0
        s_vals = s_channel[contour_pixels]
        v_vals = v_channel[contour_pixels]
        if s_vals.size == 0 or v_vals.size == 0:
            return False

        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))
        std_v = float(np.std(v_vals))

        # Background lights often have very low saturation and very high value.
        if mean_s < 30 and mean_v > 185:
            return False

        # Strong lights tend to be flat with low variance in brightness.
        if mean_v > 170 and std_v < 22:
            return False

        # Reject white/bright lights inside contour region.
        white_like = ((s_channel < 35) & (v_channel > 220)).astype(np.uint8) * 255
        white_like = cv2.bitwise_and(white_like, contour_mask)
        white_ratio = float(np.count_nonzero(white_like)) / contour_area
        if white_ratio > 0.18:
            return False

        return True

    def _overlaps_face(self, contour: np.ndarray, face_rects) -> bool:
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        for fx, fy, fw, fh in face_rects:
            if x < fx + fw and x + w > fx and y < fy + fh and y + h > fy:
                return True
        return False

    def release(self) -> None:
        self.detector.release()


def draw_gestures(frame, gestures: List[GestureMatch]) -> np.ndarray:
    """Draw cyan rectangles only for accepted palm detections."""
    for contour, gesture_name, confidence in gestures:
        contour_int = contour.astype(np.int32)
        x, y, w, h = cv2.boundingRect(contour_int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

        label = f"{gesture_name.upper()} {confidence:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(0, y - text_h - 10)
        cv2.rectangle(frame, (x, top), (x + text_w + 10, y), (255, 255, 0), -1)
        cv2.putText(
            frame,
            label,
            (x + 5, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame
