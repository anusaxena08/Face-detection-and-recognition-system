"""Hand gesture detector using OpenCV hand detection.

Detects hand regions and extracts contour-based features for gesture classification.
This approach is simpler and more reliable on Windows than MediaPipe.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

HandLandmarks = np.ndarray  # Shape (N, 2) — hand contour points


class HandDetector:
    """Detects hands using skin color + contour analysis.

    Simpler alternative to MediaPipe that works reliably on Windows.
    Extracts hand contours and computes shape features.
    """

    def __init__(self, confidence: float = 0.7) -> None:
        """Initialize hand detector.
        
        Parameters
        ----------
        confidence : float
            Minimum confidence threshold (0-1). Higher = stricter detection.
        """
        self.confidence = confidence

    def detect(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Detect hands and return list of contour arrays.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input frame in BGR format.

        Returns
        -------
        contours : list of np.ndarray
            Each array contains contour points (N, 2) for detected hands.
        """
        # Convert to HSV for skin detection
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color (more restrictive)
        # Lower range: H=0-20 (red/skin), S=10-150, V=60-255
        lower_skin1 = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin1 = np.array([20, 150, 255], dtype=np.uint8)
        
        mask = cv2.inRange(frame_hsv, lower_skin1, upper_skin1)
        
        # Also check for darker skin tones
        lower_skin2 = np.array([0, 5, 30], dtype=np.uint8)
        upper_skin2 = np.array([20, 120, 200], dtype=np.uint8)
        mask2 = cv2.inRange(frame_hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask, mask2)
        
        # Apply morphological operations to clean up while keeping finger detail
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by multiple criteria
        h, w = frame_bgr.shape[:2]
        min_area = (h * w) * 0.01  # Minimum 1% so farther hands are still detected
        max_area = (h * w) * 0.45  # Allow large close-up hands
        
        hand_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filtering
            if not (min_area < area < max_area):
                continue
            
            # Perimeter and circularity filtering
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 35:
                continue
                
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Hands are NOT very circular (faces might be)
            # This filters out round objects and faces
            if circularity > 0.85:
                continue
            
            # Convexity filtering
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Hands have moderate solidity (faces/flat regions are often very solid)
            if solidity < 0.25 or solidity > 0.95:
                continue
            
            # Bounding box aspect ratio
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            
            # Keep wide range; actual rejection happens in gesture recognizer distance gate
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            hand_contours.append(contour.reshape(-1, 2).astype(np.float32))

        # Largest contours first helps with multiple visible palms.
        hand_contours.sort(key=lambda c: cv2.contourArea(c.astype(np.int32).reshape(-1, 1, 2)), reverse=True)
        return hand_contours[:8]

    def extract_features(self, landmarks: HandLandmarks) -> np.ndarray:
        """Extract gesture features from hand contour.

        Computes shape descriptors that are rotation-invariant:
        - Contour area
        - Contour perimeter  
        - Contour circularity
        - Bounding box aspect ratio
        - Convexity defects count
        
        Parameters
        ----------
        landmarks : np.ndarray
            Shape (N, 2) array of hand contour points.

        Returns
        -------
        features : np.ndarray
            Shape (10,) — normalized shape features.
        """
        contour = landmarks.astype(np.int32).reshape(-1, 1, 2)
        
        # Area
        area = cv2.contourArea(contour)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            perimeter = 1
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h) if w * h > 0 else 0
        
        # Convex hull and defects
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Hu moments (shape signature).
        # Use absolute values so mirrored hands (left/right) map similarly.
        moments = cv2.HuMoments(contour)
        hu = np.abs(moments.flatten()[:4])  # First 4 moments are sufficient here
        
        # Combine features
        features = np.array([
            np.log10(area + 1),  # Log area
            circularity,
            aspect_ratio,
            extent,
            solidity,
            *hu
        ], dtype=np.float32)
        
        # Normalize
        features = np.nan_to_num(features, 0)
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features[:10].astype(np.float32)

    def release(self) -> None:
        """Clean up resources (no-op for this implementation)."""
        pass
