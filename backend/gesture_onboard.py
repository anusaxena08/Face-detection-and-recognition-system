"""Gesture onboarding — capture training samples for custom hand gestures.

Usage:
  python src/gesture_onboard.py --gesture "palm" --samples 20
  python src/gesture_onboard.py --gesture "thumbs_up" --samples 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from backend.hand_gesture.detector import HandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture training samples for a custom hand gesture.",
    )
    parser.add_argument(
        "--gesture",
        required=True,
        help="Name of the gesture to capture (e.g., 'palm', 'thumbs_up').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to capture (default: 20).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0).",
    )
    parser.add_argument(
        "--gesture-data-dir",
        type=Path,
        default=Path("gesture_data"),
        help="Root directory for gesture training data (default: gesture_data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gesture_name = args.gesture.lower().replace(" ", "_")
    save_dir = args.gesture_data_dir / gesture_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Capturing gesture: {gesture_name} ===")
    print(f"Saving to: {save_dir}\n")

    detector = HandDetector()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}.")

    captured = 0
    print(f"Focus the camera window. Press SPACE to capture, q to quit.\n")

    try:
        while captured < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Detect hands
            hand_contours = detector.detect(frame)
            h, w = frame.shape[:2]

            # Draw frame info
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Gesture: {gesture_name}  |  Captured: {captured}/{args.samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                "SPACE = capture   q = quit",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            # Draw detected hands
            if hand_contours:
                for contour in hand_contours:
                    contour_int = contour.astype(np.int32)
                    cv2.drawContours(overlay, [contour_int], 0, (0, 255, 0), 2)
                    cv2.circle(overlay, tuple(contour_int[0]), 5, (0, 255, 0), -1)

            cv2.imshow(f"Gesture Capture — {gesture_name}", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and hand_contours:
                # Save features from the first detected hand
                features = detector.extract_features(hand_contours[0])
                file_path = save_dir / f"{gesture_name}_{captured:03d}.npy"
                np.save(file_path, features)
                captured += 1
                print(f"  Saved sample {captured}/{args.samples}")

                # Flash feedback
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), 30)
                cv2.imshow(f"Gesture Capture — {gesture_name}", flash)
                cv2.waitKey(150)

            elif key == ord("q"):
                print("Capture stopped.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()

    print(f"\n{captured} sample(s) captured for '{gesture_name}'.")
    print("Next: Run 'python src/train_gesture.py' to train the model.\n")


if __name__ == "__main__":
    main()
