from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from backend.hand_gesture.detector import HandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import palm images from a folder and convert to gesture feature files.",
    )
    parser.add_argument(
        "--palm-dir",
        type=Path,
        required=True,
        help="Folder containing palm images (jpg/png/webp).",
    )
    parser.add_argument(
        "--gesture-data-dir",
        type=Path,
        default=Path("gesture_data"),
        help="Output gesture data directory (default: gesture_data).",
    )
    parser.add_argument(
        "--class-name",
        default="palm",
        help="Gesture class name (default: palm).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum images to process (default: 500).",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing .npy files in target class folder before importing.",
    )
    return parser.parse_args()


def _iter_images(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        for fp in folder.glob(ext):
            yield fp


def _extract_main_object_ignore_white(img: np.ndarray) -> np.ndarray:
    """Remove white background and keep only the main foreground object.

    This is tuned for dataset images where the hand is centered and the
    background is plain white.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White background: very low saturation + very high brightness.
    white_mask = cv2.inRange(hsv, np.array([0, 0, 210], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8))
    fg_mask = cv2.bitwise_not(white_mask)

    # Clean up tiny holes and noise in foreground mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Keep the largest non-white region as the main object (expected hand).
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < (img.shape[0] * img.shape[1] * 0.01):
        return img

    x, y, w, h = cv2.boundingRect(largest)
    pad = int(max(w, h) * 0.08)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)

    cropped = img[y0:y1, x0:x1].copy()
    local_mask = fg_mask[y0:y1, x0:x1]

    # Zero out residual white background so only foreground drives training.
    cropped[local_mask == 0] = 0
    return cropped


def main() -> None:
    args = parse_args()

    if not args.palm_dir.exists() or not args.palm_dir.is_dir():
        raise FileNotFoundError(f"Palm folder not found: {args.palm_dir}")

    class_name = args.class_name.lower().replace(" ", "_")
    out_class_dir = args.gesture_data_dir / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_existing:
        for fp in out_class_dir.glob("*.npy"):
            fp.unlink()

    detector = HandDetector()
    try:
        kept = 0
        scanned = 0

        for image_path in _iter_images(args.palm_dir):
            scanned += 1
            if kept >= args.max_images:
                break

            img = cv2.imread(str(image_path))
            if img is None:
                continue

            img = _extract_main_object_ignore_white(img)

            contours = detector.detect(img)
            if not contours:
                continue

            # Use largest candidate contour from each image.
            best = max(
                contours,
                key=lambda c: cv2.contourArea(c.astype(np.int32).reshape(-1, 1, 2)),
            )
            features = detector.extract_features(best)

            np.save(out_class_dir / f"{class_name}_{kept:04d}.npy", features)
            kept += 1

        print("Import complete")
        print(f"  scanned images: {scanned}")
        print(f"  kept samples : {kept}")
        print(f"  output dir   : {out_class_dir}")

        if kept == 0:
            print("No usable palm samples detected. Try clearer palm images or better lighting.")
    finally:
        detector.release()


if __name__ == "__main__":
    main()
