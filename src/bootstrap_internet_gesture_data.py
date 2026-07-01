from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as: python src/bootstrap_internet_gesture_data.py
sys.path.insert(0, str(Path(__file__).parent))

from hand_gesture.detector import HandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download internet images and convert them into gesture feature samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("internet_samples"),
        help="Directory where downloaded images are stored.",
    )
    parser.add_argument(
        "--gesture-data-dir",
        type=Path,
        default=Path("gesture_data"),
        help="Directory where .npy gesture features are written.",
    )
    parser.add_argument(
        "--palm-limit",
        type=int,
        default=120,
        help="Max images to download for palm query.",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=120,
        help="Max images to download for not_palm query.",
    )
    return parser.parse_args()


def _download_images(query: str, limit: int, out_dir: Path) -> Path:
    from bing_image_downloader import downloader

    downloader.download(
        query=query,
        limit=limit,
        output_dir=str(out_dir),
        adult_filter_off=True,
        force_replace=False,
        timeout=30,
        verbose=False,
    )
    return out_dir / query


def _to_safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def _extract_features_from_folder(
    detector: HandDetector,
    source_dir: Path,
    class_name: str,
    target_dir: Path,
    max_samples: int,
) -> int:
    target_class_dir = target_dir / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_files.extend(source_dir.glob(ext))

    kept = 0
    for idx, image_path in enumerate(image_files):
        if kept >= max_samples:
            break

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        contours = detector.detect(img)
        if not contours:
            continue

        # Keep the largest contour from this image.
        best = max(
            contours,
            key=lambda c: cv2.contourArea(c.astype(np.int32).reshape(-1, 1, 2)),
        )
        features = detector.extract_features(best)

        out_file = target_class_dir / f"{class_name}_web_{kept:04d}.npy"
        np.save(out_file, features)
        kept += 1

    return kept


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gesture_data_dir.mkdir(parents=True, exist_ok=True)

    # palm: positive class
    palm_query = "human open palm hand"
    # not_palm: hard negatives that can be skin-colored and confuse detector
    negative_query = "human face portrait close up"

    print(f"Downloading palm images: {palm_query}")
    palm_dir = _download_images(palm_query, args.palm_limit, args.output_dir)

    print(f"Downloading negative images: {negative_query}")
    neg_dir = _download_images(negative_query, args.negative_limit, args.output_dir)

    detector = HandDetector()
    try:
        palm_class = _to_safe_name("palm")
        neg_class = _to_safe_name("not_palm")

        palm_kept = _extract_features_from_folder(
            detector=detector,
            source_dir=palm_dir,
            class_name=palm_class,
            target_dir=args.gesture_data_dir,
            max_samples=args.palm_limit,
        )

        neg_kept = _extract_features_from_folder(
            detector=detector,
            source_dir=neg_dir,
            class_name=neg_class,
            target_dir=args.gesture_data_dir,
            max_samples=args.negative_limit,
        )

        print("\nInternet bootstrap complete")
        print(f"  palm samples kept: {palm_kept}")
        print(f"  not_palm samples kept: {neg_kept}")
        print(f"  gesture data dir: {args.gesture_data_dir}")
    finally:
        detector.release()


if __name__ == "__main__":
    main()
