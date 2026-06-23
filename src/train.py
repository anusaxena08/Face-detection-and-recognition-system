"""Standalone training entry point.

Reads labeled images from known_faces/, generates DeepFace embeddings,
and saves the encoding database to models/encodings.pkl.

Usage
-----
  python src/train.py
  python src/train.py --known-faces-dir known_faces --output models/encodings.pkl
  python src/train.py --model ArcFace
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face recognition encodings.")
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("known_faces"),
        metavar="DIR",
        help="Folder of labeled face images (default: known_faces).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/encodings.pkl"),
        help="Output path for encodings file (default: models/encodings.pkl).",
    )
    parser.add_argument(
        "--model",
        default="Facenet512",
        metavar="MODEL",
        help=(
            "DeepFace model to use for embeddings. "
            "Options: Facenet512, VGG-Face, ArcFace, Facenet, SFace, GhostFaceNet "
            "(default: Facenet512)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from face_detection.trainer import train  # noqa: PLC0415

    train(
        known_faces_dir=args.known_faces_dir,
        output_path=args.output,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
