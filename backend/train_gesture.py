"""Gesture model trainer entry point.

Usage:
  python src/train_gesture.py
  python src/train_gesture.py --gesture-data-dir gesture_data --output models/gesture_model.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hand gesture recognition model.")
    parser.add_argument(
        "--gesture-data-dir",
        type=Path,
        default=Path("gesture_data"),
        help="Directory containing gesture training data (default: gesture_data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/gesture_model.pkl"),
        help="Output path for trained model (default: models/gesture_model.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from backend.hand_gesture.trainer import GestureTrainer  # noqa: PLC0415

    trainer = GestureTrainer()
    trainer.train(gesture_data_dir=args.gesture_data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
