from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from face_detection.detector import FaceDetector, draw_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect faces in an image.")
    parser.add_argument("input", type=Path, help="Path to input image")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detected.jpg"),
        help="Path for output image (default: outputs/detected.jpg)",
    )
    parser.add_argument(
        "--recognize",
        action="store_true",
        help="Identify faces using trained encodings (requires models/encodings.pkl).",
    )
    parser.add_argument(
        "--encodings",
        default="models/encodings.pkl",
        metavar="FILE",
        help="Path to encodings file (default: models/encodings.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    detector = FaceDetector()
    faces = detector.detect(image)

    if args.recognize:
        from face_detection.recognizer import FaceRecognizer, draw_recognition
        recognizer = FaceRecognizer(encodings_path=args.encodings)
        matches = recognizer.recognize(image, faces)
        output = draw_recognition(image, matches)
        names = [name for _, name, _ in matches]
        print(f"Detected {len(faces)} face(s): {', '.join(names) if names else 'none'}")
    else:
        output = draw_faces(image, faces)
        print(f"Detected {len(faces)} face(s).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
