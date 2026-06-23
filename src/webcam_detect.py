from __future__ import annotations

import argparse

import cv2

from face_detection.detector import FaceDetector, draw_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect faces from webcam stream.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Recognition similarity threshold 0-1 (default: 0.6). Lower = more matches.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=5,
        metavar="N",
        help="Run recognition every N frames for real-time speed (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = FaceDetector()
    recognizer = None

    if args.recognize:
        from face_detection.recognizer import FaceRecognizer, draw_recognition
        recognizer = FaceRecognizer(
            encodings_path=args.encodings,
            threshold=args.threshold,
            skip_frames=args.skip_frames,
        )
        print(f"Recognition enabled. Loaded encodings from {args.encodings}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    print("Press 'q' or Esc in the camera window to quit (click the window first).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            # Mirror view so movement matches user expectation in webcam preview.
            frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)

            if recognizer is not None:
                matches = recognizer.recognize(frame, faces)
                frame = draw_recognition(frame, matches)
                label = f"Faces: {len(faces)}"
            else:
                frame = draw_faces(frame, faces)
                label = f"Faces: {len(faces)}"

            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Face Detection (Webcam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
