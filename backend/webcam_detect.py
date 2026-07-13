from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from backend.face_detection.detector import FaceDetector, draw_faces


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
    parser.add_argument(
        "--gesture",
        action="store_true",
        help="Enable hand gesture recognition (requires models/gesture_model.pkl).",
    )
    parser.add_argument(
        "--gesture-model",
        default="models/gesture_model.pkl",
        metavar="FILE",
        help="Path to gesture model (default: models/gesture_model.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = FaceDetector(scale_factor=1.08, min_neighbors=5)
    recognizer = None
    gesture_recognizer = None

    if args.recognize:
        from backend.face_detection.recognizer import FaceRecognizer, draw_recognition

        recognizer = FaceRecognizer(
            encodings_path=args.encodings,
            threshold=args.threshold,
            skip_frames=args.skip_frames,
        )
        print(f"Face recognition enabled. Loaded encodings from {args.encodings}")

    if args.gesture:
        try:
            from backend.hand_gesture.recognizer import (GestureRecognizer,
                                                 draw_gestures)

            gesture_recognizer = GestureRecognizer(model_path=args.gesture_model)
            print(f"Gesture recognition enabled. Loaded model from {args.gesture_model}")
        except FileNotFoundError as error:
            print(error)
            print("Run: python src/gesture_onboard.py --gesture <name> --samples 20")
            print("     python src/train_gesture.py")

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

            frame = cv2.flip(frame, 1)
            analysis_frame = frame.copy()
            faces = detector.detect(analysis_frame)
            face_rects = [(x, y, w, h) for x, y, w, h in faces]

            if recognizer is not None:
                matches = recognizer.recognize(analysis_frame, faces)
                frame = draw_recognition(frame, matches)
            else:
                frame = draw_faces(frame, faces)

            if gesture_recognizer is not None:
                hand_contours = gesture_recognizer.detector.detect(analysis_frame)
                gestures = gesture_recognizer.recognize_contours(
                    analysis_frame,
                    hand_contours,
                    face_rects=face_rects,
                )
                frame = draw_gestures(frame, gestures)

            cv2.imshow("Face Detection (Webcam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if gesture_recognizer is not None:
            gesture_recognizer.release()


if __name__ == "__main__":
    main()
