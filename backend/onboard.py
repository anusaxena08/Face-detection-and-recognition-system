"""Onboarding script — register a new person for face recognition.

Two capture modes:
  webcam   Open the camera and capture N photos interactively.
  folder   Import all images from an existing folder.

Usage examples
--------------
  # Capture 5 webcam photos for Alice, then train immediately
  python src/onboard.py --name Alice --webcam --photos 5 --train

  # Import photos from a folder, skip training for now
  python src/onboard.py --name Bob --folder "C:/photos/bob"

  # Import a single image
  python src/onboard.py --name Carol --folder "C:/photos/carol/headshot.jpg"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2

_SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a person for face recognition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Person's name (used as folder label).")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--webcam",
        action="store_true",
        help="Capture photos from the webcam.",
    )
    source.add_argument(
        "--folder",
        type=Path,
        metavar="PATH",
        help="Path to a folder of images (or a single image file) to import.",
    )

    parser.add_argument(
        "--photos",
        type=int,
        default=5,
        metavar="N",
        help="Number of webcam photos to capture (default: 5).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam camera index (default: 0).",
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("known_faces"),
        metavar="DIR",
        help="Root directory for known faces (default: known_faces).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Re-run training immediately after capturing photos.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Webcam capture
# ---------------------------------------------------------------------------

def capture_from_webcam(
    name: str,
    save_dir: Path,
    n_photos: int,
    camera_index: int,
) -> int:
    """Open the webcam, let the user capture N photos, save to save_dir.

    Controls (OpenCV window must be focused):
      SPACE  — capture current frame
      q/Esc  — quit early

    Returns the number of photos actually captured.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    captured = 0
    print(f"\nCapturing {n_photos} photo(s) for '{name}'.")
    print("Focus the camera window, then press SPACE to capture. Press q or Esc to quit.\n")

    try:
        while captured < n_photos:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame. Stopping.")
                break

            # Overlay instructions
            overlay = frame.copy()
            remaining = n_photos - captured
            cv2.putText(
                overlay,
                f"{name}  |  Captured: {captured}/{n_photos}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                "SPACE = capture   q/Esc = quit",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow(f"Onboarding — {name}", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                file_path = save_dir / f"{name}_{captured + 1:03d}.jpg"
                cv2.imwrite(str(file_path), frame)
                captured += 1
                print(f"  Saved {file_path.name}  ({captured}/{n_photos})")

                # Flash feedback
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), 20)
                cv2.imshow(f"Onboarding — {name}", flash)
                cv2.waitKey(150)

            elif key in (ord("q"), 27):
                print("Capture stopped early.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured


# ---------------------------------------------------------------------------
# Folder / file import
# ---------------------------------------------------------------------------

def import_from_folder(source: Path, save_dir: Path) -> int:
    """Copy images from source (file or folder) into save_dir.

    Returns the number of images imported.
    """
    if source.is_file():
        sources = [source] if source.suffix.lower() in _SUPPORTED else []
    elif source.is_dir():
        sources = [p for p in source.iterdir() if p.suffix.lower() in _SUPPORTED]
    else:
        raise FileNotFoundError(f"Source path not found: {source}")

    if not sources:
        raise ValueError(f"No supported images found at: {source}")

    imported = 0
    for img_path in sorted(sources):
        dest = save_dir / img_path.name
        # Avoid overwriting existing files
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = save_dir / f"{stem}_dup{imported}{suffix}"
        shutil.copy2(img_path, dest)
        imported += 1
        print(f"  Imported {img_path.name}")

    return imported


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Sanitize name for use as a directory name
    safe_name = args.name.strip().replace(" ", "_")
    person_dir = args.known_faces_dir / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Onboarding: {args.name} ===")
    print(f"Saving photos to: {person_dir}\n")

    # Capture or import
    if args.webcam:
        count = capture_from_webcam(
            name=safe_name,
            save_dir=person_dir,
            n_photos=args.photos,
            camera_index=args.camera,
        )
    else:
        count = import_from_folder(source=args.folder, save_dir=person_dir)

    if count == 0:
        print("\nNo photos captured. Onboarding aborted.")
        sys.exit(1)

    print(f"\n{count} photo(s) saved for '{args.name}'.")

    # Optional: re-train immediately
    if args.train:
        print("\nStarting training...\n")
        from backend.face_detection.trainer import train  # noqa: PLC0415
        train(known_faces_dir=args.known_faces_dir)
    else:
        print(
            "\nTo train the model with the new person run:\n"
            "  python src/train.py\n"
        )


if __name__ == "__main__":
    main()
