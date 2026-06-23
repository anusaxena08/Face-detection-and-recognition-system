"""Face recognition trainer using DeepFace.

Expects a `known_faces/` directory structured as:

    known_faces/
        Alice/
            photo1.jpg
            photo2.jpg
        Bob/
            photo1.jpg

Each sub-folder name becomes that person's label. The trainer generates a
face embedding for every image using DeepFace (default model: Facenet512)
and saves all encodings to a pickle file for use by the recogniser.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from deepface import DeepFace


EncodingDB = Dict[str, List]


def train(
    known_faces_dir: str | Path = "known_faces",
    output_path: str | Path = "models/encodings.pkl",
    model_name: str = "Facenet512",
    verbose: bool = True,
) -> EncodingDB:
    """Build face embeddings from a labeled image directory and save to disk.

    Algorithm:
    1. Walk each sub-folder of `known_faces_dir`; the folder name is the label.
    2. Load each image and compute a face embedding using DeepFace
       (default: Facenet512 — 512-d vector, high accuracy).
    3. Collect all embeddings per label.
    4. Persist the mapping {label: [embeddings]} to a pickle file.

    Parameters
    ----------
    known_faces_dir:
        Root folder containing one sub-folder per person.
    output_path:
        Where to save the resulting encodings pickle file.
    model_name:
        DeepFace model to use. Options: Facenet512, VGG-Face, ArcFace,
        Facenet, DeepFace, DeepID, SFace, GhostFaceNet.
    verbose:
        Print progress while encoding.

    Returns
    -------
    db : dict
        Mapping of label -> list of face embeddings.
    """
    known_faces_dir = Path(known_faces_dir)
    output_path = Path(output_path)

    if not known_faces_dir.exists():
        raise FileNotFoundError(
            f"Known-faces directory not found: {known_faces_dir}\n"
            "Create it and add one sub-folder per person with their photos inside."
        )

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    db: EncodingDB = {}

    for person_dir in sorted(known_faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        label = person_dir.name
        db[label] = []

        images = [p for p in person_dir.iterdir() if p.suffix.lower() in supported]
        if not images:
            if verbose:
                print(f"  [skip] {label}: no supported images found.")
            continue

        if verbose:
            print(f"Encoding {label} ({len(images)} image(s))...")

        for img_path in images:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                if verbose:
                    print(f"  [warn] Cannot read {img_path.name}, skipping.")
                continue

            # Produce two variants: original + horizontal flip for augmentation
            variants = [
                ("orig", _preprocess(img_bgr)),
                ("flip", _preprocess(cv2.flip(img_bgr, 1))),
            ]

            for variant_name, processed in variants:
                try:
                    result = DeepFace.represent(
                        img_path=processed,
                        model_name=model_name,
                        enforce_detection=True,
                    )
                    db[label].append(result[0]["embedding"])
                except ValueError:
                    if verbose and variant_name == "orig":
                        print(f"  [warn] No face found in {img_path.name}, skipping.")
                except Exception as exc:
                    if verbose:
                        print(f"  [warn] {img_path.name} ({variant_name}): {exc}")

        if verbose:
            print(f"  -> {len(db[label])} encoding(s) stored for {label}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model_name, "encodings": db}, f)


# ---------------------------------------------------------------------------
# Shared preprocessing (must match recognizer preprocessing)
# ---------------------------------------------------------------------------

def _preprocess(bgr_img: np.ndarray) -> np.ndarray:
    """Apply CLAHE lighting normalisation so training matches inference.

    Converts to YCrCb, equalises the Y (luma) channel with CLAHE,
    then converts back to BGR.  This makes embeddings robust to lighting
    differences between training photos and real-time webcam frames.
    """
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if verbose:
        total = sum(len(v) for v in db.values())
        print(f"\nDone. {total} encoding(s) across {len(db)} person(s) saved to {output_path}")

    return db


if __name__ == "__main__":
    train()
