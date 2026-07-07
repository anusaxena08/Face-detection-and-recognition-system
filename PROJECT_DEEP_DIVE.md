# Project Deep Dive

## 1. Purpose and System Overview

This repository is a computer-vision project that combines:

1. Face detection and face recognition.
2. Hand gesture recognition using two paths:
   - Custom-trained contour/KNN pipeline.
   - MediaPipe pre-trained gesture recognizer pipeline.
3. A Gradio web interface that links gestures to actions (voice-open file, page up/down, close file).

At runtime, the project can run in three different operating styles:

1. CLI webcam mode for local OpenCV windows.
2. CLI image mode for static image processing.
3. Browser mode with streaming webcam UI and gesture-triggered control.

## 2. How the Codebase Is Organized

### 2.1 Top-level source/documentation/config files

- .gitignore
- GESTURE_GUIDE.md
- README.md
- requirements.txt
- test_mediapipe.py
- test_mediapipe2.py

### 2.2 Source package files under src

- src/bootstrap_internet_gesture_data.py
- src/gesture_onboard.py
- src/image_detect.py
- src/import_palm_images.py
- src/onboard.py
- src/test_mediapipe.py
- src/test_mediapipe2.py
- src/train.py
- src/train_gesture.py
- src/webcam_detect.py
- src/web_website.py
- src/face_detection/detector.py
- src/face_detection/recognizer.py
- src/face_detection/trainer.py
- src/face_detection/__init__.py
- src/hand_gesture/detector.py
- src/hand_gesture/mp_recognizer.py
- src/hand_gesture/recognizer.py
- src/hand_gesture/trainer.py
- src/hand_gesture/voice_file_opener.py
- src/hand_gesture/__init__.py

### 2.3 Data/model/output folders and what they mean

- gesture_data: training features for custom gestures, primarily .npy vectors.
- internet_samples: downloaded internet images used for gesture bootstrapping.
- known_faces: person folders containing images used for face embedding training.
- models: persisted model artifacts (encodings and gesture models).
- outputs: generated/processed outputs.
- palm: mixed scraped/sample assets used as input material.
- Photos: extra face/image assets.
- test: test assets (for example office document files).

## 3. End-to-End Runtime Pipelines

## 3.1 Face path

1. Capture frame or load image.
2. Detect face boxes with Haar cascade logic in face_detection/detector.py.
3. Optional recognition in face_detection/recognizer.py by DeepFace embeddings plus cosine matching against trained encodings.
4. Draw annotations and show/save output.

## 3.2 Custom gesture path (OpenCV contour + KNN)

1. Detect skin-like hand contours in hand_gesture/detector.py.
2. Extract 10-dimensional shape features from contours.
3. Classify with KNN model from hand_gesture/trainer.py via hand_gesture/recognizer.py.
4. Apply geometric/color gates to reject non-hand false positives.

## 3.3 MediaPipe gesture path

1. Use hand_gesture/mp_recognizer.py with gesture_recognizer.task.
2. Get gestures like Open_Palm, Thumb_Up, Thumb_Down, Closed_Fist.
3. Draw landmarks and labels.
4. In web_website.py, map gestures to actions.

## 4. File-by-File Deep Explanation

## 4.1 .gitignore

Why it exists:
- Prevents committing generated and environment-specific files.

Main logic:
- Ignores Python cache files, virtualenvs, build artifacts, IDE files, and project output folders.

Notable detail:
- It ignores outputs, known_faces, and Photos. That means this repository keeps generated/training image data local by default.

## 4.2 requirements.txt

Why it exists:
- Single dependency manifest for environment setup.

Dependency meaning:
- opencv-python: image/video I/O, cascades, contour operations.
- numpy: numeric arrays and vector math.
- deepface and tf-keras: face embedding generation.
- mediapipe: pre-trained gesture recognition.
- scikit-learn: KNN model for custom gestures.
- gradio: browser UI.
- SpeechRecognition: speech-to-text for voice command.
- pyautogui: page navigation keypress automation.
- psutil: process management for closing opened file apps.

## 4.3 README.md

Why it exists:
- Primary setup and usage document.

What it contains:
- Project features.
- Installation and run commands.
- Face and gesture pipeline descriptions.

Important note:
- Some README gesture architecture text describes MediaPipe landmark feature extraction for training, while current training implementation for custom gestures is contour-based in hand_gesture/detector.py and hand_gesture/trainer.py.

## 4.4 GESTURE_GUIDE.md

Why it exists:
- Dedicated guide for gesture-related workflow.

What it contains:
- Capture, train, and run commands.
- Troubleshooting notes.
- Conceptual architecture.

Important note:
- Like README, parts of this guide describe an older MediaPipe-training-style feature pipeline. Current custom training code uses contour-derived feature vectors.

## 4.5 test_mediapipe.py (root)

Why it exists:
- Quick health-check for MediaPipe hand landmarker availability.

Logic:
1. Computes bundled model path from MediaPipe package.
2. Attempts to instantiate HandLandmarker.
3. Prints success or full exception traceback.

## 4.6 test_mediapipe2.py (root)

Why it exists:
- Quick health-check for MediaPipe GestureRecognizer availability.

Logic:
1. Attempts to create GestureRecognizer.
2. If that fails, falls back to HandLandmarker creation.
3. Prints errors for diagnostics.

## 4.7 src/face_detection/__init__.py

Why it exists:
- Package marker for imports under face_detection.

Logic:
- No executable logic.

## 4.8 src/face_detection/detector.py

Why this module exists:
- Core face detection using OpenCV Haar cascades, tuned for speed and practical filtering.

External modules used and why:
- cv2: cascade classifiers, color conversions, image ops.
- numpy: low-light boost math.
- pathlib: robust cascade file path construction.
- typing: typed tuple/list signatures.

Primary symbols:
- Rect type alias = tuple of (x, y, w, h).
- class FaceDetector.
- draw_faces(frame, faces).

FaceDetector constructor:
- Loads frontal face cascade and eye cascade from cv2 data path.
- Stores scale_factor and min_neighbors.

FaceDetector.detect(frame, strict=False):
1. BGR to grayscale conversion.
2. Dynamic minimum face pixel size based on frame resolution.
3. Optional strict mode doubling neighbor requirement.
4. Primary detection on CLAHE-enhanced grayscale.
5. Optional low-light fallback if nothing detected and frame is dark.
6. Candidate filtering by area and aspect ratio.
7. Human verification by eye cascade and skin tone checks.
8. Duplicate suppression using IoU/containment rules.
9. Bounding box trim for tighter visual fit.

Internal helper logic:
- _enhance_gray: CLAHE and slight blur.
- _boost_low_light: gamma transform in dark scenes.
- _filter_face_candidates: removes unlikely boxes.
- _verify_human_faces: rejects non-human detections (for example animals/objects).
- _has_human_skin: YCrCb skin-mask ratio test.
- _suppress_duplicate_faces and _is_duplicate_face: keep one box per face.
- _trim_face_box: removes detector padding.

draw_faces:
- Draws green rectangles for each face box.

## 4.9 src/face_detection/recognizer.py

Why this module exists:
- Assigns person identity labels to detected faces using trained embeddings.

External modules used and why:
- pickle: load stored encodings database.
- cv2 and numpy: image preprocessing and vector math.
- deepface: embedding generation through DeepFace.represent.

Primary symbols:
- Match type alias = (rect, name, similarity).
- class FaceRecognizer.
- _preprocess function.
- draw_recognition function.

FaceRecognizer constructor:
1. Loads encodings pickle.
2. Reads model name and label-to-embeddings database.
3. Pre-normalizes stored embeddings for fast cosine dot product.
4. Warms up DeepFace model to avoid first-frame lag.

recognize(frame, faces):
- Runs full recognition every skip_frames; remaps old labels on skipped frames for speed.
- For each face box:
  1. Expands crop with padding.
  2. Applies preprocessing.
  3. Human-skin pre-check.
  4. Embedding extraction via DeepFace.
  5. Cosine similarity against all stored embeddings.
  6. Threshold decision: known name or Unknown.

Key helper logic:
- _remap_labels: nearest-center tracking with movement/scale sanity checks.
- _is_human_face_crop: skin-density gate to block non-human crops.
- _match_crop: embedding + similarity + robust retry path.
- _preprocess: YCrCb CLAHE normalization, shared with training pipeline.

draw_recognition:
- Draws bounding boxes and labels with score for known faces.
- Uses red-ish styling for Unknown and green for recognized.

## 4.10 src/face_detection/trainer.py

Why this module exists:
- Builds and saves face embedding database used by FaceRecognizer.

External modules used and why:
- pickle: persists model/encoding mapping.
- cv2 and numpy: image loading and preprocessing.
- deepface: embedding extraction model.

Primary symbols:
- EncodingDB type alias.
- train(...).
- _preprocess(...).
- _boost_low_light(...).

train workflow:
1. Walks known_faces directory.
2. Treats each subfolder as person label.
3. Loads supported image files.
4. Creates augmentation variants per image:
   - original
   - horizontal flip
   - low-light boosted
   - boosted + flip
5. Runs DeepFace embedding extraction on each variant.
6. Saves pickle as {model, encodings}.

_preprocess:
- CLAHE on luma channel for lighting robustness.

_boost_low_light:
- HSV value scaling for dark training images.

Important code observation:
- There is unreachable code after pickle dump where verbose summary and return db appear indented under _boost_low_light. Runtime still writes model successfully, but that summary/return block is not executed as intended.

## 4.11 src/hand_gesture/__init__.py

Why it exists:
- Package marker for hand_gesture imports.

Logic:
- No executable logic.

## 4.12 src/hand_gesture/detector.py

Why this module exists:
- Detects hand-like regions without MediaPipe, then extracts robust contour features for custom gesture classification.

External modules used and why:
- cv2: HSV masking, morphology, contour extraction, geometry metrics, Hu moments.
- numpy: feature vectors and normalization.

Primary symbols:
- HandLandmarks type alias (contour points array).
- class HandDetector with detect, extract_features, release.

detect(frame_bgr) logic:
1. Convert BGR to HSV.
2. Build skin masks using two HSV ranges for broad tones.
3. Morphological open/close to reduce noise.
4. Find contours.
5. Filter contours by area, perimeter, circularity, solidity, aspect ratio.
6. Sort by contour area descending and return top contours (max 8).

extract_features(landmarks) logic:
1. Compute area, perimeter, circularity.
2. Compute bounding-box aspect ratio and extent.
3. Compute solidity from convex hull.
4. Compute first four absolute Hu moments.
5. Build 10-d feature vector.
6. NaN-safe normalization.

release:
- No-op, present for interface symmetry.

## 4.13 src/hand_gesture/trainer.py

Why this module exists:
- Trains a KNN classifier on stored gesture .npy vectors.

External modules used and why:
- numpy: stack and distance statistics.
- scikit-learn KNeighborsClassifier: classifier fit and neighbor distance queries.
- pickle: model persistence.

Primary symbols:
- class GestureTrainer.
- train(...), _build_distance_thresholds(...), _build_one_class_profile(...).

train workflow:
1. Reads gesture_data subfolders as classes.
2. Loads all .npy samples.
3. Builds X matrix and y labels.
4. Trains KNN model.
5. Calculates per-class acceptance distance thresholds using percentile statistics.
6. Optional one-class palm profile if only class is palm.
7. Saves model package with thresholds.

_build_distance_thresholds:
- Uses same-class neighbor mean distance.
- Per-class threshold = percentile 95 plus margin and clamped range.

_build_one_class_profile:
- Centroid plus max-distance boundary for outlier rejection in palm-only setup.

## 4.14 src/hand_gesture/recognizer.py

Why this module exists:
- Applies trained gesture KNN model to detected contours with strict gating.

External modules used and why:
- pickle: load trained model package.
- cv2/numpy: contour geometry, skin gating, confidence calculation.
- hand_gesture.detector: feature extractor and contour detector.

Primary symbols:
- GestureMatch type alias.
- class GestureRecognizer.
- draw_gestures(frame, gestures).

Constructor logic:
1. Load model file.
2. Extract classes, thresholds, optional one-class profile.
3. Initialize HandDetector.
4. Set min confidence.

recognize_contours(frame_bgr, contours, face_rects=None):
1. Optionally skip contours overlapping face boxes.
2. Apply shape gate (_passes_hand_shape_gate).
3. Extract features.
4. If palm-only profile exists, apply one-class gate.
5. Compute nearest-neighbor distances and predicted class.
6. Reject if above class/global distance threshold.
7. Convert distance to confidence.
8. Keep only stable palm detections with confidence >= min_confidence.

_passing gates include:
- area ratio limits.
- aspect ratio, extent, solidity, perimeter constraints.
- skin-gate in HSV plus YCrCb with glare checks.
- convexity defect requirement for open-palm-like contours.

draw_gestures:
- Draws cyan rectangle and uppercase confidence label for accepted gestures.

## 4.15 src/hand_gesture/mp_recognizer.py

Why this module exists:
- Uses MediaPipe pre-trained gesture task for direct gesture recognition without custom training.

External modules used and why:
- mediapipe tasks API: recognizer runtime.
- cv2/numpy: frame conversion and drawing.
- pathlib: model path handling.

Primary symbols:
- DEFAULT_MODEL_PATH.
- GestureResult named tuple.
- class MpGestureRecognizer with recognize and close.
- draw_mp_gestures(...).

Constructor logic:
1. Validates gesture_recognizer.task path.
2. Builds recognizer options for detection/presence/tracking confidence.
3. Instantiates recognizer.

recognize(frame_bgr) logic:
1. Convert BGR to RGB.
2. Create mediapipe Image.
3. Call recognizer.
4. For each hand, pick top gesture category.
5. Skip none/low-confidence outputs.
6. Return GestureResult list containing name, score, handedness, landmarks.

draw_mp_gestures logic:
- Draws full hand skeleton, landmark points, and label near wrist.

## 4.16 src/hand_gesture/voice_file_opener.py

Why this module exists:
- Implements voice-driven file search and file opening used in web UI when open-palm gesture is shown.

External modules used and why:
- speech_recognition: microphone audio transcription.
- pathlib/os: path and OS-specific behavior.
- webbrowser/subprocess/os.startfile: opening files appropriately.

Primary symbols:
- query cleanup constants and maps.
- _extract_extension_hint(words).
- parse_voice_query(text).
- _get_search_dirs().
- _clean_query(spoken_text).
- transcribe_audio(audio_path).
- search_files_by_name(...).
- open_file_by_voice(audio_path).

Core logic:
- Parses spoken text into folder hint, cleaned filename query, and optional extension constraints.
- Searches prioritized directories recursively with depth limit.
- Matches by exact stem, prefix, substring, and spacing variants.
- Chooses best candidates by latest modification time.
- Opens browser-renderable types in browser tab; others in default OS app.

## 4.17 src/webcam_detect.py

Why this module exists:
- Primary local webcam CLI runner combining face detection, optional face recognition, and optional custom gesture recognition.

External modules used and why:
- argparse: CLI options.
- cv2: webcam and rendering loop.
- face_detection and hand_gesture modules: detection and recognition pipelines.

Functions:
- parse_args.
- main.

main logic:
1. Parse arguments.
2. Initialize FaceDetector.
3. Optionally initialize FaceRecognizer.
4. Optionally initialize GestureRecognizer.
5. Open webcam stream.
6. For each frame:
   - mirror frame.
   - detect faces.
   - draw recognition or raw face boxes.
   - detect contours and recognize gestures if enabled.
   - draw overlays.
7. Quit on q or Esc.
8. Cleanup camera/windows/resources.

## 4.18 src/image_detect.py

Why this module exists:
- CLI entry for processing one image file with face detection and optional recognition.

Functions:
- parse_args.
- main.

main logic:
1. Read input image.
2. Detect faces.
3. Optional recognition and labeling.
4. Save annotated output image.
5. Print summary.

## 4.19 src/onboard.py

Why this module exists:
- Collects or imports labeled photos for a person and optionally triggers retraining.

Functions:
- parse_args.
- capture_from_webcam(name, save_dir, n_photos, camera_index).
- import_from_folder(source, save_dir).
- main.

Key logic:
- Mutually exclusive source selection (webcam or folder).
- Sanitizes person name to folder-safe naming.
- Webcam mode captures when spacebar is pressed and saves numbered images.
- Folder mode copies supported image files, avoiding overwrite collisions.
- Optional immediate call to face trainer.

## 4.20 src/train.py

Why this module exists:
- Thin CLI wrapper for face training.

Functions:
- parse_args.
- main.

main logic:
- Parses known-faces input path, output path, and DeepFace model name, then delegates to face_detection.trainer.train.

## 4.21 src/gesture_onboard.py

Why this module exists:
- Interactive capture of gesture samples saved as feature vectors.

Functions:
- parse_args.
- main.

main logic:
1. Parse class name and capture settings.
2. Open webcam and mirror frames.
3. Detect hand contours live.
4. On spacebar, extract features from first contour and save .npy.
5. Continue until sample target met or user quits.

## 4.22 src/train_gesture.py

Why this module exists:
- Thin CLI wrapper to train custom gesture KNN model.

Functions:
- parse_args.
- main.

main logic:
- Creates GestureTrainer and runs trainer.train with CLI-provided paths.

## 4.23 src/bootstrap_internet_gesture_data.py

Why this module exists:
- Semi-automated dataset bootstrap from Bing image download.

Functions:
- parse_args.
- _download_images(query, limit, out_dir).
- _to_safe_name(name).
- _extract_features_from_folder(...).
- main.

main logic:
1. Download positive palm and negative face-like image sets.
2. Detect contours and extract features from downloaded images.
3. Save feature vectors under gesture_data classes.

## 4.24 src/import_palm_images.py

Why this module exists:
- Converts a local folder of palm images into training feature vectors.

Functions:
- parse_args.
- _iter_images(folder).
- _extract_main_object_ignore_white(img).
- main.

main logic:
1. Iterate supported image files.
2. Remove plain white background and crop likely foreground object.
3. Detect contours and choose largest.
4. Extract and save .npy features.

## 4.25 src/web_website.py

Why this module exists:
- Full browser UI that fuses face recognition, MediaPipe gesture recognition, and gesture-triggered actions.

Key globals and purpose:
- ENCODINGS_PATH and GESTURE_MODEL_PATH: model file locations.
- _PIPELINE: cached face detector/recognizer.
- _GESTURE_RECOGNIZER plus init flag: lazy singleton MediaPipe recognizer.
- _LAST_NAV_TIME: rate limiter for navigation keypresses.
- _OPENED_FILE_PATH and _OPENED_PIDS: track opened file app for close action.
- _NAV_INTERVAL: gesture navigation debounce interval.

Core functions and logic:
- _get_pipeline and _get_cached_pipeline: initialize/cache face pipeline.
- _get_gesture_recognizer: lazy-init MediaPipe recognizer once.
- _process_bgr_frame:
  1. detect faces.
  2. recognize faces if recognizer exists.
  3. optional gesture recognition and overlay.
  4. return annotated frame, summary text, active gesture.
- _focus_opened_file_window: best-effort app window focus for keypress landing.
- _nav_action: sends pageup/pagedown with rate limiting.
- _focus_browser_window: bring browser back after close action.
- _close_opened_file: three-step closing strategy (known exe, tracked PIDs, window close fallback).
- open_file_by_voice_wrapper:
  1. normalize Gradio audio input into path.
  2. snapshot current PIDs.
  3. call voice opener.
  4. track newly spawned PIDs.
- run_webcam_stream: streaming handler tying gesture states to actions and UI updates.
- stop_webcam/start_webcam: UI state transitions.
- _decode_uploaded_image, _is_video, _is_image: upload helpers.
- run_uploaded_file: processes uploaded image/video and returns annotated outputs.
- build_app: complete Gradio UI definition with tabs, controls, streaming wiring, and event handlers.
- main: launch app on first available port from 7860-7863.

Gesture action mapping in web UI:
- Open_Palm: show microphone and allow voice file open.
- Thumb_Up: page up.
- Thumb_Down: page down.
- Closed_Fist: close opened document app.

## 4.26 src/test_mediapipe.py

Why this module exists:
- Same diagnostic purpose as root test_mediapipe.py for convenience when running from src context.

Logic:
- Checks bundled hand landmarker path and instantiation.

## 4.27 src/test_mediapipe2.py

Why this module exists:
- Same diagnostic purpose as root test_mediapipe2.py for convenience when running from src context.

Logic:
- Gesture recognizer init attempt with fallback to hand landmarker.

## 5. Data and Artifact Files: What Each Category Is For

This repository contains many non-code files. Their logic role is:

1. Training assets:
   - known_faces/<person>/*.jpg or .jpeg
   - gesture_data/<gesture>/*.npy
2. Model artifacts:
   - models/encodings.pkl
   - models/gesture_model.pkl
   - models/gesture_recognizer.task
3. Generated outputs:
   - outputs/** mirrored processed results.
4. Raw/experimental input pools:
   - palm/** and Photos/** and internet_samples/**

Current extension-level inventory snapshot:

- gesture_data: mostly .npy, with a few image files.
- internet_samples: image files from internet bootstrap.
- known_faces: image files per person.
- models: .pkl and .task model files.
- outputs: processed images/videos/features/models copies.
- palm: mostly downloaded web artifacts plus sample images.
- Photos: local image assets.
- test: contains a .pptx test asset.

## 6. Important Cross-Module Couplings

1. face_detection/trainer.py must run before recognition if models/encodings.pkl is missing.
2. hand_gesture/trainer.py must run before custom gesture recognition in webcam_detect.py.
3. web_website.py uses MediaPipe recognizer task file, not the custom KNN gesture model.
4. voice_file_opener.py is only invoked by web_website.py gesture-triggered voice workflow.

## 7. Why Each Major Module Exists (High-Level)

- face_detection/detector.py: fast candidate face localization.
- face_detection/trainer.py: create face identity embedding database.
- face_detection/recognizer.py: runtime identity matching.
- hand_gesture/detector.py: robust contour feature extraction without full landmark stack.
- hand_gesture/trainer.py: train custom gesture classifier.
- hand_gesture/recognizer.py: runtime contour classification with strict gates.
- hand_gesture/mp_recognizer.py: plug-and-play pre-trained gesture path.
- hand_gesture/voice_file_opener.py: voice query to file-open utility.
- webcam_detect.py/image_detect.py: practical CLI execution frontends.
- web_website.py: unified interactive browser UX and automation actions.

## 8. Practical Run Order for the Full System

1. Install dependencies and activate environment.
2. Collect known face images with onboard.py or manual folder placement.
3. Train face model with train.py.
4. For custom gesture path:
   - capture gesture samples with gesture_onboard.py
   - train with train_gesture.py
   - run webcam_detect.py --gesture
5. For web experience:
   - ensure models/encodings.pkl and models/gesture_recognizer.task exist
   - run web_website.py

## 9. Summary

The codebase has a clear separation between:

1. Data preparation scripts.
2. Model training scripts.
3. Runtime recognition frontends.
4. Web integration and action automation.

Most heavy logic lives in the face_detection and hand_gesture packages, while top-level src scripts are orchestrators around those reusable modules.

## 10. File Inventory Snapshot (Non-.venv)

This snapshot captures the current workspace file state excluding virtual-environment and __pycache__ internals.

- gesture_data: 17 files (.npy, .webp, .jpg)
- internet_samples: 8 files (.jpg, .jpeg, .gif, .png)
- known_faces: 45 files (.jpg, .jpeg)
- models: 3 files (.pkl, .task)
- outputs: 74 files (.jpg, .jpeg, .npy, .webp, .pkl, .task, .gif, .png)
- palm: 219 files (mixed downloaded/web artifacts and sample images)
- Photos: 22 files (.jpeg, .jpg, .webp)
- test: 1 file (.pptx)

Interpretation:

1. Source modules and scripts hold logic.
2. Media/image/audio/model files are runtime assets, training sets, and generated outputs consumed by those modules.
3. The behavior of each asset category is explained in Sections 2, 5, and the per-file module sections above.
