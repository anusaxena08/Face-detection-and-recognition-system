"""Streamlit frontend for Face Recognition & Hand Gesture Recognition.

Replaces the old Gradio-based web_website.py with equivalent functionality:
- Live webcam face detection & recognition
- Hand gesture recognition with PPT controls
- Image/video upload with face detection
- Voice-based file opening (Open Palm gesture)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on the path for backend imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.face_detection.detector import FaceDetector, draw_faces
from backend.face_detection.recognizer import FaceRecognizer, draw_recognition
from backend.hand_gesture.mp_recognizer import MpGestureRecognizer, draw_mp_gestures
from backend.hand_gesture.voice_file_opener import open_file_by_voice

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

try:
    import pyautogui  # type: ignore
    pyautogui.FAILSAFE = False
except ImportError:
    pyautogui = None  # type: ignore


# --- Constants ----------------------------------------------------------------
ENCODINGS_PATH = "models/encodings.pkl"
GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
NAV_INTERVAL = 0.8  # seconds between page-nav keypresses


# --- Session State Initialization ---------------------------------------------
def _init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "webcam_running": False,
        "last_nav_time": 0.0,
        "opened_file_path": None,
        "opened_pids": set(),
        "last_voice_status": "",
        "pipeline": None,
        "gesture_recognizer": None,
        "gesture_init_attempted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# --- Pipeline Initialization --------------------------------------------------
def _get_pipeline() -> Tuple[FaceDetector, Optional[FaceRecognizer]]:
    """Create face detection and recognition pipeline."""
    if st.session_state.pipeline is not None:
        return st.session_state.pipeline

    detector = FaceDetector(scale_factor=1.08, min_neighbors=5)
    recognizer = None
    try:
        recognizer = FaceRecognizer(encodings_path=ENCODINGS_PATH, threshold=0.6, skip_frames=1)
    except Exception:
        recognizer = None

    st.session_state.pipeline = (detector, recognizer)
    return detector, recognizer


def _get_gesture_recognizer() -> Optional[MpGestureRecognizer]:
    """Get or initialize the gesture recognizer."""
    if st.session_state.gesture_init_attempted:
        return st.session_state.gesture_recognizer

    st.session_state.gesture_init_attempted = True
    try:
        st.session_state.gesture_recognizer = MpGestureRecognizer(model_path=GESTURE_MODEL_PATH)
    except Exception:
        st.session_state.gesture_recognizer = None

    return st.session_state.gesture_recognizer


# --- Frame Processing ---------------------------------------------------------
def _process_bgr_frame(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    recognizer: Optional[FaceRecognizer],
    detect_gestures: bool = False,
    strict_mode: bool = False,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """Process one BGR frame and return annotated frame, summary, and gesture name."""
    faces = detector.detect(frame_bgr, strict=strict_mode)

    if recognizer is None:
        annotated = draw_faces(frame_bgr.copy(), faces)
        face_summary = f"{len(faces)} face(s) detected."
    else:
        matches = recognizer.recognize(frame_bgr, faces)
        if faces and not matches:
            matches = recognizer.recognize(frame_bgr, faces)
        annotated = draw_recognition(frame_bgr.copy(), matches)
        names = [name for _, name, _ in matches]
        face_summary = f"Faces: {', '.join(names)}" if names else "No face recognised"

    active_gesture: Optional[str] = None
    if detect_gestures:
        gr_inst = _get_gesture_recognizer()
        if gr_inst is not None:
            try:
                gesture_results = gr_inst.recognize(frame_bgr)
                if gesture_results:
                    active_gesture = gesture_results[0].gesture_name
                    annotated = draw_mp_gestures(annotated, gesture_results)
            except Exception:
                pass

    if active_gesture:
        summary = face_summary + f" | Gesture: {active_gesture}"
    else:
        summary = face_summary

    return annotated, summary, active_gesture


# --- Gesture Actions ----------------------------------------------------------
def _nav_action(gesture_name: str) -> str:
    """Send a Page Up / Down keypress (rate-limited)."""
    if pyautogui is None:
        return "pyautogui not installed — navigation unavailable."

    now = time.time()
    if now - st.session_state.last_nav_time < NAV_INTERVAL:
        return ""
    st.session_state.last_nav_time = now

    key = "pageup" if gesture_name.lower() in ("thumb_up", "thumbup") else "pagedown"
    try:
        pyautogui.press(key)
        direction = "↑ Up" if key == "pageup" else "↓ Down"
        return f"Slide {direction}"
    except Exception as exc:
        return f"Navigation error: {exc}"


def _close_opened_file() -> str:
    """Force-close the application opened by voice command."""
    _PROC_NAME_MAP = {
        ".pptx": "POWERPNT.EXE", ".ppt": "POWERPNT.EXE",
        ".docx": "WINWORD.EXE", ".doc": "WINWORD.EXE",
        ".xlsx": "EXCEL.EXE", ".xls": "EXCEL.EXE",
        ".pdf": "AcroRd32.exe",
    }

    killed: list = []
    opened_path = st.session_state.opened_file_path
    opened_pids = st.session_state.opened_pids

    if opened_path and psutil is not None:
        suffix = Path(opened_path).suffix.lower()
        target_exe = _PROC_NAME_MAP.get(suffix)
        if target_exe:
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if proc.name().upper() == target_exe.upper():
                        killed.append(proc.name())
                        proc.kill()
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    if not killed and opened_pids and psutil is not None:
        for pid in list(opened_pids):
            try:
                p = psutil.Process(pid)
                killed.append(p.name())
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    st.session_state.opened_file_path = None
    st.session_state.opened_pids = set()

    if killed:
        return f"Closed: {', '.join(killed)}"
    return "No tracked file to close (open a file first with Open Palm)."


def _open_file_from_audio_input(audio_value) -> str:
    """Save recorded audio to a temp file, then open the requested document."""
    if audio_value is None:
        return "No audio received. Record a command and try again."

    audio_name = getattr(audio_value, "name", "voice-command.wav")
    suffix = Path(audio_name).suffix or ".wav"

    if hasattr(audio_value, "getvalue"):
        audio_bytes = audio_value.getvalue()
    else:
        audio_bytes = bytes(audio_value)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(audio_bytes)
        tmp.close()

        before_pids: set[int] = set()
        if psutil is not None:
            try:
                before_pids = set(psutil.pids())
            except Exception:
                pass

        status, opened_path = open_file_by_voice(tmp.name)
        if opened_path:
            st.session_state.opened_file_path = opened_path
            if psutil is not None:
                time.sleep(2.0)
                try:
                    st.session_state.opened_pids = set(psutil.pids()) - before_pids
                except Exception:
                    st.session_state.opened_pids = set()

        return status
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# --- File Upload Processing ---------------------------------------------------
def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _process_uploaded_file(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[str], str]:
    """Process an uploaded image or video file."""
    if uploaded_file is None:
        return None, None, "Please upload an image or video file."

    # Save uploaded file to temp location
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    file_path = Path(tmp.name)

    detector, recognizer = _get_pipeline()

    if _is_image(file_path):
        data = np.fromfile(str(file_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, None, "Could not decode uploaded image."

        annotated_bgr, summary, _ = _process_bgr_frame(
            img_bgr, detector, recognizer, strict_mode=True
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        os.unlink(tmp.name)
        return annotated_rgb, None, summary

    if _is_video(file_path):
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            os.unlink(tmp.name)
            return None, None, "Could not open uploaded video."

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_out.close()
        out_path = tmp_out.name

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        frame_count = 0
        kept_faces = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                annotated_bgr, summary, _ = _process_bgr_frame(
                    frame_bgr, detector, recognizer, strict_mode=True
                )
                writer.write(annotated_bgr)
                frame_count += 1
                faces = detector.detect(frame_bgr)
                kept_faces += len(faces)
        finally:
            cap.release()
            writer.release()

        os.unlink(tmp.name)
        summary = (
            f"Processed video frames: {frame_count}. "
            f"Total face detections across frames: {kept_faces}."
        )
        return None, out_path, summary

    os.unlink(tmp.name)
    return None, None, "Unsupported file type. Upload image or video."


# --- Page Layout --------------------------------------------------------------
def main():
    _init_session_state()

    st.set_page_config(page_title="Face Recognition App", layout="wide")
    st.title("🎯 Face Recognition & Gesture Control")
    st.markdown(
        "Live face recognition + hand gesture controls. "
        "Upload images/videos for offline processing."
    )

    tab_webcam, tab_upload = st.tabs(["📷 Webcam", "📁 Upload Image/Video"])

    # --- Webcam Tab ---
    with tab_webcam:
        st.markdown("### Live Webcam Feed")
        st.markdown(
            "Start the webcam for real-time face detection, recognition, "
            "and hand gesture recognition."
        )

        col1, col2 = st.columns(2)
        with col1:
            start = st.button("▶ Start Webcam", type="primary")
        with col2:
            stop = st.button("⏹ Stop Webcam")

        if start:
            st.session_state.webcam_running = True
        if stop:
            st.session_state.webcam_running = False

        # Gesture control reference
        with st.expander("🤚 Gesture Controls", expanded=False):
            st.markdown("""
| Gesture | Action |
|---|---|
| ✋ Open Palm | Use the voice recorder below to open a file |
| 👍 Thumb Up | Page Up (slide navigation) |
| 👎 Thumb Down | Page Down (slide navigation) |
| ✊ Closed Fist | Close the currently open file |
            """)
            st.info(
                "🎤 **Voice format** — say folder name FIRST, then file name:\n"
                "- *\"Documents quarterly report\"*\n"
                "- *\"Downloads presentation slides\"*\n\n"
                "Recognised folders: **Desktop · Documents · Downloads · Pictures · Videos**"
            )

        st.markdown("### 🎤 Voice File Opener")
        st.caption("Show **Open Palm**, then record a command like `Documents quarterly report ppt`.")

        recorded_audio = None
        fallback_audio = None
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record folder name first, then file name")
        else:
            st.warning("This Streamlit version does not support in-browser recording. Upload a short audio clip instead.")
            fallback_audio = st.file_uploader(
                "Upload recorded audio",
                type=["wav", "mp3", "m4a", "ogg"],
                key="voice_audio_upload",
            )

        if st.button("Open file by voice", key="open_file_by_voice"):
            audio_source = recorded_audio if recorded_audio is not None else fallback_audio
            if audio_source is None:
                st.session_state.last_voice_status = "Record audio first, then click 'Open file by voice'."
            else:
                with st.spinner("Processing voice command..."):
                    st.session_state.last_voice_status = _open_file_from_audio_input(audio_source)

        if st.session_state.last_voice_status:
            if st.session_state.last_voice_status.startswith("✅"):
                st.success(st.session_state.last_voice_status)
            else:
                st.info(st.session_state.last_voice_status)

        # Webcam display area
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        nav_placeholder = st.empty()

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Check camera permissions.")
                st.session_state.webcam_running = False
            else:
                detector, recognizer = _get_pipeline()
                status_placeholder.info("Webcam active. Press 'Stop Webcam' to end.")

                while st.session_state.webcam_running:
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Failed to read frame from camera.")
                        break

                    frame = cv2.flip(frame, 1)
                    annotated_bgr, summary, active_gesture = _process_bgr_frame(
                        frame, detector, recognizer, detect_gestures=True
                    )
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                    status_placeholder.text(summary)

                    # Handle gesture actions
                    gesture_lower = (active_gesture or "").lower()
                    nav_status = ""
                    if gesture_lower in ("thumb_up", "thumbup"):
                        nav_status = _nav_action(active_gesture)
                    elif gesture_lower in ("thumb_down", "thumbdown"):
                        nav_status = _nav_action(active_gesture)
                    elif gesture_lower in ("closed_fist", "closedfist"):
                        nav_status = _close_opened_file()

                    if nav_status:
                        nav_placeholder.success(nav_status)

                cap.release()
                frame_placeholder.empty()
                status_placeholder.info("Webcam stopped.")

    # --- Upload Tab ---
    with tab_upload:
        st.markdown("### Upload Image or Video")
        st.markdown("Upload an image or video file for face detection and recognition.")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv", "webm"],
        )

        if st.button("🔍 Run Recognition", type="primary"):
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    image_result, video_path, summary = _process_uploaded_file(uploaded_file)

                st.markdown(f"**Result:** {summary}")

                if image_result is not None:
                    st.image(image_result, caption="Detection Result", use_container_width=True)

                if video_path is not None:
                    st.video(video_path)
                    # Clean up temp file after display
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass
            else:
                st.warning("Please upload a file first.")


if __name__ == "__main__":
    main()
