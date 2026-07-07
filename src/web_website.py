from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv
load_dotenv()
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

try:
    import pyautogui  # type: ignore
    pyautogui.FAILSAFE = False
except ImportError:
    pyautogui = None  # type: ignore

# Allow direct execution: python src/web_website.py
sys.path.insert(0, str(Path(__file__).parent))

from face_detection.detector import FaceDetector, draw_faces
from face_detection.recognizer import FaceRecognizer, draw_recognition
from hand_gesture.mp_recognizer import MpGestureRecognizer, draw_mp_gestures
from hand_gesture.voice_file_opener import open_file_by_voice


ENCODINGS_PATH = "models/encodings.pkl"
GESTURE_MODEL_PATH = "models/gesture_recognizer.task"

_PIPELINE: Tuple[FaceDetector, FaceRecognizer | None] | None = None
_GESTURE_RECOGNIZER: MpGestureRecognizer | None = None
_GESTURE_INIT_ATTEMPTED: bool = False

# --- Gesture action state ---------------------------------------------------
_LAST_NAV_TIME: float = 0.0          # timestamp of last page nav keypress
_OPENED_FILE_PATH: Optional[str] = None   # path of last file opened by voice
_OPENED_PIDS: set[int] = set()       # PIDs spawned when that file was opened
_NAV_INTERVAL: float = 0.8           # seconds between page-nav keypresses


def _get_pipeline() -> Tuple[FaceDetector, FaceRecognizer | None]:
    detector = FaceDetector(scale_factor=1.08, min_neighbors=5)
    recognizer = None
    try:
        recognizer = FaceRecognizer(encodings_path=ENCODINGS_PATH, threshold=0.6, skip_frames=1)
    except Exception:
        recognizer = None
    return detector, recognizer


def _get_cached_pipeline() -> Tuple[FaceDetector, FaceRecognizer | None]:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _get_pipeline()
    return _PIPELINE


def _get_gesture_recognizer() -> MpGestureRecognizer | None:
    global _GESTURE_RECOGNIZER, _GESTURE_INIT_ATTEMPTED
    if _GESTURE_INIT_ATTEMPTED:
        return _GESTURE_RECOGNIZER
    _GESTURE_INIT_ATTEMPTED = True
    try:
        _GESTURE_RECOGNIZER = MpGestureRecognizer(model_path=GESTURE_MODEL_PATH)
    except Exception:
        _GESTURE_RECOGNIZER = None
    return _GESTURE_RECOGNIZER


def _process_bgr_frame(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    recognizer: FaceRecognizer | None,
    detect_gestures: bool = False,
    strict_mode: bool = False,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """Process one BGR frame.

    Returns
    -------
    (annotated_bgr, summary, active_gesture_name | None)
        active_gesture_name is the top MediaPipe gesture category (e.g.
        "Open_Palm", "Thumb_Up", "Closed_Fist") or None when no hand is seen.
    """
    faces = detector.detect(frame_bgr, strict=strict_mode)
    face_rects = list(faces)

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


def _focus_opened_file_window() -> None:
    """Bring the opened file's application window to the foreground (best-effort).

    Uses pygetwindow (bundled with pyautogui on Windows) to find and activate
    the correct window so keystrokes land in PowerPoint/Word/etc., not the browser.
    """
    if not _OPENED_FILE_PATH:
        return
    try:
        stem = Path(_OPENED_FILE_PATH).stem
        suffix = Path(_OPENED_FILE_PATH).suffix.lower()

        if suffix in (".pptx", ".ppt"):
            keywords = [stem, "PowerPoint"]
        elif suffix in (".docx", ".doc"):
            keywords = [stem, "Word"]
        elif suffix in (".xlsx", ".xls"):
            keywords = [stem, "Excel"]
        elif suffix == ".pdf":
            keywords = [stem, "Adobe", "Acrobat"]
        else:
            keywords = [stem]

        import pygetwindow as gw  # bundled with pyautogui on Windows
        for kw in keywords:
            if not kw:
                continue
            wins = gw.getWindowsWithTitle(kw)
            if wins:
                try:
                    wins[0].activate()
                except Exception:
                    pass
                return
    except Exception:
        pass  # best-effort; keypress still fires, just may miss focus


def _nav_action(gesture_name: str) -> str:
    """Send a Page Up / Down keypress (rate-limited to _NAV_INTERVAL).

    Auto-focuses the opened document window first so the keypress lands in
    PowerPoint/PDF viewer rather than the browser, without any manual clicking.
    """
    global _LAST_NAV_TIME
    if pyautogui is None:
        return "pyautogui not installed — navigation unavailable."
    now = time.time()
    if now - _LAST_NAV_TIME < _NAV_INTERVAL:
        return ""  # still within rate-limit window
    _LAST_NAV_TIME = now

    # Auto-focus the document window before sending the keystroke.
    _focus_opened_file_window()

    key = "pageup" if gesture_name.lower() in ("thumb_up", "thumbup") else "pagedown"
    try:
        pyautogui.press(key)
        direction = "↑ Up" if key == "pageup" else "↓ Down"
        return f"Slide {direction}"
    except Exception as exc:
        return f"Navigation error: {exc}"


def _focus_browser_window() -> None:
    """Bring the Gradio browser tab back into focus after closing a document."""
    try:
        import pygetwindow as gw
        for term in ("localhost:786", "Face Recognition", "Chrome", "Edge", "Firefox", "Mozilla"):
            wins = [w for w in gw.getWindowsWithTitle(term) if w.title]
            if wins:
                try:
                    wins[0].activate()
                    return
                except Exception:
                    pass
    except Exception:
        pass  # best-effort


# Maps file extension to the Windows process executable that handles it.
_PROC_NAME_MAP: dict[str, str] = {
    ".pptx": "POWERPNT.EXE", ".ppt": "POWERPNT.EXE",
    ".docx": "WINWORD.EXE",  ".doc": "WINWORD.EXE",
    ".xlsx": "EXCEL.EXE",    ".xls": "EXCEL.EXE",
    ".pdf":  "AcroRd32.exe",
}


def _close_opened_file() -> str:
    """Force-close the application opened by voice command, then refocus browser."""
    global _OPENED_FILE_PATH, _OPENED_PIDS

    killed: list[str] = []

    # Strategy 1 (most reliable on Windows): kill by known exe name.
    # Office apps memory-map files so open_files() returns nothing; killing
    # POWERPNT.EXE / WINWORD.EXE works regardless of how the app was launched.
    if _OPENED_FILE_PATH and psutil is not None:
        suffix = Path(_OPENED_FILE_PATH).suffix.lower()
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

    # Strategy 2: kill PIDs captured right after os.startfile() was called.
    if not killed and _OPENED_PIDS and psutil is not None:
        for pid in list(_OPENED_PIDS):
            try:
                p = psutil.Process(pid)
                killed.append(p.name())
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # Strategy 3: send WM_CLOSE to the window via pygetwindow (last resort).
    if not killed and _OPENED_FILE_PATH:
        try:
            import pygetwindow as gw
            stem = Path(_OPENED_FILE_PATH).stem
            for win in gw.getWindowsWithTitle(stem):
                try:
                    win.close()
                    killed.append(win.title[:40])
                except Exception:
                    pass
        except Exception:
            pass

    _OPENED_FILE_PATH = None
    _OPENED_PIDS.clear()

    if killed:
        _focus_browser_window()
        return f"Closed: {', '.join(killed)}"
    return "No tracked file to close (open a file first with Open Palm)."


def open_file_by_voice_wrapper(audio) -> str:
    """Gradio handler: transcribe mic audio, find file, open it, track PID."""
    global _OPENED_FILE_PATH, _OPENED_PIDS

    if audio is None:
        return "No audio received. Try again."

    # audio may be a file path string or a tuple (sr, data) depending on Gradio version.
    audio_path: Optional[str] = None
    if isinstance(audio, str):
        audio_path = audio
    elif isinstance(audio, (tuple, list)) and len(audio) == 2:
        # Gradio type='numpy' returns (sample_rate, numpy_array).
        import wave
        sr_val, data = audio
        data = np.asarray(data)
        if data.ndim > 1:          # stereo → take first channel
            data = data[:, 0]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr_val))
            wf.writeframes(data.astype(np.int16).tobytes())
        audio_path = tmp.name
    else:
        return "Unsupported audio format from Gradio."

    # Snapshot PIDs before opening.
    before_pids: set[int] = set()
    if psutil is not None:
        try:
            before_pids = set(psutil.pids())
        except Exception:
            pass

    status, opened_path = open_file_by_voice(audio_path)

    if opened_path:
        _OPENED_FILE_PATH = opened_path
        # Track new PIDs in a background thread (give apps ~2 s to start).
        def _track_pids(snap: set[int]) -> None:
            global _OPENED_PIDS
            time.sleep(2.0)
            if psutil is not None:
                try:
                    _OPENED_PIDS = set(psutil.pids()) - snap
                except Exception:
                    pass
        threading.Thread(target=_track_pids, args=(before_pids,), daemon=True).start()

    return status


def run_webcam_stream(webcam_image: np.ndarray):
    """Live webcam stream handler.

    Returns
    -------
    (annotated_rgb, summary, gesture_name, voice_input_update, nav_status)
    """
    if webcam_image is None:
        return None, "Waiting for webcam...", None, gr.update(visible=False), ""

    detector, recognizer = _get_cached_pipeline()
    frame_bgr = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)
    annotated_bgr, summary, active_gesture = _process_bgr_frame(
        frame_bgr, detector, recognizer, detect_gestures=True
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    nav_status = ""
    gesture_lower = (active_gesture or "").lower()

    if gesture_lower in ("thumb_up", "thumbup"):
        nav_status = _nav_action(active_gesture)
    elif gesture_lower in ("thumb_down", "thumbdown"):
        nav_status = _nav_action(active_gesture)
    elif gesture_lower in ("closed_fist", "closedfist"):
        nav_status = _close_opened_file()

    is_palm = gesture_lower in ("open_palm", "openpalm")
    # Only make mic visible on palm — never force-hide it via stream.
    # Hiding it every frame would kill an in-progress recording.
    voice_update = gr.update(visible=True) if is_palm else gr.update()

    return annotated_rgb, summary, active_gesture, voice_update, nav_status


def stop_webcam():
    """Stop the webcam, clear outputs, and reset button states."""
    return (
        gr.update(visible=False),                                # hide webcam_input
        None,                                                     # clear webcam_live
        "Webcam stopped. Click 'Start Webcam' to begin again.",  # webcam_text
        None,                                                     # gesture_state
        gr.update(visible=True,  interactive=True),              # show start_btn
        gr.update(visible=False, interactive=False),             # hide stop_btn
    )


def start_webcam():
    """Reveal the webcam component and swap Start → Stop button."""
    return (
        gr.update(visible=True),                                 # show webcam_input
        gr.update(visible=False, interactive=False),             # hide start_btn
        gr.update(visible=True,  interactive=True),              # show stop_btn
    )





def _decode_uploaded_image(file_path: str) -> np.ndarray:
    data = np.fromfile(file_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image.")
    return img_bgr


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None, None, "Please upload an image or video file."

    file_path = Path(uploaded_file.name) if hasattr(uploaded_file, 'name') else Path(str(uploaded_file))
    detector, recognizer = _get_pipeline()

    if _is_image(file_path):
        img_bgr = _decode_uploaded_image(str(file_path))
        annotated_bgr, summary, _ = _process_bgr_frame(img_bgr, detector, recognizer, strict_mode=True)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, None, summary

    if _is_video(file_path):
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
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
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_count = 0
        kept_faces = 0

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                annotated_bgr, summary, _ = _process_bgr_frame(frame_bgr, detector, recognizer, strict_mode=True)
                writer.write(annotated_bgr)
                frame_count += 1

                # Quick aggregate count for summary.
                faces = detector.detect(frame_bgr)
                kept_faces += len(faces)
        finally:
            cap.release()
            writer.release()

        summary = (
            f"Processed video frames: {frame_count}. "
            f"Total face detections across frames: {kept_faces}."
        )
        return None, out_path, summary

    return None, None, "Unsupported file type. Upload image or video."


def build_app():
    with gr.Blocks(title="Face Recognition Website") as demo:
        gr.Markdown("# Face Recognition Website")
        gr.Markdown(
            "Webcam tab: live face recognition + hand gesture controls.  "
            "Make sure the target document window is in the foreground for "
            "navigation and close gestures."
        )

        with gr.Tabs():
            with gr.TabItem("Webcam"):
                gesture_state = gr.State(value=None)

                with gr.Row():
                    start_btn = gr.Button("▶ Start Webcam", variant="primary", scale=1)
                    stop_btn  = gr.Button("⏹ Stop Webcam",  variant="stop",    scale=1,
                                          visible=False, interactive=False)

                # Hidden until Start is clicked — camera LED stays off on page load.
                webcam_input = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    streaming=True,
                    label="Webcam Feed",
                    height=280,
                    visible=False,
                )
                webcam_live = gr.Image(
                    type="numpy",
                    label="Live Feed (with recognition)",
                    interactive=False,
                )
                webcam_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Click 'Start Webcam' to begin.",
                )

                gr.Markdown("---")
                gr.Markdown(
                    "### Gesture Controls\n"
                    "| Gesture | Action |\n"
                    "|---|---|\n"
                    "| ✋ Open Palm | Mic appears — **say folder + file name** to open in new tab |\n"
                    "| 👍 Thumb Up | Page Up while held (0.8 s / page) |\n"
                    "| 👎 Thumb Down | Page Down while held (0.8 s / page) |\n"
                    "| ✊ Closed Fist | Close the currently open file |\n"
                )
                gr.Markdown(
                    "> 🎤 **Voice format — say folder name FIRST, then file name:**  \n"
                    "> &nbsp;&nbsp;• *\"Documents quarterly report\"*  \n"
                    "> &nbsp;&nbsp;• *\"Downloads presentation slides\"*  \n"
                    "> &nbsp;&nbsp;• *\"Desktop budget 2025\"*  \n"
                    "> Recognised folders: **Desktop · Documents · Downloads · Pictures · Videos**"
                )

                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎤 Say the file name (appears when Open Palm is detected)",
                    visible=False,
                )
                voice_status = gr.Textbox(
                    label="Voice File Open",
                    interactive=False,
                    placeholder="Speak a file name while Open Palm is active…",
                )
                nav_status = gr.Textbox(
                    label="Navigation / Close Status",
                    interactive=False,
                    placeholder="Thumb Up / Down / Closed Fist actions appear here…",
                )

                # Start: reveal webcam input and swap button visibility.
                start_btn.click(
                    fn=start_webcam,
                    inputs=None,
                    outputs=[webcam_input, start_btn, stop_btn],
                )

                # Stop: kill camera tracks via JS, hide webcam, restore Start button.
                stop_btn.click(
                    fn=stop_webcam,
                    inputs=None,
                    outputs=[webcam_input, webcam_live, webcam_text, gesture_state,
                              start_btn, stop_btn],
                    js="""
                    () => {
                        document.querySelectorAll('video').forEach((v) => {
                            if (v.srcObject) {
                                v.srcObject.getTracks().forEach(t => t.stop());
                                v.srcObject = null;
                            }
                        });
                        return [];
                    }
                    """,
                )

                webcam_input.stream(
                    fn=run_webcam_stream,
                    inputs=[webcam_input],
                    outputs=[webcam_live, webcam_text, gesture_state, voice_input, nav_status],
                    stream_every=0.1,
                    show_progress="hidden",
                )

                voice_input.stop_recording(
                    fn=open_file_by_voice_wrapper,
                    inputs=[voice_input],
                    outputs=[voice_status],
                )

            with gr.TabItem("Upload Image/Video"):
                file_input = gr.File(label="Upload Image or Video")
                file_button = gr.Button("Run Recognition")
                file_image_output = gr.Image(type="numpy", label="Image Result")
                file_video_output = gr.Video(label="Processed Video")
                file_text = gr.Textbox(label="Summary")

                file_button.click(
                    fn=run_uploaded_file,
                    inputs=[file_input],
                    outputs=[file_image_output, file_video_output, file_text],
                )

    return demo


def main() -> None:
    app = build_app()

    port = int(
        os.environ.get("PORT")
        or os.environ.get("WEBSITES_PORT")
        or os.environ.get("SERVER_PORT")
        or os.environ.get("GRADIO_SERVER_PORT")
        or 8000
    )

    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")

app.launch(
    server_name=host,
    server_port=port,
    inbrowser=False,
    share=False,
)


if __name__ == "__main__":
    main()
