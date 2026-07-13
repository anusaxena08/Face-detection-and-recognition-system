"""Voice-based file opener for the gesture-controlled UI.

Workflow:
  1. Gradio's gr.Audio (browser microphone) records the user's speech and
     saves it to a temporary file.
  2. transcribe_audio() converts the audio file to text using Google Speech API.
  3. search_files_by_name() walks common user directories to find a matching
     file by stem name.
  4. open_file_by_voice() ties all steps together and returns a status string
     plus the resolved path (so the caller can track the opened process).
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Filler words stripped before file-name matching
# ---------------------------------------------------------------------------
_FILLER_WORDS = {
    "open", "the", "file", "please", "show", "me", "my", "a", "an",
    "document", "load", "play", "run", "start",
}

# ---------------------------------------------------------------------------
# Folder-name map: spoken word → actual folder name on disk
# ---------------------------------------------------------------------------
_FOLDER_MAP: dict[str, str] = {
    "desktop": "Desktop",
    "documents": "Documents",
    "downloads": "Downloads",
    "pictures": "Pictures",
    "videos": "Videos",
    "music": "Music",
    "my documents": "Documents",
    "my downloads": "Downloads",
    "my desktop": "Desktop",
    "my pictures": "Pictures",
}

# ---------------------------------------------------------------------------
# Extension-word map: spoken file-type word → matching extensions
# Allows "test ppt" → query="test", extensions=[".pptx",".ppt"]
# ---------------------------------------------------------------------------
_EXTENSION_WORDS: dict[str, list[str]] = {
    "ppt":          [".pptx", ".ppt"],
    "pptx":         [".pptx"],
    "powerpoint":   [".pptx", ".ppt"],
    "presentation": [".pptx", ".ppt"],
    "pdf":          [".pdf"],
    "word":         [".docx", ".doc"],
    "doc":          [".docx", ".doc"],
    "docx":         [".docx"],
    "excel":        [".xlsx", ".xls"],
    "spreadsheet":  [".xlsx", ".xls"],
    "xls":          [".xlsx", ".xls"],
    "xlsx":         [".xlsx"],
    "txt":          [".txt"],
    "text":         [".txt"],
    "image":        [".jpg", ".jpeg", ".png", ".webp"],
    "photo":        [".jpg", ".jpeg", ".png"],
    "video":        [".mp4", ".avi", ".mov", ".mkv"],
    "mp4":          [".mp4"],
    "mp3":          [".mp3"],
    "csv":          [".csv"],
    "zip":          [".zip"],
}

# File types the browser can render natively (open in new tab).
# Everything else is opened with the OS default application.
_BROWSER_RENDERABLE = {
    ".pdf", ".html", ".htm", ".txt",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp",
    ".mp4", ".webm", ".mp3", ".wav", ".ogg",
}


def _extract_extension_hint(words: list[str]) -> tuple[list[str], list[str]]:
    """Remove file-type words; return (remaining_words, extension_list).

    Example::

        ["test", "ppt"]          →  (["test"], [".pptx", ".ppt"])
        ["quarterly", "report"]  →  (["quarterly", "report"], [])
    """
    remaining: list[str] = []
    extensions: list[str] = []
    for w in words:
        if w in _EXTENSION_WORDS:
            extensions.extend(_EXTENSION_WORDS[w])
        else:
            remaining.append(w)
    return remaining, extensions


def parse_voice_query(text: str) -> tuple[str | None, str, list[str]]:
    """Split a voice utterance into (folder_hint, filename_query, extensions).

    Extension words (ppt, pdf, doc, …) are stripped from the filename query
    and returned separately as extension filters.

    Examples::

        "documents test ppt"   →  ("Documents", "test",  [".pptx", ".ppt"])
        "downloads report pdf" →  ("Downloads", "report", [".pdf"])
        "test ppt"             →  (None,         "test",  [".pptx", ".ppt"])
        "test"                 →  (None,         "test",  [])

    Returns
    -------
    (folder_name_on_disk | None, cleaned_filename_query, extension_list)
    """
    words = text.lower().strip().split()
    if not words:
        return None, "", []

    def _finish(folder: str | None, rest_words: list[str]):
        clean_words, exts = _extract_extension_hint(rest_words)
        query = " ".join(w for w in clean_words if w not in _FILLER_WORDS).strip()
        # Deduplicate extensions while preserving order
        seen_ext: dict[str, None] = {}
        for e in exts:
            seen_ext[e] = None
        return folder, query, list(seen_ext)

    # Two-word folder prefix (e.g. "my documents test ppt")
    if len(words) >= 3:
        two_word = f"{words[0]} {words[1]}"
        if two_word in _FOLDER_MAP:
            return _finish(_FOLDER_MAP[two_word], words[2:])

    # Single-word folder prefix (e.g. "documents test ppt")
    if words[0] in _FOLDER_MAP and len(words) > 1:
        folder, query, exts = _finish(_FOLDER_MAP[words[0]], words[1:])
        if query:
            return folder, query, exts

    # No folder hint
    return _finish(None, words)

# Directories searched for files (depth limited inside search_files_by_name)
_SEARCH_DIRS: List[Path] = []


def _get_search_dirs() -> List[Path]:
    """Build the list of directories to search, evaluated lazily."""
    if _SEARCH_DIRS:
        return _SEARCH_DIRS

    home = Path.home()
    candidates = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home,
        Path.cwd(),
    ]
    # Also add any drive-root Documents folder on Windows (e.g., D:\Documents)
    if os.name == "nt":
        import string
        for drive in string.ascii_uppercase:
            d = Path(f"{drive}:\\")
            if d.exists():
                for sub in ("Desktop", "Documents", "Downloads"):
                    p = d / sub
                    if p.exists():
                        candidates.append(p)

    seen: set[Path] = set()
    for p in candidates:
        try:
            rp = p.resolve()
            if rp not in seen and rp.is_dir():
                seen.add(rp)
                _SEARCH_DIRS.append(rp)
        except (PermissionError, OSError):
            pass
    return _SEARCH_DIRS


def _clean_query(spoken_text: str) -> str:
    """Strip filler words and return a normalised query string."""
    words = spoken_text.lower().strip().split()
    meaningful = [w for w in words if w not in _FILLER_WORDS]
    return " ".join(meaningful).strip()


def transcribe_audio(audio_path: str) -> tuple[str, str]:
    """Transcribe an audio file to text using Google Speech API.

    Returns
    -------
    (transcribed_text, error_message)
        On success: (text, "").  On failure: ("", error_description).
    """
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        return "", "SpeechRecognition library not installed."

    if not audio_path or not Path(audio_path).exists():
        return "", f"Audio file not found: {audio_path}"

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(audio_path)) as source:
            # Calibrate to ambient noise using the first 0.2 s so that background
            # hum / mic gain shifts do not fool the energy-threshold filter.
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text, ""
    except sr.UnknownValueError:
        return "", (
            "Could not understand speech. "
            "Try speaking more slowly, e.g. 'Documents test ppt'."
        )
    except sr.RequestError as exc:
        return "", f"Google Speech API error: {exc}. Check internet connection."
    except Exception as exc:
        return "", f"Transcription error: {exc}"


def search_files_by_name(
    query: str,
    folder_hint: str | None = None,
    extensions: list[str] | None = None,
    max_depth: int = 3,
) -> List[Path]:
    """Search user directories for files matching *query*.

    When *folder_hint* is given (e.g. ``"Documents"``) that folder is
    searched first across all drives, giving faster and more accurate results.
    When *extensions* is given (e.g. ``[".pptx", ".ppt"]``) only files with
    those suffixes are matched.

    Matching rules (case-insensitive):
      - Exact stem or full-name match
      - Stem starts with query
      - Query is a substring of the stem
      - Space-separated query also tried with ``_`` / ``-`` / no-space variants

    Returns up to 5 paths sorted by modification time (newest first).
    """
    if not query:
        return []

    query_lower = query.lower()
    matches: List[Path] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for entry in directory.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    _walk(entry, depth + 1)
                elif entry.is_file():
                    stem_lower = entry.stem.lower()
                    full_lower = entry.name.lower()
                    variants = {
                        query_lower,
                        query_lower.replace(" ", "_"),
                        query_lower.replace(" ", "-"),
                        query_lower.replace(" ", ""),
                    }
                    if any(
                        stem_lower == v
                        or full_lower == v
                        or stem_lower.startswith(v)
                        or v in stem_lower
                        for v in variants
                        if v
                    ):
                        # Apply extension filter when a file type was spoken
                        if extensions and entry.suffix.lower() not in extensions:
                            continue
                        matches.append(entry)
        except (PermissionError, OSError):
            pass

    # Build ordered search list: hinted folder first, then general dirs.
    ordered: List[Path] = []
    if folder_hint:
        candidate_bases: List[Path] = [Path.home()]
        if os.name == "nt":
            import string
            candidate_bases += [
                Path(f"{d}:\\")
                for d in string.ascii_uppercase
                if Path(f"{d}:\\").exists()
            ]
        for base in candidate_bases:
            p = base / folder_hint
            if p.is_dir():
                ordered.append(p)
    ordered += _get_search_dirs()

    seen_dirs: set[Path] = set()
    for search_dir in ordered:
        try:
            rd = search_dir.resolve()
        except OSError:
            continue
        if rd in seen_dirs:
            continue
        seen_dirs.add(rd)
        _walk(search_dir, 0)
        if len(matches) >= 20:
            break

    # Sort by modification time, newest first; deduplicate by resolved path.
    seen: set[Path] = set()
    unique: List[Path] = []
    for p in sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                unique.append(p)
        except OSError:
            pass
    return unique[:5]


def open_file_by_voice(audio_path: str) -> Tuple[str, Optional[str]]:
    """Transcribe *audio_path*, find the best matching file, and open it.

    Returns
    -------
    (status_message, opened_filepath_str | None)
        *opened_filepath_str* is ``None`` when nothing was opened.
    """
    # Guard: reject recordings that are too short to contain speech
    # (~5 kB ≈ 0.15 s of 16-bit 16 kHz mono audio + 44-byte WAV header).
    try:
        if Path(audio_path).stat().st_size < 5_000:
            return (
                "Recording too short — hold Open Palm, then speak the file name clearly.",
                None,
            )
    except OSError:
        pass

    # 1. Transcribe
    spoken, err = transcribe_audio(audio_path)
    if not spoken:
        return err or "Could not understand audio. Please speak clearly and try again.", None

    # 2. Parse folder hint + filename + extension hint
    folder_hint, query, extensions = parse_voice_query(spoken)
    if not query:
        return (
            f'Heard: "{spoken}" — no file name found. '
            "Just say the file name, e.g. 'test' or 'test ppt'.",
            None,
        )

    # 3. Search (folder-first; extension-filtered when type was spoken)
    results = search_files_by_name(
        query,
        folder_hint=folder_hint,
        extensions=extensions or None,
    )
    if not results:
        folder_info = f" in '{folder_hint}'" if folder_hint else ""
        ext_info    = f" ({'/'.join(extensions)})" if extensions else ""
        return (
            f'Heard: "{spoken}" → "{query}"{ext_info}{folder_info} — file not found.',
            None,
        )

    best = results[0]

    # 4. Smart open: browser tab for renderable types, native app for everything else
    try:
        if best.suffix.lower() in _BROWSER_RENDERABLE:
            webbrowser.open_new_tab(best.resolve().as_uri())
            return f"✅ Opened in new browser tab: {best.name}", str(best)
        else:
            # Office docs (PPTX, DOCX, XLSX), archives, etc.
            if os.name == "nt":
                os.startfile(str(best))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(best)])
            return f"✅ Opened with default app: {best.name}", str(best)
    except Exception as exc:
        return (f'Found "{best.name}" but could not open: {exc}', None)
