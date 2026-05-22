from pathlib import Path
from typing import List

from config import (
    INPUT_PENDING_FOLDER,
    INPUT_PROCESSING_FOLDER,
    INPUT_COMPLETED_FOLDER,
    INPUT_FAILED_FOLDER,
    OUTPUT_RUNS_FOLDER,
    SUPPORTED_VIDEO_EXTENSIONS,
)


def ensure_directory(folder_path: str) -> Path:
    """
    Create a directory if it does not already exist.

    Returns:
        Path object for the directory.
    """
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_batch_folders() -> None:
    """
    Create all folders required for local batch video processing.

    Folder structure:
        input_videos/pending
        input_videos/processing
        input_videos/completed
        input_videos/failed
        outputs/runs
    """
    ensure_directory(INPUT_PENDING_FOLDER)
    ensure_directory(INPUT_PROCESSING_FOLDER)
    ensure_directory(INPUT_COMPLETED_FOLDER)
    ensure_directory(INPUT_FAILED_FOLDER)
    ensure_directory(OUTPUT_RUNS_FOLDER)


def get_pending_video_files() -> List[Path]:
    """
    Return all supported video files from the pending folder.

    For now, only .mp4 is supported unless config.py is changed.
    """
    pending_folder = ensure_directory(INPUT_PENDING_FOLDER)

    supported_extensions = {
        extension.lower() for extension in SUPPORTED_VIDEO_EXTENSIONS
    }

    video_files = []

    for file_path in pending_folder.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in supported_extensions:
            video_files.append(file_path)

    return sorted(video_files)


def clean_filename_stem(filename: str) -> str:
    """
    Convert a filename into a safe folder-name stem.

    Example:
        My Test Video.mp4 -> My_Test_Video
    """
    path = Path(filename)
    stem = path.stem.strip()

    safe_characters = []

    for character in stem:
        if character.isalnum():
            safe_characters.append(character)
        elif character in ["-", "_"]:
            safe_characters.append(character)
        else:
            safe_characters.append("_")

    cleaned = "".join(safe_characters)

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    cleaned = cleaned.strip("_")

    if not cleaned:
        return "video"

    return cleaned


def build_run_folder_name(video_file: Path, timestamp: str) -> str:
    """
    Build a unique run folder name from the video filename and timestamp.

    Example:
        test_drive_001_2026-05-22_09-35-10
    """
    safe_stem = clean_filename_stem(video_file.name)
    return f"{safe_stem}_{timestamp}"


def get_output_runs_folder() -> Path:
    """
    Return the outputs/runs folder, creating it if needed.
    """
    return ensure_directory(OUTPUT_RUNS_FOLDER)


def get_processing_folder() -> Path:
    """
    Return the processing folder, creating it if needed.
    """
    return ensure_directory(INPUT_PROCESSING_FOLDER)


def get_completed_folder() -> Path:
    """
    Return the completed folder, creating it if needed.
    """
    return ensure_directory(INPUT_COMPLETED_FOLDER)


def get_failed_folder() -> Path:
    """
    Return the failed folder, creating it if needed.
    """
    return ensure_directory(INPUT_FAILED_FOLDER)