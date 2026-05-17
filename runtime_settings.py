import json
import os
from pathlib import Path

from config import (
    VIDEO_PATH,
    MIN_CONFIDENCE,
    PROCESS_EVERY_N_FRAMES,
    ACTIVE_PROFILE
)


# =========================
# FILE / FOLDER SETTINGS
# =========================

RUNTIME_SETTINGS_FILE = "runtime_settings.json"

SAMPLE_VIDEOS_FOLDER = "videos"
UPLOADS_FOLDER = "uploads"

ALLOWED_VIDEO_EXTENSIONS = [".mp4"]


# =========================
# DEFAULT SETTINGS
# =========================

DEFAULT_RUNTIME_SETTINGS = {
    "selected_video_path": VIDEO_PATH,
    "minimum_confidence": MIN_CONFIDENCE,
    "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
    "active_profile": ACTIVE_PROFILE
}


# =========================
# HELPER FUNCTIONS
# =========================

def ensure_runtime_folders_exist():
    os.makedirs(SAMPLE_VIDEOS_FOLDER, exist_ok=True)
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)


def normalize_path_for_app(file_path):
    return str(file_path).replace("\\", "/")


def is_allowed_video_file(filename):
    file_extension = Path(filename).suffix.lower()
    return file_extension in ALLOWED_VIDEO_EXTENSIONS


def get_safe_upload_filename(filename):
    # Keep this simple and safe for now.
    # Removes folder path tricks and spaces.
    base_name = os.path.basename(filename)
    safe_name = base_name.replace(" ", "_")
    return safe_name


def video_file_exists(video_path):
    if not video_path:
        return False

    return os.path.exists(video_path)


def list_video_files():
    ensure_runtime_folders_exist()

    videos = []

    folders_to_scan = [
        {
            "folder": SAMPLE_VIDEOS_FOLDER,
            "source": "sample"
        },
        {
            "folder": UPLOADS_FOLDER,
            "source": "uploaded"
        }
    ]

    for folder_info in folders_to_scan:
        folder = folder_info["folder"]
        source = folder_info["source"]

        if not os.path.exists(folder):
            continue

        for filename in sorted(os.listdir(folder)):
            if not is_allowed_video_file(filename):
                continue

            full_path = os.path.join(folder, filename)
            app_path = normalize_path_for_app(full_path)

            videos.append({
                "label": filename,
                "path": app_path,
                "source": source
            })

    return videos


def load_runtime_settings():
    ensure_runtime_folders_exist()

    if not os.path.exists(RUNTIME_SETTINGS_FILE):
        save_runtime_settings(DEFAULT_RUNTIME_SETTINGS)
        return DEFAULT_RUNTIME_SETTINGS.copy()

    try:
        with open(RUNTIME_SETTINGS_FILE, "r", encoding="utf-8") as file:
            saved_settings = json.load(file)

        runtime_settings = DEFAULT_RUNTIME_SETTINGS.copy()
        runtime_settings.update(saved_settings)

        return runtime_settings

    except Exception:
        save_runtime_settings(DEFAULT_RUNTIME_SETTINGS)
        return DEFAULT_RUNTIME_SETTINGS.copy()


def save_runtime_settings(settings):
    ensure_runtime_folders_exist()

    with open(RUNTIME_SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    return settings


def validate_runtime_settings(settings):
    errors = []

    selected_video_path = settings.get("selected_video_path")
    minimum_confidence = settings.get("minimum_confidence")
    process_every_n_frames = settings.get("process_every_n_frames")
    active_profile = settings.get("active_profile")

    if not selected_video_path:
        errors.append("selected_video_path is required.")
    elif not video_file_exists(selected_video_path):
        errors.append(f"Selected video file does not exist: {selected_video_path}")

    try:
        confidence_value = float(minimum_confidence)

        if confidence_value < 0 or confidence_value > 1:
            errors.append("minimum_confidence must be between 0 and 1.")
    except Exception:
        errors.append("minimum_confidence must be a number between 0 and 1.")

    try:
        frame_value = int(process_every_n_frames)

        if frame_value < 1:
            errors.append("process_every_n_frames must be 1 or greater.")
    except Exception:
        errors.append("process_every_n_frames must be a whole number.")

    if active_profile != "driving":
        errors.append("Only the driving profile is supported for now.")

    return errors


def update_runtime_settings(new_values):
    current_settings = load_runtime_settings()

    updated_settings = current_settings.copy()

    if "selected_video_path" in new_values:
        updated_settings["selected_video_path"] = new_values["selected_video_path"]

    if "minimum_confidence" in new_values:
        updated_settings["minimum_confidence"] = float(new_values["minimum_confidence"])

    if "process_every_n_frames" in new_values:
        updated_settings["process_every_n_frames"] = int(new_values["process_every_n_frames"])

    # Driving only for now.
    updated_settings["active_profile"] = "driving"

    errors = validate_runtime_settings(updated_settings)

    if errors:
        return {
            "status": "error",
            "errors": errors,
            "settings": current_settings
        }

    save_runtime_settings(updated_settings)

    return {
        "status": "ok",
        "message": "Runtime settings updated.",
        "settings": updated_settings
    }


def save_uploaded_video(upload_file):
    ensure_runtime_folders_exist()

    original_filename = upload_file.filename

    if not original_filename:
        return {
            "status": "error",
            "message": "No filename was provided."
        }

    if not is_allowed_video_file(original_filename):
        return {
            "status": "error",
            "message": "Only .mp4 files are supported."
        }

    safe_filename = get_safe_upload_filename(original_filename)
    destination_path = os.path.join(UPLOADS_FOLDER, safe_filename)

    with open(destination_path, "wb") as output_file:
        output_file.write(upload_file.file.read())

    app_path = normalize_path_for_app(destination_path)

    return {
        "status": "ok",
        "message": "Video uploaded successfully.",
        "filename": safe_filename,
        "path": app_path,
        "source": "uploaded"
    }