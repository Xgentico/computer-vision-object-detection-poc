import json
import os
from pathlib import Path

from config import (
    VIDEO_PATH,
    MIN_CONFIDENCE,
    PROCESS_EVERY_N_FRAMES,
    ACTIVE_PROFILE
)

from profiles.profile_loader import load_profile


# =========================
# FILE / FOLDER SETTINGS
# =========================

RUNTIME_SETTINGS_FILE = "runtime_settings.json"

SAMPLE_VIDEOS_FOLDER = "videos"
UPLOADS_FOLDER = "uploads"

ALLOWED_VIDEO_EXTENSIONS = [".mp4"]


# =========================
# OBJECT CLASS SETTINGS
# =========================

# For now we only expose the driving profile classes.
# These class IDs are COCO IDs used by YOLOv8.
DRIVING_OBJECT_CLASSES = [
    {
        "id": 0,
        "name": "person",
        "description": "Person / pedestrian"
    },
    {
        "id": 1,
        "name": "bicycle",
        "description": "Bicycle"
    },
    {
        "id": 2,
        "name": "car",
        "description": "Car"
    },
    {
        "id": 3,
        "name": "motorcycle",
        "description": "Motorcycle"
    },
    {
        "id": 5,
        "name": "bus",
        "description": "Bus"
    },
    {
        "id": 7,
        "name": "truck",
        "description": "Truck"
    },
    {
        "id": 9,
        "name": "traffic light",
        "description": "Traffic light"
    },
    {
        "id": 11,
        "name": "stop sign",
        "description": "Stop sign"
    }
]


def get_supported_object_class_ids():
    return [object_class["id"] for object_class in DRIVING_OBJECT_CLASSES]


def get_object_class_name(class_id):
    for object_class in DRIVING_OBJECT_CLASSES:
        if object_class["id"] == class_id:
            return object_class["name"]

    return f"class_{class_id}"


def get_object_class_names(class_ids):
    return [get_object_class_name(class_id) for class_id in class_ids]


# =========================
# DEFAULT SETTINGS
# =========================

def get_default_selected_class_ids():
    profile = load_profile(ACTIVE_PROFILE)
    return profile.TARGET_CLASS_IDS.copy()


DEFAULT_RUNTIME_SETTINGS = {
    "selected_video_path": VIDEO_PATH,
    "minimum_confidence": MIN_CONFIDENCE,
    "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
    "active_profile": ACTIVE_PROFILE,
    "selected_class_ids": get_default_selected_class_ids()
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


def normalize_selected_class_ids(selected_class_ids):
    if selected_class_ids is None:
        return get_default_selected_class_ids()

    normalized_ids = []

    for class_id in selected_class_ids:
        try:
            normalized_ids.append(int(class_id))
        except Exception:
            continue

    # Remove duplicates while preserving order.
    deduped_ids = []
    for class_id in normalized_ids:
        if class_id not in deduped_ids:
            deduped_ids.append(class_id)

    return deduped_ids


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

        # Backward compatibility for older runtime_settings.json files.
        if "selected_class_ids" not in runtime_settings:
            runtime_settings["selected_class_ids"] = get_default_selected_class_ids()

        runtime_settings["selected_class_ids"] = normalize_selected_class_ids(
            runtime_settings.get("selected_class_ids")
        )

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
    selected_class_ids = settings.get("selected_class_ids")

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

    normalized_class_ids = normalize_selected_class_ids(selected_class_ids)
    supported_class_ids = get_supported_object_class_ids()

    if len(normalized_class_ids) == 0:
        errors.append("At least one object class must be selected.")

    unsupported_class_ids = [
        class_id for class_id in normalized_class_ids
        if class_id not in supported_class_ids
    ]

    if unsupported_class_ids:
        errors.append(
            f"Unsupported object class IDs selected: {unsupported_class_ids}"
        )

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

    if "selected_class_ids" in new_values:
        updated_settings["selected_class_ids"] = normalize_selected_class_ids(
            new_values["selected_class_ids"]
        )

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