import csv
import json
import os
import glob
import shutil
import stat
import threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    RUN_NAME,
    ACTIVE_PROFILE,
    VIDEO_PATH,
    MODEL_NAME,
    MIN_CONFIDENCE,
    PROCESS_EVERY_N_FRAMES,
    MAX_DISTANCE_BETWEEN_PERSONS,
    INPUT_PENDING_FOLDER,
    INPUT_PROCESSING_FOLDER,
    INPUT_COMPLETED_FOLDER,
    INPUT_FAILED_FOLDER,
    OUTPUT_RUNS_FOLDER,
    BATCH_MODE_ENABLED,
)

from profiles.profile_loader import load_profile
from video_streamer import generate_annotated_frames

from runtime_settings import (
    list_video_files,
    load_runtime_settings,
    update_runtime_settings,
    save_uploaded_video,
    DRIVING_OBJECT_CLASSES,
    get_object_class_names,
)

from warning_state import (
    get_warning_events,
    clear_warning_events,
)

from openai_narrative_service import generate_narrative_from_latest_run

from batch_file_utils import (
    ensure_batch_folders,
    get_pending_video_files,
)

from batch_processor import process_pending_video


app = FastAPI(
    title="Computer Vision Object Detection API",
    version="1.1.0"
)

# Prevent two browser requests from starting two batch runs at the same time.
batch_run_lock = threading.Lock()

# Allow browser access during local development.
# Later we can restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# STATIC DASHBOARD
# =========================

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_dashboard():
    return FileResponse("static/dashboard.html")


# =========================
# BASIC API ENDPOINTS
# =========================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Computer vision API is running"
    }


@app.get("/config")
def get_config():
    profile = load_profile(ACTIVE_PROFILE)

    return {
        "run_name": RUN_NAME,
        "active_profile": ACTIVE_PROFILE,
        "detection_profile": profile.DETECTION_PROFILE,
        "video_path": VIDEO_PATH,
        "model_name": MODEL_NAME,
        "minimum_confidence": MIN_CONFIDENCE,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
        "max_distance_between_persons": MAX_DISTANCE_BETWEEN_PERSONS,
        "target_class_ids": profile.TARGET_CLASS_IDS,
        "target_class_names": get_object_class_names(profile.TARGET_CLASS_IDS),
        "person_class_id": profile.PERSON_CLASS_ID
    }


# =========================
# RUNTIME SETTINGS ENDPOINTS
# =========================

@app.get("/runtime-settings")
def get_runtime_settings():
    settings = load_runtime_settings()

    selected_class_ids = settings.get("selected_class_ids", [])

    return {
        "status": "ok",
        "settings": {
            **settings,
            "selected_class_names": get_object_class_names(selected_class_ids)
        }
    }


@app.post("/runtime-settings")
def post_runtime_settings(settings: dict):
    result = update_runtime_settings(settings)

    if result.get("status") == "ok":
        selected_class_ids = result["settings"].get("selected_class_ids", [])
        result["settings"]["selected_class_names"] = get_object_class_names(selected_class_ids)

    return result


# =========================
# OBJECT CLASS ENDPOINTS
# =========================

@app.get("/object-classes")
def get_object_classes():
    return {
        "status": "ok",
        "active_profile": "driving",
        "classes": DRIVING_OBJECT_CLASSES,
        "note": "Only driving profile object classes are supported for now."
    }


# =========================
# VIDEO FILE ENDPOINTS
# =========================

@app.get("/videos")
def get_videos():
    videos = list_video_files()

    return {
        "status": "ok",
        "video_count": len(videos),
        "videos": videos
    }


@app.post("/videos/upload")
def upload_video(file: UploadFile = File(...)):
    result = save_uploaded_video(file)

    return result


# =========================
# LIVE WARNING ENDPOINTS
# =========================

@app.get("/warnings")
def get_warnings():
    return get_warning_events()


@app.post("/warnings/clear")
def clear_warnings():
    return clear_warning_events()


# =========================
# VIDEO STREAM ENDPOINT
# =========================

@app.get("/video-stream")
def video_stream():
    # Clear the live warning panel whenever a new stream starts.
    clear_warning_events()

    return StreamingResponse(
        generate_annotated_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# NARRATIVE SUMMARY ENDPOINTS
# =========================

@app.post("/runs/latest/narrative-summary")
def post_latest_run_narrative_summary():
    return generate_narrative_from_latest_run()


# =========================
# BATCH PROCESSING HELPERS
# =========================

def get_timestamp_string() -> str:
    """
    Return timestamp formatted for safe duplicate filenames.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_safe_upload_destination(folder_path: Path, filename: str) -> Path:
    """
    Build a safe upload destination path.

    If the file already exists, append a timestamp.
    """
    original_path = Path(filename)
    safe_filename = original_path.name

    destination_path = folder_path / safe_filename

    if not destination_path.exists():
        return destination_path

    timestamp = get_timestamp_string()

    return folder_path / f"{original_path.stem}_{timestamp}{original_path.suffix}"


def get_file_info(folder_path: str):
    """
    Return basic file info for files in a folder.

    This is used by the dashboard batch status page.
    """
    folder = Path(folder_path)

    if not folder.exists():
        return []

    files = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.name == ".gitkeep":
            continue

        stat = file_path.stat()

        files.append({
            "name": file_path.name,
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        })

    return files


def read_json_file_if_exists(file_path: Path, default_value):
    """
    Read a JSON file if it exists.

    Returns default_value if the file does not exist.
    Raises an exception if the file exists but cannot be parsed.
    """
    if not file_path.exists():
        return default_value

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_preview_if_exists(file_path: Path, max_rows: int = 50):
    """
    Read a small preview of a CSV file.

    This prevents the browser from loading a huge events file.
    """
    if not file_path.exists():
        return []

    rows = []

    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader):
            if index >= max_rows:
                break

            rows.append(row)

    return rows


OUTPUTS_FOLDER = "outputs"


def normalize_path_for_compare(path_value: str) -> str:
    """
    Normalize a file path so run references can be compared safely.
    """
    if not path_value:
        return ""

    try:
        return str(Path(path_value).resolve())
    except Exception:
        return str(Path(path_value))


def is_valid_stream_run_summary_filename(filename: str) -> bool:
    return filename.startswith("run_summary_") and filename.endswith(".json")


def find_narrative_files_for_stream_run(summary_filename: str):
    """
    Find narrative summary files linked to one stream run summary.
    """
    narrative_pattern = os.path.join(OUTPUTS_FOLDER, "narrative_summary_*.json")
    linked_files = []

    for narrative_file in glob.glob(narrative_pattern):
        try:
            with open(narrative_file, "r", encoding="utf-8") as file:
                narrative_data = json.load(file)
        except Exception:
            continue

        if narrative_data.get("source_run_filename") == summary_filename:
            linked_files.append(narrative_file)

    return linked_files


def is_video_referenced_by_other_runs(
    video_path: Path,
    exclude_batch_run_id: str = None,
    exclude_stream_summary_filename: str = None,
) -> bool:
    """
    Return True if another batch or stream run still references this video file.
    """
    if not video_path:
        return False

    candidate_path = normalize_path_for_compare(str(video_path))
    candidate_name = video_path.name

    runs_folder = Path(OUTPUT_RUNS_FOLDER)
    if runs_folder.exists():
        for run_folder in runs_folder.iterdir():
            if not run_folder.is_dir():
                continue

            if exclude_batch_run_id and run_folder.name == exclude_batch_run_id:
                continue

            summary_path = run_folder / "summary.json"
            if not summary_path.exists():
                continue

            try:
                summary = read_json_file_if_exists(summary_path, {})
            except Exception:
                continue

            original_filename = summary.get("original_filename", "")
            if original_filename and original_filename == candidate_name:
                return True

    summary_pattern = os.path.join(OUTPUTS_FOLDER, "run_summary_*.json")
    for summary_file in glob.glob(summary_pattern):
        summary_filename = os.path.basename(summary_file)

        if exclude_stream_summary_filename and summary_filename == exclude_stream_summary_filename:
            continue

        try:
            with open(summary_file, "r", encoding="utf-8") as file:
                run_summary = json.load(file)
        except Exception:
            continue

        other_video_path = run_summary.get("video_path", "")
        if not other_video_path:
            continue

        if normalize_path_for_compare(other_video_path) == candidate_path:
            return True

        if Path(other_video_path).name == candidate_name:
            return True

    return False


def find_batch_original_video_path(original_filename: str):
    """
    Find the original uploaded video for one batch run by filename.
    """
    if not original_filename:
        return None

    search_folders = [
        INPUT_COMPLETED_FOLDER,
        INPUT_FAILED_FOLDER,
        INPUT_PROCESSING_FOLDER,
        INPUT_PENDING_FOLDER,
    ]

    for folder_path in search_folders:
        candidate = Path(folder_path) / original_filename
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def delete_file_safely(file_path: Path, deleted_items: list, skipped_items: list):
    """
    Delete one file and record the result.
    """
    if not file_path or not file_path.exists():
        return

    try:
        file_path.unlink()
        deleted_items.append(str(file_path))
    except Exception as error:
        skipped_items.append(f"{file_path} — could not delete: {error}")


def remove_readonly_file_error(remove_function, path, error_info):
    """
    Clear read-only flags on Windows so folder deletion can continue.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        remove_function(path)
    except Exception:
        raise error_info[1]


def delete_folder_safely(folder_path: Path, deleted_items: list, skipped_items: list) -> bool:
    """
    Delete a folder and everything inside it.

    Uses a Windows-friendly retry path when shutil.rmtree hits access errors.
    """
    if not folder_path.exists():
        return True

    try:
        shutil.rmtree(folder_path, onerror=remove_readonly_file_error)
        deleted_items.append(str(folder_path))
        return True
    except Exception as first_error:
        try:
            for item in list(folder_path.rglob("*")):
                if item.is_file() or item.is_symlink():
                    try:
                        os.chmod(item, stat.S_IWRITE)
                    except Exception:
                        pass
                    item.unlink(missing_ok=True)
                    deleted_items.append(str(item))
                elif item.is_dir():
                    try:
                        item.rmdir()
                        deleted_items.append(str(item))
                    except Exception:
                        pass

            if folder_path.exists():
                try:
                    os.chmod(folder_path, stat.S_IWRITE)
                except Exception:
                    pass

                folder_path.rmdir()
                deleted_items.append(str(folder_path))
                return True

        except Exception as fallback_error:
            skipped_items.append(
                f"{folder_path} — could not delete: {fallback_error} (initial error: {first_error})"
            )
            return False

    return True


def is_batch_run_folder_empty(run_folder: Path) -> bool:
    """
    Return True when a batch run folder has no summary and no files.
    """
    if not run_folder.exists() or not run_folder.is_dir():
        return False

    summary_path = run_folder / "summary.json"
    if summary_path.exists():
        return False

    return not any(run_folder.iterdir())


def cleanup_empty_batch_run_folders():
    """
    Remove leftover empty batch run folders that would show as unknown in the UI.
    """
    runs_folder = Path(OUTPUT_RUNS_FOLDER)
    if not runs_folder.exists():
        return

    deleted_items = []
    skipped_items = []

    for run_folder in runs_folder.iterdir():
        if not run_folder.is_dir():
            continue

        if is_batch_run_folder_empty(run_folder):
            delete_folder_safely(run_folder, deleted_items, skipped_items)


def delete_batch_run_folder(run_id: str):
    """
    Delete one batch output run folder and related files.
    """
    deleted_items = []
    skipped_items = []

    run_folder = get_safe_batch_run_folder(run_id)
    if run_folder is None:
        return {
            "status": "not_found",
            "message": f"Batch run not found: {run_id}",
            "deleted": deleted_items,
            "skipped": skipped_items,
        }

    summary = read_json_file_if_exists(run_folder / "summary.json", {})
    original_filename = summary.get("original_filename", "")
    original_video_path = find_batch_original_video_path(original_filename)

    folder_deleted = delete_folder_safely(run_folder, deleted_items, skipped_items)
    if not folder_deleted:
        return {
            "status": "error",
            "message": f"Could not delete batch run folder: {run_id}",
            "deleted": deleted_items,
            "skipped": skipped_items,
        }

    if original_video_path:
        if is_video_referenced_by_other_runs(
            original_video_path,
            exclude_batch_run_id=run_id,
        ):
            skipped_items.append(
                f"{original_video_path} — still referenced by other runs"
            )
        else:
            delete_file_safely(original_video_path, deleted_items, skipped_items)

    return {
        "status": "ok",
        "message": f"Batch run deleted: {run_id}",
        "run_id": run_id,
        "deleted": deleted_items,
        "skipped": skipped_items,
    }


def delete_stream_run_summary(filename: str):
    """
    Delete one stream run summary and linked narrative files.
    """
    deleted_items = []
    skipped_items = []

    if not is_valid_stream_run_summary_filename(filename):
        return {
            "status": "error",
            "message": "Invalid run summary filename.",
            "deleted": deleted_items,
            "skipped": skipped_items,
        }

    summary_path = Path(OUTPUTS_FOLDER) / filename
    if not summary_path.exists():
        return {
            "status": "not_found",
            "message": f"Run summary file not found: {filename}",
            "deleted": deleted_items,
            "skipped": skipped_items,
        }

    run_summary = read_json_file_if_exists(summary_path, {})
    video_path_value = run_summary.get("video_path", "")
    video_path = Path(video_path_value) if video_path_value else None

    for narrative_file in find_narrative_files_for_stream_run(filename):
        delete_file_safely(Path(narrative_file), deleted_items, skipped_items)

    delete_file_safely(summary_path, deleted_items, skipped_items)

    if video_path and video_path.exists():
        if is_video_referenced_by_other_runs(
            video_path,
            exclude_stream_summary_filename=filename,
        ):
            skipped_items.append(
                f"{video_path} — still referenced by other runs"
            )
        else:
            delete_file_safely(video_path, deleted_items, skipped_items)
    elif video_path_value:
        skipped_items.append(f"{video_path_value} — video file not found")

    return {
        "status": "ok",
        "message": f"Run summary deleted: {filename}",
        "filename": filename,
        "deleted": deleted_items,
        "skipped": skipped_items,
    }


def get_safe_batch_run_folder(run_id: str):
    """
    Resolve a batch run folder safely under OUTPUT_RUNS_FOLDER.

    This prevents path traversal like ../../somefile.
    """
    output_runs_root = Path(OUTPUT_RUNS_FOLDER).resolve()
    requested_folder = (output_runs_root / run_id).resolve()

    try:
        requested_folder.relative_to(output_runs_root)
    except ValueError:
        return None

    if not requested_folder.exists() or not requested_folder.is_dir():
        return None

    return requested_folder


def get_batch_output_runs():
    """
    Return latest batch output run folders from outputs/runs.

    Each batch run folder may contain:
    - processed_video.mp4
    - processed_video_web.mp4
    - summary.json
    - warnings.json
    - events.csv
    - narrative_summary.json
    """
    runs_folder = Path(OUTPUT_RUNS_FOLDER)

    if not runs_folder.exists():
        return []

    cleanup_empty_batch_run_folders()

    run_folders = [
        folder for folder in runs_folder.iterdir()
        if folder.is_dir() and not is_batch_run_folder_empty(folder)
    ]

    run_folders = sorted(
        run_folders,
        key=lambda folder: folder.stat().st_mtime,
        reverse=True
    )

    runs = []

    for folder in run_folders:
        summary_path = folder / "summary.json"
        warnings_path = folder / "warnings.json"
        events_path = folder / "events.csv"
        narrative_path = folder / "narrative_summary.json"
        processed_video_path = folder / "processed_video.mp4"
        processed_video_web_path = folder / "processed_video_web.mp4"

        summary = {}
        narrative = {}

        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as file:
                    summary = json.load(file)
            except Exception as error:
                summary = {
                    "status": "error",
                    "error_message": str(error)
                }

        if narrative_path.exists():
            try:
                with narrative_path.open("r", encoding="utf-8") as file:
                    narrative = json.load(file)
            except Exception as error:
                narrative = {
                    "status": "error",
                    "error_message": str(error)
                }

        runs.append({
            "run_id": folder.name,
            "folder": str(folder),
            "status": summary.get("status", "unknown"),
            "original_filename": summary.get("original_filename"),
            "created_at": summary.get("created_at"),
            "frames_read": summary.get("frames_read"),
            "frames_processed": summary.get("frames_processed"),
            "unique_persons": summary.get("unique_persons"),
            "warnings_generated": summary.get("warnings_generated"),
            "total_detection_counts": summary.get("total_detection_counts", {}),

            "summary_exists": summary_path.exists(),
            "warnings_exists": warnings_path.exists(),
            "events_exists": events_path.exists(),
            "narrative_exists": narrative_path.exists(),
            "processed_video_exists": processed_video_path.exists(),
            "processed_video_web_exists": processed_video_web_path.exists(),

            "summary_path": str(summary_path),
            "warnings_path": str(warnings_path),
            "events_path": str(events_path),
            "narrative_path": str(narrative_path),
            "processed_video_path": str(processed_video_path),
            "processed_video_web_path": str(processed_video_web_path),

            "narrative_status": narrative.get("status") if narrative else None,
            "narrative_model_name": narrative.get("model_name") if narrative else None,
        })

    return runs


# =========================
# BATCH PROCESSING ENDPOINTS
# =========================

@app.get("/batch/status")
def get_batch_status():
    """
    Return batch folder status and latest batch output runs.

    This does not start processing. It only reports current state.
    """
    ensure_batch_folders()

    pending_files = get_file_info(INPUT_PENDING_FOLDER)
    processing_files = get_file_info(INPUT_PROCESSING_FOLDER)
    completed_files = get_file_info(INPUT_COMPLETED_FOLDER)
    failed_files = get_file_info(INPUT_FAILED_FOLDER)
    batch_runs = get_batch_output_runs()

    return {
        "status": "ok",
        "batch_mode_enabled": BATCH_MODE_ENABLED,
        "is_running": batch_run_lock.locked(),
        "folders": {
            "pending": INPUT_PENDING_FOLDER,
            "processing": INPUT_PROCESSING_FOLDER,
            "completed": INPUT_COMPLETED_FOLDER,
            "failed": INPUT_FAILED_FOLDER,
            "output_runs": OUTPUT_RUNS_FOLDER
        },
        "counts": {
            "pending": len(pending_files),
            "processing": len(processing_files),
            "completed": len(completed_files),
            "failed": len(failed_files),
            "output_runs": len(batch_runs)
        },
        "files": {
            "pending": pending_files,
            "processing": processing_files,
            "completed": completed_files,
            "failed": failed_files
        },
        "runs": batch_runs[:20]
    }


@app.post("/batch/upload")
def upload_batch_videos(files: list[UploadFile] = File(...)):
    """
    Upload one or more MP4 files directly into input_videos/pending.

    This supports the Batch Processing UI.
    """
    ensure_batch_folders()

    pending_folder = Path(INPUT_PENDING_FOLDER)
    pending_folder.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    rejected_files = []

    for uploaded_file in files:
        original_filename = uploaded_file.filename or ""

        if not original_filename.lower().endswith(".mp4"):
            rejected_files.append({
                "filename": original_filename,
                "reason": "Only MP4 files are supported."
            })
            continue

        destination_path = get_safe_upload_destination(
            folder_path=pending_folder,
            filename=original_filename
        )

        try:
            with destination_path.open("wb") as output_file:
                shutil.copyfileobj(uploaded_file.file, output_file)

            uploaded_files.append({
                "filename": original_filename,
                "saved_filename": destination_path.name,
                "saved_path": str(destination_path),
                "size_bytes": destination_path.stat().st_size,
                "size_mb": round(destination_path.stat().st_size / (1024 * 1024), 2)
            })

        except Exception as error:
            rejected_files.append({
                "filename": original_filename,
                "reason": str(error)
            })

    return {
        "status": "ok" if uploaded_files else "error",
        "message": f"Uploaded {len(uploaded_files)} file(s) to pending.",
        "uploaded_count": len(uploaded_files),
        "rejected_count": len(rejected_files),
        "uploaded_files": uploaded_files,
        "rejected_files": rejected_files
    }


@app.post("/batch/run")
def post_batch_run():
    """
    Run batch processing from the dashboard/API.

    For now, this is synchronous:
    - The API request waits until batch processing finishes.
    - This is acceptable for the local POC.
    - Later we can move this to a background worker.
    """
    if not BATCH_MODE_ENABLED:
        return {
            "status": "disabled",
            "message": "Batch mode is disabled in config.py."
        }

    if not batch_run_lock.acquire(blocking=False):
        return {
            "status": "already_running",
            "message": "A batch run is already in progress."
        }

    try:
        ensure_batch_folders()

        pending_files = get_pending_video_files()

        if not pending_files:
            return {
                "status": "no_files",
                "message": "No pending MP4 files found.",
                "videos_found": 0,
                "completed": 0,
                "failed": 0,
                "results": []
            }

        results = []

        for pending_video_path in pending_files:
            result = process_pending_video(pending_video_path)
            results.append(result)

        completed_results = [
            result for result in results
            if result.get("status") == "completed"
        ]

        failed_results = [
            result for result in results
            if result.get("status") == "failed"
        ]

        return {
            "status": "ok",
            "message": "Batch processing complete.",
            "videos_found": len(results),
            "completed": len(completed_results),
            "failed": len(failed_results),
            "results": results
        }

    finally:
        batch_run_lock.release()


@app.get("/batch/runs/{run_id}")
def get_batch_run_detail(run_id: str):
    """
    Return review details for one batch output run.

    This powers the Review Run panel in the dashboard.
    """
    ensure_batch_folders()

    run_folder = get_safe_batch_run_folder(run_id)

    if run_folder is None:
        return {
            "status": "not_found",
            "message": f"Batch run not found: {run_id}"
        }

    summary_path = run_folder / "summary.json"
    warnings_path = run_folder / "warnings.json"
    events_path = run_folder / "events.csv"
    narrative_path = run_folder / "narrative_summary.json"
    processed_video_path = run_folder / "processed_video.mp4"
    processed_video_web_path = run_folder / "processed_video_web.mp4"

    try:
        summary = read_json_file_if_exists(summary_path, {})
        warnings = read_json_file_if_exists(warnings_path, [])
        narrative = read_json_file_if_exists(narrative_path, {})
        events_preview = read_csv_preview_if_exists(events_path, max_rows=50)

        if not isinstance(warnings, list):
            warnings = []

        return {
            "status": "ok",
            "run_id": run_folder.name,
            "folder": str(run_folder),
            "files": {
                "processed_video": str(processed_video_path),
                "processed_video_web": str(processed_video_web_path),
                "summary": str(summary_path),
                "warnings": str(warnings_path),
                "events": str(events_path),
                "narrative": str(narrative_path),
            },
            "exists": {
                "processed_video": processed_video_path.exists(),
                "processed_video_web": processed_video_web_path.exists(),
                "summary": summary_path.exists(),
                "warnings": warnings_path.exists(),
                "events": events_path.exists(),
                "narrative": narrative_path.exists(),
            },
            "summary": summary,
            "warnings": warnings,
            "warning_count": len(warnings),
            "events_preview": events_preview,
            "events_preview_count": len(events_preview),
            "narrative": narrative,
            "narrative_text": narrative.get("narrative_text", "") if narrative else "",
            "narrative_status": narrative.get("status", "not_found") if narrative else "not_found",
            "narrative_model_name": narrative.get("model_name", "") if narrative else "",
        }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Could not read batch run detail for {run_id}.",
            "error": str(error)
        }


@app.get("/batch/runs/{run_id}/processed-video")
def get_batch_processed_video(run_id: str):
    """
    Return the processed video for optional browser replay.

    Prefer processed_video_web.mp4 because it is browser-compatible.
    Fall back to processed_video.mp4 for older runs.
    """
    run_folder = get_safe_batch_run_folder(run_id)

    if run_folder is None:
        return {
            "status": "not_found",
            "message": f"Batch run not found: {run_id}"
        }

    browser_video_path = run_folder / "processed_video_web.mp4"
    original_processed_video_path = run_folder / "processed_video.mp4"

    if browser_video_path.exists():
        return FileResponse(
            path=str(browser_video_path),
            media_type="video/mp4",
            filename=f"{run_id}_processed_video_web.mp4"
        )

    if original_processed_video_path.exists():
        return FileResponse(
            path=str(original_processed_video_path),
            media_type="video/mp4",
            filename=f"{run_id}_processed_video.mp4"
        )

    return {
        "status": "not_found",
        "message": f"Processed video not found for batch run: {run_id}"
    }


@app.delete("/batch/runs/{run_id}")
def delete_batch_run(run_id: str):
    """
    Delete one batch output run folder and related files.

    The original uploaded video is deleted only when no other runs still
    reference that file.
    """
    return delete_batch_run_folder(run_id)


# =========================
# RUN HISTORY ENDPOINTS
# =========================

@app.get("/runs")
def get_run_history():
    output_folder = OUTPUTS_FOLDER

    if not os.path.exists(output_folder):
        return {
            "status": "not_found",
            "message": "The outputs folder does not exist yet.",
            "runs": []
        }

    summary_files = glob.glob(os.path.join(output_folder, "run_summary_*.json"))

    if len(summary_files) == 0:
        return {
            "status": "not_found",
            "message": "No run summary files found.",
            "runs": []
        }

    summary_files = sorted(summary_files, key=os.path.getmtime, reverse=True)

    runs = []

    for summary_file in summary_files:
        try:
            with open(summary_file, "r", encoding="utf-8") as file:
                run_summary = json.load(file)

            filename = os.path.basename(summary_file)

            selected_class_ids = run_summary.get("selected_class_ids", run_summary.get("target_class_ids", []))

            runs.append({
                "filename": filename,
                "summary_file": summary_file,
                "run_name": run_summary.get("run_name"),
                "active_profile": run_summary.get("active_profile"),
                "detection_profile": run_summary.get("detection_profile"),
                "run_timestamp": run_summary.get("run_timestamp"),
                "video_path": run_summary.get("video_path"),
                "model_name": run_summary.get("model_name"),
                "minimum_confidence": run_summary.get("minimum_confidence"),
                "process_every_n_frames": run_summary.get("process_every_n_frames"),
                "stream_frame_width": run_summary.get("stream_frame_width"),
                "frames_read": run_summary.get("frames_read"),
                "frames_processed_by_yolo": run_summary.get("frames_processed_by_yolo"),
                "unique_persons": run_summary.get("unique_persons"),
                "total_detection_counts": run_summary.get("total_detection_counts", {}),
                "selected_class_ids": selected_class_ids,
                "selected_class_names": run_summary.get(
                    "selected_class_names",
                    get_object_class_names(selected_class_ids)
                ),
                "llm_enabled": run_summary.get("llm_enabled"),
                "llm_provider": run_summary.get("llm_provider"),
                "llm_model": run_summary.get("llm_model"),
                "llm_warnings_generated": run_summary.get("llm_warnings_generated", 0),
                "llm_warning_events": run_summary.get("llm_warning_events", [])
            })

        except Exception as error:
            runs.append({
                "summary_file": summary_file,
                "error": str(error)
            })

    return {
        "status": "ok",
        "run_count": len(runs),
        "runs": runs
    }


@app.get("/runs/latest")
def get_latest_run_summary():
    output_folder = OUTPUTS_FOLDER

    if not os.path.exists(output_folder):
        return {
            "status": "not_found",
            "message": "The outputs folder does not exist yet."
        }

    summary_files = glob.glob(os.path.join(output_folder, "run_summary_*.json"))

    if len(summary_files) == 0:
        return {
            "status": "not_found",
            "message": "No run summary files found."
        }

    latest_file = max(summary_files, key=os.path.getmtime)

    with open(latest_file, "r", encoding="utf-8") as file:
        run_summary = json.load(file)

    return {
        "status": "ok",
        "filename": os.path.basename(latest_file),
        "summary_file": latest_file,
        "run_summary": run_summary
    }


@app.delete("/runs/{filename}")
def delete_run_by_filename(filename: str):
    """
    Delete one stream run summary and linked files.

    The source video is deleted only when no other runs still reference it.
    """
    return delete_stream_run_summary(filename)


@app.get("/runs/{filename}")
def get_run_by_filename(filename: str):
    output_folder = OUTPUTS_FOLDER

    if not is_valid_stream_run_summary_filename(filename):
        return {
            "status": "error",
            "message": "Invalid run summary filename."
        }

    file_path = os.path.join(output_folder, filename)

    if not os.path.exists(file_path):
        return {
            "status": "not_found",
            "message": f"Run summary file not found: {filename}"
        }

    with open(file_path, "r", encoding="utf-8") as file:
        run_summary = json.load(file)

    return {
        "status": "ok",
        "filename": filename,
        "summary_file": file_path,
        "run_summary": run_summary
    }