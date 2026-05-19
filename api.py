import json
import os
import glob

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
    MAX_DISTANCE_BETWEEN_PERSONS
)

from profiles.profile_loader import load_profile
from video_streamer import generate_annotated_frames

from runtime_settings import (
    list_video_files,
    load_runtime_settings,
    update_runtime_settings,
    save_uploaded_video,
    DRIVING_OBJECT_CLASSES,
    get_object_class_names
)

from warning_state import (
    get_warning_events,
    clear_warning_events
)

from openai_narrative_service import generate_narrative_from_latest_run


app = FastAPI(
    title="Computer Vision Object Detection API",
    version="0.6.0"
)

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
# RUN HISTORY ENDPOINTS
# =========================

@app.get("/runs")
def get_run_history():
    output_folder = "outputs"

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
    output_folder = "outputs"

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


@app.get("/runs/{filename}")
def get_run_by_filename(filename: str):
    output_folder = "outputs"

    if not filename.startswith("run_summary_") or not filename.endswith(".json"):
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