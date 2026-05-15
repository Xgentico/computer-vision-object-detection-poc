import json
import os
import glob
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


app = FastAPI(
    title="Computer Vision Object Detection API",
    version="0.1.0"
)

# Allow the local dashboard.html file to call the API.
# This is okay for local development.
# Later, for production, we will lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "person_class_id": profile.PERSON_CLASS_ID
    }


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

    # Sort newest first
    summary_files = sorted(summary_files, key=os.path.getmtime, reverse=True)

    runs = []

    for summary_file in summary_files:
        try:
            with open(summary_file, "r", encoding="utf-8") as file:
                run_summary = json.load(file)

            filename = os.path.basename(summary_file)

            runs.append({
                "filename": filename,
                "summary_file": summary_file,
                "run_name": run_summary.get("run_name"),
                "active_profile": run_summary.get("active_profile"),
                "detection_profile": run_summary.get("detection_profile"),
                "run_timestamp": run_summary.get("run_timestamp"),
                "model_name": run_summary.get("model_name"),
                "minimum_confidence": run_summary.get("minimum_confidence"),
                "process_every_n_frames": run_summary.get("process_every_n_frames"),
                "frames_read": run_summary.get("frames_read"),
                "frames_processed_by_yolo": run_summary.get("frames_processed_by_yolo"),
                "unique_persons": run_summary.get("unique_persons"),
                "total_detection_counts": run_summary.get("total_detection_counts", {})
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

    # Security guard:
    # Only allow reading files that look like our run summary files.
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