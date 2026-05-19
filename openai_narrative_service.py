import glob
import json
import os
from datetime import datetime
from pathlib import Path

from config import (
    OPENAI_NARRATIVE_ENABLED,
    OPENAI_NARRATIVE_MODEL,
    OPENAI_NARRATIVE_MAX_WORDS,
    OPENAI_NARRATIVE_MAX_WARNING_EVENTS
)


# =========================
# FILE HELPERS
# =========================

OUTPUTS_FOLDER = "outputs"


def ensure_outputs_folder_exists():
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)


def get_latest_run_summary_file():
    if not os.path.exists(OUTPUTS_FOLDER):
        return None

    summary_files = glob.glob(os.path.join(OUTPUTS_FOLDER, "run_summary_*.json"))

    if len(summary_files) == 0:
        return None

    return max(summary_files, key=os.path.getmtime)


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_narrative_output_path():
    ensure_outputs_folder_exists()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return os.path.join(
        OUTPUTS_FOLDER,
        f"narrative_summary_{timestamp}.json"
    )


def save_narrative_summary(narrative_payload):
    output_path = create_narrative_output_path()

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(narrative_payload, file, indent=4)

    return output_path


# =========================
# WARNING EVENT PREPARATION
# =========================

def get_warning_events_from_run_summary(run_summary):
    warning_events = run_summary.get("llm_warning_events", [])

    if not warning_events:
        return []

    sorted_events = sorted(
        warning_events,
        key=lambda event: int(event.get("frame_number", 0))
    )

    return sorted_events[:OPENAI_NARRATIVE_MAX_WARNING_EVENTS]


def build_warning_timeline(warning_events):
    timeline = []

    previous_frame = None

    for event in warning_events:
        frame_number = int(event.get("frame_number", 0))
        person_id = event.get("person_id")
        confidence = event.get("confidence")
        center_point = event.get("center_point", {})

        if previous_frame is None:
            frames_since_previous = 0
        else:
            frames_since_previous = frame_number - previous_frame

        previous_frame = frame_number

        timeline.append({
            "person_id": person_id,
            "frame_number": frame_number,
            "frames_since_previous_new_person": frames_since_previous,
            "confidence": confidence,
            "center_x": center_point.get("x"),
            "center_y": center_point.get("y")
        })

    return timeline


def build_local_fallback_narrative(run_summary, warning_timeline):
    warning_count = len(warning_timeline)
    frames_read = run_summary.get("frames_read", 0)
    video_path = run_summary.get("video_path", "selected video")

    if warning_count == 0:
        return (
            f"Run complete. No new unique-person warnings were generated in {video_path}. "
            f"The run processed {frames_read} frames."
        )

    first_event = warning_timeline[0]
    last_event = warning_timeline[-1]

    largest_gap = max(
        warning_timeline,
        key=lambda event: int(event.get("frames_since_previous_new_person", 0))
    )

    return (
        f"Run complete. {warning_count} unique-person warning events were generated. "
        f"The first person warning occurred on frame {first_event['frame_number']}. "
        f"The last person warning occurred on frame {last_event['frame_number']}. "
        f"The longest gap between new-person warnings was "
        f"{largest_gap['frames_since_previous_new_person']} frames before person "
        f"{largest_gap['person_id']} was detected."
    )


# =========================
# OPENAI NARRATIVE GENERATION
# =========================

def build_narrative_prompt(run_summary, warning_timeline):
    return {
        "task": "Summarize object detection warning events for a computer vision demo.",
        "style": {
            "tone": "clear, calm, operational",
            "audience": "executive demo audience",
            "max_words": OPENAI_NARRATIVE_MAX_WORDS
        },
        "rules": [
            "Use only the structured data provided.",
            "Do not invent people, objects, locations, risks, or distances.",
            "Mention frame numbers where useful.",
            "Mention long gaps between new-person warnings when useful.",
            "Keep the summary concise and easy to speak aloud.",
            "Do not claim danger, collision, crime, or intent.",
            "Use the phrase 'new person' rather than identifying anyone."
        ],
        "run_summary": {
            "run_name": run_summary.get("run_name"),
            "video_path": run_summary.get("video_path"),
            "model_name": run_summary.get("model_name"),
            "minimum_confidence": run_summary.get("minimum_confidence"),
            "process_every_n_frames": run_summary.get("process_every_n_frames"),
            "frames_read": run_summary.get("frames_read"),
            "frames_processed_by_yolo": run_summary.get("frames_processed_by_yolo"),
            "unique_persons": run_summary.get("unique_persons"),
            "llm_warnings_generated": run_summary.get("llm_warnings_generated"),
            "total_detection_counts": run_summary.get("total_detection_counts", {})
        },
        "warning_timeline": warning_timeline
    }


def call_openai_for_narrative(run_summary, warning_timeline):
    if not OPENAI_NARRATIVE_ENABLED:
        return build_local_fallback_narrative(run_summary, warning_timeline)

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in PowerShell before generating a narrative."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install openai"
        ) from error

    client = OpenAI(api_key=api_key)

    narrative_prompt = build_narrative_prompt(
        run_summary=run_summary,
        warning_timeline=warning_timeline
    )

    response = client.responses.create(
        model=OPENAI_NARRATIVE_MODEL,
        instructions=(
            "You summarize structured computer vision warning events. "
            "Be factual, concise, and suitable for spoken narration. "
            "Do not invent facts."
        ),
        input=json.dumps(narrative_prompt, indent=2)
    )

    narrative_text = response.output_text.strip()

    if not narrative_text:
        return build_local_fallback_narrative(run_summary, warning_timeline)

    return narrative_text


# =========================
# PUBLIC SERVICE FUNCTION
# =========================

def generate_narrative_from_latest_run():
    latest_file = get_latest_run_summary_file()

    if latest_file is None:
        return {
            "status": "not_found",
            "message": "No run summary files found."
        }

    run_summary = load_json_file(latest_file)
    warning_events = get_warning_events_from_run_summary(run_summary)
    warning_timeline = build_warning_timeline(warning_events)

    try:
        narrative_text = call_openai_for_narrative(
            run_summary=run_summary,
            warning_timeline=warning_timeline
        )

        status = "ok"
        error_message = None

    except Exception as error:
        narrative_text = build_local_fallback_narrative(
            run_summary=run_summary,
            warning_timeline=warning_timeline
        )

        status = "fallback"
        error_message = str(error)

    narrative_payload = {
        "status": status,
        "message": "Narrative summary generated.",
        "source_run_file": latest_file,
        "source_run_filename": Path(latest_file).name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": OPENAI_NARRATIVE_MODEL,
        "openai_enabled": OPENAI_NARRATIVE_ENABLED,
        "warning_count": len(warning_timeline),
        "narrative_text": narrative_text,
        "warning_timeline": warning_timeline,
        "error_message": error_message
    }

    output_path = save_narrative_summary(narrative_payload)

    narrative_payload["narrative_file_path"] = output_path
    narrative_payload["narrative_filename"] = Path(output_path).name

    return narrative_payload