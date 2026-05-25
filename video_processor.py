import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from config import (
    ACTIVE_PROFILE,
    MODEL_NAME,
    MIN_CONFIDENCE,
    PROCESS_EVERY_N_FRAMES,
    MAX_DISTANCE_BETWEEN_PERSONS,
    SAVE_PROCESSED_VIDEO,
    SAVE_EVENTS_CSV,
    SAVE_WARNINGS_JSON,
    SAVE_SUMMARY_JSON,
    RUNTIME_SETTINGS_FILE,
)

from video_web_converter import convert_video_to_browser_mp4


# COCO class names used by YOLOv8.
DEFAULT_CLASS_ID_TO_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}


DEFAULT_SELECTED_CLASS_IDS = [0, 1, 2, 3, 5, 7, 9, 11]


def get_timestamp_string() -> str:
    """
    Return timestamp formatted for filenames and run IDs.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_iso_timestamp() -> str:
    """
    Return ISO-style timestamp for events.
    """
    return datetime.now().isoformat(timespec="seconds")


def load_runtime_settings() -> Dict:
    """
    Load runtime settings from runtime_settings.json if it exists.

    The dashboard already persists settings there.
    Batch mode should reuse those settings when available.

    Falls back to config.py defaults if runtime settings are missing.
    """
    settings = {
        "minimum_confidence": MIN_CONFIDENCE,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
        "selected_class_ids": DEFAULT_SELECTED_CLASS_IDS,
        "selected_class_names": [
            DEFAULT_CLASS_ID_TO_NAME[class_id]
            for class_id in DEFAULT_SELECTED_CLASS_IDS
            if class_id in DEFAULT_CLASS_ID_TO_NAME
        ],
        "active_profile": ACTIVE_PROFILE,
    }

    runtime_settings_path = Path(RUNTIME_SETTINGS_FILE)

    if not runtime_settings_path.exists():
        return settings

    try:
        with runtime_settings_path.open("r", encoding="utf-8") as file:
            runtime_settings = json.load(file)

        settings["minimum_confidence"] = float(
            runtime_settings.get("minimum_confidence", settings["minimum_confidence"])
        )

        settings["process_every_n_frames"] = int(
            runtime_settings.get(
                "process_every_n_frames",
                settings["process_every_n_frames"],
            )
        )

        selected_class_ids = runtime_settings.get("selected_class_ids")

        if isinstance(selected_class_ids, list) and selected_class_ids:
            settings["selected_class_ids"] = [int(class_id) for class_id in selected_class_ids]

        selected_class_names = runtime_settings.get("selected_class_names")

        if isinstance(selected_class_names, list) and selected_class_names:
            settings["selected_class_names"] = selected_class_names
        else:
            settings["selected_class_names"] = [
                DEFAULT_CLASS_ID_TO_NAME[class_id]
                for class_id in settings["selected_class_ids"]
                if class_id in DEFAULT_CLASS_ID_TO_NAME
            ]

        settings["active_profile"] = runtime_settings.get("active_profile", ACTIVE_PROFILE)

    except Exception as error:
        print(f"WARNING: Could not load runtime settings. Using defaults. Error: {error}")

    return settings


def calculate_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def calculate_distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return math.sqrt(
        ((point_a[0] - point_b[0]) ** 2) +
        ((point_a[1] - point_b[1]) ** 2)
    )


def determine_screen_region(center_x: int, frame_width: int) -> str:
    """
    Determine whether a detected object appears on the left, center, or right.
    """
    one_third = frame_width / 3
    two_thirds = one_third * 2

    if center_x < one_third:
        return "left"

    if center_x > two_thirds:
        return "right"

    return "center"


def build_warning_text(screen_region: str) -> str:
    """
    Build the same short warning format used by the dashboard voice layer.
    """
    if screen_region == "left":
        return "Warning. Person on the left."

    if screen_region == "right":
        return "Warning. Person on the right."

    if screen_region == "center":
        return "Warning. Person in the center."

    return "Warning. Person detected."


def draw_detection(
    frame,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    """
    Draw a bounding box and label on a video frame.
    """
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

    label_text = f"{label} {confidence:.2f}"

    cv2.putText(
        frame,
        label_text,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 180, 0),
        2,
    )


def draw_overlay(
    frame,
    frame_number: int,
    current_frame_counts: Dict[str, int],
    unique_person_count: int,
    latest_warning_text: Optional[str],
) -> None:
    """
    Draw summary overlay on each processed frame.
    """
    y = 28

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    y += 30

    cv2.putText(
        frame,
        f"Unique persons: {unique_person_count}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    y += 30

    counts_text = ", ".join(
        f"{class_name}: {count}"
        for class_name, count in current_frame_counts.items()
        if count > 0
    )

    if counts_text:
        cv2.putText(
            frame,
            counts_text[:120],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

    if latest_warning_text:
        frame_height = frame.shape[0]

        cv2.putText(
            frame,
            latest_warning_text,
            (20, frame_height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
        )


def write_json_file(path: Path, data) -> None:
    """
    Write JSON with consistent formatting.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_events_csv(path: Path, events: List[Dict]) -> None:
    """
    Write warning/event rows to CSV.
    """
    columns = [
        "timestamp",
        "frame_number",
        "event_type",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "person_id",
        "screen_region",
        "warning_text",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for event in events:
            writer.writerow({
                "timestamp": event.get("timestamp", ""),
                "frame_number": event.get("frame_number", ""),
                "event_type": event.get("event_type", ""),
                "class": event.get("class", ""),
                "confidence": event.get("confidence", ""),
                "x1": event.get("box", {}).get("x1", ""),
                "y1": event.get("box", {}).get("y1", ""),
                "x2": event.get("box", {}).get("x2", ""),
                "y2": event.get("box", {}).get("y2", ""),
                "person_id": event.get("person_id", ""),
                "screen_region": event.get("screen_region", ""),
                "warning_text": event.get("warning_text", ""),
            })


def process_video_file(
    video_path: str,
    output_folder: str,
    runtime_settings: Optional[Dict] = None,
    generate_narrative: bool = False,
) -> Dict:
    """
    Process one MP4 video file and create reviewable outputs.

    Outputs:
    - processed_video.mp4
    - processed_video_web.mp4
    - summary.json
    - warnings.json
    - events.csv
    """
    del generate_narrative

    source_video_path = Path(video_path)
    run_output_folder = Path(output_folder)
    run_output_folder.mkdir(parents=True, exist_ok=True)

    if not source_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_video_path}")

    settings = runtime_settings or load_runtime_settings()

    minimum_confidence = float(settings.get("minimum_confidence", MIN_CONFIDENCE))
    process_every_n_frames = int(settings.get("process_every_n_frames", PROCESS_EVERY_N_FRAMES))
    selected_class_ids = [int(class_id) for class_id in settings.get("selected_class_ids", DEFAULT_SELECTED_CLASS_IDS)]

    if process_every_n_frames < 1:
        process_every_n_frames = 1

    selected_class_names = [
        DEFAULT_CLASS_ID_TO_NAME[class_id]
        for class_id in selected_class_ids
        if class_id in DEFAULT_CLASS_ID_TO_NAME
    ]

    run_id = run_output_folder.name

    processed_video_path = run_output_folder / "processed_video.mp4"
    processed_video_web_path = run_output_folder / "processed_video_web.mp4"
    summary_path = run_output_folder / "summary.json"
    warnings_path = run_output_folder / "warnings.json"
    events_path = run_output_folder / "events.csv"

    model = YOLO(MODEL_NAME)

    capture = cv2.VideoCapture(str(source_video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {source_video_path}")

    original_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not original_fps or original_fps <= 0:
        original_fps = 30.0

    video_writer = None

    if SAVE_PROCESSED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(processed_video_path),
            fourcc,
            original_fps,
            (frame_width, frame_height),
        )

        if not video_writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create processed video: {processed_video_path}")

    frames_read = 0
    frames_processed = 0

    total_detection_counts: Dict[str, int] = {
        class_name: 0 for class_name in selected_class_names
    }

    known_person_centers: List[Tuple[int, int]] = []
    unique_person_count = 0

    warning_events: List[Dict] = []
    csv_events: List[Dict] = []

    latest_warning_text = None
    latest_warning_display_counter = 0

    print("")
    print(f"Processing video: {source_video_path}")
    print(f"Output folder: {run_output_folder}")
    print(f"Confidence: {minimum_confidence}")
    print(f"Process every N frames: {process_every_n_frames}")
    print(f"Selected classes: {', '.join(selected_class_names)}")
    print("")

    while True:
        success, frame = capture.read()

        if not success:
            break

        frames_read += 1

        current_frame_counts: Dict[str, int] = {
            class_name: 0 for class_name in selected_class_names
        }

        should_process_frame = (
            frames_read == 1 or frames_read % process_every_n_frames == 0
        )

        if should_process_frame:
            frames_processed += 1

            results = model(
                frame,
                conf=minimum_confidence,
                classes=selected_class_ids,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes

                if boxes is None:
                    continue

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id not in selected_class_ids:
                        continue

                    class_name = DEFAULT_CLASS_ID_TO_NAME.get(class_id, str(class_id))

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_tuple = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )

                    current_frame_counts[class_name] = current_frame_counts.get(class_name, 0) + 1
                    total_detection_counts[class_name] = total_detection_counts.get(class_name, 0) + 1

                    draw_detection(frame, box_tuple, class_name, confidence)

                    if class_name == "person":
                        center_x, center_y = calculate_center(box_tuple)

                        matched_existing_person = False

                        for known_center in known_person_centers:
                            distance = calculate_distance((center_x, center_y), known_center)

                            if distance <= MAX_DISTANCE_BETWEEN_PERSONS:
                                matched_existing_person = True
                                break

                        if not matched_existing_person:
                            unique_person_count += 1
                            known_person_centers.append((center_x, center_y))

                            screen_region = determine_screen_region(center_x, frame_width)
                            warning_text = build_warning_text(screen_region)

                            latest_warning_text = warning_text
                            latest_warning_display_counter = 30

                            warning_event = {
                                "timestamp": get_iso_timestamp(),
                                "frame_number": frames_read,
                                "event_type": "unique_person_detected",
                                "severity": "medium",
                                "class": "person",
                                "confidence": confidence,
                                "person_id": unique_person_count,
                                "screen_region": screen_region,
                                "center_point": {
                                    "x": center_x,
                                    "y": center_y,
                                },
                                "box": {
                                    "x1": box_tuple[0],
                                    "y1": box_tuple[1],
                                    "x2": box_tuple[2],
                                    "y2": box_tuple[3],
                                },
                                "warning_text": warning_text,
                                "overlay_warning_text": warning_text,
                            }

                            warning_events.append(warning_event)
                            csv_events.append(warning_event)

        if latest_warning_display_counter > 0:
            latest_warning_display_counter -= 1
        else:
            latest_warning_text = None

        draw_overlay(
            frame=frame,
            frame_number=frames_read,
            current_frame_counts=current_frame_counts,
            unique_person_count=unique_person_count,
            latest_warning_text=latest_warning_text,
        )

        if video_writer:
            video_writer.write(frame)

        if frames_read % 100 == 0:
            print(f"Frames read: {frames_read} / {total_video_frames}")

    capture.release()

    if video_writer:
        video_writer.release()

    web_video_conversion = {
        "status": "skipped",
        "message": "Processed video was not created, so web conversion was skipped.",
        "output_video_path": str(processed_video_web_path),
        "error_message": None,
    }

    if SAVE_PROCESSED_VIDEO and processed_video_path.exists():
        print("")
        print("Creating browser-compatible replay video...")

        web_video_conversion = convert_video_to_browser_mp4(
            input_video_path=str(processed_video_path),
            output_video_path=str(processed_video_web_path),
        )

        print(f"Web video conversion status: {web_video_conversion.get('status')}")
        print(f"Web video path: {processed_video_web_path}")

        if web_video_conversion.get("error_message"):
            print(f"Web video warning: {web_video_conversion.get('error_message')}")

    summary = {
        "run_id": run_id,
        "original_filename": source_video_path.name,
        "status": "completed",
        "source": "batch_processor",
        "detection_profile": ACTIVE_PROFILE,
        "active_profile": ACTIVE_PROFILE,
        "model": MODEL_NAME,
        "minimum_confidence": minimum_confidence,
        "process_every_n_frames": process_every_n_frames,
        "selected_class_ids": selected_class_ids,
        "selected_class_names": selected_class_names,
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "video_total_frames_reported": total_video_frames,
        "video_fps": original_fps,
        "video_width": frame_width,
        "video_height": frame_height,
        "unique_persons": unique_person_count,
        "total_detection_counts": total_detection_counts,
        "warnings_generated": len(warning_events),
        "processed_video_path": str(processed_video_path),
        "processed_video_web_path": str(processed_video_web_path),
        "web_video_conversion": web_video_conversion,
        "summary_path": str(summary_path),
        "warnings_path": str(warnings_path),
        "events_path": str(events_path),
        "created_at": get_iso_timestamp(),
    }

    if SAVE_SUMMARY_JSON:
        write_json_file(summary_path, summary)

    if SAVE_WARNINGS_JSON:
        write_json_file(warnings_path, warning_events)

    if SAVE_EVENTS_CSV:
        write_events_csv(events_path, csv_events)

    print("")
    print("VIDEO PROCESSING COMPLETE")
    print(f"Run ID: {run_id}")
    print(f"Frames read: {frames_read}")
    print(f"Frames processed: {frames_processed}")
    print(f"Unique persons: {unique_person_count}")
    print(f"Warnings generated: {len(warning_events)}")
    print(f"Output folder: {run_output_folder}")
    print("")

    return summary


def main() -> None:
    """
    Simple direct test entry point.

    Example:
        python video_processor.py input_videos/pending/sample_driving.mp4 outputs/runs/manual_test
    """
    import sys

    if len(sys.argv) != 3:
        print("")
        print("Usage:")
        print("python video_processor.py <video_path> <output_folder>")
        print("")
        print("Example:")
        print("python video_processor.py input_videos/pending/sample_driving.mp4 outputs/runs/manual_test")
        print("")
        return

    video_path = sys.argv[1]
    output_folder = sys.argv[2]

    process_video_file(
        video_path=video_path,
        output_folder=output_folder,
        runtime_settings=None,
        generate_narrative=False,
    )


if __name__ == "__main__":
    main()