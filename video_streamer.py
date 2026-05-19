import cv2
import math
import json
import os
from datetime import datetime
from ultralytics import YOLO

from config import (
    RUN_NAME,
    MODEL_NAME,
    MAX_DISTANCE_BETWEEN_PERSONS,
    LLM_ENABLED,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_WARNING_DISPLAY_FRAMES
)

from profiles.profile_loader import load_profile
from runtime_settings import (
    load_runtime_settings,
    get_object_class_names
)
from warning_state import add_warning_event


# =========================
# STREAM PERFORMANCE SETTINGS
# =========================

# Resize video frames before YOLO detection and before browser streaming.
# Recommended values:
# 640 = faster
# 960 = good local testing balance
# 1280 = near full-width for 720p/1080p source videos
STREAM_FRAME_WIDTH = 1280


# =========================
# MODEL SETUP
# =========================

model = YOLO(MODEL_NAME)


# =========================
# HELPER FUNCTIONS
# =========================

def resize_frame_for_streaming(frame):
    height, width = frame.shape[:2]

    if width <= STREAM_FRAME_WIDTH:
        return frame

    scale_ratio = STREAM_FRAME_WIDTH / width
    target_height = int(height * scale_ratio)

    resized_frame = cv2.resize(frame, (STREAM_FRAME_WIDTH, target_height))

    return resized_frame


def get_center_point(x1, y1, x2, y2):
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_screen_region(center_x, frame_width):
    left_boundary = frame_width / 3
    right_boundary = (frame_width / 3) * 2

    if center_x < left_boundary:
        return "left"

    if center_x < right_boundary:
        return "center"

    return "right"


def build_region_warning_messages(screen_region):
    if screen_region == "left":
        return "Warning. Person on the left.", "Person left."

    if screen_region == "right":
        return "Warning. Person on the right.", "Person right."

    return "Warning. Person in the center.", "Person center."


def distance_between_points(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def find_matching_person(center_point, tracked_persons):
    for person in tracked_persons:
        distance = distance_between_points(center_point, person["center"])

        if distance < MAX_DISTANCE_BETWEEN_PERSONS:
            return person

    return None


def draw_detection_box(frame, x1, y1, x2, y2, label):
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def draw_screen_text(frame, text, x, y):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


def draw_warning_overlay(frame, warning_text):
    if not warning_text:
        return

    height, width = frame.shape[:2]

    box_width = 300
    box_height = 50
    margin = 15

    x1 = max(margin, width - box_width - margin)
    y1 = margin
    x2 = min(width - margin, x1 + box_width)
    y2 = y1 + box_height

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        warning_text,
        (x1 + 12, y1 + 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )


def increment_count(counts_dictionary, class_name):
    if class_name not in counts_dictionary:
        counts_dictionary[class_name] = 0

    counts_dictionary[class_name] += 1


def create_timestamped_summary_path():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"outputs/run_summary_{timestamp}.json"


def save_run_summary_to_json(run_summary, output_file_path):
    output_folder = os.path.dirname(output_file_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=4)


def create_unique_person_event(
    confidence,
    frame_number,
    person_id,
    center_point,
    screen_region,
    warning_text,
    overlay_warning_text
):
    return {
        "event_type": "unique_person_detected",
        "severity": "medium",
        "class": "person",
        "confidence": confidence,
        "frame_number": frame_number,
        "person_id": person_id,
        "center_point": {
            "x": center_point[0],
            "y": center_point[1]
        },
        "screen_region": screen_region,
        "message": "New unique person detected.",
        "warning_text": warning_text,
        "overlay_warning_text": overlay_warning_text
    }


def build_run_summary(
    runtime_settings,
    detection_profile,
    target_class_ids,
    selected_class_ids,
    selected_class_names,
    person_class_id,
    frame_number,
    processed_frame_count,
    unique_person_count,
    total_detection_counts,
    llm_warnings_generated,
    llm_warning_events,
    summary_file_path
):
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "run_name": RUN_NAME,
        "active_profile": runtime_settings["active_profile"],
        "detection_profile": detection_profile,
        "run_timestamp": run_timestamp,
        "video_path": runtime_settings["selected_video_path"],
        "model_name": MODEL_NAME,
        "minimum_confidence": runtime_settings["minimum_confidence"],
        "process_every_n_frames": runtime_settings["process_every_n_frames"],
        "max_distance_between_persons": MAX_DISTANCE_BETWEEN_PERSONS,
        "stream_frame_width": STREAM_FRAME_WIDTH,
        "target_class_ids": target_class_ids,
        "selected_class_ids": selected_class_ids,
        "selected_class_names": selected_class_names,
        "person_class_id": person_class_id,
        "frames_read": frame_number,
        "frames_processed_by_yolo": processed_frame_count,
        "unique_persons": unique_person_count,
        "total_detection_counts": total_detection_counts,
        "llm_enabled": LLM_ENABLED,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_warnings_generated": llm_warnings_generated,
        "llm_warning_events": llm_warning_events,
        "summary_file_path": summary_file_path,
        "source": "browser_video_stream"
    }


def save_stream_run_summary(
    runtime_settings,
    detection_profile,
    target_class_ids,
    selected_class_ids,
    selected_class_names,
    person_class_id,
    frame_number,
    processed_frame_count,
    unique_person_count,
    total_detection_counts,
    llm_warnings_generated,
    llm_warning_events
):
    if frame_number == 0:
        return None

    summary_file_path = create_timestamped_summary_path()

    run_summary = build_run_summary(
        runtime_settings=runtime_settings,
        detection_profile=detection_profile,
        target_class_ids=target_class_ids,
        selected_class_ids=selected_class_ids,
        selected_class_names=selected_class_names,
        person_class_id=person_class_id,
        frame_number=frame_number,
        processed_frame_count=processed_frame_count,
        unique_person_count=unique_person_count,
        total_detection_counts=total_detection_counts,
        llm_warnings_generated=llm_warnings_generated,
        llm_warning_events=llm_warning_events,
        summary_file_path=summary_file_path
    )

    save_run_summary_to_json(run_summary, summary_file_path)

    print("")
    print("======================================")
    print("STREAM RUN SUMMARY SAVED")
    print("======================================")
    print(f"Summary saved to: {summary_file_path}")
    print(f"Video path: {runtime_settings['selected_video_path']}")
    print(f"Confidence: {runtime_settings['minimum_confidence']}")
    print(f"Process every N frames: {runtime_settings['process_every_n_frames']}")
    print(f"Selected classes: {selected_class_names}")
    print(f"Frames read: {frame_number}")
    print(f"Frames processed by YOLO: {processed_frame_count}")
    print(f"Unique persons: {unique_person_count}")
    print(f"Warnings generated: {llm_warnings_generated}")
    print("======================================")
    print("")

    return run_summary


# =========================
# VIDEO STREAM GENERATOR
# =========================

def generate_annotated_frames():
    runtime_settings = load_runtime_settings()

    selected_video_path = runtime_settings["selected_video_path"]
    minimum_confidence = float(runtime_settings["minimum_confidence"])
    process_every_n_frames = int(runtime_settings["process_every_n_frames"])
    active_profile = runtime_settings["active_profile"]

    profile = load_profile(active_profile)

    detection_profile = profile.DETECTION_PROFILE
    target_class_ids = profile.TARGET_CLASS_IDS
    person_class_id = profile.PERSON_CLASS_ID

    selected_class_ids = runtime_settings.get("selected_class_ids", target_class_ids)
    selected_class_names = get_object_class_names(selected_class_ids)

    video_capture = cv2.VideoCapture(selected_video_path)

    frame_number = 0
    processed_frame_count = 0
    unique_person_count = 0
    tracked_persons = []
    total_detection_counts = {}

    llm_warnings_generated = 0
    llm_warning_events = []

    latest_overlay_warning = ""
    warning_frames_remaining = 0

    summary_saved = False

    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video file: {selected_video_path}")

    try:
        while True:
            success, frame = video_capture.read()

            if not success:
                break

            frame = resize_frame_for_streaming(frame)

            frame_number += 1
            annotated_frame = frame.copy()

            should_process_frame = frame_number % process_every_n_frames == 0

            if should_process_frame:
                processed_frame_count += 1

                results = model(frame, show=False, verbose=False)

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])

                        if cls not in selected_class_ids:
                            continue

                        if confidence < minimum_confidence:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = model.names[cls]

                        increment_count(total_detection_counts, class_name)

                        if cls == person_class_id:
                            center_point = get_center_point(x1, y1, x2, y2)

                            matching_person = find_matching_person(center_point, tracked_persons)

                            if matching_person is None:
                                unique_person_count += 1

                                matching_person = {
                                    "id": unique_person_count,
                                    "center": center_point,
                                    "first_seen_frame": frame_number,
                                    "last_seen_frame": frame_number
                                }

                                tracked_persons.append(matching_person)

                                frame_width = frame.shape[1]
                                screen_region = get_screen_region(center_point[0], frame_width)

                                warning_text, overlay_warning_text = build_region_warning_messages(screen_region)

                                warning_event = create_unique_person_event(
                                    confidence=confidence,
                                    frame_number=frame_number,
                                    person_id=matching_person["id"],
                                    center_point=center_point,
                                    screen_region=screen_region,
                                    warning_text=warning_text,
                                    overlay_warning_text=overlay_warning_text
                                )

                                llm_warning_events.append(warning_event)
                                add_warning_event(warning_event)

                                llm_warnings_generated += 1

                                latest_overlay_warning = overlay_warning_text
                                warning_frames_remaining = LLM_WARNING_DISPLAY_FRAMES

                                print(f"LIVE WARNING: {warning_text}")

                            else:
                                matching_person["center"] = center_point
                                matching_person["last_seen_frame"] = frame_number

                            label = f"person #{matching_person['id']}: {confidence:.2f}"

                            draw_detection_box(
                                annotated_frame,
                                x1,
                                y1,
                                x2,
                                y2,
                                label
                            )

                            cv2.circle(
                                annotated_frame,
                                center_point,
                                5,
                                (0, 255, 0),
                                -1
                            )

                        else:
                            label = f"{class_name}: {confidence:.2f}"

                            draw_detection_box(
                                annotated_frame,
                                x1,
                                y1,
                                x2,
                                y2,
                                label
                            )

            draw_screen_text(
                annotated_frame,
                f"Profile: {detection_profile}",
                20,
                40
            )

            draw_screen_text(
                annotated_frame,
                f"Video: {os.path.basename(selected_video_path)}",
                20,
                80
            )

            draw_screen_text(
                annotated_frame,
                f"Frame: {frame_number}",
                20,
                120
            )

            draw_screen_text(
                annotated_frame,
                f"Processed frames: {processed_frame_count}",
                20,
                160
            )

            draw_screen_text(
                annotated_frame,
                f"Unique persons: {unique_person_count}",
                20,
                200
            )

            draw_screen_text(
                annotated_frame,
                f"Min confidence: {minimum_confidence}",
                20,
                240
            )

            draw_screen_text(
                annotated_frame,
                f"Processing every {process_every_n_frames} frame(s)",
                20,
                280
            )

            draw_screen_text(
                annotated_frame,
                f"Classes: {', '.join(selected_class_names)}",
                20,
                320
            )

            if warning_frames_remaining > 0:
                draw_warning_overlay(
                    annotated_frame,
                    latest_overlay_warning
                )

                warning_frames_remaining -= 1

            success, encoded_image = cv2.imencode(".jpg", annotated_frame)

            if not success:
                continue

            frame_bytes = encoded_image.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )

        save_stream_run_summary(
            runtime_settings=runtime_settings,
            detection_profile=detection_profile,
            target_class_ids=target_class_ids,
            selected_class_ids=selected_class_ids,
            selected_class_names=selected_class_names,
            person_class_id=person_class_id,
            frame_number=frame_number,
            processed_frame_count=processed_frame_count,
            unique_person_count=unique_person_count,
            total_detection_counts=total_detection_counts,
            llm_warnings_generated=llm_warnings_generated,
            llm_warning_events=llm_warning_events
        )
        summary_saved = True

    finally:
        video_capture.release()

        if not summary_saved and frame_number > 0:
            save_stream_run_summary(
                runtime_settings=runtime_settings,
                detection_profile=detection_profile,
                target_class_ids=target_class_ids,
                selected_class_ids=selected_class_ids,
                selected_class_names=selected_class_names,
                person_class_id=person_class_id,
                frame_number=frame_number,
                processed_frame_count=processed_frame_count,
                unique_person_count=unique_person_count,
                total_detection_counts=total_detection_counts,
                llm_warnings_generated=llm_warnings_generated,
                llm_warning_events=llm_warning_events
            )