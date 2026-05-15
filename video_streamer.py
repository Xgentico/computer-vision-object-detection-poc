import cv2
import math
import json
import os
from datetime import datetime
from ultralytics import YOLO

from config import (
    RUN_NAME,
    VIDEO_PATH,
    MODEL_NAME,
    MIN_CONFIDENCE,
    PROCESS_EVERY_N_FRAMES,
    MAX_DISTANCE_BETWEEN_PERSONS,
    ACTIVE_PROFILE
)

from profiles.profile_loader import load_profile


# =========================
# PROFILE / MODEL SETUP
# =========================

profile = load_profile(ACTIVE_PROFILE)

DETECTION_PROFILE = profile.DETECTION_PROFILE
TARGET_CLASS_IDS = profile.TARGET_CLASS_IDS
PERSON_CLASS_ID = profile.PERSON_CLASS_ID

model = YOLO(MODEL_NAME)


# =========================
# HELPER FUNCTIONS
# =========================

def get_center_point(x1, y1, x2, y2):
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


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
        0.8,
        (255, 255, 255),
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


def build_run_summary(
    frame_number,
    processed_frame_count,
    unique_person_count,
    total_detection_counts,
    summary_file_path
):
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "run_name": RUN_NAME,
        "active_profile": ACTIVE_PROFILE,
        "detection_profile": DETECTION_PROFILE,
        "run_timestamp": run_timestamp,
        "video_path": VIDEO_PATH,
        "model_name": MODEL_NAME,
        "minimum_confidence": MIN_CONFIDENCE,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
        "max_distance_between_persons": MAX_DISTANCE_BETWEEN_PERSONS,
        "target_class_ids": TARGET_CLASS_IDS,
        "person_class_id": PERSON_CLASS_ID,
        "frames_read": frame_number,
        "frames_processed_by_yolo": processed_frame_count,
        "unique_persons": unique_person_count,
        "total_detection_counts": total_detection_counts,
        "summary_file_path": summary_file_path,
        "source": "browser_video_stream"
    }


def save_stream_run_summary(
    frame_number,
    processed_frame_count,
    unique_person_count,
    total_detection_counts
):
    if frame_number == 0:
        return None

    summary_file_path = create_timestamped_summary_path()

    run_summary = build_run_summary(
        frame_number=frame_number,
        processed_frame_count=processed_frame_count,
        unique_person_count=unique_person_count,
        total_detection_counts=total_detection_counts,
        summary_file_path=summary_file_path
    )

    save_run_summary_to_json(run_summary, summary_file_path)

    print("")
    print("======================================")
    print("STREAM RUN SUMMARY SAVED")
    print("======================================")
    print(f"Summary saved to: {summary_file_path}")
    print(f"Frames read: {frame_number}")
    print(f"Frames processed by YOLO: {processed_frame_count}")
    print(f"Unique persons: {unique_person_count}")
    print("======================================")
    print("")

    return run_summary


# =========================
# VIDEO STREAM GENERATOR
# =========================

def generate_annotated_frames():
    video_capture = cv2.VideoCapture(VIDEO_PATH)

    frame_number = 0
    processed_frame_count = 0
    unique_person_count = 0
    tracked_persons = []
    total_detection_counts = {}

    summary_saved = False

    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

    try:
        while True:
            success, frame = video_capture.read()

            if not success:
                break

            frame_number += 1
            annotated_frame = frame.copy()

            should_process_frame = frame_number % PROCESS_EVERY_N_FRAMES == 0

            if should_process_frame:
                processed_frame_count += 1

                results = model(frame, show=False, verbose=False)

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])

                        if cls not in TARGET_CLASS_IDS:
                            continue

                        if confidence < MIN_CONFIDENCE:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = model.names[cls]

                        increment_count(total_detection_counts, class_name)

                        if cls == PERSON_CLASS_ID:
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
                f"Profile: {DETECTION_PROFILE}",
                20,
                40
            )

            draw_screen_text(
                annotated_frame,
                f"Frame: {frame_number}",
                20,
                80
            )

            draw_screen_text(
                annotated_frame,
                f"Processed frames: {processed_frame_count}",
                20,
                120
            )

            draw_screen_text(
                annotated_frame,
                f"Unique persons: {unique_person_count}",
                20,
                160
            )

            draw_screen_text(
                annotated_frame,
                f"Min confidence: {MIN_CONFIDENCE}",
                20,
                200
            )

            draw_screen_text(
                annotated_frame,
                f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s)",
                20,
                240
            )

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
            frame_number=frame_number,
            processed_frame_count=processed_frame_count,
            unique_person_count=unique_person_count,
            total_detection_counts=total_detection_counts
        )
        summary_saved = True

    finally:
        video_capture.release()

        # If the browser disconnects early, still save a partial summary.
        if not summary_saved and frame_number > 0:
            save_stream_run_summary(
                frame_number=frame_number,
                processed_frame_count=processed_frame_count,
                unique_person_count=unique_person_count,
                total_detection_counts=total_detection_counts
            )