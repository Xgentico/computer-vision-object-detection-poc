import cv2
import math
import json
import os
from datetime import datetime
from video_input import get_video_frames
from ultralytics import YOLO

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


# =========================
# PROFILE SETUP
# =========================

profile = load_profile(ACTIVE_PROFILE)

DETECTION_PROFILE = profile.DETECTION_PROFILE
TARGET_CLASS_IDS = profile.TARGET_CLASS_IDS
PERSON_CLASS_ID = profile.PERSON_CLASS_ID


# =========================
# MODEL SETUP
# =========================

model = YOLO(MODEL_NAME)


# =========================
# TRACKING VARIABLES
# =========================

frame_number = 0
processed_frame_count = 0

unique_person_count = 0
tracked_persons = []

# Total detections across the whole video.
# This is NOT unique object tracking.
# It counts each detection event on each processed frame.
total_detection_counts = {}


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


def print_final_summary(run_summary):
    print("")
    print("======================================")
    print("FINAL RUN SUMMARY")
    print("======================================")
    print(f"Run name: {run_summary['run_name']}")
    print(f"Active profile: {run_summary['active_profile']}")
    print(f"Detection profile: {run_summary['detection_profile']}")
    print(f"Run timestamp: {run_summary['run_timestamp']}")
    print(f"Video path: {run_summary['video_path']}")
    print(f"Model: {run_summary['model_name']}")
    print(f"Minimum confidence: {run_summary['minimum_confidence']}")
    print(f"Process every N frames: {run_summary['process_every_n_frames']}")
    print("--------------------------------------")
    print(f"Frames read: {run_summary['frames_read']}")
    print(f"Frames processed by YOLO: {run_summary['frames_processed_by_yolo']}")
    print(f"Unique persons: {run_summary['unique_persons']}")
    print("--------------------------------------")
    print("Total detection counts:")

    if len(run_summary["total_detection_counts"]) == 0:
        print("No detections found.")
    else:
        for class_name, count in run_summary["total_detection_counts"].items():
            print(f"{class_name}: {count}")

    print("--------------------------------------")
    print(f"Summary saved to: {run_summary['summary_file_path']}")
    print("======================================")
    print("")


# =========================
# MAIN VIDEO LOOP
# =========================

for frame in get_video_frames(VIDEO_PATH):
    frame_number += 1

    annotated_frame = frame.copy()

    # Counts what is visible in the current processed frame only.
    current_frame_counts = {}

    should_process_frame = frame_number % PROCESS_EVERY_N_FRAMES == 0

    if should_process_frame:
        processed_frame_count += 1

        results = model(frame, show=False, verbose=False)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only process the classes we care about
                if cls not in TARGET_CLASS_IDS:
                    continue

                # Ignore weak detections
                if confidence < MIN_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = model.names[cls]

                # Count this object for the current processed frame
                increment_count(current_frame_counts, class_name)

                # Count this object across the whole video
                increment_count(total_detection_counts, class_name)

                # Special logic for persons:
                # draw them AND count unique people
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

                        print(
                            f"Unique persons detected: {unique_person_count} "
                            f"| frame: {frame_number} "
                            f"| processed frame: {processed_frame_count} "
                            f"| confidence: {confidence:.2f}"
                        )

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

                # All other selected object classes:
                # draw them but do not uniquely count them yet
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

    # =========================
    # VIDEO DISPLAY TEXT
    # =========================

    draw_screen_text(
        annotated_frame,
        f"Run: {RUN_NAME}",
        20,
        40
    )

    draw_screen_text(
        annotated_frame,
        f"Active profile: {ACTIVE_PROFILE}",
        20,
        80
    )

    draw_screen_text(
        annotated_frame,
        f"Profile: {DETECTION_PROFILE}",
        20,
        120
    )

    draw_screen_text(
        annotated_frame,
        f"Frame: {frame_number}",
        20,
        160
    )

    draw_screen_text(
        annotated_frame,
        f"Processed frames: {processed_frame_count}",
        20,
        200
    )

    draw_screen_text(
        annotated_frame,
        f"Unique persons: {unique_person_count}",
        20,
        240
    )

    draw_screen_text(
        annotated_frame,
        f"Min confidence: {MIN_CONFIDENCE}",
        20,
        280
    )

    draw_screen_text(
        annotated_frame,
        f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s)",
        20,
        320
    )

    # Show current-frame object counts
    draw_screen_text(
        annotated_frame,
        "Current frame counts:",
        20,
        370
    )

    y_position = 410

    for class_name, count in current_frame_counts.items():
        draw_screen_text(
            annotated_frame,
            f"{class_name}: {count}",
            20,
            y_position
        )
        y_position += 35

    # Show total detection counts across the video so far
    draw_screen_text(
        annotated_frame,
        "Total detection counts:",
        420,
        370
    )

    y_position = 410

    for class_name, count in total_detection_counts.items():
        draw_screen_text(
            annotated_frame,
            f"{class_name}: {count}",
            420,
            y_position
        )
        y_position += 35

    # Display the annotated video
    cv2.imshow("Detected Video", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cv2.destroyAllWindows()


# =========================
# RUN SUMMARY
# =========================

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
summary_file_path = create_timestamped_summary_path()

run_summary = {
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
    "summary_file_path": summary_file_path
}


# =========================
# SAVE SUMMARY
# =========================

save_run_summary_to_json(run_summary, summary_file_path)


# =========================
# FINAL TERMINAL OUTPUT
# =========================

print_final_summary(run_summary)