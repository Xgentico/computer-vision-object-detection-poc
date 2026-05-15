# =========================
# RUN SETTINGS
# =========================

RUN_NAME = "driving_test_001"

# This controls which profile main.py loads.
# Current options:
# "driving"
# "airport"
ACTIVE_PROFILE = "driving"


# =========================
# VIDEO / MODEL SETTINGS
# =========================

#VIDEO_PATH = r"C:\Users\jcano\OneDrive\Documents\Vertex\4644521-hd_1282_720_60fps.mp4"
VIDEO_PATH = "videos/sample_driving.mp4"

MODEL_NAME = "yolov8n.pt"


# =========================
# DETECTION RUNTIME SETTINGS
# =========================

# Minimum confidence required to draw/count a detection
MIN_CONFIDENCE = 0.50

# Process every N frames
# 1 = every frame, best accuracy
# 2 = every other frame
# 4 = every 4th frame
PROCESS_EVERY_N_FRAMES = 90

# How close a new person detection must be to an existing person
# to be considered the same person
MAX_DISTANCE_BETWEEN_PERSONS = 80