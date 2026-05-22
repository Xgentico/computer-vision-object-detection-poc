import os
from dotenv import load_dotenv

# =========================
# ENVIRONMENT SETUP
# =========================

# Loads local secrets/settings from a .env file in the project root.
# Example .env:
# OPENAI_API_KEY=sk-proj-your-key-here
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# =========================
# GENERAL RUN SETTINGS
# =========================

RUN_NAME = "vertex_computer_vision_poc"

# For now, we are only using the driving profile.
ACTIVE_PROFILE = "driving"


# =========================
# VIDEO SETTINGS
# =========================

# Default video path. Runtime settings can override this from the dashboard.
VIDEO_PATH = "videos/sample.mp4"

# Process every N frames.
# Higher number = faster but less accurate.
# Lower number = slower but more complete.
PROCESS_EVERY_N_FRAMES = 90


# =========================
# MODEL SETTINGS
# =========================

# YOLOv8 nano model.
# This is lightweight and works well for a local POC.
MODEL_NAME = "yolov8n.pt"

# Default minimum confidence threshold.
# Runtime settings can override this from the dashboard.
MIN_CONFIDENCE = 0.50

# Alias kept for readability and future use.
# Existing app files may still import MIN_CONFIDENCE.
MINIMUM_CONFIDENCE = MIN_CONFIDENCE


# =========================
# PERSON TRACKING SETTINGS
# =========================

# Used to estimate whether a detected person is the same person
# across processed frames.
MAX_DISTANCE_BETWEEN_PERSONS = 80


# =========================
# LIVE WARNING SETTINGS
# =========================

# This controls the local warning layer.
# The live warning does not need OpenAI.
LLM_ENABLED = True
LLM_PROVIDER = "local"
LLM_MODEL = "local-warning-interpreter"

# Number of streamed frames to keep the warning overlay visible.
LLM_WARNING_DISPLAY_FRAMES = 30


# =========================
# OPENAI NARRATIVE SUMMARY SETTINGS
# =========================

# This controls the end-of-run narrative summary.
# If OPENAI_API_KEY is not present, the app should fall back
# to a local summary.
OPENAI_ENABLED = bool(OPENAI_API_KEY)

# Existing narrative service code expects this name.
OPENAI_NARRATIVE_ENABLED = OPENAI_ENABLED

# Keep this inexpensive for the POC.
OPENAI_NARRATIVE_MODEL = "gpt-4o-mini"

# Existing service code may expect this name.
OPENAI_MODEL = OPENAI_NARRATIVE_MODEL

# Maximum target length for the generated run narrative.
OPENAI_NARRATIVE_MAX_WORDS = 120

# Maximum number of warning events to send to OpenAI.
# Keeps the prompt small and inexpensive.
OPENAI_NARRATIVE_MAX_WARNING_EVENTS = 25


# =========================
# FILE STORAGE SETTINGS
# =========================

UPLOADS_FOLDER = "uploads"
OUTPUTS_FOLDER = "outputs"
STATIC_FOLDER = "static"
VIDEOS_FOLDER = "videos"

RUNTIME_SETTINGS_FILE = "runtime_settings.json"


# =========================
# BATCH PROCESSING SETTINGS
# =========================

# Batch mode lets the user drop multiple MP4 files into a folder
# and process them one by one from the terminal.
BATCH_MODE_ENABLED = True

INPUT_VIDEOS_FOLDER = "input_videos"

INPUT_PENDING_FOLDER = "input_videos/pending"
INPUT_PROCESSING_FOLDER = "input_videos/processing"
INPUT_COMPLETED_FOLDER = "input_videos/completed"
INPUT_FAILED_FOLDER = "input_videos/failed"

OUTPUT_RUNS_FOLDER = "outputs/runs"

SUPPORTED_VIDEO_EXTENSIONS = [".mp4"]

SAVE_PROCESSED_VIDEO = True
SAVE_EVENTS_CSV = True
SAVE_WARNINGS_JSON = True
SAVE_SUMMARY_JSON = True

# Batch mode should not call OpenAI by default.
# The current dashboard can still call OpenAI for narrative summaries.
BATCH_GENERATE_NARRATIVE_DEFAULT = False