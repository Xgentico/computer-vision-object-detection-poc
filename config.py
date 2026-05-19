# =========================
# RUN SETTINGS
# =========================

RUN_NAME = "driving_test_001"

# For now, only the driving profile is active.
ACTIVE_PROFILE = "driving"


# =========================
# VIDEO / MODEL SETTINGS
# =========================

# Render-safe relative path.
# Uploaded videos are selected through runtime_settings.json.
VIDEO_PATH = "videos/sample_driving.mp4"

# Lightweight YOLO model for proof-of-concept use.
MODEL_NAME = "yolov8n.pt"

# Default confidence threshold.
# Runtime settings can override this value.
MIN_CONFIDENCE = 0.50

# Default frame sampling.
# Runtime settings can override this value.
PROCESS_EVERY_N_FRAMES = 90


# =========================
# UNIQUE PERSON TRACKING
# =========================

# Approximate pixel distance used to decide whether a detected person
# is likely the same person as one already seen.
MAX_DISTANCE_BETWEEN_PERSONS = 80


# =========================
# LLM / INTERPRETER SETTINGS
# =========================

# Keep this False for now.
# The current live warning implementation uses a local fallback interpreter.
# No real LLM API call is made for live warnings.
LLM_ENABLED = False

# Local interpreter for live warnings.
LLM_PROVIDER = "local"

# Local placeholder model name for traceability in summaries.
LLM_MODEL = "local-warning-interpreter"

# Maximum intended length for warning messages.
LLM_WARNING_MAX_WORDS = 20

# Number of streamed frames the warning should remain visible on the overlay.
LLM_WARNING_DISPLAY_FRAMES = 90


# =========================
# OPENAI NARRATIVE SUMMARY SETTINGS
# =========================

# This controls the end-of-run narrative summary only.
# It does not affect YOLO detection or the live warning interpreter.
OPENAI_NARRATIVE_ENABLED = True

# Low-cost model for summarizing structured warning events.
OPENAI_NARRATIVE_MODEL = "gpt-5.4-nano"

# Keep the narrative concise for demo use.
OPENAI_NARRATIVE_MAX_WORDS = 140

# Limit warning events sent to OpenAI to avoid unnecessary token use.
OPENAI_NARRATIVE_MAX_WARNING_EVENTS = 50