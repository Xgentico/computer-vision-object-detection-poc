# Roadmap — Computer Vision Object Detection POC

## Purpose

This roadmap describes the planned evolution of the Computer Vision Object Detection POC.

The goal is to grow the project in small, safe, testable steps.

This project should remain beginner-friendly, modular, and deployment-safe. Each enhancement should preserve the working baseline before adding new behavior.

## Current Product Direction

The application is becoming a modular video intelligence platform.

The first use case is driving footage, where the system detects people, vehicles, bikes, traffic lights, stop signs, and other configured object classes.

The architecture should also support future detection scenarios such as:

* airports
* prison yards
* buildings
* facilities
* event spaces
* security camera footage
* drone/object detection

## Current Known Capabilities

The project currently supports or is expected to support:

* FastAPI backend
* static dashboard served by FastAPI
* MP4 video processing
* YOLOv8 object detection through `ultralytics`
* OpenCV video processing
* annotated output videos
* runtime settings
* configurable detection classes
* active detection profiles
* multi-class detection
* current-frame object counts
* total detection event counts
* approximate unique-person counting
* timestamped JSON run summaries
* OpenAI/LLM summaries
* Render deployment through GitHub
* batch processing through `batch_processor.py`

## Roadmap Principles

All roadmap work should follow these rules:

1. **Small changes first**

   * Make one logical improvement at a time.

2. **Preserve working behavior**

   * Do not break video processing, dashboard loading, JSON summaries, or unique-person counting while adding features.

3. **Keep the system modular**

   * Scenario-specific logic should live in profiles or configuration.

4. **Keep outputs reviewable**

   * Users should be able to review videos, warnings, settings, and summaries from the dashboard.

5. **Keep deployment safe**

   * Changes must work locally and on Render.

6. **Keep LLMs grounded**

   * Detection logic creates facts.
   * LLMs summarize facts.
   * LLMs must not invent detections or warnings.

## Phase 1 — Stabilize the Current POC

### Goal

Make sure the current working version is documented, understandable, and safe to modify.

### Tasks

| Task                                          |      Status | Notes                                                     |
| --------------------------------------------- | ----------: | --------------------------------------------------------- |
| Move project out of OneDrive                  |        Done | Recommended location is `C:\dev\VERTEX`.                  |
| Add Cursor project rules                      |        Done | Stored under `.cursor/rules/`.                            |
| Add coding style rules                        |        Done | Beginner-friendly Python workflow.                        |
| Add safety review rules                       |        Done | Prevents unapproved broad changes.                        |
| Add architecture documentation                | In progress | Stored in `docs/architecture.md`.                         |
| Add setup documentation                       | In progress | Stored in `docs/setup.md`.                                |
| Add roadmap documentation                     | In progress | This file.                                                |
| Add decisions documentation                   |     Planned | Stored in `docs/decisions.md`.                            |
| Confirm `.gitignore` excludes generated files |     Planned | Important before pushing to GitHub.                       |
| Confirm Render settings                       |     Planned | Must match `uvicorn api:app --host 0.0.0.0 --port $PORT`. |

### Acceptance Criteria

Phase 1 is complete when:

* Cursor rules exist.
* Documentation files exist.
* Local setup instructions are clear.
* The app runs locally from `C:\dev\VERTEX`.
* The dashboard opens.
* A small MP4 can be processed.
* JSON summaries are created.
* Generated outputs are not accidentally committed.

## Phase 2 — Selected Run Detail Panel

### Goal

Allow a user to click a run from Run History and see detailed information without manually opening JSON files.

### Why This Matters

The project already creates run summaries. The next logical step is making those summaries easy to review from the dashboard.

### Planned Features

The Selected Run Detail panel should display:

* selected video
* processed output video
* runtime settings
* active profile
* detection profile
* confidence threshold
* frame sampling value
* selected class IDs
* detection counts
* unique person count
* warning events
* LLM summary if available
* JSON summary filename or reference

### Likely Files Affected

| File                    | Possible Change                                              |
| ----------------------- | ------------------------------------------------------------ |
| `api.py`                | Add or verify endpoint for reading one selected run summary. |
| `static/dashboard.html` | Add selected run detail UI.                                  |
| output JSON structure   | May need to confirm fields are consistent.                   |

### Safety Notes

* Do not change JSON structure unless needed.
* Handle older JSON files gracefully.
* Do not break existing Run History.
* Do not break existing video processing.
* Keep UI plain HTML, CSS, and JavaScript.

### Acceptance Criteria

This phase is complete when:

* Run History still displays.
* A user can select one run.
* The dashboard displays run details.
* Missing fields do not crash the UI.
* Warning events are visible if present.
* The selected run detail can be tested locally.

## Phase 3 — Warning Event Improvements

### Goal

Improve how warning events are created, stored, and displayed.

### Example Warning Types

* person detected
* person close by
* object detected in area of concern
* vehicle detected
* bike detected
* stop sign detected
* traffic light detected
* configured high-priority object detected

### Warning Event Structure

Warning events should be stored as structured data.

Recommended fields:

| Field               | Purpose                             |
| ------------------- | ----------------------------------- |
| `warning_type`      | Category of warning.                |
| `object_class`      | Detected object class.              |
| `frame_number`      | Frame where warning occurred.       |
| `timestamp_seconds` | Approximate video timestamp.        |
| `confidence`        | Detection confidence.               |
| `position`          | Optional location or side of frame. |
| `severity`          | Low, medium, high, or similar.      |
| `message`           | Human-readable warning message.     |

### Safety Notes

* Detection logic should create warning events.
* LLMs may summarize warnings.
* LLMs must not invent warning events.
* Warning logic should be configurable by profile when possible.

### Acceptance Criteria

This phase is complete when:

* Warning events are structured.
* Warning events are saved to JSON summaries.
* Warning events display in the dashboard.
* OpenAI summaries do not replace raw warning data.

## Phase 4 — Batch Processing Workflow

### Goal

Support dropping multiple MP4 files into a folder and processing them in the background or through a controlled batch process.

### Planned Capabilities

* input folder for MP4 files
* batch processor reads pending files
* each file gets processed independently
* each file gets its own JSON summary
* each file gets its own output video
* failed files are reported clearly
* successful and failed runs are visible later

### Likely Files Affected

| File                    | Possible Change                  |
| ----------------------- | -------------------------------- |
| `batch_processor.py`    | Main batch processing workflow.  |
| `api.py`                | Optional batch status endpoints. |
| `static/dashboard.html` | Optional batch status UI.        |
| output folders          | Store per-file results.          |

### Safety Notes

* Preserve single-video processing.
* Avoid overwriting files.
* Avoid blocking the dashboard.
* Handle bad video files clearly.
* Track each file separately.
* Do not commit bulk videos or generated outputs.

### Acceptance Criteria

This phase is complete when:

* Multiple MP4 files can be processed.
* Each processed file has an output video.
* Each processed file has a JSON summary.
* Failed files are reported without crashing the whole batch.
* Results can be reviewed later.

## Phase 5 — Profile Management

### Goal

Make detection profiles easier to add, switch, and understand.

### Planned Capabilities

* profile list
* active profile selection
* profile-specific class IDs
* profile-specific warning rules
* profile-specific thresholds if needed
* profile metadata shown in dashboard
* profile included in JSON summary

### Example Profiles

| Profile           | Purpose                                   |
| ----------------- | ----------------------------------------- |
| `driving`         | Road/driving footage.                     |
| `facility`        | Indoor or building footage.               |
| `prison_yard`     | Correctional facility outdoor monitoring. |
| `airport`         | Airport or restricted area monitoring.    |
| `event`           | Crowd and venue monitoring.               |
| `drone_detection` | Drone/object-focused detection.           |

### Safety Notes

* Keep generic processing separate from scenario-specific profile logic.
* Do not hard-code profile behavior into `api.py` or generic processing code.
* New profiles should not require rewriting the app.

### Acceptance Criteria

This phase is complete when:

* Profiles are modular.
* Active profile can be selected or configured.
* Profile metadata is visible.
* JSON summaries include the profile used.
* Adding a new profile is straightforward.

## Phase 6 — Detection Benchmarking

### Goal

Compare detection results across confidence thresholds, frame sampling values, and profiles.

### Why This Matters

Frame skipping and confidence thresholds affect accuracy and performance. The project should make these tradeoffs visible.

### Planned Capabilities

* compare run summaries
* show detection counts across runs
* show unique person counts across runs
* show settings used for each run
* show differences between profiles
* identify accuracy/performance tradeoffs

### Example Metrics

| Metric                 | Purpose                             |
| ---------------------- | ----------------------------------- |
| total processing time  | Performance measurement.            |
| frame sampling value   | Shows how many frames were skipped. |
| confidence threshold   | Shows detection strictness.         |
| total detection events | Overall detection volume.           |
| unique person count    | Approximate people count.           |
| warning count          | Number of warning events.           |

### Acceptance Criteria

This phase is complete when:

* Runs can be compared.
* Settings are visible per run.
* Accuracy/performance tradeoffs are easier to understand.
* Results can guide future configuration choices.

## Phase 7 — Persistence and Reporting

### Goal

Move beyond JSON files if the project needs stronger history, search, reporting, or multi-user review.

### Possible Future Storage Options

| Option     | Use Case                                        |
| ---------- | ----------------------------------------------- |
| JSON files | Good for early POC and simple local review.     |
| SQLite     | Good for local structured storage.              |
| PostgreSQL | Good for hosted app, dashboards, and reporting. |

### Planned Capabilities

* persistent run database
* searchable run history
* warning event table
* profile tracking
* processing metrics
* report exports
* better dashboard analytics

### Safety Notes

* Do not add a database until JSON summaries are no longer sufficient.
* Adding a database is an architecture change and requires approval.
* Render deployment impact must be reviewed before adding database dependencies.

### Acceptance Criteria

This phase is complete when:

* A storage decision is approved.
* Migration path from JSON summaries is clear.
* Dashboard can read from the selected storage layer.
* Render deployment impact is understood.

## Phase 8 — Deployment Hardening

### Goal

Make the Render deployment more reliable.

### Planned Capabilities

* verify Render startup command
* verify environment variables
* verify output folder behavior
* verify static file serving
* verify video processing limits
* verify OpenAI failure handling
* add health check endpoint if not present
* document deployment steps

### Known Render Settings

| Setting       | Value                                         |
| ------------- | --------------------------------------------- |
| Runtime       | Python                                        |
| Service Type  | Web Service                                   |
| Build Command | `pip install -r requirements.txt`             |
| Start Command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Branch        | `main`                                        |

### Safety Notes

* Do not change Render start command unless approved.
* Do not commit `.env`.
* Do not expose `OPENAI_API_KEY`.
* Do not rely on local Windows paths.
* Be careful with large video processing on hosted infrastructure.

### Acceptance Criteria

This phase is complete when:

* Render deployment starts reliably.
* Dashboard loads from Render.
* Required environment variables are documented.
* Missing optional variables do not crash the app.
* Deployment risks are documented.

## Phase 9 — LLM Interpretation Improvements

### Goal

Improve how the LLM summarizes detection results while keeping facts grounded in detection data.

### Planned Capabilities

* structured prompt inputs
* consistent summary format
* warning summary
* risk summary
* plain-English explanation of what was detected
* graceful failure when OpenAI is unavailable
* clear separation of facts and interpretation

### LLM Rules

* The LLM summarizes detection facts.
* The LLM does not create detection facts.
* The LLM does not invent warning events.
* The LLM should not receive unnecessary video data.
* The LLM should receive structured summary data.
* OpenAI calls should happen server-side.

### Acceptance Criteria

This phase is complete when:

* Summaries are grounded in JSON run data.
* LLM failure does not break video processing.
* Summaries are useful but clearly secondary to detection facts.
* API keys remain secure.

## Phase 10 — Production Considerations

### Goal

Identify what would be needed before treating the POC like a real product.

### Possible Future Needs

* authentication
* user roles
* secure file upload
* stronger file validation
* job queue
* database
* object storage
* monitoring
* logs
* audit trail
* retention policy
* privacy review
* model performance benchmarks
* deployment scaling plan

### Important Note

This project is currently a proof of concept. It should not be treated as production-ready until security, scale, data retention, privacy, and operational concerns are reviewed.

## Current Recommended Next Step

The recommended next feature is:

```text
Selected Run Detail panel under Run History
```

This is the best next step because it builds directly on the existing JSON summaries and makes the app easier to review without manually opening output files.

## Near-Term Task List

| Priority | Task                                          | Status      |
| -------: | --------------------------------------------- | ----------- |
|        1 | Confirm project runs from `C:\dev\VERTEX`     | Planned     |
|        2 | Confirm `.gitignore` excludes generated files | Planned     |
|        3 | Add docs files                                | In progress |
|        4 | Add Selected Run Detail panel                 | Planned     |
|        5 | Improve warning event display                 | Planned     |
|        6 | Improve batch processing review workflow      | Planned     |
|        7 | Add profile selection improvements            | Planned     |
|        8 | Add run comparison / benchmarking             | Future      |
|        9 | Consider database persistence                 | Future      |
|       10 | Harden Render deployment                      | Future      |

## Development Rule

Every roadmap task should follow this workflow:

1. Inspect the current code.
2. Explain what exists.
3. Propose a small plan.
4. Wait for approval.
5. Make only the approved change.
6. Test locally.
7. Review Render risk.
8. Document the result.
