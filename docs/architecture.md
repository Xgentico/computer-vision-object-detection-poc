# Architecture — Computer Vision Object Detection POC

## Purpose

This project is a lightweight computer vision proof of concept for processing MP4 video files, detecting configured object classes, generating annotated output videos, saving run summaries, and optionally using an LLM to summarize results or warning events.

The first use case is driving footage, but the project should remain modular so it can support other scenarios later, such as airports, prison yards, facilities, events, security footage, or drone/object detection.

## High-Level Architecture

The application has four main parts:

1. **FastAPI backend**

   * Serves the API.
   * Serves the static dashboard.
   * Handles video processing requests.
   * Reads and writes run summaries.
   * Integrates with OpenAI for optional LLM summaries.

2. **Static dashboard**

   * Plain HTML, CSS, and JavaScript.
   * Located at `static/dashboard.html`.
   * Used to upload/select videos, adjust settings, start processing, view run history, and review results.

3. **Computer vision processing**

   * Uses Python scripts.
   * Uses YOLOv8 through `ultralytics`.
   * Uses OpenCV for video/image processing.
   * Creates annotated output videos.
   * Tracks detection counts, unique person counts, and warning events.

4. **Output and run history**

   * Saves processed video outputs.
   * Saves timestamped JSON summaries.
   * JSON summaries support later review, comparison, and UI display.

## Known Technology Stack

| Area                                | Tool                               |
| ----------------------------------- | ---------------------------------- |
| Backend web server / API            | FastAPI                            |
| Local API runner                    | Uvicorn                            |
| UI/dashboard                        | Plain static HTML, CSS, JavaScript |
| Dashboard file                      | `static/dashboard.html`            |
| Computer vision processing          | Plain Python scripts               |
| Batch processing                    | `batch_processor.py`               |
| Object detection                    | YOLOv8 via `ultralytics`           |
| Video/image processing              | OpenCV                             |
| Browser-compatible video conversion | `imageio-ffmpeg` / FFmpeg          |
| LLM summaries                       | OpenAI Python SDK                  |
| Deployment                          | Render                             |
| Source control                      | GitHub                             |

## Backend Architecture

The backend is a FastAPI application.

Known app entry point:

```text
api.py
```

Known FastAPI app object:

```python
app = FastAPI(...)
```

Local development should run with Uvicorn.

Typical local command:

```powershell
python -m uvicorn api:app --reload
```

Render start command:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

## Static Dashboard Architecture

The dashboard is not a React, Streamlit, Gradio, or Vite application.

It is a plain static frontend served by FastAPI.

Known dashboard file:

```text
static/dashboard.html
```

The dashboard should stay simple and beginner-friendly.

The dashboard is expected to support:

* selecting or uploading MP4 files
* choosing runtime settings
* starting video processing
* viewing processing status
* reviewing run history
* reviewing selected run details
* showing processed output video links
* showing detection counts
* showing warning events
* showing LLM summaries when available

## Computer Vision Architecture

The object detection layer uses YOLOv8 through `ultralytics`.

The video processing layer uses OpenCV.

The system should preserve the distinction between:

1. **Current-frame counts**

   * Objects detected in the current processed frame.

2. **Total detection events**

   * Object detections accumulated across processed frames.

3. **Approximate unique-person count**

   * A separate tracking/counting concept used to estimate unique people seen in the video.

These should not be treated as the same metric.

## Detection Profiles

The project should support modular detection profiles.

The current primary profile is for driving footage.

A profile may define:

* scenario name
* selected class IDs
* labels
* thresholds
* warning rules
* scenario-specific settings

Scenario-specific logic should stay in profile/config files when possible.

Generic processing code should remain reusable across profiles.

Future example profiles could include:

* driving
* airport
* prison yard
* facility
* event crowd
* drone detection

## Runtime Settings

Runtime settings should remain configurable through the API or dashboard.

Settings may include:

* selected video path
* confidence threshold
* frame sampling / frame skip value
* selected class IDs
* active profile

Runtime settings should be reflected in:

* video processing behavior
* dashboard display
* JSON run summaries
* logs or status messages when useful

## Batch Processing Architecture

Batch processing is handled by:

```text
batch_processor.py
```

Batch processing should allow multiple MP4 files to be processed without breaking the single-video workflow.

Each batch item should ideally have:

* original video reference
* processed output video reference
* JSON run summary
* processing status
* detection counts
* warning events
* LLM summary if available

Batch processing should handle failures per file when practical, so one bad video does not necessarily stop the entire batch.

## Output Architecture

The system saves outputs for later review.

Outputs may include:

* processed/annotated videos
* JSON run summaries
* warning events
* LLM summaries
* metadata about settings used for a run

JSON summaries should remain machine-readable and stable enough to support:

* run history
* selected run detail panels
* later reporting
* later database persistence if needed

## LLM / OpenAI Architecture

The LLM is used for summaries and interpretation, not for ground-truth detection.

Important separation:

```text
Computer vision detection creates facts.
The LLM summarizes or explains those facts.
The LLM must not invent detections or warning events.
```

OpenAI should be called from the Python backend, not from browser JavaScript.

The expected environment variable is:

```text
OPENAI_API_KEY
```

If the API key is missing, the application should still process videos when possible and simply disable or skip LLM summaries.

## Deployment Architecture

The project deploys from GitHub to Render.

Known Render settings:

| Setting           | Value                                         |
| ----------------- | --------------------------------------------- |
| Runtime           | Python                                        |
| Service type      | Web Service                                   |
| Build command     | `pip install -r requirements.txt`             |
| Start command     | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Deployment branch | `main`                                        |

The project must avoid local-only assumptions because Render runs on Linux.

Avoid:

* hard-coded Windows paths
* local user paths like `C:\Users\...`
* committed secrets
* committed `.env` files
* committed large video outputs unless explicitly approved

## Source Control Architecture

The project uses GitHub.

The deployment branch is:

```text
main
```

Because GitHub is synced to Render, pushing to `main` may trigger a deployment.

Changes should be reviewed before pushing.

## Key Architecture Principles

1. **Keep the app modular**

   * Profiles should separate scenario-specific logic from generic processing.

2. **Preserve working behavior**

   * Unique-person counting, multi-class detection, JSON summaries, and dashboard behavior should not be broken by unrelated changes.

3. **Keep changes small**

   * This project should evolve in small, testable steps.

4. **Keep outputs reviewable**

   * Users should be able to review processed videos, warnings, and summaries without manually opening JSON files.

5. **Keep LLM output separate from facts**

   * Detection data is the source of truth.
   * LLM summaries are interpretation only.

6. **Keep deployment safe**

   * Local changes must be checked for Render impact.

## Current Known Risks

| Risk                                  | Why it matters                                                       |
| ------------------------------------- | -------------------------------------------------------------------- |
| Frame skipping affects accuracy       | Processing fewer frames can reduce unique-person detection accuracy. |
| Render file paths differ from Windows | Hard-coded local paths can break deployment.                         |
| Large video files can be slow         | Processing may take time and consume storage.                        |
| LLM summaries may hallucinate         | LLM output must stay tied to structured detection data.              |
| JSON format changes can break UI      | Dashboard and run history may depend on existing summary fields.     |
| Pushing to `main` can deploy          | Broken code can affect the Render service.                           |

## Future Architecture Ideas

Possible future improvements include:

* selected run detail panel
* stronger warning event viewer
* background queue for bulk MP4 processing
* database persistence for runs and warnings
* profile manager UI
* detection benchmark comparison across settings
* support for additional detection scenarios
* better progress tracking for long-running jobs
* improved LLM summary templates
* deployment hardening for Render
