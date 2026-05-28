# Decisions — Computer Vision Object Detection POC

## Purpose

This document records important technical and product decisions for the Computer Vision Object Detection POC.

The goal is to make the project easier to understand, maintain, and improve over time.

When a decision changes, add a new entry instead of silently deleting the old one. This helps explain how the project evolved.

## Decision Log Format

Use this format for new decisions:

```text
## Decision: Short decision title

Date: YYYY-MM-DD  
Status: Proposed / Accepted / Changed / Deprecated

### Context

What problem were we solving?

### Decision

What did we decide?

### Reason

Why did we make this decision?

### Impact

What does this affect?

### Follow-up

What should be reviewed later?
```

---

## Decision: Use FastAPI for the Backend

Date: 2026-05-28
Status: Accepted

### Context

The project needs a lightweight backend to serve the dashboard, expose API endpoints, process videos, read/write runtime settings, and return run history data.

### Decision

Use FastAPI as the backend web server and API framework.

The FastAPI app entry point is:

```text
api.py
```

The FastAPI app object is:

```python
app = FastAPI(...)
```

### Reason

FastAPI is lightweight, Python-native, and works well for this project because the computer vision processing is also written in Python.

It also works cleanly with Uvicorn for local development and Render deployment.

### Impact

The local server should run with:

```powershell
python -m uvicorn api:app --reload
```

Render should run with:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

### Follow-up

Keep the app entry point as `api.py` unless there is a strong reason to change it.

---

## Decision: Use Plain Static HTML, CSS, and JavaScript for the Dashboard

Date: 2026-05-28
Status: Accepted

### Context

The project needs a simple dashboard for selecting videos, configuring runtime settings, starting processing, and reviewing outputs.

### Decision

Use plain static HTML, CSS, and JavaScript served by FastAPI.

The known dashboard file is:

```text
static/dashboard.html
```

### Reason

The project is still a proof of concept. A simple static dashboard is easier to understand, debug, and modify than a larger frontend framework.

Juan is still learning Python and project structure, so keeping the frontend simple reduces unnecessary complexity.

### Impact

Do not introduce React, Vite, Streamlit, Gradio, or another frontend framework unless Juan explicitly approves.

### Follow-up

If the dashboard becomes too complex, revisit whether a frontend framework is justified.

---

## Decision: Use YOLOv8 Through Ultralytics for Object Detection

Date: 2026-05-28
Status: Accepted

### Context

The project needs object detection for videos.

The initial focus is driving footage, including detection of people, cars, bikes, traffic lights, stop signs, and related objects.

### Decision

Use YOLOv8 through the `ultralytics` Python package.

### Reason

YOLOv8 is practical for object detection experiments and can detect multiple object classes from video frames.

It is suitable for the current proof of concept.

### Impact

Detection logic should preserve:

* multi-class detection
* configurable class IDs
* confidence thresholds
* current-frame counts
* total detection events
* approximate unique-person counting

### Follow-up

Evaluate other models only if YOLOv8 becomes insufficient for accuracy, performance, or deployment needs.

---

## Decision: Use OpenCV for Video Processing

Date: 2026-05-28
Status: Accepted

### Context

The project needs to read videos, process frames, draw overlays, and create annotated output videos.

### Decision

Use OpenCV for video and image processing.

### Reason

OpenCV is a standard Python tool for reading video frames, writing output video, drawing boxes/text, and handling frame-level processing.

### Impact

Computer vision processing should remain clear and readable.

Frame processing changes should be made carefully because they can affect detection results and unique-person counting.

### Follow-up

If browser playback compatibility is an issue, continue using FFmpeg-related conversion tools where needed.

---

## Decision: Use imageio-ffmpeg / FFmpeg for Browser-Compatible Video Conversion

Date: 2026-05-28
Status: Accepted

### Context

Videos created by OpenCV may not always play correctly in a browser, depending on codec and format.

### Decision

Use `imageio-ffmpeg` / FFmpeg support for browser-compatible video conversion.

### Reason

The dashboard needs to display processed video outputs reliably.

Browser compatibility matters because the dashboard is the main review interface.

### Impact

Video conversion failures should be handled clearly.

The app should explain when processing worked but browser conversion failed.

### Follow-up

Confirm the exact output format and codec expected by the dashboard.

---

## Decision: Keep Runtime Settings Configurable

Date: 2026-05-28
Status: Accepted

### Context

The project needs to test different settings such as selected video, confidence threshold, frame sampling, selected class IDs, and active profile.

### Decision

Keep runtime settings configurable rather than hard-coded.

Runtime settings may be stored in:

```text
runtime_settings.json
```

and handled by:

```text
runtime_settings.py
```

### Reason

Different videos and profiles may require different detection settings.

Frame sampling and confidence thresholds affect both performance and accuracy.

### Impact

Do not hard-code:

* selected video path
* confidence threshold
* frame sampling value
* selected class IDs
* active profile

### Follow-up

Consider exposing more runtime settings in the dashboard over time.

---

## Decision: Preserve Unique-Person Counting Separately from Detection Events

Date: 2026-05-28
Status: Accepted

### Context

The project tracks object detections across frames and also estimates unique people.

These are not the same thing.

A person may be detected many times across many frames, but should not necessarily count as many unique people.

### Decision

Preserve the distinction between:

* current-frame counts
* total detection events
* approximate unique-person count

### Reason

This distinction is important for understanding results accurately.

Previous testing showed that frame skipping affects unique-person detection counts.

### Impact

Cursor and future developers should not merge these metrics or treat them as interchangeable.

Detection changes must explain how they affect unique-person counting.

### Follow-up

Consider documenting the unique-person counting method in more detail later.

---

## Decision: Keep Detection Profiles Modular

Date: 2026-05-28
Status: Accepted

### Context

The first scenario is driving footage, but the project may later support other use cases like airports, prison yards, buildings, events, or drone detection.

### Decision

Use modular detection profiles.

Scenario-specific settings should live in profile files or profile-related configuration when possible.

### Reason

Profiles make it easier to add new detection scenarios without rewriting the main app.

### Impact

Driving-specific class IDs, labels, thresholds, and warning rules should not be hard-coded into generic processing code.

### Follow-up

Add or improve profile management in the dashboard later.

---

## Decision: Use JSON Run Summaries for the Current POC

Date: 2026-05-28
Status: Accepted

### Context

The app needs to save information about each processing run.

This includes selected video, settings, detection counts, warnings, summaries, and output references.

### Decision

Use timestamped JSON run summaries for the current POC.

### Reason

JSON is simple, inspectable, and good enough for the early proof of concept.

It avoids adding database complexity too early.

### Impact

Run summaries should remain machine-readable and stable enough for:

* run history
* selected run detail panels
* future comparison views
* possible migration to a database later

### Follow-up

Consider SQLite or PostgreSQL only when JSON summaries are no longer enough.

---

## Decision: Keep LLM Summaries Separate from Detection Facts

Date: 2026-05-28
Status: Accepted

### Context

The project may use OpenAI to summarize detections or warnings.

LLMs can be useful but can also hallucinate.

### Decision

The LLM may summarize or explain detection data, but it must not create detection facts.

Important rule:

```text
Computer vision detection creates facts.
The LLM summarizes or explains those facts.
The LLM must not invent detections or warning events.
```

### Reason

Detection results must remain grounded in actual computer vision data.

LLM summaries should help humans understand the results, not replace the raw facts.

### Impact

Raw detection data and LLM summaries should be stored separately in JSON run summaries.

The app should still work if OpenAI is unavailable.

### Follow-up

Improve prompt templates and summary structure over time.

---

## Decision: Use OPENAI_API_KEY for OpenAI Access

Date: 2026-05-28
Status: Accepted

### Context

The project needs an environment variable for OpenAI API access.

### Decision

Use:

```text
OPENAI_API_KEY
```

### Reason

`OPENAI_API_KEY` is the standard and commonly recognized environment variable name for OpenAI integrations.

### Impact

The API key should be stored:

* locally in `.env`
* in Render environment variables for deployment

The API key must not be committed to GitHub.

### Follow-up

Confirm the code reads `OPENAI_API_KEY`.

If the current code uses another variable name, update the code and Render setting together after approval.

---

## Decision: Deploy Through GitHub Synced to Render

Date: 2026-05-28
Status: Accepted

### Context

The project is deployed through Render, and Render is synced to the GitHub repository.

### Decision

Use GitHub as source control and Render as the deployment target.

The deployment branch is:

```text
main
```

### Reason

This keeps deployment simple and makes it easy to deploy by pushing code.

### Impact

Pushing to `main` may trigger a Render deployment.

Deployment-impacting changes must be reviewed carefully.

### Follow-up

Before pushing to `main`, confirm local testing and review Render risk.

---

## Decision: Use Render as a Python Web Service

Date: 2026-05-28
Status: Accepted

### Context

The app needs to run as a hosted web application.

### Decision

Use Render as a Python Web Service.

Known Render settings:

| Setting       | Value                                         |
| ------------- | --------------------------------------------- |
| Runtime       | Python                                        |
| Service Type  | Web Service                                   |
| Build Command | `pip install -r requirements.txt`             |
| Start Command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Branch        | `main`                                        |

### Reason

Render supports simple Python web services and GitHub-based deployment.

### Impact

Do not change the Render start command unless Juan explicitly approves.

### Follow-up

Review Render storage and performance limits before processing large videos in production.

---

## Decision: Move Active Development Out of OneDrive

Date: 2026-05-28
Status: Accepted

### Context

The project was previously located under OneDrive/Documents.

Windows File Explorer and development tools were slow.

### Decision

Move active development to:

```text
C:\dev\VERTEX
```

### Reason

OneDrive can slow down active coding projects, especially projects with:

* virtual environments
* generated files
* videos
* Git folders
* many Python package files
* output folders

### Impact

Cursor should open the project from `C:\dev\VERTEX`.

OneDrive should not be used for active project development.

### Follow-up

If the repo is re-cloned, clone it into `C:\dev`.

---

## Decision: Require Cursor Approval Before Editing Files

Date: 2026-05-28
Status: Accepted

### Context

Cursor can edit files quickly, but broad or unapproved changes can break the project.

The project is connected to GitHub and Render, so changes can have deployment impact.

### Decision

Cursor must not edit files until Juan explicitly says:

```text
approved, make the changes
```

A plan alone is not permission to edit.

### Reason

Juan is still learning Python and wants small, controlled changes.

This protects the project from wholesale edits, surprise refactors, and deployment-breaking changes.

### Impact

Cursor may inspect code, explain code, and propose plans before approval.

Cursor may not modify files until the approval phrase is given.

### Follow-up

This rule may be relaxed later for documentation-only changes if Juan chooses.

---

## Decision: Make Small Beginner-Friendly Changes

Date: 2026-05-28
Status: Accepted

### Context

Juan is learning Python and wants to understand the project as it grows.

### Decision

Make small changes, one logical step at a time.

### Reason

Small changes are easier to understand, test, and roll back.

They also reduce the chance of breaking the app.

### Impact

Avoid:

* large rewrites
* broad refactors
* surprise cleanup
* changing many files at once
* changing unrelated behavior

### Follow-up

If a large feature is needed, break it into smaller approved steps.

---

## Decision: Add Selected Run Detail Panel as the Next Major Feature

Date: 2026-05-28
Status: Proposed

### Context

The project creates JSON run summaries, but reviewing those summaries manually is inconvenient.

### Decision

The recommended next feature is a Selected Run Detail panel under Run History.

### Reason

This builds directly on existing run summaries and makes the app easier to use.

### Expected Display Fields

The panel should show:

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
* JSON summary reference

### Impact

Likely files affected:

* `api.py`
* `static/dashboard.html`
* JSON summary reading logic

### Follow-up

Before building, inspect the current run history and JSON summary structure.

---

## Future Decisions to Make

These decisions are not final yet.

| Topic          | Question                                                        |
| -------------- | --------------------------------------------------------------- |
| Database       | When are JSON summaries no longer enough?                       |
| Queue system   | Does batch processing need a real background job queue?         |
| Storage        | Where should large videos and outputs live long term?           |
| Authentication | Does the dashboard need login protection?                       |
| Profiles       | Should profiles be editable from the dashboard?                 |
| Warning rules  | Should warning rules be configurable per profile?               |
| Deployment     | Is Render sufficient for large video workloads?                 |
| Model choice   | Is YOLOv8 good enough for all target scenarios?                 |
| Reporting      | Should the app export PDF/CSV reports?                          |
| Security       | What file upload restrictions are needed before production use? |

## Current Decision Summary

| Area                        | Decision                              |
| --------------------------- | ------------------------------------- |
| Backend                     | FastAPI                               |
| Runner                      | Uvicorn                               |
| App entry point             | `api.py`                              |
| Dashboard                   | Plain static HTML/CSS/JS              |
| Dashboard file              | `static/dashboard.html`               |
| Object detection            | YOLOv8 through `ultralytics`          |
| Video processing            | OpenCV                                |
| Video conversion            | `imageio-ffmpeg` / FFmpeg             |
| LLM summaries               | OpenAI Python SDK                     |
| OpenAI environment variable | `OPENAI_API_KEY`                      |
| Runtime settings            | Configurable, not hard-coded          |
| Detection profiles          | Modular                               |
| Current storage             | JSON run summaries                    |
| Deployment                  | Render Web Service                    |
| Source control              | GitHub                                |
| Deployment branch           | `main`                                |
| Local project path          | `C:\dev\VERTEX`                       |
| Cursor editing rule         | Wait for `approved, make the changes` |
