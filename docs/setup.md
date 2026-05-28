# Setup Guide — Computer Vision Object Detection POC

## Purpose

This guide explains how to set up and run the computer vision object detection project locally on Windows.

The project uses:

* FastAPI for the backend API
* Uvicorn to run the local web server
* Plain HTML, CSS, and JavaScript for the dashboard
* YOLOv8 through `ultralytics` for object detection
* OpenCV for video processing
* OpenAI Python SDK for optional LLM summaries
* GitHub for source control
* Render for deployment

## Project Location

Recommended local project location:

```text
C:\dev\VERTEX
```

Avoid keeping active coding projects inside OneDrive because OneDrive can slow down Python projects, virtual environments, generated video files, and Git operations.

## Expected Project Structure

The project root should look similar to this:

```text
VERTEX/
  .cursor/
  .venv/
  docs/
  input_videos/
  outputs/
  profiles/
  static/
  uploads/
  videos/
  .env
  .gitignore
  api.py
  batch_processor.py
  config.py
  detector.py
  llm_interpreter.py
  main.py
  openai_narrative_service.py
  requirements.txt
  runtime_settings.json
  runtime_settings.py
  video_input.py
  video_processor.py
  video_streamer.py
  video_web_converter.py
  warning_state.py
  yolov8n.pt
```

Some files or folders may change over time, but the important files for setup are:

| File or Folder          | Purpose                            |
| ----------------------- | ---------------------------------- |
| `api.py`                | FastAPI app entry point            |
| `requirements.txt`      | Python dependency list             |
| `static/dashboard.html` | Browser dashboard                  |
| `.env`                  | Local environment variables        |
| `.venv/`                | Local Python virtual environment   |
| `outputs/`              | Generated videos and run summaries |
| `profiles/`             | Detection profiles                 |
| `runtime_settings.json` | Runtime configuration              |

## Prerequisites

Install these before running the project:

1. Python
2. Git
3. Cursor
4. A web browser
5. FFmpeg support through project dependencies

Recommended Python version:

```text
Python 3.11 or newer
```

Do not use a very old Python version.

## Open the Project in Cursor

Open Cursor.

Then choose:

```text
File → Open Folder
```

Select:

```text
C:\dev\VERTEX
```

Do not open only one file. Open the whole project folder.

## Create the Virtual Environment

Open a terminal in Cursor or Windows PowerShell.

Make sure you are inside the project folder:

```powershell
cd C:\dev\VERTEX
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show something like:

```text
(.venv) PS C:\dev\VERTEX>
```

That means the virtual environment is active.

## If PowerShell Blocks Activation

If PowerShell blocks the virtual environment activation script, run this:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

With the virtual environment active, install the project dependencies:

```powershell
pip install -r requirements.txt
```

This installs the Python packages needed by the project.

Important packages may include:

* `fastapi`
* `uvicorn`
* `ultralytics`
* `opencv-python`
* `openai`
* `python-dotenv`
* `imageio-ffmpeg`

The actual installed packages are controlled by `requirements.txt`.

## Local Environment Variables

Create or update the local `.env` file in the project root:

```text
C:\dev\VERTEX\.env
```

Add this variable if LLM summaries are enabled:

```env
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

The `.env` file is for local development only.

## Important Security Rule

Never put the actual OpenAI API key into:

* `api.py`
* `static/dashboard.html`
* `requirements.txt`
* JSON run summaries
* logs
* Git commits
* screenshots
* documentation

The key should stay in `.env` locally and in Render environment variables for deployment.

## Run the Local API Server

From the project root, with the virtual environment active:

```powershell
python -m uvicorn api:app --reload
```

Expected result:

```text
Uvicorn running on http://127.0.0.1:8000
```

If the server starts successfully, the backend is running.

## Open the Dashboard

Open a browser and go to:

```text
http://127.0.0.1:8000
```

or, depending on how the app serves the dashboard:

```text
http://127.0.0.1:8000/dashboard
```

If unsure, check `api.py` for the route that serves `static/dashboard.html`.

## Basic Local Test

After starting the server:

1. Open the dashboard in the browser.
2. Select or upload a small MP4 file.
3. Choose runtime settings if available.
4. Start processing.
5. Confirm that the app processes the video.
6. Confirm an annotated output video is created.
7. Confirm a JSON run summary is created.
8. Confirm the dashboard still loads after processing.

## Expected Output Folders

Generated files may be saved in:

```text
outputs/
```

or another verified output folder used by the code.

Generated output may include:

* annotated videos
* JSON run summaries
* warning events
* LLM summaries

Do not commit generated outputs unless specifically approved.

## Runtime Settings

Runtime settings may be stored in:

```text
runtime_settings.json
```

Runtime settings may include:

* selected video path
* confidence threshold
* frame sampling value
* selected class IDs
* active profile

Do not hard-code runtime settings into processing logic unless approved.

## Running Batch Processing

Batch processing is handled by:

```text
batch_processor.py
```

Before running batch processing, confirm:

* input folder exists
* videos are valid MP4 files
* output folder exists or can be created
* runtime settings are correct
* enough disk space is available

A likely command may be:

```powershell
python batch_processor.py
```

Check the script before running it because command options may change over time.

## Git Workflow

Before making changes, check the current branch:

```powershell
git branch
```

The deployment branch is:

```text
main
```

Check current file changes:

```powershell
git status
```

Add files carefully:

```powershell
git add docs/setup.md
```

Commit changes:

```powershell
git commit -m "Add setup documentation"
```

Push only when ready:

```powershell
git push origin main
```

Remember: pushing to `main` may trigger a Render deployment.

## Files That Should Usually Not Be Committed

Make sure these are ignored or handled carefully:

```text
.venv/
.env
__pycache__/
outputs/
uploads/
input_videos/
videos/
*.mp4
*.avi
*.mov
*.mkv
```

Check `.gitignore` before committing.

## Render Deployment Settings

Known Render settings:

| Setting           | Value                                         |
| ----------------- | --------------------------------------------- |
| Runtime           | Python                                        |
| Service Type      | Web Service                                   |
| Build Command     | `pip install -r requirements.txt`             |
| Start Command     | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Deployment Branch | `main`                                        |

Do not change these unless the project architecture changes and Juan approves.

## Render Environment Variables

For OpenAI summaries, set this in Render:

```text
OPENAI_API_KEY
```

Do not put the actual value in GitHub.

Set it in Render's environment variable settings.

## Troubleshooting

### Problem: `python` is not recognized

Python may not be installed or may not be on the PATH.

Try:

```powershell
py --version
```

If that works, use:

```powershell
py -m venv .venv
```

instead of:

```powershell
python -m venv .venv
```

### Problem: Virtual environment will not activate

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Problem: Missing package error

Run:

```powershell
pip install -r requirements.txt
```

Make sure the virtual environment is active first.

### Problem: Server will not start

Check:

1. Are you in the project root?
2. Is `.venv` activated?
3. Did dependencies install?
4. Does `api.py` exist?
5. Does `api.py` contain `app = FastAPI(...)`?
6. Is another process already using port `8000`?

### Problem: Dashboard does not load

Check:

1. Is Uvicorn running?
2. Did you open the correct URL?
3. Does `static/dashboard.html` exist?
4. Does `api.py` serve the static dashboard?
5. Are there errors in the terminal?

### Problem: Video does not process

Check:

1. Is the video a valid MP4?
2. Does the video file exist?
3. Is the selected path correct?
4. Does the output folder exist?
5. Is YOLO model file available?
6. Are there OpenCV errors in the terminal?

### Problem: LLM summary does not work

Check:

1. Is `OPENAI_API_KEY` set in `.env` locally?
2. Is `OPENAI_API_KEY` set in Render for deployment?
3. Is the OpenAI package installed?
4. Is the app reading environment variables correctly?
5. Is the API key valid?

The video processing should still work even if the LLM summary fails.

## Quick Start Commands

Use these commands for a fresh local setup:

```powershell
cd C:\dev\VERTEX
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Setup Success Checklist

Setup is successful when:

* `.venv` activates
* dependencies install from `requirements.txt`
* Uvicorn starts without crashing
* dashboard opens in the browser
* a small MP4 can be processed
* an output video is created
* a JSON run summary is created
* no API keys are exposed in code or browser files

## Dependency Note

This project uses pinned dependency versions in `requirements.txt`. That helps make local and Render installs repeatable.

The computer vision dependencies are heavy, especially `torch`, `torchvision`, `opencv-python`, and `ultralytics`. Render builds may take longer because of these packages.