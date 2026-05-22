import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from batch_file_utils import (
    build_run_folder_name,
    ensure_batch_folders,
    get_completed_folder,
    get_failed_folder,
    get_output_runs_folder,
    get_pending_video_files,
    get_processing_folder,
)

from video_processor import process_video_file


def get_timestamp_string() -> str:
    """
    Return timestamp formatted for filenames and run IDs.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def print_header() -> None:
    """
    Print the batch processor header.
    """
    print("")
    print("=" * 60)
    print("VERTEX COMPUTER VISION BATCH PROCESSOR")
    print("=" * 60)
    print("")


def print_no_pending_files_message() -> None:
    """
    Print a helpful message when no pending files are found.
    """
    print("No pending MP4 files found.")
    print("")
    print("To test batch mode, place one or more .mp4 files here:")
    print("input_videos/pending/")
    print("")


def create_unique_destination_path(destination_folder: Path, filename: str) -> Path:
    """
    Build a destination path. If a file with the same name already exists,
    append a timestamp to avoid overwriting it.
    """
    destination_path = destination_folder / filename

    if not destination_path.exists():
        return destination_path

    timestamp = get_timestamp_string()
    original_path = Path(filename)

    return destination_folder / f"{original_path.stem}_{timestamp}{original_path.suffix}"


def move_file_safely(source_path: Path, destination_folder: Path) -> Path:
    """
    Move a file into a destination folder without overwriting an existing file.

    Returns:
        Final destination path.
    """
    destination_folder.mkdir(parents=True, exist_ok=True)

    destination_path = create_unique_destination_path(
        destination_folder=destination_folder,
        filename=source_path.name,
    )

    shutil.move(str(source_path), str(destination_path))

    return destination_path


def write_failure_summary(
    run_folder: Path,
    original_filename: str,
    error_message: str,
    traceback_text: Optional[str] = None,
) -> None:
    """
    Write a simple failure summary into the run folder if processing fails.
    """
    run_folder.mkdir(parents=True, exist_ok=True)

    failure_summary_path = run_folder / "summary.json"

    failure_summary = {
        "run_id": run_folder.name,
        "original_filename": original_filename,
        "status": "failed",
        "source": "batch_processor",
        "error_message": error_message,
        "traceback": traceback_text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    import json

    with failure_summary_path.open("w", encoding="utf-8") as file:
        json.dump(failure_summary, file, indent=2)


def process_pending_video(pending_video_path: Path) -> Dict:
    """
    Process one pending MP4 file.

    Workflow:
    - Move pending video to processing
    - Create output run folder
    - Process video
    - Move original video to completed if successful
    - Move original video to failed if unsuccessful
    """
    processing_folder = get_processing_folder()
    completed_folder = get_completed_folder()
    failed_folder = get_failed_folder()
    output_runs_folder = get_output_runs_folder()

    original_filename = pending_video_path.name
    timestamp = get_timestamp_string()
    run_folder_name = build_run_folder_name(pending_video_path, timestamp)
    run_folder = output_runs_folder / run_folder_name

    processing_video_path = None

    print("")
    print("-" * 60)
    print(f"Starting video: {original_filename}")
    print(f"Run folder: {run_folder}")
    print("-" * 60)

    try:
        processing_video_path = move_file_safely(
            source_path=pending_video_path,
            destination_folder=processing_folder,
        )

        print(f"Moved to processing: {processing_video_path}")

        summary = process_video_file(
            video_path=str(processing_video_path),
            output_folder=str(run_folder),
            runtime_settings=None,
            generate_narrative=False,
        )

        completed_video_path = move_file_safely(
            source_path=processing_video_path,
            destination_folder=completed_folder,
        )

        print(f"Moved to completed: {completed_video_path}")

        return {
            "status": "completed",
            "original_filename": original_filename,
            "processing_video_path": str(processing_video_path),
            "final_video_path": str(completed_video_path),
            "run_folder": str(run_folder),
            "summary": summary,
            "error_message": None,
        }

    except Exception as error:
        error_message = str(error)
        traceback_text = traceback.format_exc()

        print("")
        print("ERROR processing video:")
        print(original_filename)
        print(error_message)
        print("")

        write_failure_summary(
            run_folder=run_folder,
            original_filename=original_filename,
            error_message=error_message,
            traceback_text=traceback_text,
        )

        failed_video_path = None

        try:
            if processing_video_path and Path(processing_video_path).exists():
                failed_video_path = move_file_safely(
                    source_path=Path(processing_video_path),
                    destination_folder=failed_folder,
                )
            elif pending_video_path.exists():
                failed_video_path = move_file_safely(
                    source_path=pending_video_path,
                    destination_folder=failed_folder,
                )

            if failed_video_path:
                print(f"Moved to failed: {failed_video_path}")

        except Exception as move_error:
            print(f"WARNING: Could not move failed file. Error: {move_error}")

        return {
            "status": "failed",
            "original_filename": original_filename,
            "processing_video_path": str(processing_video_path) if processing_video_path else None,
            "final_video_path": str(failed_video_path) if failed_video_path else None,
            "run_folder": str(run_folder),
            "summary": None,
            "error_message": error_message,
        }


def print_batch_summary(results: List[Dict]) -> None:
    """
    Print a final batch summary after all pending videos have been attempted.
    """
    completed_results = [
        result for result in results
        if result.get("status") == "completed"
    ]

    failed_results = [
        result for result in results
        if result.get("status") == "failed"
    ]

    output_runs_folder = get_output_runs_folder()

    print("")
    print("=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print("")
    print(f"Videos found: {len(results)}")
    print(f"Completed: {len(completed_results)}")
    print(f"Failed: {len(failed_results)}")
    print("")

    if completed_results:
        print("Completed files:")

        for result in completed_results:
            print(f"- {result.get('original_filename')}")

        print("")

    if failed_results:
        print("Failed files:")

        for result in failed_results:
            print(f"- {result.get('original_filename')}: {result.get('error_message')}")

        print("")

    print("Output folder:")
    print(output_runs_folder)
    print("")


def main() -> None:
    """
    Entry point for local batch processing.

    Current Task 5 behavior:
    - Create batch folder structure
    - Scan input_videos/pending for .mp4 files
    - Process each pending MP4 file
    - Create a unique output run folder
    - Move successful originals to completed
    - Move failed originals to failed
    - Print final batch summary
    """
    print_header()

    ensure_batch_folders()

    pending_files = get_pending_video_files()

    if not pending_files:
        print_no_pending_files_message()
        print_batch_summary([])
        return

    print(f"Pending MP4 files found: {len(pending_files)}")
    print("")

    for index, file_path in enumerate(pending_files, start=1):
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        print(f"{index}. {file_path.name}")
        print(f"   Path: {file_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print("")

    results = []

    for pending_video_path in pending_files:
        result = process_pending_video(pending_video_path)
        results.append(result)

    print_batch_summary(results)


if __name__ == "__main__":
    main()