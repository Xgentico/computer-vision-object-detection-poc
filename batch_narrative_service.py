import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_NARRATIVE_ENABLED,
    OPENAI_NARRATIVE_MODEL,
    OPENAI_NARRATIVE_MAX_WORDS,
    OPENAI_NARRATIVE_MAX_WARNING_EVENTS,
)


def get_iso_timestamp() -> str:
    """
    Return ISO-style timestamp for JSON output.
    """
    return datetime.now().isoformat(timespec="seconds")


def read_json_file(file_path: Path, default_value):
    """
    Read a JSON file if it exists.

    Returns:
        Parsed JSON data or default_value.
    """
    if not file_path.exists():
        return default_value

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(file_path: Path, data: Any) -> None:
    """
    Write JSON data with consistent formatting.
    """
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_warning_timeline_summary(warnings: List[Dict]) -> Dict:
    """
    Build a small warning timeline summary for the prompt and fallback text.

    This helps the LLM explain the run without sending excessive data.
    """
    if not warnings:
        return {
            "warning_count": 0,
            "first_warning_frame": None,
            "last_warning_frame": None,
            "longest_gap_between_warnings": None,
            "regions_seen": [],
        }

    sorted_warnings = sorted(
        warnings,
        key=lambda warning: int(warning.get("frame_number", 0) or 0),
    )

    frames = [
        int(warning.get("frame_number", 0) or 0)
        for warning in sorted_warnings
    ]

    regions_seen = sorted({
        warning.get("screen_region", "unknown")
        for warning in sorted_warnings
        if warning.get("screen_region")
    })

    longest_gap = 0

    if len(frames) > 1:
        for index in range(1, len(frames)):
            gap = frames[index] - frames[index - 1]

            if gap > longest_gap:
                longest_gap = gap

    return {
        "warning_count": len(sorted_warnings),
        "first_warning_frame": frames[0],
        "last_warning_frame": frames[-1],
        "longest_gap_between_warnings": longest_gap if len(frames) > 1 else 0,
        "regions_seen": regions_seen,
    }


def build_fallback_batch_narrative(summary: Dict, warnings: List[Dict]) -> str:
    """
    Create a local fallback narrative if OpenAI is unavailable.

    This keeps the UI useful even without quota or a key.
    """
    original_filename = summary.get("original_filename", "the video")
    frames_read = summary.get("frames_read", 0)
    frames_processed = summary.get("frames_processed", 0)
    process_every_n_frames = summary.get("process_every_n_frames", "unknown")
    unique_persons = summary.get("unique_persons", 0)
    warnings_generated = summary.get("warnings_generated", len(warnings))
    total_detection_counts = summary.get("total_detection_counts", {})

    timeline = build_warning_timeline_summary(warnings)

    narrative_parts = [
        f"Batch processing is complete for {original_filename}.",
        f"The system read {frames_read} frames and processed {frames_processed} sampled frames using a sampling rate of every {process_every_n_frames} frame(s).",
        f"The run identified {unique_persons} unique person(s) and generated {warnings_generated} warning event(s).",
    ]

    if total_detection_counts:
        counts_text = ", ".join(
            f"{class_name}: {count}"
            for class_name, count in total_detection_counts.items()
        )
        narrative_parts.append(f"Total detection counts were: {counts_text}.")

    if warnings:
        narrative_parts.append(
            f"The first warning occurred on frame {timeline.get('first_warning_frame')} and the last warning occurred on frame {timeline.get('last_warning_frame')}."
        )

        if timeline.get("longest_gap_between_warnings") is not None:
            narrative_parts.append(
                f"The longest gap between warning events was {timeline.get('longest_gap_between_warnings')} frame(s)."
            )

        if timeline.get("regions_seen"):
            narrative_parts.append(
                f"Warnings appeared in these screen regions: {', '.join(timeline.get('regions_seen'))}."
            )

    return " ".join(narrative_parts)


def build_batch_prompt(summary: Dict, warnings: List[Dict]) -> str:
    """
    Build a compact prompt for OpenAI.

    We keep this prompt small and grounded in the batch output files.
    """
    limited_warnings = warnings[:OPENAI_NARRATIVE_MAX_WARNING_EVENTS]
    timeline = build_warning_timeline_summary(warnings)

    prompt_payload = {
        "summary": {
            "run_id": summary.get("run_id"),
            "original_filename": summary.get("original_filename"),
            "status": summary.get("status"),
            "source": summary.get("source"),
            "model": summary.get("model"),
            "minimum_confidence": summary.get("minimum_confidence"),
            "process_every_n_frames": summary.get("process_every_n_frames"),
            "frames_read": summary.get("frames_read"),
            "frames_processed": summary.get("frames_processed"),
            "unique_persons": summary.get("unique_persons"),
            "warnings_generated": summary.get("warnings_generated"),
            "total_detection_counts": summary.get("total_detection_counts", {}),
            "selected_class_names": summary.get("selected_class_names", []),
        },
        "warning_timeline": timeline,
        "warnings_sample": [
            {
                "person_id": warning.get("person_id"),
                "frame_number": warning.get("frame_number"),
                "screen_region": warning.get("screen_region"),
                "confidence": warning.get("confidence"),
                "warning_text": warning.get("warning_text"),
            }
            for warning in limited_warnings
        ],
    }

    return (
        "You are summarizing the results of a computer vision batch video run. "
        "Use only the data provided. Do not invent facts. "
        "Write a clear operational summary for a business reviewer. "
        f"Keep it under {OPENAI_NARRATIVE_MAX_WORDS} words. "
        "Mention the original video name, total frames read, sampled frames processed, unique persons, warnings, and notable warning timing or locations. "
        "Use plain language and do not sound overly technical.\n\n"
        f"DATA:\n{json.dumps(prompt_payload, indent=2)}"
    )


def generate_openai_batch_narrative(summary: Dict, warnings: List[Dict]) -> Dict:
    """
    Generate the batch narrative using OpenAI.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = build_batch_prompt(summary, warnings)

    response = client.chat.completions.create(
        model=OPENAI_NARRATIVE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You summarize computer vision detection runs for operational review."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    narrative_text = response.choices[0].message.content.strip()

    return {
        "status": "ok",
        "message": "Batch narrative summary generated using OpenAI.",
        "model_name": OPENAI_NARRATIVE_MODEL,
        "openai_enabled": True,
        "narrative_text": narrative_text,
        "error_message": None,
    }


def generate_batch_narrative_for_run_folder(run_folder: str) -> Dict:
    """
    Generate or regenerate an LLM narrative summary for one batch output run.

    Expected run folder files:
    - summary.json
    - warnings.json

    Output:
    - narrative_summary.json
    """
    run_folder_path = Path(run_folder)

    if not run_folder_path.exists() or not run_folder_path.is_dir():
        return {
            "status": "not_found",
            "message": f"Batch run folder not found: {run_folder}",
            "run_folder": str(run_folder_path),
            "narrative_text": "",
            "error_message": "Run folder does not exist."
        }

    summary_path = run_folder_path / "summary.json"
    warnings_path = run_folder_path / "warnings.json"
    narrative_path = run_folder_path / "narrative_summary.json"

    summary = read_json_file(summary_path, {})
    warnings = read_json_file(warnings_path, [])

    if not isinstance(warnings, list):
        warnings = []

    if not summary:
        result = {
            "status": "error",
            "message": "Cannot generate narrative because summary.json is missing or empty.",
            "generated_at": get_iso_timestamp(),
            "run_folder": str(run_folder_path),
            "summary_path": str(summary_path),
            "warnings_path": str(warnings_path),
            "narrative_path": str(narrative_path),
            "narrative_text": "",
            "error_message": "Missing summary.json"
        }

        write_json_file(narrative_path, result)
        return result

    generated_at = get_iso_timestamp()

    try:
        if OPENAI_NARRATIVE_ENABLED and OPENAI_API_KEY:
            openai_result = generate_openai_batch_narrative(summary, warnings)

            result = {
                **openai_result,
                "generated_at": generated_at,
                "run_id": summary.get("run_id", run_folder_path.name),
                "original_filename": summary.get("original_filename"),
                "run_folder": str(run_folder_path),
                "summary_path": str(summary_path),
                "warnings_path": str(warnings_path),
                "narrative_path": str(narrative_path),
                "warning_count": len(warnings),
                "warning_timeline": build_warning_timeline_summary(warnings),
            }

        else:
            fallback_text = build_fallback_batch_narrative(summary, warnings)

            result = {
                "status": "fallback",
                "message": "OpenAI is not enabled. Local fallback batch narrative generated.",
                "generated_at": generated_at,
                "run_id": summary.get("run_id", run_folder_path.name),
                "original_filename": summary.get("original_filename"),
                "run_folder": str(run_folder_path),
                "summary_path": str(summary_path),
                "warnings_path": str(warnings_path),
                "narrative_path": str(narrative_path),
                "model_name": "local-fallback",
                "openai_enabled": False,
                "warning_count": len(warnings),
                "warning_timeline": build_warning_timeline_summary(warnings),
                "narrative_text": fallback_text,
                "error_message": None,
            }

    except Exception as error:
        fallback_text = build_fallback_batch_narrative(summary, warnings)

        result = {
            "status": "fallback",
            "message": "OpenAI batch narrative failed. Local fallback batch narrative generated.",
            "generated_at": generated_at,
            "run_id": summary.get("run_id", run_folder_path.name),
            "original_filename": summary.get("original_filename"),
            "run_folder": str(run_folder_path),
            "summary_path": str(summary_path),
            "warnings_path": str(warnings_path),
            "narrative_path": str(narrative_path),
            "model_name": "local-fallback",
            "openai_enabled": False,
            "warning_count": len(warnings),
            "warning_timeline": build_warning_timeline_summary(warnings),
            "narrative_text": fallback_text,
            "error_message": str(error),
        }

    write_json_file(narrative_path, result)

    return result


def read_existing_batch_narrative_for_run_folder(run_folder: str) -> Dict:
    """
    Read an existing narrative_summary.json for one batch run.

    If it does not exist, return a not_found response.
    """
    run_folder_path = Path(run_folder)
    narrative_path = run_folder_path / "narrative_summary.json"

    if not narrative_path.exists():
        return {
            "status": "not_found",
            "message": "No batch narrative summary has been generated yet.",
            "run_folder": str(run_folder_path),
            "narrative_path": str(narrative_path),
            "narrative_text": "",
            "error_message": None,
        }

    return read_json_file(narrative_path, {})


def main() -> None:
    """
    CLI test helper.

    Example:
        python batch_narrative_service.py outputs/runs/sample_driving_2026-05-22_17-13-43
    """
    import sys

    if len(sys.argv) != 2:
        print("")
        print("Usage:")
        print("python batch_narrative_service.py <batch_run_folder>")
        print("")
        print("Example:")
        print("python batch_narrative_service.py outputs/runs/sample_driving_2026-05-22_17-13-43")
        print("")
        return

    result = generate_batch_narrative_for_run_folder(sys.argv[1])

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()