import subprocess
from pathlib import Path
from typing import Dict

import imageio_ffmpeg


def convert_video_to_browser_mp4(
    input_video_path: str,
    output_video_path: str,
) -> Dict:
    """
    Convert an OpenCV-generated MP4 into a browser-friendly MP4.

    Browser-friendly target:
    - H.264 video codec
    - yuv420p pixel format
    - faststart metadata for browser playback

    This creates:
        processed_video_web.mp4

    It does not replace the original processed_video.mp4.
    """
    input_path = Path(input_video_path)
    output_path = Path(output_video_path)

    if not input_path.exists():
        return {
            "status": "error",
            "message": f"Input video does not exist: {input_path}",
            "input_video_path": str(input_path),
            "output_video_path": str(output_path),
            "error_message": "Input video not found.",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "fast",
        "-crf",
        "23",
        str(output_path),
    ]

    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed_process.returncode != 0:
            return {
                "status": "error",
                "message": "FFmpeg conversion failed.",
                "input_video_path": str(input_path),
                "output_video_path": str(output_path),
                "error_message": completed_process.stderr,
            }

        if not output_path.exists():
            return {
                "status": "error",
                "message": "FFmpeg finished but output file was not created.",
                "input_video_path": str(input_path),
                "output_video_path": str(output_path),
                "error_message": "Missing output file.",
            }

        return {
            "status": "ok",
            "message": "Browser-compatible video created.",
            "input_video_path": str(input_path),
            "output_video_path": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "error_message": None,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": "Unexpected video conversion error.",
            "input_video_path": str(input_path),
            "output_video_path": str(output_path),
            "error_message": str(error),
        }


def main() -> None:
    """
    Manual test helper.

    Example:
        python video_web_converter.py outputs/runs/sample/processed_video.mp4 outputs/runs/sample/processed_video_web.mp4
    """
    import json
    import sys

    if len(sys.argv) != 3:
        print("")
        print("Usage:")
        print("python video_web_converter.py <input_video_path> <output_video_path>")
        print("")
        return

    result = convert_video_to_browser_mp4(
        input_video_path=sys.argv[1],
        output_video_path=sys.argv[2],
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()