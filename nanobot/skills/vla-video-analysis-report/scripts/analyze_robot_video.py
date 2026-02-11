#!/usr/bin/env python3
"""Analyze robot-arm videos with Gemini 3 Flash (google-genai SDK)."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_OUTPUT_ROOT = Path("./nanobot/workspace/output")
DEFAULT_SIZE_THRESHOLD_MB = 180.0
DEFAULT_MAX_WAIT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 3.0


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        rendered = " ".join(cmd)
        raise RuntimeError(f"Command failed ({rendered}):\n{result.stderr.strip()}")
    return result


def require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Missing required binary: {name}")
    return path


def require_genai():
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: google-genai. Install with `pip install google-genai`."
        ) from exc
    return genai


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_session_dir(output_root: Path, session_id: str | None) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not session_id:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = output_root / session_id
    if session_dir.exists():
        raise RuntimeError(f"Session already exists: {session_dir}")
    (session_dir / "cache").mkdir(parents=True, exist_ok=False)
    return session_dir


def ffprobe_metadata(video_path: Path, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run_cmd(cmd)
    return json.loads(result.stdout)


def download_via_ytdlp(url: str, cache_dir: Path) -> Path | None:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return None

    template = str(cache_dir / "source.%(ext)s")
    cmd = [
        ytdlp,
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "-o",
        template,
        url,
    ]
    try:
        run_cmd(cmd)
    except RuntimeError:
        return None

    candidates = [
        path
        for path in cache_dir.glob("source.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def download_via_ffmpeg(url: str, cache_dir: Path, ffmpeg_bin: str) -> Path:
    target = cache_dir / "source.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        url,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    run_cmd(cmd)
    return target


def _ascii_safe_copy(local_path: Path, cache_dir: Path) -> Path:
    """Ensure the local video path is ASCII-safe for downstream SDKs.

    Some SDK/tooling paths break when the filename contains non-ASCII characters.
    We copy the file into the session cache directory with an ASCII-only name and
    return that path.
    """
    try:
        local_path_str = str(local_path)
        local_path_str.encode("ascii")
        return local_path.resolve()
    except UnicodeEncodeError:
        pass

    suffix = local_path.suffix.lower() or ".mp4"
    safe_name = f"source_{int(time.time())}{suffix}"
    target = (cache_dir / safe_name).resolve()
    shutil.copy2(local_path, target)
    return target


def resolve_source(input_value: str, cache_dir: Path, ffmpeg_bin: str) -> tuple[str, Path]:
    local_path = Path(input_value).expanduser()
    if local_path.exists() and local_path.is_file():
        return "local", _ascii_safe_copy(local_path, cache_dir)

    if not is_url(input_value):
        raise RuntimeError(f"Input is neither a local file nor a valid URL: {input_value}")

    downloaded = download_via_ytdlp(input_value, cache_dir)
    if downloaded:
        return "url", downloaded.resolve()
    return "url", download_via_ffmpeg(input_value, cache_dir, ffmpeg_bin).resolve()


def select_primary_stream(metadata: dict, codec_type: str) -> dict | None:
    streams = metadata.get("streams", [])
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type:
            return stream
    return None


def should_prepare_video(
    source_path: Path,
    metadata: dict,
    size_threshold_bytes: int,
    force_compress: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    suffix = source_path.suffix.lower()
    video_stream = select_primary_stream(metadata, "video") or {}

    if suffix != ".mp4":
        reasons.append(f"container={suffix or '<none>'}")
    if video_stream.get("codec_name") != "h264":
        reasons.append(f"video_codec={video_stream.get('codec_name', 'unknown')}")
    if source_path.stat().st_size > size_threshold_bytes:
        reasons.append("size_over_threshold")
    if force_compress:
        reasons.append("force_compress")
    return bool(reasons), reasons


def prepare_video(
    source_path: Path,
    prepared_path: Path,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
) -> Path:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(prepared_path),
    ]
    run_cmd(cmd)
    return prepared_path.resolve()


def parse_file_state(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return "UNKNOWN"
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name.upper()
    text = str(state).strip()
    if text.startswith("FileState."):
        text = text.split(".", 1)[1]
    return text.upper()


def model_dump_compat(obj: Any) -> dict:
    """Best-effort conversion of SDK objects to JSON-serializable dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # type: ignore[no-any-return]
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"repr": repr(obj)}


def json_dump_safe(obj: Any) -> str:
    """json.dumps with fallback for non-serializable objects (e.g., datetime)."""
    return json.dumps(
        obj,
        indent=2,
        default=lambda x: x.isoformat() if hasattr(x, "isoformat") else repr(x),
    )


def upload_and_wait_active(
    client: Any,
    video_path: Path,
    max_wait_seconds: int,
    poll_interval_seconds: float,
) -> Any:
    uploaded = client.files.upload(file=str(video_path))
    state = parse_file_state(uploaded)
    if state in {"ACTIVE", "READY"}:
        return uploaded
    if state in {"FAILED", "ERROR"}:
        raise RuntimeError(f"Gemini file processing failed immediately: {state}")

    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        time.sleep(poll_interval_seconds)
        uploaded = client.files.get(name=uploaded.name)
        state = parse_file_state(uploaded)
        if state in {"ACTIVE", "READY"}:
            return uploaded
        if state in {"FAILED", "ERROR"}:
            raise RuntimeError(f"Gemini file processing failed: {state}")

    raise RuntimeError("Timed out waiting for uploaded file to become ACTIVE/READY.")


def build_default_prompt(task: str | None, notes: str | None) -> str:
    task_line = task.strip() if task else "Pick-and-place or manipulation validation task"
    notes_line = notes.strip() if notes else "No extra notes were provided."
    return f"""
You are analyzing a full-scene robot-arm execution video for embodied AI / VLA research.

Task objective:
{task_line}

Research notes:
{notes_line}

Return ONLY Markdown. Use this structure exactly:

# Robot Arm VLA Execution Analysis Report

## 1. Task And Success Criteria
- Restate the intended task.
- Define measurable success criteria.

## 2. Execution Timeline (Key Stages)
- Provide a concise stage-by-stage timeline.
- For each stage, include observed arm/tool/object behavior.

## 3. Failure Diagnosis
- State whether the run succeeded.
- If failed, identify primary failure point and direct evidence.
- Include root-cause hypotheses split into:
  - Perception / state estimation
  - Action generation / policy behavior
  - Motion planning / control
  - Gripper-contact and physics interaction
  - Environment or setup mismatch

## 4. Training/Data/Prompt Optimization Suggestions
- Give actionable suggestions prioritized by expected impact.
- Tie each suggestion to a diagnosed failure mechanism.
- Include what data to add, what policy behavior to constrain, and what evaluation to run next.

## 5. Next Experiment Plan
- Provide 3-5 concrete follow-up experiments with pass/fail criteria.
""".strip()


def read_prompt(prompt_file: Path | None, task: str | None, notes: str | None) -> str:
    if prompt_file:
        try:
            text = prompt_file.read_text().strip()
        except OSError as exc:
            raise RuntimeError(f"Cannot read prompt file {prompt_file}: {exc}") from exc
        if not text:
            raise RuntimeError(f"Prompt file is empty: {prompt_file}")
        return text
    return build_default_prompt(task, notes)


def write_report(report_path: Path, report_markdown: str, run_info: dict) -> Path:
    header = [
        "# Run Metadata",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Input: {run_info['input']}",
        f"- Source Type: {run_info['source_type']}",
        f"- Model: {run_info['model']}",
        f"- Prepared Video: {run_info['prepared_video_path']}",
        f"- Prepared Video Size (MB): {run_info['prepared_video_size_mb']:.2f}",
        "",
    ]
    content = "\n".join(header) + "\n" + report_markdown.strip() + "\n"
    report_path.write_text(content)
    return report_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local video path or URL")
    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Gemini API key (default: GEMINI_API_KEY env var)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--report-name", default="report.md")
    parser.add_argument("--task", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip Gemini calls")
    parser.add_argument("--keep-cache", action="store_true", help="Do not delete cache folder")
    parser.add_argument(
        "--keep-remote-file",
        action="store_true",
        help="Do not delete uploaded Gemini file after analysis",
    )
    parser.add_argument(
        "--size-threshold-mb",
        type=float,
        default=DEFAULT_SIZE_THRESHOLD_MB,
        help="Compress/prepare when source exceeds this size",
    )
    parser.add_argument("--force-compress", action="store_true")
    parser.add_argument("--compression-crf", type=int, default=20)
    parser.add_argument(
        "--compression-preset",
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-wait-seconds", type=int, default=DEFAULT_MAX_WAIT_SECONDS)
    parser.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.size_threshold_mb <= 0:
        print("Error: --size-threshold-mb must be > 0", file=sys.stderr)
        return 2
    if args.compression_crf < 0 or args.compression_crf > 51:
        print("Error: --compression-crf must be in [0, 51]", file=sys.stderr)
        return 2
    if args.max_wait_seconds <= 0:
        print("Error: --max-wait-seconds must be > 0", file=sys.stderr)
        return 2
    if args.poll_interval_seconds <= 0:
        print("Error: --poll-interval-seconds must be > 0", file=sys.stderr)
        return 2
    if not args.dry_run and not args.api_key:
        print("Error: missing API key. Set --api-key or GEMINI_API_KEY.", file=sys.stderr)
        return 2

    session_dir: Path | None = None
    run_summary: dict[str, object] = {"status": "error"}
    uploaded_name: str = ""
    client = None

    try:
        ffmpeg_bin = require_bin("ffmpeg")
        ffprobe_bin = require_bin("ffprobe")

        session_dir = resolve_session_dir(args.output_root, args.session_id)
        cache_dir = session_dir / "cache"
        report_path = session_dir / args.report_name

        source_type, source_video = resolve_source(args.input, cache_dir, ffmpeg_bin)
        source_metadata = ffprobe_metadata(source_video, ffprobe_bin)
        (session_dir / "source_metadata.json").write_text(json.dumps(source_metadata, indent=2) + "\n")

        threshold_bytes = int(args.size_threshold_mb * 1024 * 1024)
        needs_prepare, prepare_reasons = should_prepare_video(
            source_video, source_metadata, threshold_bytes, args.force_compress
        )
        if needs_prepare:
            prepared_video = prepare_video(
                source_video,
                cache_dir / "prepared.mp4",
                ffmpeg_bin,
                args.compression_crf,
                args.compression_preset,
            )
        else:
            prepared_video = source_video

        prepared_metadata = ffprobe_metadata(prepared_video, ffprobe_bin)
        (session_dir / "prepared_metadata.json").write_text(json.dumps(prepared_metadata, indent=2) + "\n")

        prompt = read_prompt(args.prompt_file, args.task, args.notes)
        (session_dir / "prompt.txt").write_text(prompt + "\n")

        run_info = {
            "input": args.input,
            "source_type": source_type,
            "source_video_path": str(source_video),
            "source_video_size_mb": source_video.stat().st_size / (1024 * 1024),
            "prepare_reasons": prepare_reasons,
            "prepared_video_path": str(prepared_video), 
            "prepared_video_size_mb": prepared_video.stat().st_size / (1024 * 1024),
            "model": args.model,
            "size_threshold_mb": args.size_threshold_mb,
        }

        if args.dry_run:
            dry_report = (
                "# Robot Arm VLA Execution Analysis Report\n\n"
                "Dry run enabled. Video preprocessing completed, but Gemini API call was skipped.\n"
            )
            write_report(report_path, dry_report, run_info)
            run_summary = {
                "status": "ok",
                "dry_run": True,
                "session_dir": str(session_dir),
                "report_path": str(report_path),
                "prepared_video_path": str(prepared_video),
                "prepare_reasons": prepare_reasons,
                "model": args.model,
            }
        else:
            genai = require_genai()
            client = genai.Client(api_key=args.api_key)
            uploaded = upload_and_wait_active(
                client,
                prepared_video,
                args.max_wait_seconds,
                args.poll_interval_seconds,
            )
            uploaded_name = str(getattr(uploaded, "name", ""))
            (session_dir / "upload_response.json").write_text(
                json_dump_safe(model_dump_compat(uploaded)) + "\n"
            )

            response = client.models.generate_content(
                model=args.model,
                contents=[uploaded, prompt],
                config={"temperature": 0.2, "max_output_tokens": args.max_output_tokens},
            )
            (session_dir / "model_response.json").write_text(
                json_dump_safe(model_dump_compat(response)) + "\n"
            )
            report_text = getattr(response, "text", "") or ""
            if not report_text.strip():
                raise RuntimeError("Model returned empty text response.")
            write_report(report_path, report_text, run_info)
            run_summary = {
                "status": "ok",
                "dry_run": False,
                "session_dir": str(session_dir),
                "report_path": str(report_path),
                "prepared_video_path": str(prepared_video),
                "prepare_reasons": prepare_reasons,
                "uploaded_file_name": uploaded_name,
                "model": args.model,
            }
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if session_dir:
            run_summary["session_dir"] = str(session_dir)
        run_summary["error"] = str(exc)
        print(json.dumps(run_summary, indent=2))
        return 1
    finally:
        if client and uploaded_name and not args.keep_remote_file and not args.dry_run:
            try:
                client.files.delete(name=uploaded_name)
            except Exception:
                pass
        if session_dir and not args.keep_cache:
            shutil.rmtree(session_dir / "cache", ignore_errors=True)
        if session_dir:
            manifest = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "args": {
                    "input": args.input,
                    "model": args.model,
                    "size_threshold_mb": args.size_threshold_mb,
                    "force_compress": args.force_compress,
                    "compression_crf": args.compression_crf,
                    "compression_preset": args.compression_preset,
                    "dry_run": args.dry_run,
                    "keep_cache": args.keep_cache,
                },
                "result": run_summary,
            }
            (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
