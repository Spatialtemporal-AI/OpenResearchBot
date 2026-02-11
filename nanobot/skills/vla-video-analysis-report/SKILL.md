---
name: vla-video-analysis-report
description: Analyze robot-arm operation videos for embodied AI / VLA research by directly calling Gemini 2.5 Flash, then generate a structured Markdown diagnosis report with failure causes and optimization suggestions. Use when users provide local video paths or video URLs and ask for run analysis, failure attribution, post-mortem reports, training-data improvements, or next-experiment plans for real-world inference trials.
description: Analyze robot-arm operation videos for embodied AI / VLA research by directly calling Gemini 3 Flash with the Gemini Files API, then generate a structured Markdown diagnosis report with failure causes and optimization suggestions. Use when users provide local video paths or video URLs and ask for run analysis, failure attribution, post-mortem reports, training-data improvements, or next-experiment plans for real-world inference trials.
---

# VLA Video Analysis Report

## Overview

Use this skill to analyze full-scene robot-arm execution videos by direct video understanding.
Ingest local path or URL, compress only when needed for upload/compatibility, then call Gemini via official `google-genai` SDK (which internally uses Files API upload + processing + file reference) to generate a research-grade Markdown report.

## Workflow

1. Collect run context
- Ask for task goal, expected success condition, and any known run anomalies.
- Request input as local video path or URL.

2. Prepare video for Gemini
- Run `scripts/analyze_robot_video.py`.
- Use direct video understanding only.
- If file is large or not upload-friendly, it transcodes to an upload-ready MP4 while preserving resolution/time structure.

3. Generate structured report
- The script calls Gemini with a robotics-focused report prompt.
- Save output as Markdown report for researchers.
- Follow schema in `references/report-schema.md`.

4. Clean cache by default
- Temporary download/transcode/upload cache is removed automatically unless `--keep-cache` is set.
- Remote Gemini file is deleted by default unless `--keep-remote-file` is set.

## Quick Start

Set API key (do not hardcode secrets in files):

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

Install SDK once:

```bash
pip install google-genai
```

Run analysis on local video:

```bash
python3 nanobot/skills/vla-video-analysis-report/scripts/analyze_robot_video.py \
  --input "/absolute/path/to/robot_run.mp4" \
  --task "Pick up cube and place into bin" \
  --notes "Policy version vla_rtx_v4, real-robot trial #28"
```

Run analysis on URL video:

```bash
python3 nanobot/skills/vla-video-analysis-report/scripts/analyze_robot_video.py \
  --input "https://example.com/robot_run.mp4" \
  --task "Open drawer and place object inside"
```

Attach optional video metadata fps:

```bash
python3 nanobot/skills/vla-video-analysis-report/scripts/analyze_robot_video.py \
  --input "/absolute/path/to/robot_run.mp4" \
  --model "gemini-3-flash-preview"
```

Control compression threshold:

```bash
python3 nanobot/skills/vla-video-analysis-report/scripts/analyze_robot_video.py \
  --input "/absolute/path/to/robot_run.avi" \
  --size-threshold-mb 120 \
  --compression-crf 18 \
  --compression-preset slow
```

Use custom prompt file:

```bash
python3 nanobot/skills/vla-video-analysis-report/scripts/analyze_robot_video.py \
  --input "/absolute/path/to/robot_run.mp4" \
  --prompt-file "/absolute/path/to/custom_prompt.md"
```

## Decision Rules

- Use direct video understanding as the default and only analysis strategy.
- Trigger preparation when one of the following is true:
  - Input container is not `.mp4`
  - Video codec is not H.264
  - File size exceeds `--size-threshold-mb`
  - User forces `--force-compress`
- Keep quality high during preparation:
  - Default `--compression-crf 20`
  - Use `--compression-crf 18` for higher-fidelity analysis if file size allows.

## Output Expectations

- Always produce a Markdown report with:
  - Task restatement and success criteria
  - Execution timeline
  - Failure diagnosis and probable root causes
  - Prioritized optimization suggestions for model/data/control
  - Next experiment plan with pass/fail metrics
- Save run artifacts under `/Users/jikangyi/Downloads/nanobot/workspace/output/<session-id>/`.

## Scripts

- `scripts/analyze_robot_video.py`
  - Main pipeline: ingest -> optional prepare/compress -> `google-genai` `files.upload` -> wait until file active -> `models.generate_content` -> report.
  - Default behavior removes temporary cache after completion.
- `scripts/cleanup_video_session.py`
  - Manually delete remaining cache or full session.

## References

- `references/report-schema.md`: standard report sections and writing rules.
