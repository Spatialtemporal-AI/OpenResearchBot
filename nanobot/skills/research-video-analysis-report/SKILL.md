---
name: research-video-analysis-report
description: Research-assistant style video understanding for scientists and engineers. Accepts a local video path or URL, lets the model infer the domain during analysis, then calls Gemini via Files API for direct video understanding and generates a structured Markdown report with timeline, key observations, failure/risk analysis, and actionable next-step recommendations.
---

# Research Video Analysis Report

## Overview

Use this skill to analyze research videos by direct video understanding.
Ingest local path or URL, compress only when needed for upload/compatibility, then call Gemini via official `google-genai` SDK (which internally uses Files API upload + processing + file reference) to generate a research-grade Markdown report.

## Workflow

1. Collect run context
- Ask for task goal, expected success condition, and any known run anomalies.
- Request input as local video path or URL.

2. Prepare video for Gemini
- Run `scripts/analyze_research_video.py`.
- Use direct video understanding only.
- For URL input, the script first downloads the video with streaming HTTP (`requests`) into session cache.
- If file is large or not upload-friendly, it transcodes to an upload-ready MP4 while preserving resolution/time structure.

3. Generate structured report
- The script calls Gemini with a domain-adaptive research report prompt.
- Default prompt is template-driven from `assets/prompt_research_assistant.md` (`{{TASK}}`/`{{NOTES}}` placeholders).
- The model infers domain/task confidence inside the same analysis pass.
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
python3 nanobot/skills/research-video-analysis-report/scripts/analyze_research_video.py \
  --input "/absolute/path/to/experiment_run.mp4" \
  --task "Describe the experiment objective and success criteria" \
  --notes "Any relevant setup details, model/version info, or anomalies"
```

Run analysis on URL video:

```bash
python3 nanobot/skills/research-video-analysis-report/scripts/analyze_research_video.py \
  --input "https://example.com/research_video.mp4" \
  --task "Summarize key events and risks"
```

Control compression threshold:

```bash
python3 nanobot/skills/research-video-analysis-report/scripts/analyze_research_video.py \
  --input "/absolute/path/to/experiment_video.avi" \
  --size-threshold-mb 120 \
  --compression-crf 18 \
  --compression-preset slow
```

Use custom prompt file:

```bash
python3 nanobot/skills/research-video-analysis-report/scripts/analyze_research_video.py \
  --input "/absolute/path/to/experiment_video.mp4" \
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
- Save run artifacts under `nanobot/skills/research-video-analysis-report/output/<session-id>/`.

## Scripts

- `scripts/analyze_research_video.py`
  - Main pipeline: ingest -> optional prepare/compress -> `google-genai` `files.upload` -> wait until file active -> `models.generate_content` -> report.
  - Default behavior removes temporary cache after completion.
  - Manually delete remaining cache or full session.

## References

- `references/report-schema.md`: standard report sections and writing rules.
- `references/router-schema.md`: optional future two-stage routing schema (not required by current script).
