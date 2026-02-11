# Router Schema (Domain Detection)

The model should first infer the likely domain and analysis plan from a small set of sampled frames / or a short clip.

Return JSON with fields:

```json
{
  "domain": "free-text inferred domain label",
  "confidence": 0.0,
  "task_guess": "...",
  "setup_guess": {
    "environment": "...",
    "agents": ["..."],
    "sensors": ["..."],
    "notable_objects": ["..."]
  },
  "analysis_plan": {
    "focus": ["..."],
    "recommended_sampling": {
      "mode": "direct_video",
      "notes": "Gemini direct video understanding; adapt focus to inferred domain evidence"
    }
  },
  "uncertainties": ["..."]
}
```
