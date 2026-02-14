"""Training tracker tool for VLA and general model training progress tracking."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.text_viz import (
    training_dashboard,
    compare_dashboard,
)
from nanobot.agent.tools.html_dashboard import generate_training_dashboard


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _short_id() -> str:
    return "run-" + uuid.uuid4().hex[:6]


class TrainingTrackerTool(Tool):
    """
    Tool for tracking model training runs, especially VLA (Vision-Language-Action) models.

    Maintains a structured JSON store of training runs with:
    - Model / dataset / hyperparameters
    - VLA-specific fields: action space, observation space, embodiment, environment
    - Status tracking (queued / running / completed / failed / stopped)
    - Metrics recording (loss, success_rate, or any custom metric)
    - Notes and checkpoints
    - Multi-run comparison
    """

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._store_path = workspace / "research" / "training_runs.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Tool interface ----------

    @property
    def name(self) -> str:
        return "training_tracker"

    @property
    def description(self) -> str:
        return (
            "Track model training runs (optimized for VLA models, also works for any model). "
            "Actions: create, update, log_metrics, list, detail, compare, delete, summary. "
            "Supports VLA-specific fields: action_space, observation_space, embodiment, environment."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create", "update", "log_metrics",
                        "list", "detail", "compare", "delete", "summary",
                        "visualize", "dashboard",
                    ],
                    "description": "Action to perform",
                },
                # Identifiers
                "run_id": {
                    "type": "string",
                    "description": "Training run ID (for update/log_metrics/detail/delete)",
                },
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of run IDs to compare (for compare)",
                },
                # Core fields (create / update)
                "name": {
                    "type": "string",
                    "description": "Human-readable name for this training run",
                },
                "model": {
                    "type": "string",
                    "description": "Model name/architecture, e.g. 'OpenVLA-7B', 'RT-2-base', 'Octo-small'",
                },
                "dataset": {
                    "type": "string",
                    "description": "Training dataset, e.g. 'Open-X-Embodiment', 'bridge_v2', 'RT-1-Robot-Action'",
                },
                "hyperparams": {
                    "type": "object",
                    "description": (
                        "Hyperparameters dict, e.g. "
                        '{"lr": 1e-4, "batch_size": 32, "epochs": 100, "optimizer": "AdamW", '
                        '"warmup_steps": 1000, "weight_decay": 0.01}'
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": ["queued", "running", "completed", "failed", "stopped"],
                    "description": "Training status",
                },
                "gpu_info": {
                    "type": "string",
                    "description": "GPU configuration, e.g. '4x A100 80GB'",
                },
                # VLA-specific fields
                "vla_config": {
                    "type": "object",
                    "description": (
                        "VLA-specific configuration. Fields: "
                        "action_space (str, e.g. '7-DoF delta EEF'), "
                        "observation_space (str, e.g. 'RGB 256x256 + proprioception'), "
                        "embodiment (str, e.g. 'Franka Panda'), "
                        "environment (str, e.g. 'real-world tabletop' or 'SIMPLER'), "
                        "task_suite (str, e.g. 'pick-and-place, drawer open/close'), "
                        "action_tokenizer (str, e.g. '256 bins per dim'), "
                        "backbone (str, e.g. 'Llama-2-7B' or 'PaLM-E')"
                    ),
                },
                # Metrics (for log_metrics)
                "metrics": {
                    "type": "object",
                    "description": (
                        "Metrics to log, e.g. "
                        '{"epoch": 45, "step": 12000, "loss": 0.234, "success_rate": 0.67, '
                        '"eval_loss": 0.31, "action_mse": 0.012}'
                    ),
                },
                # Misc
                "note": {
                    "type": "string",
                    "description": "Add a timestamped note to the run",
                },
                "checkpoint": {
                    "type": "string",
                    "description": "Record a checkpoint path/name, e.g. 'checkpoint_epoch50.pt'",
                },
                # Filters (for list)
                "status_filter": {
                    "type": "string",
                    "enum": ["queued", "running", "completed", "failed", "stopped", "all"],
                    "description": "Filter by status (for list, default: all)",
                },
                "model_filter": {
                    "type": "string",
                    "description": "Filter by model name substring (for list)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        actions = {
            "create": self._create,
            "update": self._update,
            "log_metrics": self._log_metrics,
            "list": self._list,
            "detail": self._detail,
            "compare": self._compare,
            "delete": self._delete,
            "summary": self._summary,
            "visualize": self._visualize,
            "dashboard": self._dashboard,
        }
        handler = actions.get(action)
        if not handler:
            return f"Error: unknown action '{action}'. Use: {', '.join(actions)}"
        return handler(**kwargs)

    # ---------- Storage ----------

    def _load(self) -> list[dict]:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                return data.get("runs", [])
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save(self, runs: list[dict]) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "runs": runs}
        self._store_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ---------- Actions ----------

    def _create(
        self,
        name: str = "",
        model: str = "",
        dataset: str = "",
        hyperparams: dict | None = None,
        status: str = "queued",
        gpu_info: str = "",
        vla_config: dict | None = None,
        note: str | None = None,
        **_: Any,
    ) -> str:
        if not name:
            return "Error: 'name' is required for create"

        runs = self._load()
        now = _now_iso()

        run: dict[str, Any] = {
            "id": _short_id(),
            "name": name,
            "model": model,
            "dataset": dataset,
            "hyperparams": hyperparams or {},
            "status": status,
            "gpu_info": gpu_info,
            "vla_config": vla_config or {},
            "metrics_history": [],
            "latest_metrics": {},
            "notes": [],
            "checkpoints": [],
            "created": now,
            "updated": now,
            "started": now if status == "running" else None,
            "finished": None,
        }

        if note:
            run["notes"].append({"time": now, "content": note})

        runs.append(run)
        self._save(runs)

        model_str = f", model={model}" if model else ""
        return f"🚀 Training run created: [{run['id']}] {name}{model_str} (status={status})"

    def _update(
        self,
        run_id: str = "",
        name: str | None = None,
        model: str | None = None,
        dataset: str | None = None,
        hyperparams: dict | None = None,
        status: str | None = None,
        gpu_info: str | None = None,
        vla_config: dict | None = None,
        note: str | None = None,
        checkpoint: str | None = None,
        **_: Any,
    ) -> str:
        if not run_id:
            return "Error: 'run_id' is required for update"

        runs = self._load()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            return f"Error: run '{run_id}' not found"

        changes = []
        if name is not None:
            run["name"] = name
            changes.append(f"name→{name}")
        if model is not None:
            run["model"] = model
            changes.append(f"model→{model}")
        if dataset is not None:
            run["dataset"] = dataset
            changes.append(f"dataset→{dataset}")
        if hyperparams is not None:
            run["hyperparams"].update(hyperparams)
            changes.append("hyperparams updated")
        if gpu_info is not None:
            run["gpu_info"] = gpu_info
            changes.append(f"gpu→{gpu_info}")
        if vla_config is not None:
            run.setdefault("vla_config", {}).update(vla_config)
            changes.append("vla_config updated")
        if status is not None:
            old = run["status"]
            run["status"] = status
            changes.append(f"status: {old}→{status}")
            # Auto-set timestamps
            if status == "running" and not run.get("started"):
                run["started"] = _now_iso()
            if status in ("completed", "failed", "stopped"):
                run["finished"] = _now_iso()
        if note:
            run.setdefault("notes", []).append({"time": _now_iso(), "content": note})
            changes.append("note added")
        if checkpoint:
            run.setdefault("checkpoints", []).append(checkpoint)
            changes.append(f"checkpoint: {checkpoint}")

        run["updated"] = _now_iso()
        self._save(runs)
        return f"✅ Run [{run_id}] updated: {', '.join(changes) if changes else 'no changes'}"

    def _log_metrics(
        self,
        run_id: str = "",
        metrics: dict | None = None,
        note: str | None = None,
        **_: Any,
    ) -> str:
        if not run_id:
            return "Error: 'run_id' is required for log_metrics"
        if not metrics:
            return "Error: 'metrics' dict is required for log_metrics"

        runs = self._load()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            return f"Error: run '{run_id}' not found"

        now = _now_iso()
        entry = {"time": now, **metrics}
        run.setdefault("metrics_history", []).append(entry)

        # Update latest metrics snapshot
        run.setdefault("latest_metrics", {}).update(metrics)

        if note:
            run.setdefault("notes", []).append({"time": now, "content": note})

        # Auto-set status to running if still queued
        if run.get("status") == "queued":
            run["status"] = "running"
            if not run.get("started"):
                run["started"] = now

        run["updated"] = now
        self._save(runs)

        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        return f"📈 Metrics logged for [{run_id}]: {metrics_str}"

    def _list(
        self,
        status_filter: str = "all",
        model_filter: str = "",
        **_: Any,
    ) -> str:
        runs = self._load()
        if not runs:
            return "No training runs found. Use action='create' to register one."

        # Filter
        if status_filter and status_filter != "all":
            runs = [r for r in runs if r.get("status") == status_filter]
        if model_filter:
            model_lower = model_filter.lower()
            runs = [r for r in runs if model_lower in r.get("model", "").lower()]

        if not runs:
            return "No runs match the filter."

        # Sort by updated desc
        runs.sort(key=lambda r: r.get("updated", ""), reverse=True)

        lines = ["🏋️ **Training Runs**\n"]
        status_emoji = {
            "queued": "⏳", "running": "🔄", "completed": "✅",
            "failed": "❌", "stopped": "⏹️",
        }

        for r in runs:
            emoji = status_emoji.get(r.get("status", ""), "❓")
            model_str = f" | {r['model']}" if r.get("model") else ""
            metrics_preview = ""
            lm = r.get("latest_metrics", {})
            if lm:
                parts = []
                # Show key metrics first
                for key in ["epoch", "step", "loss", "success_rate", "eval_loss", "action_mse"]:
                    if key in lm:
                        parts.append(f"{key}={lm[key]}")
                # Remaining metrics
                for key, val in lm.items():
                    if key not in ["epoch", "step", "loss", "success_rate", "eval_loss", "action_mse"]:
                        parts.append(f"{key}={val}")
                if parts:
                    metrics_preview = f" | {', '.join(parts[:5])}"

            is_vla = "🤖" if r.get("vla_config") else ""
            lines.append(
                f"  {emoji} [{r['id']}] {is_vla}{r.get('name', 'unnamed')}"
                f"{model_str}{metrics_preview}"
            )

        return "\n".join(lines)

    def _detail(self, run_id: str = "", **_: Any) -> str:
        if not run_id:
            return "Error: 'run_id' is required for detail"

        runs = self._load()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            return f"Error: run '{run_id}' not found"

        lines = [
            f"## 🏋️ Training Run: {run.get('name', 'unnamed')}",
            f"- **ID**: {run['id']}",
            f"- **Status**: {run.get('status', 'N/A')}",
            f"- **Model**: {run.get('model', 'N/A')}",
            f"- **Dataset**: {run.get('dataset', 'N/A')}",
            f"- **GPU**: {run.get('gpu_info', 'N/A')}",
            f"- **Created**: {run.get('created', 'N/A')}",
            f"- **Started**: {run.get('started', 'N/A')}",
            f"- **Finished**: {run.get('finished', 'N/A')}",
        ]

        # Hyperparameters
        hp = run.get("hyperparams", {})
        if hp:
            lines.append("\n**Hyperparameters**:")
            for k, v in hp.items():
                lines.append(f"  - {k}: {v}")

        # VLA config
        vla = run.get("vla_config", {})
        if vla:
            lines.append("\n**VLA Configuration** 🤖:")
            field_labels = {
                "action_space": "Action Space",
                "observation_space": "Observation Space",
                "embodiment": "Embodiment",
                "environment": "Environment",
                "task_suite": "Task Suite",
                "action_tokenizer": "Action Tokenizer",
                "backbone": "LLM/VLM Backbone",
            }
            for key, label in field_labels.items():
                if key in vla:
                    lines.append(f"  - {label}: {vla[key]}")
            # Any extra fields
            for k, v in vla.items():
                if k not in field_labels:
                    lines.append(f"  - {k}: {v}")

        # Latest metrics
        lm = run.get("latest_metrics", {})
        if lm:
            lines.append("\n**Latest Metrics**:")
            for k, v in lm.items():
                lines.append(f"  - {k}: {v}")

        # Metrics history (last 5 entries)
        mh = run.get("metrics_history", [])
        if mh:
            lines.append(f"\n**Metrics History** (last 5 of {len(mh)}):")
            for entry in mh[-5:]:
                t = entry.get("time", "")
                parts = [f"{k}={v}" for k, v in entry.items() if k != "time"]
                lines.append(f"  - [{t}] {', '.join(parts)}")

        # Checkpoints
        cps = run.get("checkpoints", [])
        if cps:
            lines.append(f"\n**Checkpoints** ({len(cps)}):")
            for cp in cps[-5:]:
                lines.append(f"  - {cp}")

        # Notes
        notes = run.get("notes", [])
        if notes:
            lines.append(f"\n**Notes** ({len(notes)}):")
            for n in notes[-5:]:
                lines.append(f"  - [{n.get('time', '')}] {n.get('content', '')}")

        return "\n".join(lines)

    def _compare(self, run_ids: list[str] | None = None, **_: Any) -> str:
        if not run_ids or len(run_ids) < 2:
            return "Error: provide at least 2 run_ids for compare"

        runs = self._load()
        selected = [r for r in runs if r["id"] in run_ids]
        if len(selected) < 2:
            found = [r["id"] for r in selected]
            return f"Error: found only {found}, need at least 2 runs to compare"

        # Collect all metric keys across selected runs
        all_metric_keys: list[str] = []
        for r in selected:
            for k in r.get("latest_metrics", {}):
                if k not in all_metric_keys:
                    all_metric_keys.append(k)

        # Build comparison table
        lines = ["## 📊 Training Comparison\n"]

        # Header
        header = "| Field |"
        sep = "|-------|"
        for r in selected:
            header += f" {r.get('name', r['id'])} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)

        # Basic info rows
        for field, label in [
            ("model", "Model"),
            ("dataset", "Dataset"),
            ("status", "Status"),
            ("gpu_info", "GPU"),
        ]:
            row = f"| {label} |"
            for r in selected:
                row += f" {r.get(field, 'N/A')} |"
            lines.append(row)

        # Hyperparameter comparison
        all_hp_keys: list[str] = []
        for r in selected:
            for k in r.get("hyperparams", {}):
                if k not in all_hp_keys:
                    all_hp_keys.append(k)
        for k in all_hp_keys:
            row = f"| hp.{k} |"
            for r in selected:
                row += f" {r.get('hyperparams', {}).get(k, '-')} |"
            lines.append(row)

        # VLA config comparison
        all_vla_keys: list[str] = []
        for r in selected:
            for k in r.get("vla_config", {}):
                if k not in all_vla_keys:
                    all_vla_keys.append(k)
        if all_vla_keys:
            for k in all_vla_keys:
                row = f"| vla.{k} |"
                for r in selected:
                    row += f" {r.get('vla_config', {}).get(k, '-')} |"
                lines.append(row)

        # Metrics comparison
        if all_metric_keys:
            lines.append(f"| **--- Metrics ---** |" + " |" * len(selected))
            for k in all_metric_keys:
                row = f"| {k} |"
                for r in selected:
                    val = r.get("latest_metrics", {}).get(k, "-")
                    row += f" {val} |"
                lines.append(row)

        # Duration
        row = "| Duration |"
        for r in selected:
            started = r.get("started")
            finished = r.get("finished")
            if started and finished:
                try:
                    s = datetime.strptime(started, "%Y-%m-%dT%H:%M:%S")
                    f = datetime.strptime(finished, "%Y-%m-%dT%H:%M:%S")
                    delta = f - s
                    hours = delta.total_seconds() / 3600
                    row += f" {hours:.1f}h |"
                except ValueError:
                    row += " N/A |"
            elif started:
                row += " (running) |"
            else:
                row += " N/A |"
        lines.append(row)

        return "\n".join(lines)

    def _delete(self, run_id: str = "", **_: Any) -> str:
        if not run_id:
            return "Error: 'run_id' is required for delete"

        runs = self._load()
        before = len(runs)
        runs = [r for r in runs if r["id"] != run_id]
        if len(runs) == before:
            return f"Error: run '{run_id}' not found"

        self._save(runs)
        return f"🗑️ Training run [{run_id}] deleted."

    def _summary(self, **_: Any) -> str:
        runs = self._load()
        if not runs:
            return "No training runs yet."

        total = len(runs)
        by_status: dict[str, int] = {}
        by_model: dict[str, int] = {}
        vla_count = 0

        for r in runs:
            s = r.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

            m = r.get("model", "unknown")
            by_model[m] = by_model.get(m, 0) + 1

            if r.get("vla_config"):
                vla_count += 1

        lines = [
            "📊 **Training Summary**",
            f"- Total runs: {total}  (VLA: {vla_count})",
        ]

        lines.append("- By status:")
        status_emoji = {
            "queued": "⏳", "running": "🔄", "completed": "✅",
            "failed": "❌", "stopped": "⏹️",
        }
        for s in ["running", "queued", "completed", "failed", "stopped"]:
            if s in by_status:
                lines.append(f"  - {status_emoji.get(s, '')} {s}: {by_status[s]}")

        lines.append("- By model:")
        for m, cnt in sorted(by_model.items(), key=lambda x: -x[1]):
            lines.append(f"  - {m}: {cnt}")

        # Best performing runs (by success_rate if available)
        completed = [r for r in runs if r.get("status") == "completed" and r.get("latest_metrics")]
        if completed:
            # Try to sort by success_rate, fallback to lowest loss
            with_sr = [r for r in completed if "success_rate" in r.get("latest_metrics", {})]
            if with_sr:
                best = max(with_sr, key=lambda r: r["latest_metrics"]["success_rate"])
                sr = best["latest_metrics"]["success_rate"]
                lines.append(f"\n🏆 Best success_rate: [{best['id']}] {best.get('name', '')} → {sr}")

            with_loss = [r for r in completed if "loss" in r.get("latest_metrics", {})]
            if with_loss:
                best_loss = min(with_loss, key=lambda r: r["latest_metrics"]["loss"])
                loss = best_loss["latest_metrics"]["loss"]
                lines.append(f"🏆 Lowest loss: [{best_loss['id']}] {best_loss.get('name', '')} → {loss}")

        return "\n".join(lines)

    def _visualize(
        self,
        run_id: str = "",
        run_ids: list[str] | None = None,
        **_: Any,
    ) -> str:
        """Generate a rich ASCII dashboard for training runs."""
        runs = self._load()

        # Compare mode: multiple runs
        if run_ids and len(run_ids) >= 2:
            selected = [r for r in runs if r["id"] in run_ids]
            if len(selected) < 2:
                found = [r["id"] for r in selected]
                return f"Error: found only {found}, need at least 2 runs to compare"
            return compare_dashboard(selected)

        # Single run mode
        if run_id:
            run = next((r for r in runs if r["id"] == run_id), None)
            if not run:
                return f"Error: run '{run_id}' not found"
            return training_dashboard(run)

        # No ID given: show dashboard for the most recent run
        if not runs:
            return "No training runs found."
        runs.sort(key=lambda r: r.get("updated", ""), reverse=True)
        return training_dashboard(runs[0])

    def _dashboard(
        self,
        run_id: str = "",
        run_ids: list[str] | None = None,
        **_: Any,
    ) -> str:
        """Generate an interactive HTML dashboard and return the file path."""
        path = generate_training_dashboard(
            self._workspace, run_id=run_id, run_ids=run_ids,
        )
        return f"📊 HTML dashboard generated: {path}\nOpen in browser to view interactive charts."
