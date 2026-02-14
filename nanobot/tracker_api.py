"""
Simple Python API for tracking research tasks and training runs.

Import this in your training scripts to automatically log progress
without going through the Agent — data updates in real time.

Usage:
    from nanobot.tracker_api import ResearchTracker

    tracker = ResearchTracker()

    # ── Tasks ──
    task_id = tracker.create_task("复现 OpenVLA 实验", priority="high", tags=["VLA"])
    tracker.update_task(task_id, status="doing")
    tracker.update_task(task_id, status="done", note="实验完成，success rate 78%")

    # ── Training ──
    run_id = tracker.create_run(
        name="OpenVLA-7B finetune",
        model="OpenVLA-7B",
        dataset="bridge_v2",
        hyperparams={"lr": 2e-5, "batch_size": 16, "epochs": 100},
        vla_config={"action_space": "7-DoF delta EEF", "embodiment": "WidowX"},
    )

    # In your training loop:
    for epoch in range(100):
        loss = train_one_epoch()
        tracker.log(run_id, epoch=epoch, loss=loss)

    tracker.finish_run(run_id)

    # ── Callback (for simple loops) ──
    cb = tracker.callback(run_id, log_every=10)
    for step in range(10000):
        loss = train_step()
        cb(step=step, loss=loss)   # only logs every 10 calls
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _short_id(prefix: str) -> str:
    return f"{prefix}-" + uuid.uuid4().hex[:6]


def _safe_print(msg: str) -> None:
    """Print with fallback for terminals that can't handle emoji (e.g. Windows GBK)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        ))


class ResearchTracker:
    """Standalone tracker API — use in training scripts, no Agent needed.

    Data is stored in JSON files under ``workspace/research/``, the same
    files that the Agent tools read.  Any change made here is instantly
    visible in the live dashboard (``python -m nanobot.cli_tracker live``).
    """

    def __init__(self, workspace: Optional[str | Path] = None):
        """Initialize tracker.

        Args:
            workspace: Path to the workspace directory.
                       If *None*, auto-detects by looking for ``./workspace``,
                       then ``~/.nanobot/workspace``.
        """
        if workspace:
            self._workspace = Path(workspace)
        else:
            self._workspace = self._find_workspace()

        self._tasks_path = self._workspace / "research" / "tasks.json"
        self._runs_path = self._workspace / "research" / "training_runs.json"
        self._tasks_path.parent.mkdir(parents=True, exist_ok=True)

        # Print workspace location on first use
        _safe_print(f"📂 Tracker workspace: {self._workspace}")

    # ── Workspace detection ──────────────────────────────────────

    @staticmethod
    def _find_workspace() -> Path:
        cwd = Path.cwd()
        # Check common locations
        if (cwd / "workspace").exists():
            return cwd / "workspace"
        if (cwd / "SOUL.md").exists():
            return cwd
        home_ws = Path.home() / ".nanobot" / "workspace"
        if home_ws.exists():
            return home_ws
        # Fallback: create workspace in cwd
        ws = cwd / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    # ── Storage helpers ──────────────────────────────────────────

    def _load_tasks(self) -> list[dict]:
        if self._tasks_path.exists():
            try:
                return json.loads(
                    self._tasks_path.read_text(encoding="utf-8")
                ).get("tasks", [])
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save_tasks(self, tasks: list[dict]) -> None:
        self._tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks_path.write_text(
            json.dumps({"version": 1, "tasks": tasks}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_runs(self) -> list[dict]:
        if self._runs_path.exists():
            try:
                return json.loads(
                    self._runs_path.read_text(encoding="utf-8")
                ).get("runs", [])
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save_runs(self, runs: list[dict]) -> None:
        self._runs_path.parent.mkdir(parents=True, exist_ok=True)
        self._runs_path.write_text(
            json.dumps({"version": 1, "runs": runs}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ══════════════════════════════════════════════════════════════
    #  TASK API
    # ══════════════════════════════════════════════════════════════

    def create_task(
        self,
        title: str,
        description: str = "",
        status: str = "todo",
        priority: str = "medium",
        tags: list[str] | None = None,
    ) -> str:
        """Create a research task. Returns *task_id*."""
        tasks = self._load_tasks()
        now = _now_iso()
        task_id = _short_id("t")
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "description": description,
                "status": status,
                "priority": priority,
                "tags": tags or [],
                "notes": [],
                "created": now,
                "updated": now,
            }
        )
        self._save_tasks(tasks)
        _safe_print(f"✅ Task created: [{task_id}] {title}")
        return task_id

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        priority: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
        note: str | None = None,
    ) -> None:
        """Update a task's status, priority, or add a note."""
        tasks = self._load_tasks()
        task = next((t for t in tasks if t["id"] == task_id), None)
        if not task:
            raise ValueError(f"Task '{task_id}' not found")

        changes: list[str] = []
        if status is not None:
            task["status"] = status
            changes.append(f"status→{status}")
        if priority is not None:
            task["priority"] = priority
            changes.append(f"priority→{priority}")
        if title is not None:
            task["title"] = title
            changes.append("title updated")
        if tags is not None:
            task["tags"] = tags
        if note:
            task.setdefault("notes", []).append(
                {"time": _now_iso(), "content": note}
            )
            changes.append("note added")

        task["updated"] = _now_iso()
        self._save_tasks(tasks)
        info = f" ({', '.join(changes)})" if changes else ""
        _safe_print(f"📋 Task [{task_id}] updated{info}")

    def get_task(self, task_id: str) -> dict | None:
        """Get a single task by ID."""
        tasks = self._load_tasks()
        return next((t for t in tasks if t["id"] == task_id), None)

    def list_tasks(self, status_filter: str = "all") -> list[dict]:
        """List tasks.  *status_filter*: todo / doing / done / blocked / all."""
        tasks = self._load_tasks()
        if status_filter and status_filter != "all":
            tasks = [t for t in tasks if t["status"] == status_filter]
        return tasks

    # ══════════════════════════════════════════════════════════════
    #  TRAINING API
    # ══════════════════════════════════════════════════════════════

    def create_run(
        self,
        name: str,
        model: str = "",
        dataset: str = "",
        hyperparams: dict | None = None,
        gpu_info: str = "",
        vla_config: dict | None = None,
        status: str = "running",
    ) -> str:
        """Create a training run.  Returns *run_id*."""
        runs = self._load_runs()
        now = _now_iso()
        run_id = _short_id("run")
        runs.append(
            {
                "id": run_id,
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
        )
        self._save_runs(runs)
        model_str = f" ({model})" if model else ""
        _safe_print(f"🚀 Run created: [{run_id}] {name}{model_str}")
        return run_id

    def log(self, run_id: str, **metrics: Any) -> None:
        """Log metrics for a training run.  Call this in your training loop.

        Example::

            tracker.log(run_id, epoch=1, loss=0.45, success_rate=0.72)
        """
        if not metrics:
            return
        runs = self._load_runs()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            raise ValueError(f"Run '{run_id}' not found")

        now = _now_iso()
        run.setdefault("metrics_history", []).append({"time": now, **metrics})
        run.setdefault("latest_metrics", {}).update(metrics)

        # Auto-transition queued → running
        if run.get("status") == "queued":
            run["status"] = "running"
            if not run.get("started"):
                run["started"] = now

        run["updated"] = now
        self._save_runs(runs)

        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        _safe_print(f"📈 [{run_id}] {metrics_str}")

    def finish_run(
        self, run_id: str, status: str = "completed", note: str = ""
    ) -> None:
        """Mark a training run as finished (*completed* / *failed* / *stopped*)."""
        runs = self._load_runs()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            raise ValueError(f"Run '{run_id}' not found")

        now = _now_iso()
        run["status"] = status
        run["finished"] = now
        run["updated"] = now
        if note:
            run.setdefault("notes", []).append({"time": now, "content": note})

        self._save_runs(runs)
        emoji = "✅" if status == "completed" else "⏹️"
        _safe_print(f"{emoji} Run [{run_id}] → {status}")

    def add_checkpoint(self, run_id: str, checkpoint: str) -> None:
        """Record a checkpoint path for a training run."""
        runs = self._load_runs()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            raise ValueError(f"Run '{run_id}' not found")

        run.setdefault("checkpoints", []).append(checkpoint)
        run["updated"] = _now_iso()
        self._save_runs(runs)
        _safe_print(f"💾 [{run_id}] Checkpoint: {checkpoint}")

    def add_note(self, run_id: str, note: str) -> None:
        """Add a note to a training run."""
        runs = self._load_runs()
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            raise ValueError(f"Run '{run_id}' not found")

        run.setdefault("notes", []).append({"time": _now_iso(), "content": note})
        run["updated"] = _now_iso()
        self._save_runs(runs)
        _safe_print(f"📝 [{run_id}] Note added")

    def get_run(self, run_id: str) -> dict | None:
        """Get a single training run by ID."""
        runs = self._load_runs()
        return next((r for r in runs if r["id"] == run_id), None)

    def list_runs(self, status_filter: str = "all") -> list[dict]:
        """List runs.  *status_filter*: queued / running / completed / failed / stopped / all."""
        runs = self._load_runs()
        if status_filter and status_filter != "all":
            runs = [r for r in runs if r.get("status") == status_filter]
        return runs

    # ══════════════════════════════════════════════════════════════
    #  CALLBACKS (for training loops)
    # ══════════════════════════════════════════════════════════════

    def callback(self, run_id: str, log_every: int = 1):
        """Return a callable that logs metrics every *log_every* calls.

        Usage::

            cb = tracker.callback(run_id, log_every=10)
            for step in range(10000):
                loss = train_step()
                cb(step=step, loss=loss)   # only writes every 10 steps
        """
        counter = [0]

        def _cb(**metrics: Any) -> None:
            counter[0] += 1
            if counter[0] % log_every == 0:
                self.log(run_id, **metrics)

        return _cb

    def epoch_callback(self, run_id: str):
        """Return a callable that logs metrics every call (epoch-level).

        Usage::

            on_epoch_end = tracker.epoch_callback(run_id)
            for epoch in range(100):
                loss = train_one_epoch()
                on_epoch_end(epoch=epoch, loss=loss, val_loss=evaluate())
        """

        def _cb(**metrics: Any) -> None:
            self.log(run_id, **metrics)

        return _cb
