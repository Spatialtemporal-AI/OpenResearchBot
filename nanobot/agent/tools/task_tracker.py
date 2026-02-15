"""Task tracker tool for managing research tasks and unfinished work."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.text_viz import task_dashboard
from nanobot.agent.tools.html_dashboard import generate_task_dashboard


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _short_id() -> str:
    return "t-" + uuid.uuid4().hex[:6]


class TaskTrackerTool(Tool):
    """
    Tool for tracking research tasks and unfinished work.

    Maintains a structured JSON store of tasks with status management
    (todo / doing / done / blocked), priority, tags, and notes.
    """

    # Set by feishu_bot.py at startup; when set, _dashboard returns live URL
    live_dashboard_url: str | None = None

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._store_path = workspace / "research" / "tasks.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Tool interface ----------

    @property
    def name(self) -> str:
        return "task_tracker"

    @property
    def description(self) -> str:
        return (
            "Track research tasks and unfinished work. "
            "Actions: create, update, list, detail, delete, summary. "
            "Each task has a status (todo/doing/done/blocked), priority (low/medium/high), "
            "tags, and timestamped notes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "list", "detail", "delete", "summary"],
                    "description": "Action to perform",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (for update / detail / delete)",
                },
                "title": {
                    "type": "string",
                    "description": "Task title (for create)",
                },
                "description": {
                    "type": "string",
                    "description": "Task description (for create / update)",
                },
                "status": {
                    "type": "string",
                    "enum": ["todo", "doing", "done", "blocked"],
                    "description": "Task status (for create / update)",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Task priority (for create / update)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization, e.g. ['VLA','baseline'] (for create / update)",
                },
                "note": {
                    "type": "string",
                    "description": "Append a timestamped note to the task (for update)",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["todo", "doing", "done", "blocked", "all"],
                    "description": "Filter by status (for list, default: all non-done)",
                },
                "tag_filter": {
                    "type": "string",
                    "description": "Filter by tag (for list)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        actions = {
            "create": self._create,
            "update": self._update,
            "list": self._list,
            "detail": self._detail,
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
                return data.get("tasks", [])
            except (json.JSONDecodeError, KeyError):
                return []
        return []

    def _save(self, tasks: list[dict]) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "tasks": tasks}
        self._store_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ---------- Actions ----------

    def _create(
        self,
        title: str = "",
        description: str = "",
        status: str = "todo",
        priority: str = "medium",
        tags: list[str] | None = None,
        **_: Any,
    ) -> str:
        if not title:
            return "Error: 'title' is required for create"

        tasks = self._load()
        now = _now_iso()
        task = {
            "id": _short_id(),
            "title": title,
            "description": description,
            "status": status,
            "priority": priority,
            "tags": tags or [],
            "notes": [],
            "created": now,
            "updated": now,
        }
        tasks.append(task)
        self._save(tasks)
        return f"✅ Task created: [{task['id']}] {title} (status={status}, priority={priority})"

    def _update(
        self,
        task_id: str = "",
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        tags: list[str] | None = None,
        note: str | None = None,
        **_: Any,
    ) -> str:
        if not task_id:
            return "Error: 'task_id' is required for update"

        tasks = self._load()
        task = next((t for t in tasks if t["id"] == task_id), None)
        if not task:
            return f"Error: task '{task_id}' not found"

        changes = []
        if title is not None:
            task["title"] = title
            changes.append(f"title→{title}")
        if description is not None:
            task["description"] = description
            changes.append("description updated")
        if status is not None:
            old = task["status"]
            task["status"] = status
            changes.append(f"status: {old}→{status}")
        if priority is not None:
            task["priority"] = priority
            changes.append(f"priority→{priority}")
        if tags is not None:
            task["tags"] = tags
            changes.append(f"tags→{tags}")
        if note:
            task.setdefault("notes", []).append({"time": _now_iso(), "content": note})
            changes.append(f"note added")

        task["updated"] = _now_iso()
        self._save(tasks)
        return f"✅ Task [{task_id}] updated: {', '.join(changes) if changes else 'no changes'}"

    def _list(
        self,
        status_filter: str = "",
        tag_filter: str = "",
        **_: Any,
    ) -> str:
        tasks = self._load()
        if not tasks:
            return "No tasks found. Use action='create' to add one."

        # Filter
        if status_filter and status_filter != "all":
            tasks = [t for t in tasks if t["status"] == status_filter]
        elif not status_filter:
            # Default: show non-done tasks
            tasks = [t for t in tasks if t["status"] != "done"]

        if tag_filter:
            tasks = [t for t in tasks if tag_filter in t.get("tags", [])]

        if not tasks:
            return "No tasks match the filter."

        # Sort: high→medium→low, then by updated desc
        prio_order = {"high": 0, "medium": 1, "low": 2}
        tasks.sort(key=lambda t: (prio_order.get(t.get("priority", "medium"), 1), t.get("updated", "")))

        lines = []
        # Group by status
        for status_label in ["doing", "blocked", "todo", "done"]:
            group = [t for t in tasks if t["status"] == status_label]
            if not group:
                continue
            emoji = {"doing": "🔵", "blocked": "🔴", "todo": "⚪", "done": "✅"}.get(status_label, "")
            lines.append(f"\n### {emoji} {status_label.upper()} ({len(group)})")
            for t in group:
                prio_tag = {"high": "🔥", "medium": "", "low": "💤"}.get(t.get("priority", ""), "")
                tag_str = f" [{', '.join(t.get('tags', []))}]" if t.get("tags") else ""
                lines.append(f"  - [{t['id']}] {prio_tag}{t['title']}{tag_str}")

        return "📋 **Research Tasks**" + "\n".join(lines)

    def _detail(self, task_id: str = "", **_: Any) -> str:
        if not task_id:
            return "Error: 'task_id' is required for detail"

        tasks = self._load()
        task = next((t for t in tasks if t["id"] == task_id), None)
        if not task:
            return f"Error: task '{task_id}' not found"

        lines = [
            f"## Task: {task['title']}",
            f"- **ID**: {task['id']}",
            f"- **Status**: {task['status']}",
            f"- **Priority**: {task['priority']}",
            f"- **Tags**: {', '.join(task.get('tags', [])) or 'none'}",
            f"- **Created**: {task.get('created', 'N/A')}",
            f"- **Updated**: {task.get('updated', 'N/A')}",
        ]
        if task.get("description"):
            lines.append(f"\n**Description**:\n{task['description']}")
        if task.get("notes"):
            lines.append("\n**Notes**:")
            for n in task["notes"]:
                lines.append(f"  - [{n.get('time', '')}] {n.get('content', '')}")

        return "\n".join(lines)

    def _delete(self, task_id: str = "", **_: Any) -> str:
        if not task_id:
            return "Error: 'task_id' is required for delete"

        tasks = self._load()
        before = len(tasks)
        tasks = [t for t in tasks if t["id"] != task_id]
        if len(tasks) == before:
            return f"Error: task '{task_id}' not found"

        self._save(tasks)
        return f"🗑️ Task [{task_id}] deleted."

    def _summary(self, **_: Any) -> str:
        tasks = self._load()
        if not tasks:
            return "No tasks yet."

        total = len(tasks)
        by_status = {}
        for t in tasks:
            s = t.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

        by_priority = {}
        for t in tasks:
            if t.get("status") != "done":
                p = t.get("priority", "medium")
                by_priority[p] = by_priority.get(p, 0) + 1

        lines = [
            "📊 **Task Summary**",
            f"- Total: {total}",
        ]
        for s in ["todo", "doing", "done", "blocked"]:
            if s in by_status:
                lines.append(f"  - {s}: {by_status[s]}")

        active = total - by_status.get("done", 0)
        if active > 0:
            lines.append(f"- Active (non-done): {active}")
            for p in ["high", "medium", "low"]:
                if p in by_priority:
                    lines.append(f"  - {p} priority: {by_priority[p]}")

        return "\n".join(lines)

    def _visualize(
        self,
        status_filter: str = "",
        tag_filter: str = "",
        **_: Any,
    ) -> str:
        """Generate a rich ASCII dashboard for tasks."""
        tasks = self._load()
        if not tasks:
            return "📋 No tasks to visualize."

        # Apply filters
        if status_filter and status_filter != "all":
            tasks = [t for t in tasks if t.get("status") == status_filter]
        if tag_filter:
            tasks = [t for t in tasks if tag_filter in t.get("tags", [])]

        if not tasks:
            return "No tasks match the filter."

        return task_dashboard(tasks)

    def _dashboard(self, **_: Any) -> str:
        """Return the live dashboard URL if available, otherwise generate a static HTML file."""
        url = self.__class__.live_dashboard_url
        if url:
            return (
                f"📊 实时仪表盘正在运行，请将以下链接发送给用户（保留链接格式）：\n"
                f"[点击打开实时仪表盘]({url})\n"
                f"数据每 3 秒自动刷新，包含所有训练和任务数据的交互式图表。\n"
                f"请勿返回本地文件路径，用户无法直接打开本地文件。"
            )
        # Fallback: generate static HTML (CLI usage)
        path = generate_task_dashboard(self._workspace)
        return f"📊 HTML dashboard generated: {path}\nOpen in browser to view interactive charts."
