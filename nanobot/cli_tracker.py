"""Quick CLI for task_tracker & training_tracker (no agent needed).

Usage examples:
    # 🔴 启动实时仪表盘（推荐 — 自动刷新，训练时保持打开）
    python -m nanobot.cli_tracker live
    python -m nanobot.cli_tracker live --port 9000

    # 📊 打开静态 HTML 仪表盘（一次性快照）
    python -m nanobot.cli_tracker dashboard
    python -m nanobot.cli_tracker task dashboard
    python -m nanobot.cli_tracker train dashboard

    # 📋 文本模式（终端内显示）
    python -m nanobot.cli_tracker task visualize
    python -m nanobot.cli_tracker train visualize
    python -m nanobot.cli_tracker train visualize --run-id run-abc123
    python -m nanobot.cli_tracker train visualize --run-ids run-abc run-def
    python -m nanobot.cli_tracker task list
    python -m nanobot.cli_tracker train summary
"""

import asyncio
import json
import sys
import webbrowser
from pathlib import Path


def _safe_print(msg: str) -> None:
    """Print with fallback for terminals that can't handle emoji (e.g. Windows GBK)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        ))


def _find_workspace() -> Path:
    """Try to find the workspace directory."""
    cwd = Path.cwd()
    if (cwd / "workspace").exists():
        return cwd / "workspace"
    if (cwd / "SOUL.md").exists():
        return cwd
    return cwd


def _print_help():
    print("""
🔬 Research Tracker CLI

用法:
  python -m nanobot.cli_tracker <command> [options]

快捷命令:
  live [--port PORT]        🔴 启动实时仪表盘（默认端口 8765，自动刷新）
  dashboard                 打开完整 HTML 仪表盘（静态快照）

工具命令:
  python -m nanobot.cli_tracker <tool> <action> [options]

工具 (tool):
  task      任务追踪器
  train     训练追踪器

任务追踪器动作 (task actions):
  list                    查看任务列表（文本）
  summary                 查看任务总结（文本）
  visualize               可视化任务面板（文本）
  dashboard               打开 HTML 仪表盘
  detail --id <task_id>   查看任务详情

训练追踪器动作 (train actions):
  list                    查看训练列表（文本）
  summary                 查看训练总结（文本）
  visualize               可视化最近的训练（文本）
  dashboard               打开 HTML 仪表盘
  visualize --run-id <id>           可视化指定训练
  visualize --run-ids <id1> <id2>   对比多个训练
  detail --run-id <id>    查看训练详情

示例:
  python -m nanobot.cli_tracker live
  python -m nanobot.cli_tracker dashboard
  python -m nanobot.cli_tracker task visualize
  python -m nanobot.cli_tracker train dashboard
  python -m nanobot.cli_tracker train visualize --run-ids run-abc run-def
""")


def _open_in_browser(path: Path):
    """Open a file in the default browser."""
    url = path.resolve().as_uri()
    print(f"🌐 Opening in browser: {path}")
    webbrowser.open(url)


# ─── Live Dashboard Server ───────────────────────────────────────

def _run_live_server(workspace: Path, port: int = 8765):
    """Start a local HTTP server that serves a live-updating dashboard."""
    import http.server

    from nanobot.agent.tools.html_dashboard import generate_live_html, load_tracker_data

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                # Serve the live dashboard HTML (regenerated each request)
                html = generate_live_html(workspace)
                self._respond(200, "text/html", html.encode("utf-8"))

            elif self.path == "/api/data":
                # JSON endpoint polled by the dashboard every 3 seconds
                data = load_tracker_data(workspace)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._respond(200, "application/json", body)

            else:
                self.send_error(404)

        # ── helpers ──

        def _respond(self, code: int, content_type: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            pass  # suppress per-request logs

    server = http.server.HTTPServer(("0.0.0.0", port), _Handler)

    _safe_print(f"""
 Live Research Dashboard
 =======================
 URL:          http://localhost:{port}
 Auto-refresh: every 3 seconds
 Data from:    {workspace}
 Press Ctrl+C to stop
""")

    webbrowser.open(f"http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Live server stopped.")
        server.server_close()


# ─── Main ────────────────────────────────────────────────────────

async def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return

    workspace = _find_workspace()

    # ── Shortcut: `live` starts the live dashboard server ──
    if args[0] == "live":
        port = 8765
        for i, arg in enumerate(args[1:], 1):
            if arg == "--port" and i + 1 <= len(args) - 1:
                try:
                    port = int(args[i + 1])
                except ValueError:
                    print(f"❌ Invalid port: {args[i + 1]}")
                    return
        _run_live_server(workspace, port)
        return

    # ── Shortcut: `dashboard` opens full HTML dashboard ──
    if args[0] == "dashboard":
        from nanobot.agent.tools.html_dashboard import generate_dashboard
        path = generate_dashboard(workspace)
        _open_in_browser(path)
        return

    tool_name = args[0]
    action = args[1] if len(args) > 1 else "visualize"

    # Parse optional flags
    kwargs: dict = {}
    i = 2
    while i < len(args):
        if args[i] == "--id" and i + 1 < len(args):
            kwargs["task_id"] = args[i + 1]
            i += 2
        elif args[i] == "--run-id" and i + 1 < len(args):
            kwargs["run_id"] = args[i + 1]
            i += 2
        elif args[i] == "--run-ids":
            ids = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                ids.append(args[i])
                i += 1
            kwargs["run_ids"] = ids
        elif args[i] == "--status" and i + 1 < len(args):
            kwargs["status_filter"] = args[i + 1]
            i += 2
        elif args[i] == "--tag" and i + 1 < len(args):
            kwargs["tag_filter"] = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            kwargs["model_filter"] = args[i + 1]
            i += 2
        else:
            i += 1

    # ── Dashboard action → generate HTML and open browser ──
    if action == "dashboard":
        from nanobot.agent.tools.html_dashboard import (
            generate_dashboard,
            generate_task_dashboard,
            generate_training_dashboard,
        )
        if tool_name in ("task", "tasks"):
            path = generate_task_dashboard(workspace)
        elif tool_name in ("train", "training"):
            path = generate_training_dashboard(
                workspace,
                run_id=kwargs.get("run_id", ""),
                run_ids=kwargs.get("run_ids"),
            )
        else:
            path = generate_dashboard(workspace)
        _open_in_browser(path)
        return

    # ── Text-based actions ──
    if tool_name in ("task", "tasks"):
        from nanobot.agent.tools.task_tracker import TaskTrackerTool
        tool = TaskTrackerTool(workspace=workspace)
        result = await tool.execute(action=action, **kwargs)
        print(result)

    elif tool_name in ("train", "training"):
        from nanobot.agent.tools.training_tracker import TrainingTrackerTool
        tool = TrainingTrackerTool(workspace=workspace)
        result = await tool.execute(action=action, **kwargs)
        print(result)

    else:
        print(f"❌ 未知工具: {tool_name}")
        _print_help()


if __name__ == "__main__":
    asyncio.run(main())
