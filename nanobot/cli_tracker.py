"""Quick CLI for task_tracker & training_tracker (no agent needed).

Usage examples:
    # 📊 打开 HTML 可视化仪表盘（自动打开浏览器）
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
import sys
import webbrowser
from pathlib import Path


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
  dashboard               打开完整 HTML 仪表盘（任务 + 训练）

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


async def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return

    workspace = _find_workspace()

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
