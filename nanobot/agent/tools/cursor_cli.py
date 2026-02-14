"""Tools for integrating with Cursor CLI (Agent).

These tools allow the main agent to optionally delegate analysis
to Cursor CLI's Agent in Ask/Plan modes, e.g. to figure out how
to run code in a given project directory or which changes are
needed before running.
"""

import asyncio
import os
import shutil
from typing import Any

from nanobot.agent.tools.base import Tool


class CursorCliAskTool(Tool):
    """
    Use Cursor CLI Agent in Ask/Plan mode to analyze a project.
    
    Typical use cases:
    - You want to run code in a given folder but are unsure which command to use.
    - You suspect code changes are required before it can run.
    - You prefer to get an external second opinion from Cursor Agent.
    
    This tool does NOT run the target program itself. It only calls
    `agent` (Cursor CLI) with the given question and returns its answer.
    """

    @property
    def name(self) -> str:
        return "cursor_cli_ask"

    @property
    def description(self) -> str:
        return (
            "Use Cursor CLI Agent (agent) in Ask/Plan/Agent mode to analyze or "
            "modify code in a given project directory. "
            "Use this when you are unsure which command to run, when the code "
            "might need changes before it can run, or when you want Cursor "
            "Agent to directly edit code (Agent mode)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_dir": {
                    "type": "string",
                    "description": "Absolute or relative path to the project directory to analyze",
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Your question to Cursor Agent, e.g. "
                        "\"Figure out how to run the training script in this folder.\""
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["agent", "ask", "plan"],
                    "description": (
                        "Cursor Agent mode to use. "
                        "\"agent\" (default) = full Agent mode with code edits, "
                        "\"ask\" = explore/answer only, "
                        "\"plan\" = help design a plan."
                    ),
                },
            },
            "required": ["project_dir", "question"],
        }

    async def execute(
        self,
        project_dir: str,
        question: str,
        mode: str | None = None,
        **kwargs: Any,
    ) -> str:
        # 1. Check Cursor CLI (agent) availability
        agent_path = shutil.which("agent")
        if not agent_path:
            return (
                "Error: Cursor CLI (agent) is not installed or not on PATH.\n\n"
                "To install Cursor CLI on macOS/Linux/WSL:\n"
                "  curl https://cursor.com/install -fsS | bash\n\n"
                "Then make sure ~/.local/bin is in your PATH, e.g. for bash:\n"
                '  echo \'export PATH=\"$HOME/.local/bin:$PATH\"\' >> ~/.bashrc\n'
                "  source ~/.bashrc\n\n"
                "After installation, verify with:\n"
                "  agent --version\n"
            )

        # 2. Normalize project directory
        cwd = os.path.abspath(project_dir)
        if not os.path.isdir(cwd):
            return f"Error: project_dir '{project_dir}' does not exist or is not a directory."

        # 3. Build agent CLI command in non-interactive mode
        # Default to full Agent mode so it can directly modify code.
        mode = mode or "agent"
        if mode not in ("agent", "ask", "plan"):
            mode = "agent"

        cmd = [
            agent_path,
            f"--mode={mode}",
            "--print",
            "--output-format",
            "text",
            "-p",
            question,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            parts: list[str] = []
            if stdout.strip():
                parts.append(stdout.strip())
            if stderr.strip():
                parts.append("STDERR (from agent):")
                parts.append(stderr.strip())
            if process.returncode != 0:
                parts.append(f"(agent exit code: {process.returncode})")

            result = "\n\n".join(parts) if parts else "(no output from agent)"

            # Truncate very long responses
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n\n... (truncated, {len(result) - max_len} more chars)"

            return result

        except Exception as e:
            return f"Error calling Cursor CLI (agent): {e}"

