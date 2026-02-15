"""Baseline runner logging tool for recording experiment steps."""

from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class BaselineLoggerTool(Tool):
    """Tool to record baseline running steps and fixes."""
    
    def __init__(self, workspace: Path | None = None):
        """
        Initialize the logger tool.
        
        Args:
            workspace: Workspace directory where log file will be created.
                      If None, uses current working directory.
        """
        self.workspace = workspace or Path.cwd()
        self.log_file = self.workspace / "baseline_run_log.md"
    
    @property
    def name(self) -> str:
        return "record_baseline_step"
    
    @property
    def description(self) -> str:
        return (
            "Record a step, success, or fix in the baseline run log. "
            "Automatically creates baseline_run_log.md if it doesn't exist. "
            "Use this to maintain a real-time log of the baseline setup process."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to record (step description, error message, fix, etc.)"
                },
                "category": {
                    "type": "string",
                    "enum": ["step", "success", "error", "fix", "note"],
                    "description": "Category of the log entry: 'step' (normal step), 'success' (completed successfully), 'error' (error encountered), 'fix' (fix applied), 'note' (general note)",
                    "default": "step"
                }
            },
            "required": ["content"]
        }
    
    def _ensure_log_file(self) -> None:
        """Ensure log file exists with initial structure."""
        if not self.log_file.exists():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            initial_content = f"""# Baseline Run Log

Generated: {timestamp}

## Success Path

## Pitfalls & Fixes

---
"""
            self.log_file.write_text(initial_content, encoding="utf-8")
    
    def _format_entry(self, content: str, category: str) -> str:
        """Format a log entry based on category."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        icons = {
            "step": "📝",
            "success": "✅",
            "error": "❌",
            "fix": "🔧",
            "note": "ℹ️"
        }
        
        icon = icons.get(category, "📝")
        return f"\n{icon} **[{timestamp}]** {content}\n"
    
    async def execute(self, content: str, category: str = "step", **kwargs: Any) -> str:
        """
        Record a step in the baseline run log.
        
        Args:
            content: The content to record.
            category: Category of the log entry.
        
        Returns:
            Success message.
        """
        self._ensure_log_file()
        
        log_content = self.log_file.read_text(encoding="utf-8")
        entry = self._format_entry(content, category)
        
        # Append to appropriate section
        if category in ("success", "step"):
            # Add to Success Path section
            if "## Success Path" in log_content:
                # Find the end of Success Path section
                sections = log_content.split("## Pitfalls & Fixes")
                if len(sections) > 0:
                    success_section = sections[0]
                    # Count existing steps
                    step_count = success_section.count("✅") + success_section.count("📝")
                    numbered_entry = f"{step_count + 1}. {entry.strip()}\n"
                    new_success_section = success_section.rstrip() + "\n" + numbered_entry
                    log_content = new_success_section + "\n## Pitfalls & Fixes" + sections[1] if len(sections) > 1 else new_success_section
            else:
                log_content += f"\n## Success Path\n{entry}"
        elif category in ("error", "fix"):
            # Add to Pitfalls & Fixes section
            if "## Pitfalls & Fixes" in log_content:
                sections = log_content.split("## Pitfalls & Fixes", 1)
                if len(sections) > 1:
                    log_content = sections[0] + "## Pitfalls & Fixes" + sections[1] + entry
                else:
                    log_content += f"\n## Pitfalls & Fixes\n{entry}"
            else:
                log_content += f"\n## Pitfalls & Fixes\n{entry}"
        else:
            # General note, append at the end
            log_content += entry
        
        self.log_file.write_text(log_content, encoding="utf-8")
        return f"Recorded to {self.log_file.relative_to(Path.cwd())}"

