"""Background job tools for running long-lived commands asynchronously."""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


@dataclass
class BackgroundJob:
    """Represents a long-running background job."""

    id: str
    command: str
    working_dir: str
    channel: str
    chat_id: str
    status: str = "pending"  # pending | running | succeeded | failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

    def summary(self) -> str:
        parts: list[str] = []
        parts.append(f"Job ID: {self.id}")
        parts.append(f"Status: {self.status}")
        parts.append(f"Command: {self.command}")
        parts.append(f"Working dir: {self.working_dir}")
        parts.append(f"Created at (UTC): {self.created_at.isoformat()}")
        if self.started_at:
            parts.append(f"Started at (UTC): {self.started_at.isoformat()}")
        if self.finished_at:
            parts.append(f"Finished at (UTC): {self.finished_at.isoformat()}")
        if self.return_code is not None:
            parts.append(f"Exit code: {self.return_code}")
        if self.stdout:
            truncated = self._truncate(self.stdout)
            parts.append("STDOUT (tail):")
            parts.append(truncated)
        if self.stderr:
            truncated = self._truncate(self.stderr)
            parts.append("STDERR (tail):")
            parts.append(truncated)
        return "\n".join(parts)

    @staticmethod
    def _truncate(text: str, max_len: int = 4000) -> str:
        if len(text) <= max_len:
            return text
        return text[-max_len:]


class BackgroundJobManager:
    """In-memory manager for background jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, BackgroundJob] = {}
        self._lock = asyncio.Lock()
        self._max_output_len = 20000

    async def register(self, job: BackgroundJob) -> None:
        async with self._lock:
            self._jobs[job.id] = job

    async def get(self, job_id: str) -> Optional[BackgroundJob]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    async def append_output(self, job_id: str, field: str, text: str) -> None:
        """Append incremental output to stdout/stderr for a running job."""
        if not text:
            return
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            current = getattr(job, field, "") or ""
            combined = (current + text)[-self._max_output_len :]
            setattr(job, field, combined)


_JOB_MANAGER = BackgroundJobManager()


class BackgroundJobStartTool(Tool):
    """
    Start a long-running shell command as a background job.

    The command will continue running after this tool returns.
    You can query its status with the `background_job_status` tool.
    When the job finishes, a summary message will be sent to the user.
    """

    def __init__(
        self,
        bus: MessageBus,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._bus = bus
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str) -> None:
        """Bind this tool to the current conversation context."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    @property
    def name(self) -> str:
        return "background_job_start"

    @property
    def description(self) -> str:
        return (
            "Start a long-running shell command as a background job. "
            "Use background_job_status to check progress or results later."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run in the background",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified for background job notifications"

        cwd = working_dir or os.getcwd()

        job_id = uuid.uuid4().hex[:8]
        job = BackgroundJob(
            id=job_id,
            command=command,
            working_dir=cwd,
            channel=channel,
            chat_id=chat_id,
        )
        await _JOB_MANAGER.register(job)

        loop = asyncio.get_running_loop()
        loop.create_task(self._run_job(job_id))

        return (
            f"Started background job.\n"
            f"Job ID: {job_id}\n"
            f"Command: {command}\n"
            f"Working dir: {cwd}\n"
            f"Use background_job_status with this job_id to check progress."
        )

    async def _run_job(self, job_id: str) -> None:
        job = await _JOB_MANAGER.get(job_id)
        if not job:
            return

        await _JOB_MANAGER.update(
            job_id, status="running", started_at=datetime.utcnow()
        )

        try:
            process = await asyncio.create_subprocess_shell(
                job.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=job.working_dir,
                start_new_session=True,
            )

            # Incrementally stream stdout/stderr so that status queries
            # can see running output.
            async def _stream_output(
                stream: asyncio.StreamReader | None,
                field: str,
            ) -> None:
                if stream is None:
                    return
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        text = line.decode("utf-8", errors="replace")
                        await _JOB_MANAGER.append_output(job_id, field, text)
                except Exception:
                    # Best-effort streaming; ignore read errors
                    return

            stdout_task = asyncio.create_task(_stream_output(process.stdout, "stdout"))
            stderr_task = asyncio.create_task(_stream_output(process.stderr, "stderr"))

            await process.wait()
            await asyncio.gather(stdout_task, stderr_task)

            status = "succeeded" if process.returncode == 0 else "failed"

            await _JOB_MANAGER.update(
                job_id,
                status=status,
                finished_at=datetime.utcnow(),
                return_code=process.returncode,
            )

            # Notify main agent when finished
            await self._announce_result(job_id)

        except Exception as e:
            await _JOB_MANAGER.update(
                job_id,
                status="failed",
                finished_at=datetime.utcnow(),
                stderr=f"Error running background job: {e}",
            )
            await self._announce_result(job_id)

    async def _announce_result(self, job_id: str) -> None:
        """
        Announce the background job result back into the main agent via
        a system message, so the LLM can decide whether to重试、修改命令或仅告知用户。
        """
        job = await _JOB_MANAGER.get(job_id)
        if not job:
            return

        status_text = "completed successfully" if job.status == "succeeded" else "failed"

        announce_content = f"""[Background job {status_text}]

Job ID: {job.id}
Command: {job.command}

Result summary:
{job.summary()}

You are the main agent. Based on the background job result above, decide what to do next:
1. If the job succeeded, briefly explain the result to the user.
2. If the job failed, use the error information to decide whether you should automatically adjust and retry the command, or instead explain the failure to the user and suggest next steps.
3. Keep your reply concise and natural. Do not mention internal implementation details like \"tools\" or \"background jobs\"—just talk to the user in plain language."""

        msg = InboundMessage(
            channel="system",
            sender_id="job",
            chat_id=f"{job.channel}:{job.chat_id}",
            content=announce_content,
        )

        await self._bus.publish_inbound(msg)


class BackgroundJobStatusTool(Tool):
    """Check the status of a background job."""

    @property
    def name(self) -> str:
        return "background_job_status"

    @property
    def description(self) -> str:
        return (
            "Check the status and recent output of a background job started "
            "with background_job_start."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job ID returned by background_job_start",
                }
            },
            "required": ["job_id"],
        }

    async def execute(self, job_id: str, **kwargs: Any) -> str:
        job = await _JOB_MANAGER.get(job_id)
        if not job:
            return f"Error: No background job found with ID '{job_id}'."
        return job.summary()

