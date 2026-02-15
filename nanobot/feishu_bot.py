"""
Standalone Feishu bot entry point.

Usage:
    python -m nanobot.feishu_bot
    python -m nanobot.feishu_bot --dashboard-port 9000

This starts only the Feishu channel + agent loop, without launching
other channels (Telegram, WhatsApp, etc.) or the full gateway.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path

from loguru import logger


# ── Live Dashboard Server (background) ──────────────────────────

def _start_dashboard_server(workspace: Path, port: int = 8765) -> bool:
    """Start live dashboard HTTP server in a background daemon thread.

    Returns True if the server started successfully, False if the port
    is already in use.
    """
    import http.server

    from nanobot.agent.tools.html_dashboard import generate_live_html, load_tracker_data

    class _DashboardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                html = generate_live_html(workspace)
                self._respond(200, "text/html", html.encode("utf-8"))
            elif self.path == "/api/data":
                data = load_tracker_data(workspace)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._respond(200, "application/json", body)
            else:
                self.send_error(404)

        def _respond(self, code: int, content_type: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # noqa: A002
            pass  # suppress per-request logs

    try:
        server = http.server.HTTPServer(("0.0.0.0", port), _DashboardHandler)
    except OSError:
        return False

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return True


def _get_lan_ip() -> str:
    """Get the local LAN IP address (best effort)."""
    import socket
    try:
        # Connect to an external address to determine the outbound interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── Main ────────────────────────────────────────────────────────

def main(dashboard_port: int = 8765, verbose: bool = False) -> None:
    """Start the Feishu bot with the agent loop."""
    from nanobot import __logo__
    from nanobot.bus.queue import MessageBus
    from nanobot.config.loader import load_config, get_data_dir
    from nanobot.config.schema import Config
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.feishu import FeishuChannel, FEISHU_AVAILABLE
    from nanobot.channels.manager import ChannelManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.session.manager import SessionManager
    from nanobot.providers.litellm_provider import LiteLLMProvider

    if not FEISHU_AVAILABLE:
        logger.error(
            "lark-oapi SDK not installed. Run:\n"
            "  pip install lark-oapi>=1.0.0"
        )
        sys.exit(1)

    print(f"{__logo__} Starting OpenResearchBot Feishu Bot...")

    # Load config
    config = load_config()

    if not config.channels.feishu.enabled:
        logger.error(
            "Feishu channel is not enabled in config.\n"
            "Edit ~/.nanobot/config.json and set channels.feishu.enabled = true\n"
            "along with appId / appSecret."
        )
        sys.exit(1)

    if not config.channels.feishu.app_id or not config.channels.feishu.app_secret:
        logger.error(
            "Feishu appId / appSecret not configured.\n"
            "Edit ~/.nanobot/config.json under channels.feishu section."
        )
        sys.exit(1)

    # Provider
    p = config.get_provider()
    model = config.agents.defaults.model
    if not (p and p.api_key) and not model.startswith("bedrock/"):
        logger.error("No API key configured. Set one in ~/.nanobot/config.json")
        sys.exit(1)

    provider = LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=config.get_provider_name(),
    )

    bus = MessageBus()
    session_manager = SessionManager(config.workspace_path)

    # Cron service
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Agent loop
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
    )

    # Cron callback
    async def on_cron_job(job: CronJob) -> str | None:
        response = await agent.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "feishu",
            chat_id=job.payload.to or "direct",
        )
        if job.payload.deliver and job.payload.to:
            from nanobot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "feishu",
                chat_id=job.payload.to,
                content=response or ""
            ))
        return response

    cron.on_job = on_cron_job

    # Feishu channel — inject dashboard URL (use LAN IP so phone can access)
    lan_ip = _get_lan_ip()
    dashboard_url_lan = f"http://{lan_ip}:{dashboard_port}"
    dashboard_url_local = f"http://localhost:{dashboard_port}"
    FeishuChannel.dashboard_url = dashboard_url_lan
    feishu_channel = FeishuChannel(config.channels.feishu, bus)

    # Inject live dashboard URL into tracker tools so they return
    # the live URL instead of static file paths
    from nanobot.agent.tools.task_tracker import TaskTrackerTool
    from nanobot.agent.tools.training_tracker import TrainingTrackerTool
    TaskTrackerTool.live_dashboard_url = dashboard_url_lan
    TrainingTrackerTool.live_dashboard_url = dashboard_url_lan

    # Outbound dispatcher (route agent replies to Feishu)
    async def dispatch_outbound():
        logger.info("Feishu outbound dispatcher started")
        while True:
            try:
                msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                if msg.channel == "feishu":
                    try:
                        await feishu_channel.send(msg)
                    except Exception as e:
                        logger.error(f"Error sending Feishu message: {e}")
                else:
                    logger.debug(f"Ignoring outbound for channel: {msg.channel}")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # Start live dashboard HTTP server in background
    dashboard_ok = _start_dashboard_server(config.workspace_path, dashboard_port)

    print(f"  Model: {model}")
    print(f"  Workspace: {config.workspace_path}")
    print(f"  Feishu App ID: {config.channels.feishu.app_id[:8]}...")
    print(f"  WebSocket mode (no public IP required)")
    if dashboard_ok:
        print(f"  Live Dashboard (local):  {dashboard_url_local}")
        print(f"  Live Dashboard (LAN):    {dashboard_url_lan}")
    else:
        print(f"  Dashboard: port {dashboard_port} in use, skipped")
    print()
    print("Quick commands available in Feishu chat:")
    print("  /help        - Show all features")
    print("  /tasks       - List tasks")
    print("  /trains      - List training runs")
    print("  /status      - Project overview")
    print("  /dashboard   - Open live dashboard")
    print()
    print("Bot is running! Send a message in Feishu to interact.")
    print("Press Ctrl+C to stop.\n")

    async def run():
        try:
            await cron.start()
            await asyncio.gather(
                agent.run(),
                feishu_channel.start(),
                dispatch_outbound(),
            )
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cron.stop()
            agent.stop()
            await feishu_channel.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenResearchBot Feishu Bot")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--dashboard-port", type=int, default=8765,
        help="Live dashboard HTTP port (default: 8765)",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    main(dashboard_port=args.dashboard_port, verbose=args.verbose)
