"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import re
import threading
from collections import OrderedDict
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# PatchMessage is used for updating "thinking..." card → final response.
# Older lark-oapi versions may not expose it; degrade gracefully.
FEISHU_PATCH_AVAILABLE = False
try:
    from lark_oapi.api.im.v1 import (
        PatchMessageRequest,
        PatchMessageRequestBody,
    )
    FEISHU_PATCH_AVAILABLE = True
except ImportError:
    PatchMessageRequest = None  # type: ignore[assignment,misc]
    PatchMessageRequestBody = None  # type: ignore[assignment,misc]

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}

# Default dashboard URL (overridden by feishu_bot.py at runtime)
_DASHBOARD_URL = "http://localhost:8765"

# Quick command mapping: shortcut → agent prompt
QUICK_COMMANDS: dict[str, str] = {
    "/help": (
        "请用中文列出你支持的所有功能和命令，包括任务管理、训练追踪、可视化等。"
        "以简洁的列表形式展示。"
    ),
    "/tasks": "请用 task_tracker 工具列出所有任务，以表格形式展示。",
    "/task_summary": "请用 task_tracker 工具显示任务总览统计。",
    "/trains": "请用 training_tracker 工具列出所有训练运行。",
    "/train_summary": "请用 training_tracker 工具显示训练总览统计。",
    "/dashboard": "",  # placeholder — filled dynamically in _match_quick_command
    "/status": (
        "请分别调用 task_tracker summary 和 training_tracker summary，"
        "给我一个综合的项目状态概览。"
    ),
}

# @mention pattern: <at user_id="xxx">name</at> or @_user_xxx
_AT_MENTION_RE = re.compile(
    r'<at user_id="[^"]*">[^<]*</at>\s*|@_user_\d+\s*|@\S+\s*',
    re.IGNORECASE,
)


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Features:
    - WebSocket long connection (no public IP needed)
    - Interactive card messages with markdown + table support
    - @mention handling in group chats
    - Quick command shortcuts (/help, /tasks, /trains, etc.)
    - "Thinking..." indicator while processing
    - Reaction emoji to acknowledge messages
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    # Set by feishu_bot.py at startup; used in /dashboard quick command
    dashboard_url: str = _DASHBOARD_URL
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
        # Track "thinking" card message IDs for later update
        self._thinking_cards: dict[str, str] = {}  # chat_reply_key → message_id
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread
        def run_ws():
            try:
                self._ws_client.start()
            except Exception as e:
                logger.error(f"Feishu WebSocket error: {e}")
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    # ── Reaction helpers ──────────────────────────────────────────────────
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    # ── "Thinking" indicator ──────────────────────────────────────────────

    def _send_thinking_card_sync(self, chat_id: str, receive_id_type: str) -> str | None:
        """Send a 'thinking...' card and return its message_id for later update."""
        try:
            card = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "OpenResearchBot"},
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": "**思考中...** 请稍候",
                    },
                ],
            }
            content = json.dumps(card, ensure_ascii=False)
            
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(content)
                    .build()
                ).build()
            
            response = self._client.im.v1.message.create(request)
            if response.success() and response.data and response.data.message_id:
                return response.data.message_id
            else:
                logger.warning(f"Failed to send thinking card: code={response.code}")
                return None
        except Exception as e:
            logger.warning(f"Error sending thinking card: {e}")
            return None

    def _update_card_sync(self, message_id: str, content: str) -> bool:
        """Update an existing interactive card message with new content.
        
        Returns True if successfully updated, False otherwise (caller should
        send a new message instead).
        """
        if not FEISHU_PATCH_AVAILABLE:
            return False
        try:
            elements = self._build_card_elements(content)
            card = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "OpenResearchBot"},
                    "template": "green",
                },
                "elements": elements,
            }
            card_json = json.dumps(card, ensure_ascii=False)
            
            request = PatchMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(card_json)
                    .build()
                ).build()
            
            response = self._client.im.v1.message.patch(request)
            if not response.success():
                logger.warning(
                    f"Failed to update card: code={response.code}, msg={response.msg}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Error updating card message: {e}")
            return False

    async def _send_thinking_indicator(self, chat_id: str) -> str | None:
        """Send a thinking indicator card and return its message_id."""
        if not self._client:
            return None
        receive_id_type = "chat_id" if chat_id.startswith("oc_") else "open_id"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._send_thinking_card_sync, chat_id, receive_id_type
        )

    async def _update_thinking_card(self, message_id: str, content: str) -> bool:
        """Update the thinking card with the final response. Returns True if updated."""
        if not self._client:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._update_card_sync, message_id, content)

    # ── Markdown table → Feishu table ─────────────────────────────────────

    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        split = lambda l: [c.strip() for c in l.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(l) for l in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})
        return elements or [{"tag": "markdown", "content": content}]

    # ── Send message ──────────────────────────────────────────────────────

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return
        
        try:
            # Check if we have a thinking card to update instead of sending a new message
            thinking_key = msg.chat_id
            thinking_msg_id = self._thinking_cards.pop(thinking_key, None)
            
            if thinking_msg_id:
                # Try to update existing thinking card with the response
                updated = await self._update_thinking_card(thinking_msg_id, msg.content)
                if updated:
                    logger.debug(f"Updated thinking card {thinking_msg_id} for {msg.chat_id}")
                    return
                # If update failed, fall through to send a new message
            
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"
            
            # Build card with markdown + table support
            elements = self._build_card_elements(msg.content)
            card = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "OpenResearchBot"},
                    "template": "green",
                },
                "elements": elements,
            }
            content = json.dumps(card, ensure_ascii=False)
            
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("interactive")
                    .content(content)
                    .build()
                ).build()
            
            response = self._client.im.v1.message.create(request)
            
            if not response.success():
                logger.error(
                    f"Failed to send Feishu message: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
            else:
                logger.debug(f"Feishu message sent to {msg.chat_id}")
                
        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")
    
    # ── Receive message ───────────────────────────────────────────────────

    @staticmethod
    def _strip_at_mentions(text: str) -> str:
        """Strip @mention tags from message text (common in group chats)."""
        return _AT_MENTION_RE.sub("", text).strip()

    def _handle_direct_command(self, text: str, reply_to: str) -> bool:
        """Handle commands that should be answered directly (no Agent).

        Returns True if the command was handled, False otherwise.
        """
        text_lower = text.strip().lower()

        if text_lower == "/dashboard" or text_lower.startswith("/dashboard "):
            url = self.__class__.dashboard_url
            content = (
                f"📊 **实时数据仪表盘**\n\n"
                f"点击打开：[{url}]({url})\n\n"
                f"仪表盘包含所有任务和训练数据的交互式图表，数据每 **3 秒自动刷新**。\n"
                f"请在电脑或手机浏览器中打开上方链接查看。"
            )
            # Send directly via Feishu API (no Agent, no hallucination)
            self._send_card_sync(reply_to, content)
            return True

        return False

    def _send_card_sync(self, chat_id: str, content: str) -> None:
        """Send a card message directly (synchronous, for use in direct commands)."""
        if not self._client:
            logger.warning("Feishu client not initialized, cannot send direct card")
            return
        try:
            if chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"

            elements = self._build_card_elements(content)
            card = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "OpenResearchBot"},
                    "template": "green",
                },
                "elements": elements,
            }
            card_json = json.dumps(card, ensure_ascii=False)

            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(card_json)
                    .build()
                ).build()

            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    f"Failed to send direct card: code={response.code}, "
                    f"msg={response.msg}"
                )
            else:
                logger.debug(f"Direct card sent to {chat_id}")
        except Exception as e:
            logger.error(f"Error sending direct card: {e}")

    @classmethod
    def _match_quick_command(cls, text: str) -> str | None:
        """Check if text matches a quick command. Returns the agent prompt or None."""
        text_lower = text.strip().lower()

        # /dashboard is now handled by _handle_direct_command — skip here
        if text_lower == "/dashboard" or text_lower.startswith("/dashboard "):
            return None

        # Exact match
        if text_lower in QUICK_COMMANDS:
            return QUICK_COMMANDS[text_lower]
        # Match with extra arguments (e.g. "/task create ...")
        for cmd, prompt in QUICK_COMMANDS.items():
            if text_lower.startswith(cmd + " "):
                extra = text[len(cmd):].strip()
                return f"{prompt}\n用户补充信息: {extra}"
        return None

    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "OnIt")
            
            # Parse message content
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
            
            if not content:
                return
            
            # Strip @mentions in group chat
            if chat_type == "group":
                content = self._strip_at_mentions(content)
                if not content:
                    return  # Message was only @mentions with no actual content
            
            # Determine reply target
            reply_to = chat_id if chat_type == "group" else sender_id

            # ── Direct commands (bypass Agent entirely) ──
            if self._handle_direct_command(content, reply_to):
                return

            # Check for quick commands (rewritten as Agent prompts)
            quick_prompt = self._match_quick_command(content)
            if quick_prompt:
                content = quick_prompt
            
            # Send "thinking..." indicator
            thinking_msg_id = await self._send_thinking_indicator(reply_to)
            if thinking_msg_id:
                self._thinking_cards[reply_to] = thinking_msg_id
            
            # Forward to message bus
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
