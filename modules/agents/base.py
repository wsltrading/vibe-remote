"""Abstract agent interfaces and shared dataclasses."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from modules.im import MessageContext

if TYPE_CHECKING:
    from core.status_updater import StatusUpdater

logger = logging.getLogger(__name__)


@dataclass
class AgentRequest:
    """Normalized agent invocation request."""

    context: MessageContext
    message: str
    working_path: str
    base_session_id: str
    composite_session_id: str
    settings_key: str
    ack_message_id: Optional[str] = None
    status_updater: Optional["StatusUpdater"] = None
    last_agent_message: Optional[str] = None
    last_agent_message_parse_mode: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)


@dataclass
class AgentMessage:
    """Normalized message emitted by an agent implementation."""

    text: str
    message_type: str = "assistant"
    parse_mode: str = "markdown"
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class for all agent implementations."""

    name: str

    def __init__(self, controller):
        self.controller = controller
        self.config = controller.config
        self.im_client = controller.im_client
        self.settings_manager = controller.settings_manager

    def _calculate_duration_ms(self, started_at: Optional[float]) -> int:
        if not started_at:
            return 0
        elapsed = time.monotonic() - started_at
        return max(0, int(elapsed * 1000))

    async def emit_result_message(
        self,
        context: MessageContext,
        result_text: Optional[str],
        subtype: str = "success",
        duration_ms: Optional[int] = None,
        started_at: Optional[float] = None,
        parse_mode: str = "markdown",
        suffix: Optional[str] = None,
    ) -> None:
        if duration_ms is None:
            duration_ms = self._calculate_duration_ms(started_at)
        formatted = self.im_client.formatter.format_result_message(
            subtype or "", duration_ms, result_text
        )
        if suffix:
            formatted = f"{formatted}\n{suffix}"
        await self.controller.emit_agent_message(
            context, "result", formatted, parse_mode=parse_mode
        )

    async def _finalize_ack(
        self,
        request: AgentRequest,
        delete_message: bool = True,
        final_activity: Optional[str] = None,
    ) -> None:
        if request.status_updater:
            await request.status_updater.stop(
                delete_message=delete_message,
                final_activity=final_activity,
            )
            request.status_updater = None
            request.ack_message_id = None
            return
        if delete_message:
            await self._delete_ack_message(request)

    async def _delete_ack_message(self, request: AgentRequest) -> None:
        ack_id = request.ack_message_id
        if ack_id and hasattr(self.im_client, "delete_message"):
            try:
                await self.im_client.delete_message(request.context.channel_id, ack_id)
            except Exception as err:
                logger.debug(f"Could not delete ack message: {err}")
            finally:
                request.ack_message_id = None

    @abstractmethod
    async def handle_message(self, request: AgentRequest) -> None:
        """Process a user message routed to this agent."""

    async def clear_sessions(self, settings_key: str) -> int:
        """Clear session state for a given settings key. Returns cleared count."""
        return 0

    async def handle_stop(self, request: AgentRequest) -> bool:
        """Attempt to interrupt an in-flight task. Returns True if handled."""
        return False
