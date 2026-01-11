import asyncio
import itertools
import logging
import time
from typing import Optional

from modules.im import MessageContext

logger = logging.getLogger(__name__)


class StatusUpdater:
    """Periodically updates an acknowledgement message with status."""

    def __init__(
        self,
        im_client,
        context: MessageContext,
        message_id: str,
        base_text: str,
        interval_seconds: int = 10,
    ):
        self.im_client = im_client
        self.context = context
        self.message_id = message_id
        self.base_text = base_text
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.activity = "Starting"
        self._spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._last_text: Optional[str] = None

    def start(self) -> None:
        if self._task:
            return
        self._task = asyncio.create_task(self._run())

    def update_activity(self, activity: str) -> None:
        if activity:
            self.activity = activity

    async def stop(
        self,
        delete_message: bool = True,
        final_activity: Optional[str] = None,
    ) -> None:
        if final_activity:
            self.activity = final_activity
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if delete_message and hasattr(self.im_client, "delete_message"):
            try:
                await self.im_client.delete_message(
                    self.context.channel_id, self.message_id
                )
            except Exception as err:
                logger.debug(f"Failed to delete status message: {err}")
        elif not delete_message:
            await self._update_message()

    async def _run(self) -> None:
        try:
            await self._update_message()
            while True:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.interval_seconds
                    )
                    break
                except asyncio.TimeoutError:
                    await self._update_message()
        except asyncio.CancelledError:
            pass
        except Exception as err:
            logger.debug(f"Status updater failed: {err}")

    async def _update_message(self) -> None:
        if not hasattr(self.im_client, "edit_message"):
            return
        elapsed = self._format_elapsed(time.monotonic() - self.started_at)
        spinner = next(self._spinner)
        status_line = f"{spinner} {elapsed} • {self.activity}"
        text = f"{self.base_text}\n{status_line}"
        if text == self._last_text:
            return
        self._last_text = text
        try:
            await self.im_client.edit_message(self.context, self.message_id, text=text)
        except Exception as err:
            logger.debug(f"Failed to edit status message: {err}")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        if total_seconds < 60:
            return f"{total_seconds}s"
        minutes, secs = divmod(total_seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"
