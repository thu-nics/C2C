from typing import Optional
from contextlib import contextmanager
from rich.console import Console
from rich.markup import escape

class StatusLogger:
    """Handles console status display with spinner and history."""

    def __init__(self, enabled: bool = True, max_queue_preview: Optional[int] = None):
        self.console = Console() if enabled else None
        self.max_queue_preview = max_queue_preview

    @staticmethod
    def _shorten(text: Optional[str], max_len: int = 240) -> Optional[str]:
        if not text:
            return None
        s = " ".join(text.strip().split())
        return (s[: max_len - 1] + "…") if len(s) > max_len else s

    @staticmethod
    def _fmt_current(task: Optional[str], *, index: Optional[int]) -> str:
        if not task:
            return "[bold]Current:[/bold] [dim]-[/dim]"
        t = escape(task.strip())
        return f"[bold]Current:[/bold] [{index}] {t}"

    @staticmethod
    def _fmt_queue_item(task: Optional[str], *, index: Optional[int], suffix: str = "") -> str:
        if not task:
            return "[dim]  -[/dim]"
        t = escape(task.strip())
        if suffix:
            t = f"{t} {suffix}"
        return f"[dim]  [{index}] {t}[/dim]"

    def _format_status(self, round_idx: int, tasks: list, response: Optional[str] = None, state: str = "execute", done: bool = False) -> str:
        """Format display content: tasks + optional current subagent response."""
        task_count = len(tasks)
        queue_items = tasks[1:]
        checkmark = "[bold green]✓[/bold green] " if done else ""
        lines = [
            f"{checkmark}[bold green]{escape(state)}[/bold green] [dim]|[/dim] "
            f"Step {round_idx} [dim]|[/dim] "
            f"Remaining {task_count}",
        ]
        lines.append("  " + self._fmt_current(tasks[0] if tasks else None, index=round_idx if tasks else None))
        short = self._shorten(response)
        if short:
            lines.append(f"  [bold]Response:[/bold] {escape(short)}")
        else:
            lines.append("  [bold]Response:[/bold] [dim]-[/dim]")
        if not done and queue_items:
            lines.append("  [dim]Queue:[/dim]")
            display_count = len(queue_items) if self.max_queue_preview is None else min(self.max_queue_preview, len(queue_items))
            for i in range(display_count):
                idx = round_idx + 1 + i
                suffix = ""
                if self.max_queue_preview is not None and i == self.max_queue_preview - 1 and len(queue_items) > self.max_queue_preview:
                    suffix = f"(+{len(queue_items) - self.max_queue_preview} more)"
                lines.append("  " + self._fmt_queue_item(queue_items[i], index=idx, suffix=suffix))
        return "\n".join(lines)

    @contextmanager
    def round(self, round_idx: int, tasks: list, state: str = "Execute"):
        """Context manager: show spinner; yield updater for current subagent response."""
        status_msg = self._format_status(round_idx, tasks, state=state)
        last_response: Optional[str] = None
        if self.console:
            with self.console.status(status_msg) as status:
                def update_response(resp: Optional[str]):
                    nonlocal last_response
                    last_response = resp
                    status.update(self._format_status(round_idx, tasks, response=resp, state=state))

                yield update_response
            done_msg = self._format_status(round_idx, tasks, response=last_response, state=state, done=True)
            self.console.print(done_msg, highlight=False)
        else:
            def _noop(_: Optional[str]):
                return None
            yield _noop
