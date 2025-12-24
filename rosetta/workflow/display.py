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
            return "[bold cyan]Current:[/bold cyan] [dim]-[/dim]"
        t = escape(task.strip())
        return f"[bold cyan]Current:[/bold cyan] [{index}] {t}"

    @staticmethod
    def _fmt_queue_item(task: Optional[str], *, index: Optional[int], suffix: str = "") -> str:
        if not task:
            return "[dim]  -[/dim]"
        t = escape(task.strip())
        if suffix:
            t = f"{t} {suffix}"
        return f"[dim]  [{index}] {t}[/dim]"

    def _format_status(self, round_idx: int, tasks: list, response: Optional[str] = None, state: str = "execute", done: bool = False, tools_used: Optional[list] = None) -> str:
        """Format display content: tasks + optional current subagent response + tools used."""
        task_count = len(tasks)
        queue_items = tasks[1:]
        checkmark = "[bold green]✓[/bold green] " if done else ""
        lines = [
            f"{checkmark}[bold green]{escape(state)}[/bold green] [dim]|[/dim] "
            f"Step {round_idx} [dim]|[/dim] "
            f"Remaining {task_count}",
        ]
        lines.append("  " + self._fmt_current(tasks[0] if tasks else None, index=round_idx if tasks else None))

        # Show tools used
        if tools_used:
            tools_str = ", ".join(escape(t) for t in tools_used)
            lines.append(f"  [bold cyan]Tools:[/bold cyan] {tools_str}")

        short = self._shorten(response)
        if short:
            lines.append(f"  [bold cyan]Response:[/bold cyan] {escape(short)}")
        else:
            lines.append("  [bold cyan]Response:[/bold cyan] [dim]-[/dim]")
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
        last_tools: Optional[list] = None
        if self.console:
            with self.console.status(status_msg) as status:
                def update_response(resp: Optional[str], tools_used: Optional[list] = None):
                    nonlocal last_response, last_tools
                    last_response = resp
                    last_tools = tools_used
                    status.update(self._format_status(round_idx, tasks, response=resp, state=state, tools_used=tools_used))

                yield update_response
            done_msg = self._format_status(round_idx, tasks, response=last_response, state=state, done=True, tools_used=last_tools)
            self.console.print(done_msg, highlight=False)
        else:
            def _noop(*args, **kwargs):
                return None
            yield _noop
