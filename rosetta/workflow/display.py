from typing import Optional, List
from contextlib import contextmanager
from rich.console import Console
from rich.markup import escape

class StatusLogger:
    """Handles console status display with spinner and history."""

    def __init__(self, enabled: bool = True, max_tasks_preview: Optional[int] = None):
        self.console = Console() if enabled else None
        self.max_tasks_preview = max_tasks_preview

    @staticmethod
    def _shorten(text: Optional[str], max_len: int = 240) -> Optional[str]:
        if not text:
            return None
        s = " ".join(text.strip().split())
        return (s[: max_len - 1] + "…") if len(s) > max_len else s

    @staticmethod
    def _fmt_task(task: str, marker: str, dim: bool = False) -> str:
        """Format a task with marker (✓, •, or space)."""
        t = escape(task.strip())
        if dim:
            return f"[dim]  {marker} {t}[/dim]"
        return f"  {marker} {t}"

    def _format_status(
        self,
        action: str,
        round_idx: int,
        step_idx: int,
        status_desc: str,
        finished: List[str],
        current: List[str],
        pending: List[str],
        response: Optional[str] = None,
        done: bool = False,
        tools_used: Optional[list] = None,
    ) -> str:
        """Format display content: ACTION | Round X | Step Y | Progress.

        Task markers:
            ✓ - finished tasks
            • - current task (in progress)
            ○ - pending tasks (todo)
        """
        progress = len(finished)
        total = len(finished) + len(current) + len(pending)

        checkmark = "[bold green]✓[/bold green] " if done else ""
        lines = [
            f"{checkmark}[bold green]{escape(action.capitalize())}[/bold green] [dim]|[/dim] "
            f"Round {round_idx} [dim]|[/dim] "
            f"Step {step_idx} [dim]|[/dim] "
            f"Progress {progress}/{total}",
            f"  [bold cyan]Status:[/bold cyan] {escape(status_desc)}",
        ]

        # Show tools used
        if tools_used:
            tools_str = ", ".join(escape(t) for t in tools_used)
            lines.append(f"  [bold cyan]Tools:[/bold cyan] {tools_str}")

        # Show response
        short = self._shorten(response)
        if short:
            lines.append(f"  [bold cyan]Response:[/bold cyan] {escape(short)}")

        # Show task lists only while working (not when done)
        if not done and (finished or current or pending):
            lines.append("  [bold cyan]Tasks:[/bold cyan]")
            for task in finished:
                lines.append(self._fmt_task(task, "[green]✓[/green]", dim=True))
            for task in current:
                lines.append(self._fmt_task(task, "[bold yellow]•[/bold yellow]", dim=False))
            display_pending = pending
            extra_count = 0
            if self.max_tasks_preview is not None and len(pending) > self.max_tasks_preview:
                display_pending = pending[:self.max_tasks_preview]
                extra_count = len(pending) - self.max_tasks_preview
            for task in display_pending:
                lines.append(self._fmt_task(task, "[dim]○[/dim]", dim=True))
            if extra_count > 0:
                lines.append(f"[dim]    (+{extra_count} more)[/dim]")

        return "\n".join(lines)

    @contextmanager
    def status(
        self,
        action: str,
        round_idx: int,
        step_idx: int,
        status_desc: str,
        finished: List[str],
        current: List[str],
        pending: List[str],
    ):
        """Context manager: show spinner; yield updater for current subagent response."""
        last_response: Optional[str] = None
        last_tools: Optional[list] = None
        last_tasks: tuple = (list(finished), list(current), list(pending))
        if self.console:
            status_msg = self._format_status(action, round_idx, step_idx, status_desc, *last_tasks)
            with self.console.status(status_msg) as st:
                def update_response(
                    resp: Optional[str],
                    tools_used: Optional[list] = None,
                    tasks: Optional[tuple] = None,
                ):
                    nonlocal last_response, last_tools, last_tasks
                    last_response = resp
                    last_tools = tools_used
                    if tasks:
                        last_tasks = tasks
                    st.update(self._format_status(
                        action, round_idx, step_idx, status_desc, *last_tasks,
                        response=resp, tools_used=tools_used
                    ))
                yield update_response
            done_msg = self._format_status(
                action, round_idx, step_idx, status_desc, *last_tasks,
                response=last_response, done=True, tools_used=last_tools
            )
            self.console.print(done_msg, highlight=False)
        else:
            yield lambda *a, **kw: None
