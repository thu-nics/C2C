from typing import Optional, List
from contextlib import contextmanager
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.live import Live


class ConvLogger:
    """Live conversation display that updates in place."""

    _ROLE_STYLES = {
        "system": ("bold magenta", "System"),
        "user": ("bold green", "User"),
        "assistant": ("bold blue", "Assistant"),
        "tool": ("bold yellow", "Tool"),
    }

    def __init__(self, tokenizer=None, enabled: bool = True, max_content_len: int = 300, max_messages: int = 4):
        self.console = Console() if enabled else None
        self.tokenizer = tokenizer
        self.max_content_len = max_content_len
        self.max_messages = max_messages
        self._live: Optional[Live] = None
        self._last_messages: List[dict] = []

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer, or estimate by chars."""
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4  # Rough estimate

    def _shorten(self, text: Optional[str]) -> str:
        if not text:
            return "[dim](empty)[/dim]"
        s = " ".join(text.strip().split())
        if len(s) > self.max_content_len:
            return escape(s[: self.max_content_len - 1]) + "[dim]…[/dim]"
        return escape(s)

    def _format_message(self, msg: dict, idx: int) -> Text:
        """Format a single message as Rich Text."""
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        style, label = self._ROLE_STYLES.get(role, ("white", role.capitalize()))

        # Count tokens
        token_count = self._count_tokens(content)
        lines = [f"[{style}][{idx}] {label}[/{style}] [dim]| {token_count} tokens[/dim]"]

        # Show content
        if content:
            lines.append(f"  {self._shorten(content)}")

        # Show tool calls for assistant
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if len(args) > 80:
                    args = args[:77] + "..."
                lines.append(f"  [dim]→ {name}({args})[/dim]")

        # Show tool_call_id for tool messages
        if msg.get("tool_call_id"):
            lines.append(f"  [dim]tool_call_id: {msg['tool_call_id'][:20]}...[/dim]")

        return Text.from_markup("\n".join(lines))

    def _render_all(self, messages: List[dict]) -> Group:
        """Render all messages as a Rich Group."""
        renderables = []
        
        # Calculate total token count
        total_tokens = sum(self._count_tokens(msg.get("content", "")) for msg in messages)
        header = Text.from_markup(f"[bold cyan]Total Tokens: {total_tokens}[/bold cyan] [dim]| {len(messages)} messages[/dim]")
        renderables.append(header)
        
        # Only show the latest N messages
        display_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        start_idx = len(messages) - len(display_messages)
        for i, msg in enumerate(display_messages, start=start_idx):
            renderables.append(self._format_message(msg, i))
        return Group(*renderables)

    def start(self) -> None:
        """Start live display mode."""
        if self.console and self._live is None:
            self._live = Live(console=self.console, refresh_per_second=4)
            self._live.start()

    def stop(self) -> None:
        """Stop live display mode (final state remains visible)."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(self, messages: List[dict]) -> None:
        """Update display with current messages (clears old, prints new in place)."""
        if not self.console:
            return

        self._last_messages = list(messages)

        if self._live:
            # Live mode: update in place
            self._live.update(self._render_all(messages))
        else:
            # Non-live mode: just print all
            self.print_all(messages)

    def reset(self) -> None:
        """Reset state."""
        self._last_messages = []

    def print_all(self, messages: List[dict]) -> None:
        """Print all messages (non-live, permanent output)."""
        if not self.console:
            return
        
        # Print total token count header
        total_tokens = sum(self._count_tokens(msg.get("content", "")) for msg in messages)
        header = Text.from_markup(f"[bold cyan]Total Tokens: {total_tokens}[/bold cyan] [dim]| {len(messages)} messages[/dim]")
        self.console.print(header)
        
        # Only show the latest N messages
        display_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        start_idx = len(messages) - len(display_messages)
        for i, msg in enumerate(display_messages, start=start_idx):
            self.console.print(self._format_message(msg, i))


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
