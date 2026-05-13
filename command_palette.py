"""
Command palette — an interactive, Vim-inspired command selector for the CLI.

Behavior:
  - Typing `/` at the prompt opens the palette.
  - Shows 5 commands at a time, scrollable with wrap.
  - Arrow Up/Down to navigate, Enter to select, Esc to cancel.
  - Sub-views: `/tokens` (Vim-like `:q` to exit).

All terminal I/O uses raw mode — no escape sequences leak to the user.
"""

import os
import sys
import tty
import termios
import time
from typing import Any, Callable, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
#  ANSI escape sequences
# ═══════════════════════════════════════════════════════════════════════════

CURSOR_HIDE       = "\033[?25l"    # hide blinking cursor
CURSOR_SHOW       = "\033[?25h"    # show cursor
CURSOR_SAVE       = "\033[s"       # save cursor position (anchor)
CURSOR_RESTORE    = "\033[u"       # restore to saved position
CURSOR_UP         = "\033[A"       # cursor up 1 row
CURSOR_DOWN       = "\033[B"       # cursor down 1 row
CARRIAGE_RETURN   = "\r"
CLEAR_LINE        = "\033[2K"      # erase entire current line
CLEAR_FROM_CURSOR = "\033[J"       # erase from cursor to end of screen
RESET             = "\033[0m"
BOLD              = "\033[1m"
REVERSE           = "\033[7m"
CYAN              = "\033[36m"
GRAY              = "\033[90m"
GREEN             = "\033[32m"
YELLOW            = "\033[33m"
DIM               = "\033[2m"


def cursor_up(n: int = 1) -> str:
    return f"\033[{n}A"


def cursor_down(n: int = 1) -> str:
    return f"\033[{n}B"


# ═══════════════════════════════════════════════════════════════════════════
#  Command — a single palette entry
# ═══════════════════════════════════════════════════════════════════════════


class Command:
    """A single palette command: name, description, and handler callable."""

    def __init__(self, name: str, desc: str, handler: Callable[[Any], str]):
        self.name = name
        self.desc = desc
        self.handler = handler

    def run(self, app_state: Any) -> str:
        return self.handler(app_state)


# ═══════════════════════════════════════════════════════════════════════════
#  RawTerminal — context manager for raw TTY mode
# ═══════════════════════════════════════════════════════════════════════════


class RawTerminal:
    """Context manager that switches stdin to raw mode and restores on exit."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ═══════════════════════════════════════════════════════════════════════════
#  Key reader — maps raw bytes to semantic tokens
# ═══════════════════════════════════════════════════════════════════════════


def read_key() -> str:
    """
    Read one keypress from raw stdin.
    Returns 'UP', 'DOWN', 'RIGHT', 'LEFT', 'ENTER', 'BACKSPACE', 'ESC',
    or a single-character string.
    """
    ch = sys.stdin.read(1)

    if ch == "\x1b":
        nxt = sys.stdin.read(1)
        if nxt == "[":
            seq = sys.stdin.read(1)
            while seq and seq[-1] not in "ABCDEFGH~":
                seq += sys.stdin.read(1)
            return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(seq, "ESC")
        elif nxt == "O":
            sys.stdin.read(1)
            return "ESC"
        else:
            return "ESC"

    if ch == "\r" or ch == "\n":
        return "ENTER"
    if ch == "\x7f" or ch == "\b":
        return "BACKSPACE"
    return ch


# ═══════════════════════════════════════════════════════════════════════════
#  CommandPaletteUI — owns all palette rendering state
# ═══════════════════════════════════════════════════════════════════════════
#
#  Key design: cursor save/restore is used as a fixed render anchor.
#  open() saves the cursor position. Every redraw RESTORES to that anchor,
#  clears the old palette area, draws the new one, and RESTORES again.
#  This guarantees zero positional drift — no compounding on repeated presses.


class CommandPaletteUI:
    """
    Interactive command palette that renders in a fixed terminal region.

    State:
      commands       — list of Command objects
      selected       — index of highlighted command
      active         — whether palette is currently open
      rendered_lines — number of lines in the most recent render
    """

    def __init__(self, commands, app_state, visible_count=5):
        self.commands = commands
        self.app_state = app_state
        self.visible_count = visible_count
        self.selected = 0
        self.active = False
        self.rendered_lines = 0

    # ── selection helpers ─────────────────────────────────────────

    def current_command(self):
        """Return the currently highlighted Command."""
        return self.commands[self.selected]

    def move_up(self):
        """Move selection up (wraps to last)."""
        self.selected = (self.selected - 1) % len(self.commands)
        self._redraw()

    def move_down(self):
        """Move selection down (wraps to first)."""
        self.selected = (self.selected + 1) % len(self.commands)
        self._redraw()

    # ── scroll / layout helpers ──────────────────────────────────

    def _scroll_offset(self):
        """Window start index so selected is always visible."""
        n = len(self.commands)
        if n <= self.visible_count:
            return 0
        offset = max(0, self.selected - self.visible_count + 1)
        if offset > n - self.visible_count:
            offset = n - self.visible_count
        return offset

    def _fmt_row(self, cmd, selected):
        """Format one palette row (arrow / name / gap / desc)."""
        prefix = " \u25b8 " if selected else "   "
        name = f"{BOLD}{CYAN}{cmd.name}{RESET}"
        gap = " " * max(1, 24 - len(cmd.name))
        desc = f"{DIM}{cmd.desc}{RESET}"
        if selected:
            return f"{REVERSE} {prefix}{name}{gap}{desc} {RESET}"
        return f" {prefix}{name}{gap}{desc}"

    def render_lines(self):
        """
        Build the full palette as a list of strings (header + body + footer).
        Does NOT write to the terminal.
        """
        lines = []
        lines.append(f"{BOLD} Command{RESET}  {' ' * 16}{DIM}Description{RESET}")
        lines.append(f"{DIM}{'\u2500' * 50}{RESET}")

        n = len(self.commands)
        show_count = min(self.visible_count, n)
        offset = self._scroll_offset()
        for i in range(show_count):
            idx = offset + i
            cmd = self.commands[idx]
            lines.append(self._fmt_row(cmd, idx == self.selected))

        lines.append(f"{DIM}{'\u2500' * 50}{RESET}")
        return lines

    # ── low-level terminal drawing ───────────────────────────────

    def _clear_area(self, count):
        """
        Clear `count` lines starting from the current cursor position.
        Leaves the cursor at the start of the first cleared line.
        """
        for i in range(count):
            if i > 0:
                sys.stdout.write(CURSOR_DOWN)
            sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE)
        if count > 1:
            # After the loop the cursor is on the last cleared line
            # (position L + count - 1).  Move back to L.
            sys.stdout.write(f"\033[{count - 1}A")

    def _write_render(self, lines):
        """Write the palette content lines to the terminal from anchor."""
        for line in lines:
            # \r ensures column 0 before clearing — raw mode \n is only LF
            sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE + line + "\n")

    # ── public lifecycle ─────────────────────────────────────────

    def open(self):
        """
        Save cursor as anchor, hide cursor, draw initial palette.
        Always finishes by RESTOREing to the anchor.
        """
        self.active = True
        sys.stdout.write(CURSOR_SAVE + CURSOR_HIDE)
        lines = self.render_lines()
        self.rendered_lines = len(lines)
        self._write_render(lines)
        sys.stdout.write(CURSOR_RESTORE)
        sys.stdout.flush()

    def _redraw(self):
        """
        In-place redraw — never appends new frames.
        Uses CURSOR_RESTORE to jump back to the open() anchor,
        clears old content, draws new content, and RESTOREs again.
        """
        if self.rendered_lines == 0:
            lines = self.render_lines()
            self.rendered_lines = len(lines)
            self._write_render(lines)
            sys.stdout.write(CURSOR_RESTORE)
            sys.stdout.flush()
            return

        # Step 1: clear the previously rendered area
        self._clear_area(self.rendered_lines)

        # Step 2: draw new content at the same position
        lines = self.render_lines()
        self.rendered_lines = len(lines)
        self._write_render(lines)

        # Step 3: return to anchor so the next redraw starts clean
        sys.stdout.write(CURSOR_RESTORE)
        sys.stdout.flush()

    def close(self, clear=True):
        """
        Remove the palette from the screen and show cursor.
        """
        if clear and self.rendered_lines > 0:
            self._clear_area(self.rendered_lines)
        self.rendered_lines = 0
        sys.stdout.write(CURSOR_SHOW)
        sys.stdout.flush()
        self.active = False

    def run(self):
        """
        Open the palette and handle keyboard input.

        Returns the selected command's handler result, or None if
        the user cancelled (Esc/Ctrl-C/Ctrl-D).
        """
        if len(self.commands) == 0:
            return None

        self.selected = 0

        with RawTerminal():
            try:
                self.open()
                while True:
                    key = read_key()
                    if key == "UP":
                        self.move_up()
                    elif key == "DOWN":
                        self.move_down()
                    elif key == "ENTER":
                        cmd = self.current_command()
                        self.close()
                        return cmd.run(self.app_state)
                    elif key == "ESC":
                        self.close()
                        return None
            except (KeyboardInterrupt, EOFError):
                self.close()
                return None
            except BaseException:
                self.close()
                raise


# ═══════════════════════════════════════════════════════════════════════════
#  Public API — wrapper + utility
# ═══════════════════════════════════════════════════════════════════════════


def run_palette(commands: List[Command], app_state: Any) -> Optional[str]:
    """Open the command palette.  Returns the selected command result or None."""
    ui = CommandPaletteUI(commands, app_state)
    return ui.run()


def _clear_palette(lines: int):
    """
    Clear `lines` rows from the terminal starting at cursor position.
    Used by sub-views (token, resource, model, help overlays).
    """
    for i in range(lines):
        if i > 0:
            sys.stdout.write(CURSOR_DOWN)
        sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE)
    if lines > 1:
        sys.stdout.write(f"\033[{lines - 1}A")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-views — full-screen overlays
# ═══════════════════════════════════════════════════════════════════════════


def show_token_view(agent: Any) -> None:
    """Vim-like sub-view for token counts.  Press `:q` to exit."""
    with RawTerminal():
        sys.stdout.write(CURSOR_HIDE)
        sys.stdout.flush()

        try:
            while True:
                info = _get_token_info(agent)
                lines = info.split("\n")
                n_lines = len(lines) + 2

                for line in lines:
                    sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE + line + "\n")
                sys.stdout.write(f"\n{CLEAR_LINE}{DIM}:{RESET} ")
                sys.stdout.write(cursor_up(n_lines - 1))
                sys.stdout.flush()

                while True:
                    key = read_key()

                    if key == ":":
                        sys.stdout.write("\n" + CLEAR_LINE + ":")
                        sys.stdout.flush()
                        cmd = _read_vim_command()
                        if cmd == "q":
                            _clear_palette(n_lines)
                            sys.stdout.write(CURSOR_SHOW)
                            sys.stdout.flush()
                            return
                        elif cmd == "":
                            pass
                        break

                    elif key == "q":
                        _clear_palette(n_lines)
                        sys.stdout.write(CURSOR_SHOW)
                        sys.stdout.flush()
                        return

                    elif key == "ESC":
                        break

                    elif key == "ENTER":
                        break

        except (KeyboardInterrupt, EOFError):
            _clear_palette(n_lines if "n_lines" in locals() else 5)
            sys.stdout.write(CURSOR_SHOW)
            sys.stdout.flush()


def _read_vim_command() -> str:
    """Read a short vim-style command string after ':'."""
    buf = ""
    while True:
        ch = sys.stdin.read(1)
        if ch == "\r" or ch == "\n":
            sys.stdout.write("\n")
            sys.stdout.flush()
            return buf.strip()
        if ch == "\x1b":
            return ""
        if ch == "\x7f" or ch == "\b":
            if buf:
                buf = buf[:-1]
                sys.stdout.write("\b \b")
                sys.stdout.flush()
        else:
            buf += ch
            sys.stdout.write(ch)
            sys.stdout.flush()


def _get_token_info(agent: Any) -> str:
    """Build a formatted string with estimated token counts."""
    try:
        if hasattr(agent, "token_estimate"):
            te = agent.token_estimate
            prompt_tokens = te["prompt"]
            completion_tokens = te["completion"]
            total_tokens = te["total"]
            ctx_window = te["context_window"]
            msg_count = te["messages"]
        else:
            msg_history = getattr(agent, "messages", [])
            total_chars = sum(len(m.get("content", "")) for m in msg_history)
            prompt_chars = total_chars
            completion_chars = sum(
                len(m.get("content", ""))
                for m in msg_history if m.get("role") == "assistant"
            )
            prompt_tokens = prompt_chars // 4 if prompt_chars else 0
            completion_tokens = completion_chars // 4 if completion_chars else 0
            total_tokens = prompt_tokens + completion_tokens
            ctx_window = 32768
            msg_count = len(msg_history)

        return (
            f"{BOLD}{CYAN}  Token Info{RESET}\n"
            f"{DIM}  {'\u2500' * 30}{RESET}\n"
            f"  Total tokens (est.): {total_tokens:,}\n"
            f"  Prompt tokens (est.): {prompt_tokens:,}\n"
            f"  Completion tokens:    {completion_tokens:,}\n"
            f"  Context window:       {ctx_window:,}\n"
            f"  Usage:                {total_tokens / ctx_window * 100:.1f}%\n"
            f"  Message count:        {msg_count}\n"
            f"\n"
            f"  {DIM}Press :q to exit{RESET}"
        )
    except Exception as e:
        return f"  Error fetching token info: {e}"


def show_resource_view() -> None:
    """Display CPU/memory/GPU usage as a key-to-dismiss overlay."""
    lines = []

    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        boot_ts = psutil.boot_time()
        uptime_sec = time.time() - boot_ts
        uptime_h = uptime_sec // 3600
        uptime_m = (uptime_sec % 3600) // 60
        lines.append(f"  CPU:           {cpu:.1f}% ({psutil.cpu_count()} cores)")
        lines.append(f"  Memory:        {mem.used >> 20:,} MB / {mem.total >> 20:,} MB ({mem.percent:.0f}%)")
        lines.append(f"  Uptime:        {int(uptime_h)}h {int(uptime_m)}m")
    except ImportError:
        lines.append(f"  {YELLOW}Install 'psutil' for detailed resource info{RESET}")

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split("\n")
            for gpu in gpu_lines:
                parts = [p.strip() for p in gpu.split(",")]
                if len(parts) >= 4:
                    lines.append(
                        f"  GPU ({parts[0]}):  {parts[1]}% | "
                        f"{int(parts[2])} MB / {int(parts[3])} MB"
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    output = "\n".join(lines) if lines else "  Resource info unavailable."
    _show_overlay(output)


def show_model_view(agent: Any) -> None:
    """Display model information as a key-to-dismiss overlay."""
    try:
        model_path = getattr(agent, "model_path", "unknown")

        import openvino as ov
        cores = ov.Core()
        devices = ", ".join(cores.available_devices)

        output = (
            f"  Model:         Qwen3-4B-Instruct (Int8, OpenVINO)\n"
            f"  Path:          {model_path}\n"
            f"  Device:        {agent.device}\n"
            f"  Available:     {devices}\n"
            f"  State:         Loaded\n"
            f"  Tools:         {', '.join(agent.registry.list_tools().keys())}"
        )
    except Exception as e:
        output = f"  Model info unavailable: {e}"

    _show_overlay(output)


def show_help_view(commands: List[Command]) -> None:
    """Display all available commands as a key-to-dismiss overlay."""
    lines = [f"  Available commands ({len(commands)}):\n"]
    for cmd in commands:
        lines.append(f"    {CYAN}{cmd.name}{RESET}\t{cmd.desc}")
    _show_overlay("\n".join(lines))


def _show_overlay(content: str):
    """Show content as a key-to-dismiss overlay.  Renders, waits, clears."""
    with RawTerminal():
        sys.stdout.write(CURSOR_HIDE)
        sys.stdout.flush()

        try:
            rendered = content.split("\n")
            n = len(rendered) + 1
            sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE + f"{BOLD}{'\u2500' * 50}{RESET}\n")
            for line in rendered:
                sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE + line + "\n")
            sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE + f"{DIM}  Press any key to dismiss{RESET}\n")
            sys.stdout.write(CARRIAGE_RETURN + CLEAR_LINE + f"{DIM}{'\u2500' * 50}{RESET}\n")
            sys.stdout.write(cursor_up(n))
            sys.stdout.flush()

            while True:
                k = read_key()
                if k in ("ENTER", "ESC") or len(k) == 1:
                    break

            _clear_palette(n + 1)
        finally:
            sys.stdout.write(CURSOR_SHOW)
            sys.stdout.flush()
