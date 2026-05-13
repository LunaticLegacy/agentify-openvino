#!/usr/bin/env python3
"""
Agentified LLM CLI — with colored streaming output and a command palette.

Features:
  - Interactive REPL with `/`-triggered command palette
  - Palette: navigate with Up/Down, select with Enter, cancel with Esc
  - <think> blocks in gray, <tool_call> blocks in dark orange
  - Arrow keys work in normal input mode (readline-backed)
  - Built-in commands: /exit, /model, /tokens, /resources, /help

Usage:
  python run_agent.py                          # interactive
  python run_agent.py "What's the weather?"    # one-shot
"""

import sys
import argparse
from pathlib import Path

# ── Add the agentify directory to sys.path so imports resolve ──────────────
# Without this, `from agent import ...` would fail when running from outside
# the agentify directory (e.g. from /home/luna/Documents/llm).
sys.path.insert(0, str(Path(__file__).parent))

from agent import create_agent
from command_palette import (
    Command,
    run_palette,
    show_token_view,
    show_resource_view,
    show_model_view,
    show_help_view,
)


# ═══════════════════════════════════════════════════════════════════════════
#  ANSI escape codes for terminal coloring
# ═══════════════════════════════════════════════════════════════════════════
#  90 = bright-black (gray foreground)
#  38;5;208 = 256-color "dark orange"
#  97 = bright-white foreground
GRAY        = "\033[90m"
DARK_ORANGE = "\033[38;5;208m"
WHITE       = "\033[97m"
RESET       = "\033[0m"       # reset all attributes
BOLD        = "\033[1m"
DIM         = "\033[2m"       # half-bright (used for descriptions)
CLEAR_LINE  = "\033[2K"       # erase entire current line


# ═══════════════════════════════════════════════════════════════════════════
#  SmartStreamer — colorizes <think>/<tool_call> blocks during model output
# ═══════════════════════════════════════════════════════════════════════════
#  The Qwen3 model outputs plain text containing XML-like markers:
#    <think>reasoning...</think>
#    <tool_call>{"name":"...","arguments":{...}}</tool_call>
#  We want to:
#    (a) Hide the markers themselves (never print <think>, </think>, etc.)
#    (b) Print the content inside them with a distinct color
#    (c) Handle markers that arrive split across multiple subword tokens
#        (e.g. "<" arrives in token N, "think>" arrives in token N+1)
#
#  This is a state machine with three modes:
#    None   → normal text (print white)
#    "think" → inside <think>...</think> (buffer, then print gray)
#    "tool"  → inside <tool_call>...</tool_call> (buffer, then print orange)

class SmartStreamer:

    # ── Marker strings the model emits ──────────────────────────────────
    START_THINK = "<think>"
    END_THINK   = "</think>"
    START_TOOL  = "<tool_call>"
    END_TOOL    = "</tool_call>"

    def __init__(self):
        # mode tracks which marker we're currently inside (None | "think" | "tool")
        self.mode = None
        # buffer accumulates partial text until we can flush a complete segment
        self.buffer = ""

    # ── Called by the OpenVINO GenAI pipeline for each subword token ────
    # The pipeline passes one subword at a time.  Return False to continue
    # generation (True would stop early).
    def __call__(self, subword: str) -> bool:
        self._feed(subword)
        return False

    # ── Called after generation finishes ────────────────────────────────
    # Flush any remaining buffered content so nothing is lost.
    def finish(self):
        if self.buffer:
            color = self._color_for_mode()
            self._raw_write(color + self.buffer)
            self.buffer = ""
        # Append a newline in the neutral color to separate agent output
        # from whatever comes next (tool logs, user prompt).
        self._raw_write(RESET + "\n")

    # ── Feed text into the state machine ────────────────────────────────
    def _feed(self, text: str):
        self.buffer += text        # append new subword to working buffer
        self._try_flush()          # attempt to emit as much as possible

    # ── Repeatedly scan the buffer for markers and flush what we can ────
    def _try_flush(self):
        while True:
            if self.mode == "think":
                # We're inside a <think> block — look for the </think> end marker
                did = self._flush_until_marker(self.END_THINK, GRAY, None)
            elif self.mode == "tool":
                # We're inside a <tool_call> block — look for </tool_call>
                did = self._flush_until_marker(self.END_TOOL, DARK_ORANGE, None)
            else:
                # We're in normal text — look for any *start* marker
                did = self._flush_start_markers()
            if not did:
                break  # nothing more we can flush with current buffer

    # ── Look for <think> or <tool_call> in normal text ─────────────────
    def _flush_start_markers(self):
        # Find the earliest occurrence of either start marker
        t_start = self.buffer.find(self.START_THINK)
        tc_start = self.buffer.find(self.START_TOOL)

        if t_start == -1 and tc_start == -1:
            # No complete marker found — but there might be a partial marker
            # at the end of the buffer (e.g. buffer ends with "<th").
            # Compute how many trailing chars could be a marker prefix,
            # emit the safe prefix, and keep the partial for later.
            partial = self._longest_partial_prefix()
            safe = len(self.buffer) - partial if partial > 0 else len(self.buffer)
            if safe > 0:
                self._emit(self.buffer[:safe], WHITE)
                self.buffer = self.buffer[safe:]
            return False

        # Pick whichever marker appears first
        picked = None
        if t_start >= 0:
            picked = (t_start, self.START_THINK, "think")
        if tc_start >= 0 and (picked is None or tc_start < picked[0]):
            picked = (tc_start, self.START_TOOL, "tool")
        if not picked:
            return False

        idx, marker, nxt = picked
        # Emit text that appears *before* the marker (this is plain content)
        if idx > 0:
            self._emit(self.buffer[:idx], WHITE)
        # Discard the marker itself — we never print it
        self.buffer = self.buffer[idx + len(marker):]
        # Switch mode so future flushes use the correct color
        self.mode = nxt
        return True

    # ── While inside a colored block, look for the end marker ──────────
    def _flush_until_marker(self, end_marker, color, next_mode):
        idx = self.buffer.find(end_marker)
        if idx < 0:
            # End marker not found yet — check if the buffer ends with
            # a partial match (e.g. "</tool_cal" — the model might send
            # "l" in the next subword).  Only emit what's *definitely* not
            # part of a future marker.
            partial = self._partial_match(self.buffer, end_marker)
            if partial is not None and partial > 0:
                safe = len(self.buffer) - partial
                if safe > 0:
                    self._emit(self.buffer[:safe], color)
                    self.buffer = self.buffer[safe:]
                return False  # wait for more data
            # No partial match — emit the whole buffer in the current color
            if self.buffer:
                self._emit(self.buffer, color)
                self.buffer = ""
            return False

        # Found the end marker — emit content before it
        if idx > 0:
            self._emit(self.buffer[:idx], color)
        # Discard the end marker
        self.buffer = self.buffer[idx + len(end_marker):]
        # Switch back to normal text mode
        self.mode = next_mode
        # Return True so _try_flush loops again — there might be another
        # marker immediately after this one.
        return True

    # ── Write colored text to stdout ───────────────────────────────────
    def _emit(self, text, color):
        if not text:
            return
        self._raw_write(color + text + RESET)

    @staticmethod
    def _raw_write(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # ── Checks if the tail of the buffer could be a partial marker ─────
    # e.g. buffer ends with "</t" — that's the first 3 chars of "</think>"
    def _longest_partial_prefix(self):
        best = 0
        for m in (self.START_THINK, self.START_TOOL):
            n = self._partial_match(self.buffer, m)
            if n is not None and n > best:
                best = n
        return best

    @staticmethod
    def _partial_match(text, marker):
        # Walk backwards to find the longest suffix that matches the start
        # of the marker.  Max length is len(marker)-1 because a complete
        # marker isn't "partial".
        max_n = min(len(text), len(marker) - 1)
        for n in range(max_n, 0, -1):
            if marker.startswith(text[-n:]):
                return n
        return None

    def _color_for_mode(self):
        if self.mode == "think":
            return GRAY
        if self.mode == "tool":
            return DARK_ORANGE
        return WHITE


# ═══════════════════════════════════════════════════════════════════════════
#  Command handlers — each corresponds to a palette entry
# ═══════════════════════════════════════════════════════════════════════════

# /exit — raises SystemExit which is caught in main()'s except clause
def cmd_exit(app: dict) -> str:
    raise SystemExit(0)

# /model — shows model info as a key-to-dismiss overlay
def cmd_model(app: dict) -> str:
    agent = app["agent"]
    show_model_view(agent)
    return ""

# /tokens — opens the Vim-like token viewer (press :q to exit)
def cmd_tokens(app: dict) -> str:
    agent = app["agent"]
    show_token_view(agent)
    return ""

# /resources — shows CPU/RAM/GPU as a key-to-dismiss overlay
def cmd_resources(app: dict) -> str:
    show_resource_view()
    return ""

# /help — shows all available commands as a key-to-dismiss overlay
def cmd_help(app: dict) -> str:
    commands = app["commands"]
    show_help_view(commands)
    return ""


def build_commands(agent, app_state: dict) -> list:
    """Build the list of Command objects that the palette displays."""
    return [
        Command("/exit",      "Exit the CLI",                    cmd_exit),
        Command("/model",     "Show model information",           cmd_model),
        Command("/tokens",    "Show the current context token count", cmd_tokens),
        Command("/resources", "Show CPU, memory and GPU usage",   cmd_resources),
        Command("/help",      "Show available commands",         cmd_help),
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Custom line reader — detects "/" and opens the palette
# ═══════════════════════════════════════════════════════════════════════════
#  Normal input() with readline handles arrow keys fine, but it can't
#  intercept the "/" keystroke before it goes into the edit buffer.
#  Solution: read the first byte in raw mode ourselves; if it's "/", open
#  the palette; otherwise echo it back and let input() handle the rest.

def _read_line_or_palette(agent, commands, app_state: dict) -> str:
    """
    Custom line reader: prints prompt, reads first char via raw mode.
    If first char is `/`, invokes the palette. Otherwise falls back to
    readline for the rest of the line.
    """
    import tty, termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)       # save current terminal settings

    sys.stdout.write("You: ")
    sys.stdout.flush()

    # ── Grab the first keystroke in raw mode (no echo, no buffering) ──
    try:
        tty.setraw(fd)                # switch stdin to raw mode
        first = sys.stdin.read(1)     # block until one byte arrives
    finally:
        # Restore terminal settings immediately, even if read() throws
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ── First character is "/" → open the command palette ─────────────
    if first == "/":
        # Erase the "You: /" text we already wrote, since the palette
        # will render its own UI in its place.
        sys.stdout.write("\r" + CLEAR_LINE)
        sys.stdout.flush()
        result = run_palette(commands, app_state)
        if result is not None:
            return result
        # Palette was cancelled (Esc) → return empty string; main loop
        # will treat this as "skip to next prompt".
        return ""

    # ── Empty input (Enter with no text, or Ctrl+D) ───────────────────
    if first in ("\r", "\n", "\x04"):
        sys.stdout.write("\n")
        sys.stdout.flush()
        return ""

    # ── Normal text — let readline handle the rest of the line ────────
    # Enable GNU readline so arrow keys do in-place editing (no ^[[A junk)
    try:
        import readline  # noqa: F401
    except ImportError:
        pass  # non-Linux fallback — arrow keys may show escape sequences

    # Write the first character back to the terminal (readline won't
    # see it since we already consumed it from stdin).
    sys.stdout.write(first)
    sys.stdout.flush()

    # Read the remainder of the line with input().  readline's history
    # and editing apply to everything typed *after* the first char.
    rest = input("")
    return first + rest


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agentified LLM")
    parser.add_argument("message", nargs="*",
                        help="One-shot message (omit for interactive mode)")
    parser.add_argument("--model",
                        default="/home/luna/Documents/llm/models/qwen3-4b-int8-ov/",
                        help="Path to the OpenVINO model")
    parser.add_argument("--device", default="CPU",
                        help="Device (CPU, GPU, NPU)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming output")

    args = parser.parse_args()

    # ── Boot the agent (loads the OpenVINO model + tool registry) ──────
    print("🚀 Initializing agent...", file=sys.stderr)
    agent = create_agent(model_path=args.model, device=args.device)

    # Build commands list and shared app state dict (passed into handlers)
    commands = build_commands(agent, {})
    app_state = {"agent": agent, "commands": commands}

    print(f"✅ Agent ready!  Tools: {', '.join(agent.registry.list_tools().keys())}",
          file=sys.stderr)
    print(file=sys.stderr)

    # ── One-shot mode (message passed on command line) ─────────────────
    if args.message:
        msg = " ".join(args.message)
        print(f"\nYou: {msg}")
        print("Agent: ", end="", flush=True)
        streamer = SmartStreamer() if not args.no_stream else None
        agent.chat(msg, stream=not args.no_stream, external_streamer=streamer)
        print()
        return

    # ── Interactive REPL ───────────────────────────────────────────────
    print("Interactive mode. Type / to open command palette.\n")

    while True:
        try:
            # _read_line_or_palette handles both normal text input and
            # the "/" palette trigger transparently.
            user_input = _read_line_or_palette(agent, commands, app_state)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        print("Agent: ", end="", flush=True)
        streamer = SmartStreamer() if not args.no_stream else None
        try:
            agent.chat(user_input, stream=not args.no_stream,
                       external_streamer=streamer)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        print()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass  # /exit raises SystemExit(0) — don't print a traceback
