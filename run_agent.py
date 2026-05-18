#!/usr/bin/env python3
"""
Agentified LLM CLI — with colored streaming output and a command palette.

Features:
  - Interactive REPL with `/`-triggered command palette
  - Palette: navigate with Up/Down, select with Enter, cancel with Esc
  - <think> blocks in gray, <tool_call> blocks in dark orange
  - Arrow keys work in normal input mode (readline-backed)
  - Ctrl+C interrupts generation without polluting conversation history
  - Double Ctrl+C exits the CLI
  - /exit, /model, /tokens, /resources, /help commands

Usage:
  python run_agent.py                          # interactive
  python run_agent.py "What's the weather?"    # one-shot
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import create_agent, GenerationInterrupted
from command_palette import (
    Command,
    run_palette,
    show_token_view,
    show_resource_view,
    show_model_view,
    show_help_view,
)


# ── ANSI colors ──────────────────────────────────────────────────────────
from .streamers import (
    GRAY,
    DARK_ORANGE,
    WHITE,
    RESET,
    SmartStreamer
)

# ═══════════════════════════════════════════════════════════════════════════
#  Command handlers
# ═══════════════════════════════════════════════════════════════════════════

def cmd_exit(app: dict) -> str:
    raise SystemExit(0)


def cmd_model(app: dict) -> str:
    show_model_view(app["agent"])
    return ""


def cmd_tokens(app: dict) -> str:
    show_token_view(app["agent"])
    return ""


def cmd_resources(app: dict) -> str:
    show_resource_view()
    return ""


def cmd_help(app: dict) -> str:
    show_help_view(app["commands"])
    return ""


def build_commands(agent, app_state: dict) -> list:
    return [
        Command("/exit",      "Exit the CLI",                    cmd_exit),
        Command("/model",     "Show model information",           cmd_model),
        Command("/tokens",    "Show the current context token count", cmd_tokens),
        Command("/resources", "Show CPU, memory and GPU usage",   cmd_resources),
        Command("/help",      "Show available commands",         cmd_help),
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Line reader — detects "/" to open palette, handles Ctrl+C properly
# ═══════════════════════════════════════════════════════════════════════════
#
#  Backspace fix: we read the first keystroke in raw mode to intercept "/",
#  then push it into readline's buffer via set_startup_hook().  This ensures
#  readline knows about the first character and backspace can reach it.
#
#  Ctrl+C fix: in raw mode, Ctrl+C arrives as the byte \x03.  We translate
#  it to KeyboardInterrupt so the main loop can handle double-Ctrl+C.

def _read_line_or_palette(agent, commands, app_state: dict) -> str:
    import tty, termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    sys.stdout.write("You: ")
    sys.stdout.flush()

    # Read first byte in raw mode (no echo, immediate)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # Ctrl+C in raw mode → translate to KeyboardInterrupt
    if first == "\x03":
        raise KeyboardInterrupt()

    # "/" → open command palette
    if first == "/":
        sys.stdout.write("\r" + CLEAR_LINE)
        sys.stdout.flush()
        result = run_palette(commands, app_state)
        if result is not None:
            return result
        return ""

    # Empty input (bare Enter / Ctrl+D)
    if first in ("\r", "\n", "\x04"):
        sys.stdout.write("\n")
        sys.stdout.flush()
        return ""

    # Normal text — push first char into readline so backspace can delete it
    try:
        import readline as _rl
    except ImportError:
        _rl = None

    if _rl is not None:
        def _startup_hook():
            _rl.insert_text(first)
            _rl.redisplay()
        _rl.set_startup_hook(_startup_hook)
        try:
            line = input("")
        finally:
            _rl.set_startup_hook(None)
        # input() already includes the first char (inserted by the hook)
        return line
    else:
        # Fallback for systems without readline
        sys.stdout.write(first)
        sys.stdout.flush()
        return first + input("")


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

    # ── Boot agent ─────────────────────────────────────────────────────
    print("🚀 Initializing agent...", file=sys.stderr)
    agent = create_agent(model_path=args.model, device=args.device)

    commands = build_commands(agent, {})
    app_state = {"agent": agent, "commands": commands}

    print(f"✅ Agent ready!  Tools: {', '.join(agent.registry.list_tools().keys())}",
          file=sys.stderr)
    print(file=sys.stderr)

    # ── One-shot mode ──────────────────────────────────────────────────
    if args.message:
        msg = " ".join(args.message)
        print(f"\nYou: {msg}")
        print("Agent: ", end="", flush=True)
        streamer = SmartStreamer() if not args.no_stream else None
        try:
            agent.chat(msg, stream=not args.no_stream,
                       external_streamer=streamer)
        except GenerationInterrupted:
            print("\n(interrupted)")
        print()
        return

    # ── Interactive REPL ───────────────────────────────────────────────
    print("Interactive mode. Type / to open command palette.")
    print("Press Ctrl+C to interrupt the model; press Ctrl+C twice to exit.\n")

    pending_exit = False  # track first Ctrl+C for double-press-to-exit

    while True:
        try:
            user_input = _read_line_or_palette(agent, commands, app_state)
        except (EOFError, KeyboardInterrupt):
            if pending_exit:
                # Second consecutive Ctrl+C → exit
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            # First Ctrl+C → show hint, stay in the loop
            pending_exit = True
            sys.stdout.write("\nPress Ctrl+C again to exit\n")
            sys.stdout.flush()
            continue

        # If we get here, the user typed something (not Ctrl+C).
        # Clear the "Press Ctrl+C again" message if it was shown.
        if pending_exit:
            sys.stdout.write(CURSOR_UP + CLEAR_LINE + "\r" + CLEAR_LINE)
            sys.stdout.flush()
            pending_exit = False

        if not user_input:
            continue

        print("Agent: ", end="", flush=True)
        streamer = SmartStreamer() if not args.no_stream else None
        try:
            agent.chat(user_input, stream=not args.no_stream,
                       external_streamer=streamer)
        except GenerationInterrupted:
            # Ctrl+C during generation — partial output discarded,
            # nothing added to conversation history.
            print("\n(interrupted)")
            continue
        print()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass  # /exit raises SystemExit(0)
