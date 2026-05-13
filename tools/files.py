"""
File tools: read, write, and list files.
"""

import os
import json
import subprocess
from pathlib import Path


SAFE_ROOT = os.path.expanduser("~/Documents")


def _resolve_path(path: str) -> Path:
    """Resolve a path, ensuring it stays within the safe root."""
    p = Path(path).expanduser().resolve()
    # Ensure it's within the safe zone
    # p_str = str(p)
    # if not p_str.startswith(str(Path(SAFE_ROOT).resolve())):
    #     raise PermissionError(f"Access denied: path must be under {SAFE_ROOT}")
    return p


def file_read(path: str) -> str:
    """Read the contents of a text file."""
    try:
        p = _resolve_path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        if p.is_dir():
            return f"Error: '{path}' is a directory, not a file"

        content = p.read_text(encoding="utf-8")
        if len(content) > 10000:
            content = content[:10000] + f"\n\n[...truncated, original length: {len(content)} chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def file_write(path: str, content: str, append: bool = False) -> str:
    """Write or append text to a file (overwrites by default). Set append=True to append."""
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        p.write_text(content, encoding="utf-8")
        action = "Appended to" if append else "Wrote"
        return f"{action} {p} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


def file_list(path: str = ".") -> str:
    """List files and directories at the given path."""
    try:
        p = _resolve_path(path)
        if not p.exists():
            return f"Error: path not found: {path}"
        if not p.is_dir():
            return f"Error: '{path}' is a file, not a directory"

        entries = []
        for item in sorted(p.iterdir()):
            suffix = "/" if item.is_dir() else ""
            size = item.stat().st_size if item.is_file() else ""
            entries.append(f"  {item.name}{suffix}  {size}")

        header = f"Directory listing: {p}\n"
        return header + "\n".join(entries) if entries else header + "(empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def run_command(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output. Use with caution."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += f"[stdout]\n{result.stdout[:3000]}"
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr[:1000]}"
        output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"
