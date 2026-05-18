import sys
from .colors import (
    GRAY,
    DARK_ORANGE,
    WHITE,
    RESET
)

class SmartStreamer:

    START_THINK = "<think>"
    END_THINK   = "</think>"
    START_TOOL  = "<tool_call>"
    END_TOOL    = "</tool_call>"

    def __init__(self):
        self.mode = None
        self.buffer = ""
        self.collected_text = [] 

    def __call__(self, subword: str) -> bool:
        self._feed(subword)
        return False

    def finish(self):
        if self.buffer:
            color = self._color_for_mode()
            self._raw_write(color + self.buffer)
            self.buffer = ""
        self._raw_write(RESET + "\n")

    def _feed(self, text: str):
        self.buffer += text
        self._try_flush()

    def _try_flush(self):
        while True:
            if self.mode == "think":
                did = self._flush_until_marker(self.END_THINK, GRAY, None)
            elif self.mode == "tool":
                did = self._flush_until_marker(self.END_TOOL, DARK_ORANGE, None)
            else:
                did = self._flush_start_markers()
            if not did:
                break

    def _flush_start_markers(self):
        t_start = self.buffer.find(self.START_THINK)
        tc_start = self.buffer.find(self.START_TOOL)
        if t_start == -1 and tc_start == -1:
            partial = self._longest_partial_prefix()
            safe = len(self.buffer) - partial if partial > 0 else len(self.buffer)
            if safe > 0:
                self._emit(self.buffer[:safe], WHITE)
                self.buffer = self.buffer[safe:]
            return False
        picked = None
        if t_start >= 0:
            picked = (t_start, self.START_THINK, "think")
        if tc_start >= 0 and (picked is None or tc_start < picked[0]):
            picked = (tc_start, self.START_TOOL, "tool")
        if not picked:
            return False
        idx, marker, nxt = picked
        if idx > 0:
            self._emit(self.buffer[:idx], WHITE)
        self.buffer = self.buffer[idx + len(marker):]
        self.mode = nxt
        return True

    def _flush_until_marker(self, end_marker, color, next_mode):
        idx = self.buffer.find(end_marker)
        if idx < 0:
            partial = self._partial_match(self.buffer, end_marker)
            if partial is not None and partial > 0:
                safe = len(self.buffer) - partial
                if safe > 0:
                    self._emit(self.buffer[:safe], color)
                    self.buffer = self.buffer[safe:]
                return False
            if self.buffer:
                self._emit(self.buffer, color)
                self.buffer = ""
            return False
        if idx > 0:
            self._emit(self.buffer[:idx], color)
        self.buffer = self.buffer[idx + len(end_marker):]
        self.mode = next_mode
        return True

    def _emit(self, text, color):
        if not text:
            return
        self._raw_write(color + text + RESET)

    @staticmethod
    def _raw_write(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def _longest_partial_prefix(self):
        best = 0
        for m in (self.START_THINK, self.START_TOOL):
            n = self._partial_match(self.buffer, m)
            if n is not None and n > best:
                best = n
        return best

    @staticmethod
    def _partial_match(text, marker):
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

    def get_full_response(self) -> str:
        """Get the complete generated text (including think tags)."""
        return "".join(self.collected_text)
