import sys

from .colors import GRAY, WHITE, RESET

class ThinkColorStreamer:
    START_THINK = "<think>"
    END_THINK   = "</think>"
    
    def __init__(self) -> None:
        self.in_think = False
        self.buffer = ""
        self.collected_text = []  # Collect all generated text for saving to context

    def __call__(self, msg: str) -> bool:
        self.feed(msg)
        return False

    def feed(self, msg: str) -> None:
        self.buffer += msg
        self.collected_text.append(msg)  # Collect raw text for context saving

        while self.buffer:
            marker = self.END_THINK if self.in_think else self.START_THINK
            idx = self.buffer.find(marker)

            if idx != -1:
                # 输出 marker 前面的内容
                self._emit(self.buffer[:idx])

                # 吃掉 marker 本身，不显示 <think> / </think>
                self.buffer = self.buffer[idx + len(marker):]

                # 切换状态
                self.in_think = not self.in_think
                continue

            # 没找到完整 marker，保留可能被拆开的 marker 前缀
            keep = self._possible_marker_prefix_len(self.buffer, marker)

            if keep == len(self.buffer):
                # 整个 buffer 都可能是半个 marker，先不输出
                return

            emit_text = self.buffer[:-keep] if keep > 0 else self.buffer
            self.buffer = self.buffer[-keep:] if keep > 0 else ""

            self._emit(emit_text)

    def finish(self) -> None:
        # 把最后残留的非完整 marker 输出
        if self.buffer:
            self._emit(self.buffer)
            self.buffer = ""

        sys.stdout.write(RESET + "\n")
        sys.stdout.flush()

    def get_full_response(self) -> str:
        """Get the complete generated text (including think tags)."""
        return "".join(self.collected_text)

    def _emit(self, text: str) -> None:
        if not text:
            return

        color = GRAY if self.in_think else WHITE
        sys.stdout.write(color + text + RESET)
        sys.stdout.flush()

    @staticmethod
    def _possible_marker_prefix_len(text: str, marker: str) -> int:
        """
        返回 text 末尾有多少字符可能是 marker 的开头。

        例如：
        text = "abc<th"
        marker = "<think>"
        返回 3，因为 "<th" 可能是下一段继续补完的 marker。
        """
        max_len = min(len(text), len(marker) - 1)

        for n in range(max_len, 0, -1):
            if marker.startswith(text[-n:]):
                return n

        return 0
