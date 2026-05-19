"""
Agent Core — the orchestrator that wraps the LLM in a tool-using agent loop.

Architecture overview:
  1. User sends a message
  2. Agent builds a prompt (system + conversation history + user message)
  3. Prompt is fed to the OpenVINO GenAI LLM pipeline
  4. Output is scanned for <tool_call> blocks
  5. If found, the named tool is executed and the result is injected back
     into the conversation history
  6. Steps 3-5 repeat (up to max_tool_rounds) until the model responds
     without tool calls
"""

import json
import re
import sys
import time
from typing import List, Optional
from pathlib import Path


class GenerationInterrupted(Exception):
    """
    Raised when the user presses Ctrl+C during model generation.
    The partial output is discarded and NOT added to conversation history.
    """
    

import openvino as ov
import openvino_genai as ov_genai

from .tool_registry import Tool, ToolRegistry


# ── Regex that extracts JSON tool calls from model output ──────────────────
# The Qwen3 chat template emits tool calls wrapped in <tool_call> tags:
#   <tool_call>
#   {"name": "web_search", "arguments": {"query": "weather"}}
#   </tool_call>
# This pattern matches the opening tag, captures the JSON object, and
# ignores whitespace between the tag and the JSON.
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# ── Default system prompt used when none is provided ──────────────────────
# This tells the model what it is and when to use tools.  The tool
# definitions (from ToolRegistry.to_system_prompt()) are appended to this.
SYSTEM_PROMPT_BASE = (
    "You are a helpful AI assistant with access to tools. "
    "You can use tools to search the web, read/write files, and run commands. "
    "Think briefly and only use a tool when it adds new information. "
    "Do not repeat the same tool call with the same arguments. "
    "After one useful tool result, summarize and answer directly."
)


class Agent:
    """
    An LLM agent that uses tools.

    Fields:
      pipe         — OpenVINO GenAI LLMPipeline (the raw model)
      registry     — ToolRegistry holding all available tools
      messages     — conversation history (list of dicts with role/content)
      system_prompt — base system prompt (tool definitions are auto-appended)
      max_tool_rounds — max number of tool-calling iterations per user message
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        device_config: Optional[dict] = None,
        max_new_tokens: int = 8192,
        system_prompt: Optional[str] = None,
        max_tool_rounds: int = 10,
    ):
        self.model_path = model_path
        self.device = device
        self.max_tool_rounds = max_tool_rounds
        self.max_new_tokens = max_new_tokens
        self.device_config = dict(device_config or {})

        # ── Load the OpenVINO GenAI model ────────────────────────────────
        # LLMPipeline wraps tokenizer + model + generation config.
        # It's the main inference interface.
        # Device-specific options are useful for NPU, where prompt length
        # and response length limits are different from CPU/GPU defaults.
        self.pipe = ov_genai.LLMPipeline(model_path, device, self.device_config)

        # ── Register built-in tools (web search, file I/O, shell) ────────
        self.registry = ToolRegistry()
        self._register_default_tools()

        # ── Set system prompt (appended with tool definitions later) ─────
        self.system_prompt = system_prompt or SYSTEM_PROMPT_BASE

        # ── Initialize empty conversation history ────────────────────────
        # Each entry: {"role": "system"|"user"|"assistant"|"tool",
        #              "content": "..."}
        self.messages: List[dict] = []

    def _register_default_tools(self):
        """Register the six built-in tools into self.registry."""
        from .tools.web import web_search, web_fetch
        from .tools.files import file_read, file_write, file_list, run_command

        # Each Tool has a name, description (for the model), and a Python
        # function that executes it.  Parameters are auto-derived from the
        # function signature unless explicitly provided.
        self.registry.register(Tool(
            "web_search",
            "Search the internet. Use this when you need current information.",
            web_search,
        ))
        self.registry.register(Tool(
            "web_fetch",
            "Fetch and extract text content from a URL.",
            web_fetch,
        ))
        self.registry.register(Tool(
            "file_read",
            "Read the contents of a text file from the filesystem.",
            file_read,
        ))
        self.registry.register(Tool(
            "file_write",
            "Write or append text to a file on the filesystem.",
            file_write,
        ))
        self.registry.register(Tool(
            "file_list",
            "List files and directories at a given path.",
            file_list,
        ))
        self.registry.register(Tool(
            "run_command",
            "Run a shell command and return its output. Use with caution — can modify the system.",
            run_command,
        ))

    def _build_prompt(self, user_message: str) -> str:
        """
        Build the full prompt in Qwen3 chat-template format.

        The Qwen3 instruct model expects the ChatML-style template:
          <|im_start|>system
          ...system content...
          <|im_end|>
          <|im_start|>user
          ...user message...
          <|im_end|>
          <|im_start|>assistant

        Tool definitions (from the registry) are injected into the system
        message so the model knows what tools it can call and what format
        to use (<tool_call>{...}</tool_call>).
        """
        parts = []

        # System prompt — base instructions + tool definitions
        sys_with_tools = self.system_prompt + self.registry.to_system_prompt()
        parts.append(f"<|im_start|>system\n{sys_with_tools}<|im_end|>")

        # Conversation history — all previous turns
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Current user message
        parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")

        # Assistant prompt — tells the model it's its turn to generate
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _parse_tool_calls(self, text: str) -> List[dict]:
        """
        Extract tool calls from raw model output.

        Scans the text for <tool_call>...</tool_call> blocks and attempts
        to JSON-parse the content inside each block.  Returns a list of
        dicts like:
          [{"name": "web_search", "arguments": {"query": "..."}}, ...]
        Malformed blocks are silently skipped.
        """
        calls = []
        for match in TOOL_CALL_PATTERN.finditer(text):
            try:
                obj = json.loads(match.group(1))
                if "name" in obj and "arguments" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass  # malformed tool call — ignore it
        return calls

    def _execute_tool_call(self, call: dict) -> str:
        """
        Execute a single tool call and return the result as a string.

        The result is also printed to stderr for visibility in the CLI.
        """
        name = call["name"]
        args = call.get("arguments", {})

        print(f"\n  🛠️  Calling tool: {name}({json.dumps(args, ensure_ascii=False)})")
        result = self.registry.execute(name, **args)
        print(f"  ✅ Result: {result[:200]}{'...' if len(result) > 200 else ''}")
        return result

    def _tool_call_signature(self, tool_calls: List[dict]) -> str:
        """Create a stable signature for a batch of tool calls."""
        return json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)

    def chat(self, message: str, stream: bool = True,
             external_streamer: Optional[callable] = None) -> str:
        """
        Send a user message to the agent.

        The agent loop:
          1. Build prompt with conversation history
          2. Generate model output (may contain <tool_call> blocks)
          3. Record assistant response in history
          4. If the output contains tool calls, execute them, inject results
             back into history, and loop back to step 1
          5. If no tool calls, return the final response

        Returns the final (non-tool-call) text from the model.
        """
        full_response = ""
        tool_rounds = 0
        last_tool_call_signature = ""

        while tool_rounds < self.max_tool_rounds:
            # Step 1: build the prompt from history + current message
            prompt = self._build_prompt(message)

            # Step 2: generate (streams output via external_streamer if provided)
            raw_output = self._generate(prompt, stream=stream,
                                        external_streamer=external_streamer)

            # Step 3: record in history (the raw output, markers included)
            self.messages.append({"role": "assistant", "content": raw_output})

            # Step 4: check for tool calls
            tool_calls = self._parse_tool_calls(raw_output)

            if not tool_calls:
                full_response = raw_output
                break  # normal response — exit the loop

            tool_call_signature = self._tool_call_signature(tool_calls)
            if tool_call_signature == last_tool_call_signature:
                full_response = (
                    "I stopped because the model repeated the same tool call. "
                    "Please refine the request or ask me to summarize the results."
                )
                break
            last_tool_call_signature = tool_call_signature

            # Execute each tool call and inject results into history
            for tc in tool_calls:
                result = self._execute_tool_call(tc)
                self.messages.append({
                    "role": "tool",
                    "content": f'<tool_response>\n{result}\n</tool_response>',
                })

            tool_rounds += 1

            # On the next iteration, the history includes the tool responses,
            # so the model can generate the final answer or call more tools.
            # Override user_message with a continuation prompt.
            message = "Continue with the tool results above and provide your response."

        # Fallback if we hit the round limit without a non-tool-call response
        if tool_rounds >= self.max_tool_rounds and not full_response:
            full_response = raw_output if 'raw_output' in locals() \
                else "I've reached the maximum number of tool calls."

        # Save a clean version (without tool_call markup) to history.
        # Avoid duplicating the final assistant message when the last raw
        # model output was already the final response.
        if not self.messages or self.messages[-1].get("content") != full_response:
            self.messages.append({"role": "assistant", "content": full_response})

        return full_response

    def _generate(
        self,
        prompt: str,
        stream: bool = True,
        external_streamer: Optional[callable] = None,
    ) -> str:
        """
        Run model generation via the OpenVINO GenAI pipeline.

        Args:
          prompt: the full text prompt (system + history + user)
          stream: if True, display output token by token as it's generated
          external_streamer: optional callable that receives each subword
                             (used by run_agent.py's SmartStreamer for
                              colored <think>/<tool_call> display)

        Returns:
          The full generated text as a string.
        """
        config = ov_genai.GenerationConfig()
        config.do_sample = True
        config.temperature = 0.7
        config.top_p = 0.9
        config.top_k = 50
        config.max_new_tokens = self.max_new_tokens

        if stream:
            collected = []  # accumulate subwords to reconstruct the full text

            # Default streamer — just prints each subword to stdout
            def default_streamer(subword: str) -> bool:
                print(subword, end="", flush=True)
                collected.append(subword)
                return False  # False = continue generation

            # Use the external streamer if provided, otherwise the default
            callback = external_streamer or default_streamer

            # Wrapper that collects AND calls the user's callback
            # We need to collect here because the external streamer may
            # colorize but not return the raw text.
            def wrapper_streamer(subword: str) -> bool:
                collected.append(subword)
                return callback(subword)

            try:
                self.pipe.generate(prompt, generation_config=config,
                                   streamer=wrapper_streamer)
            except KeyboardInterrupt:
                # Let external streamers flush their buffers (e.g. SmartStreamer
                # needs to emit remaining colored content before we raise).
                if external_streamer:
                    external_streamer.finish() if hasattr(external_streamer, 'finish') else None
                else:
                    print(flush=True)
                # Don't add partial output to history — let the caller know.
                raise GenerationInterrupted()
            finally:
                if external_streamer:
                    external_streamer.finish() if hasattr(external_streamer, 'finish') else None
                else:
                    print(flush=True)

            return "".join(collected)
        else:
            # Non-streaming: generate the full output in one shot
            result = self.pipe.generate(prompt, generation_config=config)
            return str(result)

    @property
    def token_estimate(self) -> dict:
        """
        Rough token count estimate from conversation history.

        Uses the heuristic ~4 characters ≈ 1 token, which is reasonable
        for mixed Chinese + English text (common for Qwen models).
        The actual token count depends on the tokenizer, but this gives
        a ballpark usage figure.
        """
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        prompt_chars = total_chars
        completion_chars = sum(
            len(m.get("content", ""))
            for m in self.messages if m.get("role") == "assistant"
        )
        return {
            "prompt": prompt_chars // 4,
            "completion": completion_chars // 4,
            "total": total_chars // 4,
            "messages": len(self.messages),
            "context_window": 32768,  # Qwen3-4B default context window
        }

    def reset(self):
        """Clear conversation history (starts a fresh session)."""
        self.messages = []


def create_agent(
    model_path: str = "/home/luna/Documents/llm/models/qwen3-4b-int8-ov/",
    device: str = "CPU",
    device_config: Optional[dict] = None,
    max_new_tokens: int = 8192,
    max_tool_rounds: int = 3,
) -> Agent:
    """Factory function to create a ready-to-use agent with defaults."""
    return Agent(
        model_path=model_path,
        device=device,
        device_config=device_config,
        max_new_tokens=max_new_tokens,
        max_tool_rounds=max_tool_rounds,
    )
