import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Optional, List, Dict

import openvino as ov
import openvino_genai as ov_genai

from agentify_openvino.agent import create_agent
from agentify_openvino.streamers import SmartStreamer


SUPPORTED_DEVICES = ("CPU", "GPU", "NPU")


def normalize_device_name(value: str) -> str:
    """Normalize a requested OpenVINO device name."""
    device = value.strip().upper()
    if device not in SUPPORTED_DEVICES:
        raise argparse.ArgumentTypeError(
            f"Unsupported device '{value}'. Choose one of: {', '.join(SUPPORTED_DEVICES)}"
        )
    return device


def build_device_config(device: str) -> dict:
    """
    Build OpenVINO GenAI pipeline config for a requested device.

    NPU needs an explicit prompt-length budget for chat workloads, because the
    default NPU pipeline settings are much tighter than CPU/GPU.
    """
    if device == "NPU":
        return {
            "MAX_PROMPT_LEN": 16384,
            "MIN_RESPONSE_LEN": 1,
        }
    return {}


def device_is_available(requested: str, available_devices: list[str]) -> bool:
    """Check whether the requested OpenVINO device is present."""
    if requested == "GPU":
        return any(dev == "GPU" or dev.startswith("GPU.") for dev in available_devices)
    return requested in available_devices


# ── Workspace and Context Management ──────────────────────────────────────
# The workspace system provides:
#   1. Persistent conversation context across sessions
#   2. Markdown-based system prompts that auto-load
#   3. Organized file storage for agent operations
#
# Workspace structure:
#   workspace/
#   ├── system_prompt.md    # System prompt in markdown format
#   ├── context.json        # Saved conversation history
#   └── files/              # Working directory for file operations
#

def build_prompt_with_history(messages: List[dict]) -> str:
    """
    Build a prompt with conversation history in ChatML format.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        Formatted prompt string
    """
    parts = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Use proper ChatML format with im_start and im_end tokens
        # Building strings to avoid XML tag interpretation issues
        start_tag = "<" + "|im_start|" + ">"
        end_tag = "<" + "|im_end|" + ">"
        parts.append(f"{start_tag}{role}\n{content}{end_tag}")
    
    # Add assistant prompt to indicate it's time to generate
    start_tag = "<" + "|im_start|" + ">"
    parts.append(f"{start_tag}assistant\n")
    
    return "\n".join(parts)

class WorkspaceManager:
    """
    Manages the agent workspace including context persistence and system prompts.
    
    The workspace directory structure:
    workspace/
    ├── system_prompt.md    # System prompt in markdown format
    ├── context.json        # Conversation context/history
    └── files/              # Working directory for file operations
    """
    
    def __init__(self, workspace_path: Optional[str] = None):
        if workspace_path:
            self.workspace_dir = Path(workspace_path).expanduser().resolve()
        else:
            # Default workspace in current directory
            self.workspace_dir = Path.cwd() / "workspace"
        
        # Ensure workspace directories exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "files").mkdir(exist_ok=True)
        
        self.system_prompt_file = self.workspace_dir / "system_prompt.md"
        self.context_file = self.workspace_dir / "context.json"
        
        # Create default system prompt if it doesn't exist
        self._ensure_default_system_prompt()
    
    def _ensure_default_system_prompt(self):
        """Create a default system prompt file if it doesn't exist."""
        if not self.system_prompt_file.exists():
            default_prompt = """# AI Assistant System Prompt

## Role
You are a helpful AI assistant with advanced reasoning capabilities. You should think briefly, use tools only when they add new information, and answer directly.

## Capabilities
- Answer questions accurately and comprehensively
- Think through problems briefly using <think> tags
- Use available tools when needed (web search, file operations, commands)
- Provide clear, well-structured responses

## Guidelines
1. **Think Briefly**: Use <think>...</think> tags for short internal reasoning, then answer directly
2. **Be Helpful**: Provide accurate, relevant, and complete information
3. **Use Tools Once**: When you need external information, use the appropriate tool once and then summarize the result
4. **Be Clear**: Structure your responses clearly with proper formatting
5. **Stay Focused**: Address the user's question directly

## Response Format
- Use <think> tags for internal reasoning (these will be hidden from the user)
- Provide final answers in clear, natural language
- When using tools, explain what you're doing and then conclude

## Available Tools
The system will automatically inject tool definitions here based on registered tools.
"""
            self.system_prompt_file.write_text(default_prompt, encoding="utf-8")
            print(f"Created default system prompt at: {self.system_prompt_file}")
    
    def load_system_prompt(self) -> str:
        """Load system prompt from markdown file."""
        if self.system_prompt_file.exists():
            return self.system_prompt_file.read_text(encoding="utf-8")
        return ""
    
    def save_context(self, messages: List[dict], metadata: Optional[dict] = None):
        """
        Save conversation context to JSON file.
        
        Args:
            messages: List of message dicts with role and content
            metadata: Additional metadata (timestamp, token counts, etc.)
        """
        context_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "metadata": metadata or {},
        }
        
        try:
            with open(self.context_file, "w", encoding="utf-8") as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Context saved to: {self.context_file}")
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to save context: {e}")
    
    def load_context(self) -> Optional[dict]:
        """
        Load conversation context from JSON file.
        
        Returns:
            Dict with 'messages' and 'metadata', or None if file doesn't exist
        """
        if not self.context_file.exists():
            return None
        
        try:
            with open(self.context_file, "r", encoding="utf-8") as f:
                context_data = json.load(f)
            print(f"\n📂 Loaded context from: {self.context_file}")
            print(f"   Messages: {len(context_data.get('messages', []))}")
            return context_data
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to load context: {e}")
            return None
    
    def clear_context(self):
        """Clear the saved context file."""
        if self.context_file.exists():
            self.context_file.unlink()
            print(f"🗑️  Cleared context from: {self.context_file}")
    
    @property
    def workspace_info(self) -> dict:
        """Get information about the workspace."""
        return {
            "path": str(self.workspace_dir),
            "system_prompt_exists": self.system_prompt_file.exists(),
            "context_exists": self.context_file.exists(),
            "files_dir": str(self.workspace_dir / "files"),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LLM inference using OpenVINO with Qwen3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm.py --message "What is Python?"
  python llm.py -m "Explain quantum computing" --temperature 0.9
  python llm.py -m "Hello" --model-path /path/to/model --device GPU
  python llm.py -m "Test" --workspace ./my_workspace --load-context
        """
    )
    
    # Required arguments
    ap.add_argument(
        "-m", "--message", 
        type=str, 
        help="Message to send to the LLM model (required)"
    )
    
    # Workspace and context arguments
    ap.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Path to workspace directory (default: ./workspace)"
    )
    
    ap.add_argument(
        "--load-context",
        action="store_true",
        default=False,
        help="Load previous conversation context from workspace"
    )
    
    ap.add_argument(
        "--clear-context",
        action="store_true",
        default=False,
        help="Clear saved context before starting"
    )
    
    # Model configuration arguments
    ap.add_argument(
        "--model-path",
        type=str,
        default="~/Documents/llm/models/Qwen3-8B-int4-cw-ov/",
        help="Path to the OpenVINO model directory (default: ~/Documents/llm/models/qwen3-4b-int8-ov)"
    )
    
    ap.add_argument(
        "--device",
        type=normalize_device_name,
        default="CPU",
        choices=list(SUPPORTED_DEVICES),
        help=(
            "OpenVINO device to run inference on: CPU, GPU, or NPU "
            "(GPU here means Intel OpenVINO GPU, not NVIDIA CUDA)"
        )
    )
    
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation (default: 0.8, range: 0.0-2.0)"
    )
    
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold (default: 0.9, range: 0.0-1.0)"
    )
    
    ap.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    
    ap.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Enable sampling during generation (enabled by default)"
    )
    
    ap.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (greedy decoding)"
    )
    
    ap.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (overrides workspace system_prompt.md)"
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate (default: 8192)"
    )

    ap.add_argument(
        "--verbose-info",
        action="store_true",
        help="Verbose detailed runtime information."
    )

    if argv is not None and len(argv) == 0:
        ap.print_help()
        raise SystemExit(0)
    
    return ap.parse_args()

def run_llm_agent(args: argparse.Namespace, workspace: WorkspaceManager):
    # Display workspace information
    if args.verbose_info:
        print("\n" + "="*60)
        print("📁 Workspace Information")
        print("="*60)
        for key, value in workspace.workspace_info.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
    
    # Load system prompt from workspace (markdown file)
    if args.system_prompt:
        system_prompt = args.system_prompt
        if args.verbose_info:
            print("Using custom system prompt from command line")
    else:
        system_prompt = workspace.load_system_prompt()
        if args.verbose_info:
            if system_prompt:
                print(f"Loaded system prompt from: {workspace.system_prompt_file}")
            else:
                print(f"No system prompt found at: {workspace.system_prompt_file}; using empty prompt")
        elif not system_prompt:
            print(f"No system prompt found at: {workspace.system_prompt_file}; using empty prompt")

    # Initialize OpenVINO
    core = ov.Core()
    model_path: Path = Path(args.model_path).expanduser().resolve()
    available_devices = list(core.available_devices)
    device_config = build_device_config(args.device)

    if args.verbose_info:
        print(f"\nAvailable devices: {core.available_devices}")
        print(f"Model path: {model_path}, exists: {model_path.exists()}")
        print(f"Requested device: {args.device}")
        if device_config:
            print(f"Device config: {device_config}")

    if not device_is_available(args.device, available_devices):
        raise SystemExit(
            f"Requested device '{args.device}' is not available in OpenVINO. "
            f"Available devices: {available_devices}"
        )

    if args.device == "GPU":
        print(
            "Note: OpenVINO's GPU backend targets Intel GPU devices, not NVIDIA GPUs."
        )

    agent = create_agent(
        model_path,
        args.device,
        device_config=device_config,
        max_new_tokens=args.max_new_tokens,
    )
    agent.system_prompt = f"{system_prompt}\nYour workspace: {workspace.workspace_info}"

    streamer = SmartStreamer()

    loaded_messages: List[Dict[str, str]] = []

    # Load previous context if requested
    if args.load_context:
        context_data = workspace.load_context()
        if context_data:
            previous_messages = context_data.get("messages", [])
            filtered_messages = [msg for msg in previous_messages if msg.get("role") != "system"]
            agent.messages.extend(filtered_messages)
            loaded_messages = filtered_messages
            print(f"Loaded {len(filtered_messages)} messages from previous context")

    if args.verbose_info:
        print("\n" + "="*60)
        print("🤖 Starting LLM Generation")
        print("="*60 + "\n")
    
    try:
        # Generate response
        assistant_response = agent.chat(args.message, stream=True, external_streamer=streamer)

        # Save context after successful generation
        saved_messages = [{"role": "system", "content": agent.system_prompt}]
        saved_messages.extend(loaded_messages)
        saved_messages.append({"role": "user", "content": args.message})
        saved_messages.append({"role": "assistant", "content": assistant_response})
        workspace.save_context(
            messages=saved_messages,
            metadata={
                "model_path": str(model_path),
                "device": args.device,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        # Still save context even if interrupted (with what we have so far)
        saved_messages = [{"role": "system", "content": agent.system_prompt}]
        saved_messages.extend(loaded_messages)
        saved_messages.append({"role": "user", "content": args.message})
        if streamer.collected_text:
            assistant_response = streamer.get_full_response()
            saved_messages.append({"role": "assistant", "content": assistant_response})
        workspace.save_context(
            messages=saved_messages,
            metadata={"interrupted": True}
        )
    finally:
        streamer.finish()
        print("\n✅ Session complete. Context has been saved.")


def main():
    args = parse_args(sys.argv[1:])

    # Initialize workspace manager
    workspace = WorkspaceManager(args.workspace)

    # go into message
    if args.message:
        run_llm_agent(args, workspace)
    
    # Clear context if requested
    if args.clear_context:
        workspace.clear_context()

if __name__ == "__main__":
    main()
