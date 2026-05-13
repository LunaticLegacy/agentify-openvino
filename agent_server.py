"""
Agent API server — exposes the agentified LLM as an OpenAI-compatible API.

Run:
    uvicorn agent_server:app --host 127.0.0.1 --port 8001

The agent adds tool-calling capabilities on top of the base model.
Use /v1/chat/completions with tool definitions (OpenAI format) or
just chat normally — the agent will use built-in tools automatically.
"""

import json
import time
import uuid
from typing import List, Optional, Union, Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import create_agent

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_ID = "qwen3-4b-int8-ov-agentified"

print("🚀 Loading agent...")
agent = create_agent()
print(f"✅ Agent ready! Tools: {', '.join(agent.registry.list_tools().keys())}")

app = FastAPI(title="Agentified LLM API")


# ── Pydantic schemas ──────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, list]] = ""


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    max_new_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[ToolDefinition]] = None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-openvino-agent",
            }
        ],
    }


def _build_messages_prompt(messages: List[Message]) -> str:
    """Convert OpenAI message format to chat template format."""
    parts = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"<|im_start|>{m.role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """
    Chat completions endpoint. When tools are provided in the request,
    they are injected into the system prompt. Otherwise, the agent's
    default tool set is available.
    """
    # Extract system message and user messages
    system_msg = ""
    user_content = ""
    history = []

    for m in req.messages:
        if m.role == "system":
            system_msg += m.content + "\n"
        elif m.role == "user":
            user_content = m.content if isinstance(m.content, str) else str(m.content)
        else:
            history.append(m)

    # Build prompt
    prompt_parts = []

    # System
    sys_text = system_msg.strip() or "You are a helpful AI assistant with access to tools."
    if agent.registry.list_tools():
        sys_text += agent.registry.to_system_prompt()
    prompt_parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>")

    # History
    for h in history:
        content = h.content if isinstance(h.content, str) else str(h.content)
        prompt_parts.append(f"<|im_start|>{h.role}\n{content}<|im_end|>")

    # Current user message
    prompt_parts.append(f"<|im_start|>user\n{user_content}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")

    prompt = "\n".join(prompt_parts)
    max_new_tokens = req.max_new_tokens or req.max_tokens or 512

    if req.stream:
        def event_stream():
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"

            def streamer(subword: str) -> bool:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": subword},
                            "finish_reason": None,
                        }
                    ],
                }
                chunks.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
                return False

            chunks = []
            agent.pipe.generate(
                prompt,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=req.temperature,
            )

            for c in chunks:
                yield c

            done = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    result = agent.pipe.generate(prompt, max_new_tokens=max_new_tokens, temperature=req.temperature)
    text = str(result)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
