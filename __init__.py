"""
Agentify — turn your OpenVINO GenAI LLM into an autonomous agent.

Modules:
  tool_registry  – Define, register, and execute tools.
  agent          – The main agent loop (LLM + tools + memory).
  tools/         – Built-in tools (web, files, shell).

Quick start:
    from agentify.agent import create_agent
    agent = create_agent()
    response = agent.chat("What's the weather in Shanghai?")
"""

from .agent import Agent, create_agent
from .tool_registry import Tool, ToolRegistry

__all__ = ["Agent", "create_agent", "Tool", "ToolRegistry"]
