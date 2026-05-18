"""
Tool registry — define, register, and execute tools that the agent can call.
"""

import inspect
import json
from typing import Any, Callable, Dict, Optional


class Tool:
    """A single tool an agent can call."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn

        if parameters:
            self.parameters = parameters
        else:
            # Auto-generate parameters from the function signature
            sig = inspect.signature(fn)
            props = {}
            required = []
            for p_name, p_param in sig.parameters.items():
                p_type = str(p_param.annotation) if p_param.annotation != inspect.Parameter.empty else "string"
                type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
                js_type = type_map.get(p_type, "string")
                props[p_name] = {"type": js_type, "description": f"Parameter {p_name}"}
                if p_param.default == inspect.Parameter.empty:
                    required.append(p_name)
            self.parameters = {
                "type": "object",
                "properties": props,
                "required": required,
            }

    def run(self, **kwargs) -> str:
        """Execute the tool with given kwargs and return a string result."""
        try:
            result = self.fn(**kwargs)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            return result
        except Exception as e:
            return f"Error: {e}"

    def to_definition(self) -> Dict[str, Any]:
        """Return the tool definition for embedding in the system prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Collection of registered tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def register_fn(self, name: str, description: str, fn: Callable, parameters: Optional[Dict] = None):
        self._tools[name] = Tool(name, description, fn, parameters)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, Tool]:
        return dict(self._tools)

    def definitions(self) -> list:
        return [t.to_definition() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        return tool.run(**kwargs)

    def to_system_prompt(self) -> str:
        """Generate the system prompt section describing available tools."""
        if not self._tools:
            return ""

        lines = [
            "\nYou have access to the following tools. Use them when needed.",
            "To call a tool, respond with EXACTLY this format (no extra text around it):",
            "",
            '<tool_call>',
            '{"name": "<tool_name>", "arguments": {...}}',
            '</tool_call>',
            "",
            "After the tool responds, continue the conversation naturally.",
            "",
            "Available tools:",
        ]

        for t in self._tools.values():
            lines.append(f"\n  - {t.name}: {t.description}")
            lines.append(f"    Parameters: {json.dumps(t.parameters, ensure_ascii=False)}")
            lines.append(f"    Param types: {t.parameters['properties']}")

        lines.append("")
        return "\n".join(lines)
