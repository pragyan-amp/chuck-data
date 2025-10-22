"""LLM Provider Protocol."""

from typing import Protocol, Optional, List, Dict, Any
from openai.types.chat import ChatCompletion


class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tool_choice: str = "auto",
    ) -> ChatCompletion:
        """Send chat request to LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (provider-specific)
            tools: Optional tool definitions (OpenAI format)
            stream: Whether to stream response
            tool_choice: "auto", "required", or "none"

        Returns:
            OpenAI ChatCompletion object
        """
        ...
