"""LLM client for backward compatibility.

DEPRECATED: This client delegates to DatabricksProvider via the factory.
New code should use LLMProviderFactory.create() directly.
"""

import logging
from chuck_data.llm.providers.databricks import DatabricksProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """Backward-compatible LLM client.

    Maintains existing interface while delegating to DatabricksProvider.
    This allows existing code to continue working without changes.

    For new code, prefer:
        from chuck_data.llm.factory import LLMProviderFactory
        provider = LLMProviderFactory.create()
    """

    def __init__(self):
        """Initialize LLM client with Databricks provider."""
        self._provider = DatabricksProvider()

    def chat(self, messages, model=None, tools=None, stream=False, tool_choice="auto"):
        """Send chat request to LLM.

        Delegates to DatabricksProvider for backward compatibility.

        Args:
            messages: List of message objects
            model: Model to use (default from config)
            tools: List of tools to provide
            stream: Whether to stream the response
            tool_choice: "auto", "required", or "none"

        Returns:
            Response from the API
        """
        return self._provider.chat(
            messages=messages,
            model=model,
            tools=tools,
            stream=stream,
            tool_choice=tool_choice,
        )
