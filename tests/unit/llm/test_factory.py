"""Tests for LLMProviderFactory."""

import pytest
import os
from unittest.mock import patch, MagicMock
from chuck_data.llm.factory import LLMProviderFactory


class TestLLMProviderFactory:
    """Test LLM provider factory behavior."""

    def test_get_available_providers(self):
        """Factory returns list of supported providers."""
        providers = LLMProviderFactory.get_available_providers()
        assert "databricks" in providers
        assert "aws_bedrock" in providers
        assert "mock" in providers

    def test_unknown_provider_raises_error(self):
        """Factory raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.create("nonexistent_provider")

    @patch.dict(os.environ, {}, clear=True)
    def test_provider_selection_precedence_default(self):
        """Default provider is databricks when no config."""
        with patch("chuck_data.config.get_config_manager") as mock_config:
            mock_config.return_value.get_config.return_value = MagicMock(
                llm_provider=None
            )

            provider_name = LLMProviderFactory._resolve_provider_name()
            assert provider_name == "databricks"

    @patch.dict(os.environ, {"CHUCK_LLM_PROVIDER": "mock"}, clear=True)
    def test_provider_selection_precedence_env_var(self):
        """Environment variable overrides config."""
        with patch("chuck_data.config.get_config_manager") as mock_config:
            mock_config.return_value.get_config.return_value = MagicMock(
                llm_provider="databricks"
            )

            provider_name = LLMProviderFactory._resolve_provider_name()
            assert provider_name == "mock"

    def test_provider_selection_precedence_explicit(self):
        """Explicit parameter has highest priority."""
        with patch.dict(os.environ, {"CHUCK_LLM_PROVIDER": "openai"}, clear=True):
            provider_name = LLMProviderFactory._resolve_provider_name("aws_bedrock")
            assert provider_name == "aws_bedrock"

    def test_get_provider_config_returns_empty_dict_on_error(self):
        """Provider config returns empty dict when config unavailable."""
        with patch(
            "chuck_data.config.get_config_manager", side_effect=Exception("No config")
        ):
            config = LLMProviderFactory._get_provider_config("databricks")
            assert config == {}
