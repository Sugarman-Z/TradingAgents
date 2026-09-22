"""A stalled LLM HTTP connection must self-heal.

Left unset, the openai SDK's implicit defaults governed the request: a 600s
read timeout (connect 5s) compounded by the default retry budget, so one
wedged gateway connection stalled the graph for ten-plus minutes — and on
Windows the native socket read even ignored Ctrl+C — under a ceiling nobody
had chosen. Every OpenAI-compatible chat client now carries an explicit 300s
default request timeout, overridable via the existing ``timeout`` passthrough
kwarg.
"""

import pytest

from tradingagents.llm_clients.openai_client import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    OpenAIClient,
)


@pytest.mark.unit
def test_default_timeout_is_injected():
    llm = OpenAIClient(provider="glm-cn", model="glm-5.3-flash").get_llm()
    assert llm.request_timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS == 300


@pytest.mark.unit
def test_explicit_timeout_passthrough_wins():
    llm = OpenAIClient(provider="glm-cn", model="glm-5.3-flash", timeout=60).get_llm()
    assert llm.request_timeout == 60


@pytest.mark.unit
def test_default_applies_across_compatible_providers():
    for provider in ("openai", "deepseek", "glm-cn"):
        llm = OpenAIClient(provider=provider, model="any-model").get_llm()
        assert llm.request_timeout == 300
