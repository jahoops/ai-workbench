import json

import httpx
import pytest

from ai_workbench.config import ModelConfiguration
from ai_workbench.models import ModelRequestError, OpenAICompatibleAdapter


def configuration() -> ModelConfiguration:
    return ModelConfiguration(
        name="configured",
        adapter="openai-compatible",
        base_url="https://models.example/v1",
        model="example-model",
        api_key_env="EXAMPLE_KEY",
        thinking=True,
        reasoning_effort="max",
    )


def test_chat_completion_request_uses_configuration() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "proposal"}}]})

    adapter = OpenAICompatibleAdapter(configuration(), "top-secret", transport=httpx.MockTransport(handler))
    result = adapter.complete("project context", "make a change")

    assert result == "proposal"
    assert captured["url"] == "https://models.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer top-secret"
    assert captured["body"] == {
        "model": "example-model",
        "messages": [
            {"role": "system", "content": "project context"},
            {"role": "user", "content": "make a change"},
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }


def test_request_failure_is_sanitized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="provider echoed top-secret")

    adapter = OpenAICompatibleAdapter(configuration(), "top-secret", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelRequestError) as caught:
        adapter.complete("context", "prompt")

    assert "top-secret" not in str(caught.value)
    assert "provider echoed" not in str(caught.value)
    assert caught.value.__cause__ is None
