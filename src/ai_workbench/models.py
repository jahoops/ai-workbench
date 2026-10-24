from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import ModelConfiguration


class ModelAdapter(ABC):
    @abstractmethod
    def complete(self, context: str, prompt: str) -> str:
        """Return model text without receiving filesystem or command capabilities."""


class FakeModelAdapter(ModelAdapter):
    def complete(self, context: str, prompt: str) -> str:
        del context, prompt
        return (
            "<<<FILE AI_WORKBENCH_EXAMPLE.md>>>\n"
            "# Fake model proposal\n\n"
            "This file was proposed by the test adapter. Review before accepting.\n"
            "<<<END FILE>>>"
        )


class ModelRequestError(RuntimeError):
    """A deliberately sanitized model request failure."""


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        configuration: ModelConfiguration,
        api_key: str,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An API credential is required.")
        self.configuration = configuration
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    def complete(self, context: str, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.configuration.model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            **self.configuration.request_options(),
        }
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(
                    f"{self.configuration.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                document = response.json()
            content = document["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise KeyError("content")
            return content
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            del exc
            raise ModelRequestError(
                "The model request failed. Check the model configuration, credential, and service status."
            ) from None
