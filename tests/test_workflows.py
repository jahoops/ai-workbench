from pathlib import Path

import pytest

from ai_workbench.models import ModelAdapter, ModelRequestError
from ai_workbench.workflows import PromptWorkflow


class RecordingAdapter(ModelAdapter):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, context: str, prompt: str) -> str:
        self.calls.append((context, prompt))
        return self.response


def test_ask_calls_adapter_as_read_only_and_does_not_parse_proposal(tmp_path: Path) -> None:
    del tmp_path
    model_text = "<<<FILE should-not-be-reviewed.txt>>>\ntext\n<<<END FILE>>>"
    adapter = RecordingAdapter(model_text)

    result = PromptWorkflow().ask(adapter, "assembled context", "Explain this code")

    assert result.response == model_text
    assert result.proposal is None
    assert adapter.calls[0][0] == "assembled context"
    assert "ASK (read-only explanation)" in adapter.calls[0][1]
    assert "Explain this code" in adapter.calls[0][1]
    assert "Do not propose file replacements" in adapter.calls[0][1]


def test_change_calls_adapter_and_routes_response_to_proposal_parser(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE changed.py>>>\nnew contents\n<<<END FILE>>>")

    result = PromptWorkflow().change(adapter, tmp_path, "context", "Make the change")

    assert result.proposal is not None
    assert result.proposal.files[0].relative_path == Path("changed.py")
    assert result.proposal.files[0].content == "new contents\n"
    assert "CHANGE (reviewed whole-file proposal)" in adapter.calls[0][1]
    assert "Make the change" in adapter.calls[0][1]
    assert "<<<FILE relative/path.py>>>" in adapter.calls[0][1]
    assert ".ai/next-steps.md" in adapter.calls[0][1]
    assert ".ai/memory.md" in adapter.calls[0][1]


def test_model_failure_propagates_only_sanitized_adapter_error() -> None:
    class FailingAdapter(ModelAdapter):
        def complete(self, context: str, prompt: str) -> str:
            del context, prompt
            raise ModelRequestError("The model request failed. Check service status.")

    with pytest.raises(ModelRequestError, match="Check service status"):
        PromptWorkflow().ask(FailingAdapter(), "context", "question")
