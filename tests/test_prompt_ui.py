import asyncio
import inspect
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import flet as ft
import pytest
import yaml

from ai_workbench.app import WorkbenchUI
from ai_workbench.autopilot_runner import AutopilotStepResult, AutopilotStepStatus
from ai_workbench.config import ConfigurationService
from ai_workbench.history import HistoryService
from ai_workbench.models import ModelAdapter, ModelRequestError
from ai_workbench.verification import (
    VerificationCommandResult,
    VerificationResult,
    VerificationStatus,
)
from ai_workbench.workspace import WorkspaceService


class FakePage:
    def __init__(self) -> None:
        self.title = ""
        self.padding = 0
        self.controls: list[object] = []
        self.services: list[object] = []
        self.update_count = 0
        self.dialog: object | None = None

    def add(self, *controls: object) -> None:
        self.controls.extend(controls)

    def update(self) -> None:
        self.update_count += 1

    def show_dialog(self, dialog: object) -> None:
        self.dialog = dialog

    def pop_dialog(self) -> object | None:
        dialog = self.dialog
        self.dialog = None
        return dialog


class RecordingAdapter(ModelAdapter):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, context: str, prompt: str) -> str:
        self.calls.append((context, prompt))
        return self.response


class FailingAdapter(ModelAdapter):
    def complete(self, context: str, prompt: str) -> str:
        del context, prompt
        raise ModelRequestError(
            "The model request failed. Check the model configuration, credential, and service status."
        )


class BlockingAdapter(ModelAdapter):
    def __init__(self, response: str) -> None:
        self.response = response
        self.entered = threading.Event()
        self.release = threading.Event()

    def complete(self, context: str, prompt: str) -> str:
        del context, prompt
        self.entered.set()
        assert self.release.wait(timeout=3)
        return self.response


def make_ui(tmp_path: Path, adapter: ModelAdapter) -> tuple[WorkbenchUI, Path]:
    project = tmp_path / "project"
    (project / ".ai").mkdir(parents=True)
    (project / ".ai" / "context.md").write_text("project guidance", encoding="utf-8")
    (project / "existing.py").write_text("original\n", encoding="utf-8")
    configuration = ConfigurationService(tmp_path / "user-config")
    ui = WorkbenchUI(FakePage(), adapter=adapter, configuration=configuration)  # type: ignore[arg-type]
    ui.workspace = WorkspaceService(HistoryService(tmp_path / "history"))
    ui.build()
    assert ui.open_project_path(str(project))
    return ui, project


def test_ask_invokes_adapter_and_never_creates_an_accepted_proposal(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE existing.py>>>\nchanged\n<<<END FILE>>>")
    ui, project = make_ui(tmp_path, adapter)
    ui.prompt.value = "Explain the existing module"

    asyncio.run(ui.ask(None))  # type: ignore[arg-type]

    assert len(adapter.calls) == 1
    context, instruction = adapter.calls[0]
    assert "project guidance" in context
    assert "Explain the existing module" in instruction
    assert "read-only" in instruction
    assert ui.proposal is None
    assert [entry.data["kind"] for entry in ui.transcript.controls] == [
        "user", "activity", "model"
    ]
    model_entry = ui.transcript.controls[-1]
    assert isinstance(model_entry, ft.Container)
    assert model_entry.data["body"] == adapter.response
    assert (project / "existing.py").read_text(encoding="utf-8") == "original\n"


def test_change_routes_proposal_to_review_without_writing(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE existing.py>>>\nchanged\n<<<END FILE>>>")
    ui, project = make_ui(tmp_path, adapter)
    ui.prompt.value = "Update the module"

    asyncio.run(ui.change(None))  # type: ignore[arg-type]

    assert len(adapter.calls) == 1
    assert "Update the module" in adapter.calls[0][1]
    assert "reviewed whole-file proposal" in adapter.calls[0][1]
    assert ui.proposal is not None
    assert ui.proposal.files[0].relative_path == Path("existing.py")
    assert len(ui.transactions) == 1
    assert ui.transactions[0].container in ui.transcript.controls
    assert ui.transactions[0].status.value == "Awaiting review"
    assert (project / "existing.py").read_text(encoding="utf-8") == "original\n"


def test_context_inspection_uses_dialog_and_run_controls_never_disappear(tmp_path: Path) -> None:
    ui, _project = make_ui(tmp_path, RecordingAdapter("unused"))

    ui.inspect_assembled_context(None)  # type: ignore[arg-type]

    assert isinstance(ui.page.dialog, ft.AlertDialog)
    assert ui.autopilot_button.visible is True
    assert ui.autopilot_run_button.visible is True
    assert ui.autopilot_button.disabled is True
    assert ui.autopilot_run_button.disabled is True
    assert ui.autopilot_run_button.tooltip


def test_continue_uses_next_steps_as_one_change_task(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE continued.txt>>>\ndone\n<<<END FILE>>>")
    ui, project = make_ui(tmp_path, adapter)
    (project / ".ai" / "next-steps.md").write_text(
        "# Next step\n\nImplement the bounded follow-up.\n", encoding="utf-8"
    )
    ui.prompt.value = "this prompt must not be used"

    asyncio.run(ui.continue_next_step(None))  # type: ignore[arg-type]

    assert len(adapter.calls) == 1
    assert "Implement the bounded follow-up." in adapter.calls[0][1]
    assert "this prompt must not be used" not in adapter.calls[0][1]
    assert ui.proposal is not None
    assert not (project / "continued.txt").exists()
    continue_entry = next(
        entry for entry in ui.transcript.controls
        if entry.data.get("kind") == "user" and entry.data.get("mode") == "continue"
    )
    assert "Implement the bounded follow-up." in continue_entry.data["task"]


def test_continue_without_next_step_does_not_call_model(tmp_path: Path) -> None:
    adapter = RecordingAdapter("unexpected")
    ui, _project = make_ui(tmp_path, adapter)

    asyncio.run(ui.continue_next_step(None))  # type: ignore[arg-type]

    assert adapter.calls == []
    assert "No actionable next step" in (ui.status.value or "")
    assert ui.transcript.controls[-1].data["kind"] == "error"


def test_model_failure_is_user_friendly_and_clears_busy_state(tmp_path: Path) -> None:
    ui, _project = make_ui(tmp_path, FailingAdapter())
    ui.prompt.value = "Explain this"

    asyncio.run(ui.ask(None))  # type: ignore[arg-type]

    assert "model request failed" in (ui.status.value or "").lower()
    assert ui.transcript.controls[-1].data["kind"] == "error"
    assert "model request failed" in ui.transcript.controls[-1].data["text"].lower()
    assert ui.busy is False
    assert ui.progress.visible is False
    assert ui.ask_button.disabled is False
    assert ui.change_button.disabled is False
    assert ui.continue_button.disabled is False


def test_button_callbacks_are_async_and_busy_feedback_is_visible_during_request(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingAdapter(ModelAdapter):
        def complete(self, context: str, prompt: str) -> str:
            del context, prompt
            started.set()
            assert release.wait(timeout=2)
            return "answer"

    ui, _project = make_ui(tmp_path, BlockingAdapter())
    ui.prompt.value = "Explain this"

    async def exercise() -> None:
        task = asyncio.create_task(ui.ask(None))  # type: ignore[arg-type]
        assert await asyncio.to_thread(started.wait, 2)
        assert ui.status.value == "Fake model — thinking (ASK)..."
        assert ui.progress.visible is True
        assert ui.ask_button.disabled is True
        assert ui.change_button.disabled is True
        assert ui.continue_button.disabled is True
        release.set()
        await task

    assert inspect.iscoroutinefunction(ui.ask_button.on_click)
    assert inspect.iscoroutinefunction(ui.change_button.on_click)
    assert inspect.iscoroutinefunction(ui.continue_button.on_click)
    asyncio.run(exercise())
    assert ui.transcript.controls[-1].data["body"] == "answer"


def test_proposed_file_and_diff_are_inspectable(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE existing.py>>>\nchanged\n<<<END FILE>>>")
    ui, _project = make_ui(tmp_path, adapter)
    ui.prompt.value = "Update it"
    asyncio.run(ui.change(None))  # type: ignore[arg-type]
    transaction = ui.transactions[0]

    proposal_column = transaction.container.content
    assert isinstance(proposal_column, ft.Column)
    file_row = proposal_column.controls[2]
    assert isinstance(file_row, ft.Row)
    view_file, view_diff = file_row.controls[1:3]
    view_file.on_click(None)
    assert ui.page.dialog.title == "Proposed file · existing.py"  # type: ignore[union-attr]
    assert ui.page.dialog.content.value == "changed\n"  # type: ignore[union-attr]
    ui.page.pop_dialog()
    view_diff.on_click(None)
    assert ui.page.dialog.title == "Unified diff · existing.py"  # type: ignore[union-attr]
    assert "-original" in ui.page.dialog.content.value  # type: ignore[union-attr]
    assert "+changed" in ui.page.dialog.content.value  # type: ignore[union-attr]


def test_accept_uses_workspace_once_and_marks_transaction_accepted(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE existing.py>>>\nchanged\n<<<END FILE>>>")
    ui, project = make_ui(tmp_path, adapter)
    original_accept = ui.workspace.accept
    ui.workspace.accept = Mock(wraps=original_accept)  # type: ignore[method-assign]
    ui.prompt.value = "Update it"
    asyncio.run(ui.change(None))  # type: ignore[arg-type]
    transaction = ui.transactions[0]

    ui.accept_transaction(transaction, None)  # type: ignore[arg-type]
    ui.accept_transaction(transaction, None)  # type: ignore[arg-type]

    assert ui.workspace.accept.call_count == 1  # type: ignore[attr-defined]
    assert (project / "existing.py").read_text(encoding="utf-8") == "changed\n"
    assert transaction.status.value == "Accepted"
    assert transaction.accept_button.disabled is True
    assert transaction.reject_button.disabled is True
    assert any((tmp_path / "history").rglob("manifest.json"))


def test_reject_marks_transaction_and_leaves_files_unchanged(tmp_path: Path) -> None:
    adapter = RecordingAdapter("<<<FILE existing.py>>>\nchanged\n<<<END FILE>>>")
    ui, project = make_ui(tmp_path, adapter)
    ui.prompt.value = "Update it"
    asyncio.run(ui.change(None))  # type: ignore[arg-type]
    transaction = ui.transactions[0]

    ui.reject_transaction(transaction, None)  # type: ignore[arg-type]

    assert transaction.status.value == "Rejected"
    assert transaction.resolved is True
    assert (project / "existing.py").read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize(
    ("managed_name", "original", "replacement"),
    [
        ("next-steps.md", "Print the name.\n", "Print the name in green when -n is used.\n"),
        ("memory.md", "The CLI prints names.\n", "The CLI supports a green -n mode.\n"),
    ],
)
def test_managed_ai_file_is_automatic_context_and_valid_proposal_target(
    tmp_path: Path,
    managed_name: str,
    original: str,
    replacement: str,
) -> None:
    project = tmp_path / "project"
    control_dir = project / ".ai"
    control_dir.mkdir(parents=True)
    (control_dir / "context.md").write_text("Project guidance.\n", encoding="utf-8")
    (control_dir / managed_name).write_text(original, encoding="utf-8")
    (project / "main.py").write_text("print('name')\n", encoding="utf-8")
    target = f".ai/{managed_name}"
    adapter = RecordingAdapter(f"<<<FILE {target}>>>\n{replacement}<<<END FILE>>>")
    ui = WorkbenchUI(
        FakePage(),
        adapter=adapter,
        configuration=ConfigurationService(tmp_path / "user-config"),
    )  # type: ignore[arg-type]
    ui.workspace = WorkspaceService(HistoryService(tmp_path / "history"))
    ui.build()
    assert ui.open_project_path(str(project))
    ui.prompt.value = f"Update {target}"

    asyncio.run(ui.change(None))  # type: ignore[arg-type]

    assembled_context, _instruction = adapter.calls[0]
    assert f"{managed_name}" in assembled_context
    assert original.strip() in assembled_context
    assert target not in [control.label for control in ui.files.controls]
    assert [item.relative_path.as_posix() for item in ui.transactions[0].proposal.files] == [target]
    assert ui.transactions[0].status.value == "Awaiting review"
    assert (control_dir / managed_name).read_text(encoding="utf-8") == original

    ui.accept_transaction(ui.transactions[0], None)  # type: ignore[arg-type]

    assert (control_dir / managed_name).read_text(encoding="utf-8") == replacement
    assert replacement.strip() in ui.assemble_context()


def test_run_one_autopilot_step_control_executes_once_and_appends_result(tmp_path: Path) -> None:
    project = tmp_path / "autopilot-project"
    control = project / ".ai"
    control.mkdir(parents=True)
    (control / "goal.md").write_text("Deliver a verified result.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("Perform one step.\n", encoding="utf-8")
    (control / "project.yml").write_text(
        yaml.safe_dump({
            "verify": {"commands": [[sys.executable, "-c", "print('ui verified')"]]},
            "autopilot": {
                "max_steps": 25,
                "max_consecutive_failures": 4,
                "protected_paths": [".ai/goal.md"],
                "verification_contract_paths": ["tests/**"],
            },
        }),
        encoding="utf-8",
    )
    adapter = RecordingAdapter(
        "<<<FILE .ai/next-steps.md>>>\nPerform the next step.\n<<<END FILE>>>"
    )
    ui = WorkbenchUI(
        FakePage(),
        adapter=adapter,
        configuration=ConfigurationService(tmp_path / "user-config"),
    )  # type: ignore[arg-type]
    ui.workspace = WorkspaceService(HistoryService(tmp_path / "history"))
    ui.autopilot_runner.workspace = ui.workspace
    ui.build()
    assert ui.open_project_path(str(project))

    assert ui.autopilot_button.visible is True
    asyncio.run(ui.run_one_autopilot_step(None))  # type: ignore[arg-type]

    assert len(adapter.calls) == 1
    assert (control / "next-steps.md").read_text(encoding="utf-8") == "Perform the next step.\n"
    result_entry = next(
        entry for entry in reversed(ui.transcript.controls)
        if entry.data.get("kind") == "model"
    )
    assert result_entry.data["title"] == "Autopilot · SUCCESS"
    assert "ui verified" in result_entry.data["body"]
    assert ui.busy is False


def test_run_autopilot_control_runs_until_completion_and_reports_summary(tmp_path: Path) -> None:
    project = tmp_path / "autopilot-loop-project"
    control = project / ".ai"
    control.mkdir(parents=True)
    (control / "goal.md").write_text("Deliver a verified result.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("Perform the bounded step.\n", encoding="utf-8")
    (control / "project.yml").write_text(yaml.safe_dump({
        "verify": {"commands": [[sys.executable, "-c", "print('passed')"]]},
        "autopilot": {
            "max_steps": 3,
            "max_consecutive_failures": 2,
            "protected_paths": [".ai/goal.md"],
            "verification_contract_paths": ["acceptance/**"],
        },
    }), encoding="utf-8")
    adapter = RecordingAdapter(
        "<<<FILE .ai/next-steps.md>>>\n# Complete\n<<<END FILE>>>"
    )
    ui = WorkbenchUI(
        FakePage(),
        adapter=adapter,
        configuration=ConfigurationService(tmp_path / "user-config"),
    )  # type: ignore[arg-type]
    ui.workspace = WorkspaceService(HistoryService(tmp_path / "history"))
    ui.autopilot_runner.workspace = ui.workspace
    ui.build()
    assert ui.open_project_path(str(project))

    asyncio.run(ui.run_autopilot(None))  # type: ignore[arg-type]

    assert len(adapter.calls) == 1
    assert "Autopilot completed: COMPLETED" in ui.status.value
    assert any(
        entry.data.get("title") == "Attempt 1 result · SUCCESS"
        for entry in ui.transcript.controls
    )
    assert ui.autopilot_run_button.visible is True
    assert ui.autopilot_run_button.disabled is True
    assert ui.autopilot_pause_button.visible is True
    assert ui.autopilot_stop_button.visible is True
    assert ui.autopilot_pause_button.disabled is True
    assert ui.autopilot_stop_button.disabled is True


def test_autopilot_streams_context_and_waiting_state_before_model_returns(tmp_path: Path) -> None:
    project = tmp_path / "visible-autopilot-project"
    control = project / ".ai"
    control.mkdir(parents=True)
    (control / "goal.md").write_text("Visible goal.\n", encoding="utf-8")
    (control / "context.md").write_text("Visible context.\n", encoding="utf-8")
    (control / "memory.md").write_text("Visible memory.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("Implement Step 1.\n", encoding="utf-8")
    (project / "textstats.py").write_text("def count():\n    return 1\n", encoding="utf-8")
    acceptance = project / "acceptance"
    acceptance.mkdir()
    (acceptance / "test_cli.py").write_text("def test_cli():\n    assert True\n", encoding="utf-8")
    (control / "project.yml").write_text(yaml.safe_dump({
        "verify": {"commands": [[sys.executable, "-c", "print('passed')"]]},
        "autopilot": {
            "max_steps": 2,
            "max_consecutive_failures": 2,
            "protected_paths": [".ai/goal.md"],
            "verification_contract_paths": ["acceptance/**"],
            "context_paths": ["textstats.py", "acceptance/**"],
        },
    }), encoding="utf-8")
    adapter = BlockingAdapter(
        "<<<FILE .ai/next-steps.md>>>\n# Complete\n<<<END FILE>>>"
    )
    ui = WorkbenchUI(
        FakePage(), adapter=adapter,
        configuration=ConfigurationService(tmp_path / "user-config"),
    )  # type: ignore[arg-type]
    ui.workspace = WorkspaceService(HistoryService(tmp_path / "history"))
    ui.autopilot_runner.workspace = ui.workspace
    ui.build()
    assert ui.open_project_path(str(project))
    assert ui.selected_files() == [Path("acceptance/test_cli.py"), Path("textstats.py")]
    assert any(control.data == Path("textstats.py") for control in ui.files.controls)
    assert ui.project_files_heading.value == "Project files    2 selected / 2 files"
    assert ui.files.scroll is ft.ScrollMode.ALWAYS
    assert ui.context_summary.value == (
        "2 project files selected · 1 writable · 1 protected · 4 managed AI files"
    )
    assert ui.autopilot_status.value == "STATUS: READY"
    assert ui.autopilot_goal.value == "Visible goal."
    assert "Implement Step 1." in ui.autopilot_plan.value
    assert ui.autopilot_context_summary.value == "6 files, 2 project + 4 managed"
    assert "print('passed')" in ui.autopilot_verification.value

    async def exercise() -> None:
        running = asyncio.create_task(ui.run_autopilot(None))  # type: ignore[arg-type]
        assert await asyncio.to_thread(adapter.entered.wait, 2)
        await asyncio.sleep(0.15)
        entries = [entry.data for entry in ui.transcript.controls]
        attempt = next(item for item in entries if item.get("title") == "Autopilot attempt 1")
        assert "Implement Step 1." in attempt["body"]
        context = next(item for item in entries if item.get("text", "").startswith("Managed context:"))
        assert ".ai/goal.md" in context["text"]
        assert ".ai/context.md" in context["text"]
        assert ".ai/memory.md" in context["text"]
        assert ".ai/next-steps.md" in context["text"]
        assert "textstats.py  writable" in context["text"]
        assert "acceptance/test_cli.py  protected" in context["text"]
        assert any(item.get("text") == "Requesting model response..." for item in entries)
        adapter.release.set()
        await running

    asyncio.run(exercise())

    activity = [
        entry.data.get("text", "") for entry in ui.transcript.controls
        if entry.data.get("kind") == "activity"
    ]
    assert "Model response received." in activity
    assert any(text.startswith("Proposal files:\n  .ai/next-steps.md") for text in activity)
    assert "Applying speculative transaction..." in activity
    assert "Running verification..." in activity
    result_entry = next(
        entry.data for entry in ui.transcript.controls
        if entry.data.get("title") == "Attempt 1 result · SUCCESS"
    )
    assert "**Verification:** PASS" in result_entry["body"]
    assert "**Result:** advance to next step" in result_entry["body"]
    textstats = next(control for control in ui.files.controls if control.data == Path("textstats.py"))
    textstats.value = False
    ui.autopilot_context_selection_changed(None)  # type: ignore[arg-type]
    saved = yaml.safe_load((control / "project.yml").read_text(encoding="utf-8"))
    assert saved["autopilot"]["context_paths"] == ["acceptance/test_cli.py"]


def test_autopilot_failure_transcript_reports_rollback_and_diagnostics(tmp_path: Path) -> None:
    result = AutopilotStepResult(
        status=AutopilotStepStatus.VERIFICATION_FAILED,
        step="Repair the feature.",
        changed_paths=(Path("app.py"), Path(".ai/next-steps.md")),
        verification=VerificationResult(
            VerificationStatus.FAIL,
            (VerificationCommandResult(
                ("pytest", "-q"),
                VerificationStatus.FAIL,
                1,
                "14 passed, 1 failed\n",
                "assertion detail\n",
            ),),
        ),
        applied=True,
        snapshot_path=tmp_path / "snapshot",
        rollback_attempted=True,
        rollback_succeeded=True,
        changes_kept=False,
    )

    text = WorkbenchUI.format_autopilot_result(result)

    assert "**Verification:** FAIL" in text
    assert "**Changes rolled back:** yes" in text
    assert "Working state restored to the complete pre-step snapshot." in text
    assert "14 passed, 1 failed" in text
    assert "assertion detail" in text
