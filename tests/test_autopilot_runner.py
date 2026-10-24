import json
import sys
from pathlib import Path

import pytest
import yaml

from ai_workbench.autopilot_runner import AutopilotRunner, AutopilotStepStatus
from ai_workbench.history import HistoryService
from ai_workbench.models import ModelAdapter, ModelRequestError
from ai_workbench.verification import VerificationStatus
from ai_workbench.workspace import WorkspaceService


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
        raise ModelRequestError("sanitized failure")


def project(
    tmp_path: Path,
    *,
    verify_code: str = "print('verified')",
    timeout: float = 2,
) -> tuple[Path, Path]:
    root = tmp_path / "project"
    control = root / ".ai"
    control.mkdir(parents=True)
    (control / "goal.md").write_text("Deliver the tested implementation.\n", encoding="utf-8")
    (control / "context.md").write_text("Keep changes bounded.\n", encoding="utf-8")
    (control / "memory.md").write_text("Initial memory.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("Implement the current feature.\n", encoding="utf-8")
    (control / "project.yml").write_text(
        yaml.safe_dump({
            "verify": {
                "timeout_seconds": timeout,
                "commands": [[sys.executable, "-c", verify_code]],
            },
            "autopilot": {
                "max_steps": 25,
                "max_consecutive_failures": 4,
                "protected_paths": [".ai/goal.md", "acceptance/**"],
                "verification_contract_paths": ["test_*.py", "tests/**", "acceptance/**"],
                "context_paths": ["app.py", "acceptance/**"],
            },
        }),
        encoding="utf-8",
    )
    source = root / "app.py"
    source.write_text("old\n", encoding="utf-8")
    return root, source


def runner(tmp_path: Path) -> AutopilotRunner:
    return AutopilotRunner(
        workspace=WorkspaceService(HistoryService(tmp_path / "history"))
    )


def proposal(*files: tuple[str, str]) -> str:
    return "\n".join(
        f"<<<FILE {path}>>>\n{content}<<<END FILE>>>" for path, content in files
    )


def test_successful_single_step_applies_updates_verifies_and_snapshots(tmp_path: Path) -> None:
    root, source = project(
        tmp_path,
        verify_code="from pathlib import Path; assert Path('app.py').read_text() == 'new\\n'; print('ok')",
    )
    adapter = RecordingAdapter(proposal(
        ("app.py", "new\n"),
        (".ai/next-steps.md", "Run the next bounded task.\n"),
    ))

    result = runner(tmp_path).run(root, adapter)

    assert result.status is AutopilotStepStatus.SUCCESS
    assert result.applied is True
    assert result.changed_paths == (Path("app.py"), Path(".ai/next-steps.md"))
    assert source.read_text(encoding="utf-8") == "new\n"
    assert (root / ".ai/next-steps.md").read_text(encoding="utf-8") == "Run the next bounded task.\n"
    assert result.verification is not None
    assert result.verification.status is VerificationStatus.PASS
    assert result.verification.commands[0].stdout == "ok\n"
    assert result.snapshot_path is not None and result.snapshot_path.is_dir()
    assert result.rollback_attempted is False
    assert result.rollback_succeeded is None
    assert result.changes_kept is True
    manifest = json.loads((result.snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == ["app.py", ".ai/next-steps.md"]
    assert (result.snapshot_path / "files/0000.snapshot").read_text(encoding="utf-8") == "old\n"
    context, prompt = adapter.calls[0]
    assert "Project goal: .ai/goal.md" in context
    assert "Deliver the tested implementation." in context
    assert "exactly this one current project step" in prompt
    assert ".ai/goal.md" in prompt
    assert ".ai/memory.md" in prompt
    assert ".ai/next-steps.md" in prompt


@pytest.mark.parametrize(
    ("path", "expected_status"),
    [
        (".ai/goal.md", AutopilotStepStatus.PROTECTED_PATH_VIOLATION),
        (".ai/project.yml", AutopilotStepStatus.PROTECTED_PATH_VIOLATION),
        ("acceptance/private.txt", AutopilotStepStatus.PROTECTED_PATH_VIOLATION),
        ("test_generated.py", AutopilotStepStatus.PROTECTED_PATH_VIOLATION),
        ("tests/test_generated.py", AutopilotStepStatus.PROTECTED_PATH_VIOLATION),
        ("../escape.txt", AutopilotStepStatus.INVALID_PROPOSAL),
    ],
)
def test_rejected_proposal_paths_write_nothing(
    tmp_path: Path,
    path: str,
    expected_status: AutopilotStepStatus,
) -> None:
    root, source = project(tmp_path)
    adapter = RecordingAdapter(proposal((path, "forbidden\n"), ("app.py", "changed\n")))

    result = runner(tmp_path).run(root, adapter)

    assert result.status is expected_status
    assert result.applied is False
    assert source.read_text(encoding="utf-8") == "old\n"
    assert (root / ".ai/goal.md").read_text(encoding="utf-8") == "Deliver the tested implementation.\n"
    assert not (tmp_path / "history").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_existing_verification_contract_cannot_be_modified(tmp_path: Path) -> None:
    root, source = project(tmp_path)
    contract = root / "tests" / "test_existing.py"
    contract.parent.mkdir()
    contract.write_text("def test_contract():\n    assert True\n", encoding="utf-8")

    result = runner(tmp_path).run(
        root,
        RecordingAdapter(proposal(
            ("tests/test_existing.py", "def test_contract():\n    assert False\n"),
            ("app.py", "changed\n"),
        )),
    )

    assert result.status is AutopilotStepStatus.PROTECTED_PATH_VIOLATION
    assert "assert True" in contract.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == "old\n"


def test_protected_acceptance_file_is_readable_context_but_not_writable(tmp_path: Path) -> None:
    root, source = project(tmp_path)
    acceptance = root / "acceptance" / "test_cli.py"
    acceptance.parent.mkdir()
    acceptance.write_text("def test_cli():\n    assert True\n", encoding="utf-8")
    adapter = RecordingAdapter(proposal(("app.py", "new\n")))

    result = runner(tmp_path).run(root, adapter)

    assert result.status is AutopilotStepStatus.SUCCESS
    assert "acceptance/test_cli.py" in adapter.calls[0][0]
    assert "assert True" in adapter.calls[0][0]
    assert source.read_text(encoding="utf-8") == "new\n"


def test_not_ready_does_not_call_model(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    adapter = RecordingAdapter("unused")

    result = runner(tmp_path).run(root, adapter)

    assert result.status is AutopilotStepStatus.CONFIGURATION_ERROR
    assert result.applied is False
    assert adapter.calls == []
    assert result.problems


def test_malformed_proposal_writes_nothing(tmp_path: Path) -> None:
    root, source = project(tmp_path)

    result = runner(tmp_path).run(root, RecordingAdapter("not a proposal"))

    assert result.status is AutopilotStepStatus.INVALID_PROPOSAL
    assert result.applied is False
    assert source.read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "history").exists()


def test_model_failure_is_sanitized_and_writes_nothing(tmp_path: Path) -> None:
    root, source = project(tmp_path)

    result = runner(tmp_path).run(root, FailingAdapter())

    assert result.status is AutopilotStepStatus.MODEL_ERROR
    assert result.message == "The model request failed."
    assert result.applied is False
    assert source.read_text(encoding="utf-8") == "old\n"


def test_verification_failure_restores_complete_transaction_and_keeps_diagnostics(
    tmp_path: Path,
) -> None:
    root, source = project(
        tmp_path,
        verify_code="import sys; print('failed out'); print('failed err', file=sys.stderr); sys.exit(3)",
    )

    memory = root / ".ai/memory.md"
    next_steps = root / ".ai/next-steps.md"
    result = runner(tmp_path).run(root, RecordingAdapter(proposal(
        ("app.py", "changed\n"),
        ("created.py", "new file\n"),
        (".ai/memory.md", "Attempt claimed a discovery.\n"),
        (".ai/next-steps.md", "Incorrectly marked complete.\n"),
    )))

    assert result.status is AutopilotStepStatus.VERIFICATION_FAILED
    assert result.applied is True
    assert source.read_text(encoding="utf-8") == "old\n"
    assert not (root / "created.py").exists()
    assert memory.read_text(encoding="utf-8") == "Initial memory.\n"
    assert next_steps.read_text(encoding="utf-8") == "Implement the current feature.\n"
    assert result.snapshot_path is not None and result.snapshot_path.exists()
    assert result.rollback_attempted is True
    assert result.rollback_succeeded is True
    assert result.changes_kept is False
    assert result.verification is not None
    assert result.verification.commands[0].stdout == "failed out\n"
    assert result.verification.commands[0].stderr == "failed err\n"


def test_verification_timeout_rolls_back_applied_files(tmp_path: Path) -> None:
    root, source = project(
        tmp_path,
        verify_code="import time; time.sleep(1)",
        timeout=0.05,
    )

    result = runner(tmp_path).run(root, RecordingAdapter(proposal(("app.py", "changed\n"))))

    assert result.status is AutopilotStepStatus.VERIFICATION_TIMEOUT
    assert result.applied is True
    assert source.read_text(encoding="utf-8") == "old\n"
    assert result.verification is not None
    assert result.verification.status is VerificationStatus.TIMEOUT
    assert result.rollback_succeeded is True
    assert result.changes_kept is False


def test_rollback_failure_reports_uncertain_state_safely(tmp_path: Path) -> None:
    root, source = project(
        tmp_path,
        verify_code="import sys; sys.exit(1)",
    )

    class FailingRestoreWorkspace(WorkspaceService):
        def restore(self, snapshot: Path) -> Path:
            del snapshot
            raise OSError("secret-bearing low-level detail")

    workspace = FailingRestoreWorkspace(HistoryService(tmp_path / "history"))
    result = AutopilotRunner(workspace=workspace).run(
        root,
        RecordingAdapter(proposal(("app.py", "changed\n"))),
    )

    assert result.status is AutopilotStepStatus.ROLLBACK_ERROR
    assert result.rollback_attempted is True
    assert result.rollback_succeeded is False
    assert result.changes_kept is None
    assert "secret-bearing" not in (result.message or "")
    assert "filesystem state is uncertain" in (result.message or "")
    assert source.read_text(encoding="utf-8") == "changed\n"
    assert result.snapshot_path is not None and result.snapshot_path.exists()


@pytest.mark.parametrize(
    ("managed_path", "replacement"),
    [
        (".ai/memory.md", "Durable discovery.\n"),
        (".ai/next-steps.md", "Perform the following step.\n"),
    ],
)
def test_mutable_managed_state_may_be_updated(
    tmp_path: Path,
    managed_path: str,
    replacement: str,
) -> None:
    root, _source = project(tmp_path)

    result = runner(tmp_path).run(
        root,
        RecordingAdapter(proposal((managed_path, replacement))),
    )

    assert result.status is AutopilotStepStatus.SUCCESS
    assert result.applied is True
    assert (root / managed_path).read_text(encoding="utf-8") == replacement
