import sys
from pathlib import Path

import yaml

from ai_workbench.autopilot_loop import (
    AutopilotLoopRunner,
    AutopilotRunStatus,
    AutopilotStopReason,
)
from ai_workbench.autopilot_runner import AutopilotStepResult, AutopilotStepStatus
from ai_workbench.history import HistoryService
from ai_workbench.models import ModelAdapter
from ai_workbench.workspace import WorkspaceService


class SequenceAdapter(ModelAdapter):
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, context: str, prompt: str) -> str:
        self.calls.append((context, prompt))
        return self.responses.pop(0)


def proposal(*files: tuple[str, str]) -> str:
    return "\n".join(
        f"<<<FILE {path}>>>\n{content}<<<END FILE>>>" for path, content in files
    )


def project(
    tmp_path: Path,
    *,
    verify_code: str = "print('ok')",
    max_steps: int = 8,
    max_failures: int = 3,
) -> Path:
    root = tmp_path / "project"
    control = root / ".ai"
    control.mkdir(parents=True)
    (control / "goal.md").write_text("Deliver the goal.\n", encoding="utf-8")
    (control / "context.md").write_text("Bounded project context.\n", encoding="utf-8")
    (control / "memory.md").write_text("Durable facts.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("First logical step.\n", encoding="utf-8")
    (control / "project.yml").write_text(yaml.safe_dump({
        "verify": {
            "timeout_seconds": 2,
            "commands": [[sys.executable, "-c", verify_code]],
        },
        "autopilot": {
            "max_steps": max_steps,
            "max_consecutive_failures": max_failures,
            "protected_paths": [".ai/goal.md"],
            "verification_contract_paths": ["test_*.py", "tests/**"],
            "context_paths": ["app.py"],
        },
    }), encoding="utf-8")
    (root / "app.py").write_text("old\n", encoding="utf-8")
    return root


def loop(tmp_path: Path) -> AutopilotLoopRunner:
    from ai_workbench.autopilot_runner import AutopilotRunner

    step_runner = AutopilotRunner(
        workspace=WorkspaceService(HistoryService(tmp_path / "history"))
    )
    return AutopilotLoopRunner(step_runner)


def test_two_passing_steps_complete_when_next_step_is_empty(tmp_path: Path) -> None:
    root = project(tmp_path)
    adapter = SequenceAdapter(
        proposal(("app.py", "one\n"), (".ai/next-steps.md", "Second logical step.\n")),
        proposal(("app.py", "two\n"), (".ai/next-steps.md", "# Complete\n")),
    )

    result = loop(tmp_path).run(root, adapter)

    assert result.status is AutopilotRunStatus.COMPLETED
    assert result.stop_reason is AutopilotStopReason.COMPLETED
    assert result.total_attempts == 2
    assert result.successful_steps == 2
    assert (root / "app.py").read_text(encoding="utf-8") == "two\n"


def test_file_created_by_first_step_is_context_for_second_attempt(tmp_path: Path) -> None:
    root = project(tmp_path)
    configuration_path = root / ".ai/project.yml"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    configuration["autopilot"]["context_paths"] = ["future.py"]
    configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    adapter = SequenceAdapter(
        proposal(
            ("future.py", "created = True\n"),
            (".ai/next-steps.md", "Use the created implementation.\n"),
        ),
        proposal((".ai/next-steps.md", "DONE Project complete.\n")),
    )

    result = loop(tmp_path).run(root, adapter)

    assert result.status is AutopilotRunStatus.COMPLETED
    assert "future.py" not in adapter.calls[0][0]
    assert "future.py" in adapter.calls[1][0]
    assert "created = True" in adapter.calls[1][0]


def test_failure_retries_same_step_with_transient_diagnostics_then_passes(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        verify_code="from pathlib import Path; assert Path('app.py').read_text() == 'good\\n'",
    )
    adapter = SequenceAdapter(
        proposal(("app.py", "bad\n"), (".ai/next-steps.md", "Wrongly advanced.\n")),
        proposal(("app.py", "good\n"), (".ai/next-steps.md", "# Complete\n")),
    )

    events = []
    result = loop(tmp_path).run(root, adapter, on_event=events.append)

    assert result.status is AutopilotRunStatus.COMPLETED
    assert result.total_attempts == 2
    assert "PREVIOUS AUTOPILOT ATTEMPT FAILED" in adapter.calls[1][1]
    assert "First logical step." in adapter.calls[1][1]
    assert "Wrongly advanced" not in adapter.calls[1][1]
    assert "PREVIOUS AUTOPILOT" not in (root / ".ai/memory.md").read_text(encoding="utf-8")
    starts = [event for event in events if event.kind == "attempt_started"]
    assert starts[0].has_retry_context is False
    assert starts[1].has_retry_context is True


def test_pass_after_failure_resets_failure_count_and_advances(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        verify_code=(
            "from pathlib import Path; "
            "assert Path('app.py').read_text() not in {'bad\\n', 'bad-again\\n'}"
        ),
        max_failures=2,
    )
    adapter = SequenceAdapter(
        proposal(("app.py", "bad\n")),
        proposal(("app.py", "first-pass\n"), (".ai/next-steps.md", "Second logical step.\n")),
        proposal(("app.py", "bad-again\n")),
        proposal(("app.py", "second-pass\n"), (".ai/next-steps.md", "# Complete\n")),
    )

    result = loop(tmp_path).run(root, adapter)

    assert result.status is AutopilotRunStatus.COMPLETED
    assert result.total_attempts == 4
    assert result.successful_steps == 2
    assert result.consecutive_failures == 0
    assert "Second logical step." in adapter.calls[2][1]
    assert "PREVIOUS AUTOPILOT ATTEMPT FAILED" not in adapter.calls[2][1]


def test_repeated_failures_stop_at_configured_limit(tmp_path: Path) -> None:
    root = project(tmp_path, verify_code="raise SystemExit(7)", max_failures=2)
    adapter = SequenceAdapter(*(proposal(("app.py", f"bad {index}\n")) for index in range(3)))

    result = loop(tmp_path).run(root, adapter)

    assert result.stop_reason is AutopilotStopReason.MAX_CONSECUTIVE_FAILURES
    assert result.total_attempts == 2
    assert result.consecutive_failures == 2
    assert len(adapter.calls) == 2


def test_max_steps_counts_all_attempts(tmp_path: Path) -> None:
    root = project(tmp_path, verify_code="raise SystemExit(1)", max_steps=2, max_failures=5)
    adapter = SequenceAdapter(*(proposal(("app.py", "bad\n")) for _ in range(3)))

    result = loop(tmp_path).run(root, adapter)

    assert result.stop_reason is AutopilotStopReason.MAX_STEPS
    assert result.total_attempts == 2
    assert len(adapter.calls) == 2


def test_pause_at_boundary_and_resume_preserves_run_counts(tmp_path: Path) -> None:
    root = project(tmp_path)
    adapter = SequenceAdapter(
        proposal((".ai/next-steps.md", "Second logical step.\n")),
        proposal((".ai/next-steps.md", "# Complete\n")),
    )
    runner = loop(tmp_path)

    def pause_after_first(event) -> None:
        if event.kind == "attempt_finished":
            runner.request_pause()

    paused = runner.run(root, adapter, on_event=pause_after_first)
    resumed = runner.resume(root, adapter)

    assert paused.status is AutopilotRunStatus.PAUSED
    assert paused.total_attempts == 1
    assert resumed.status is AutopilotRunStatus.COMPLETED
    assert resumed.total_attempts == 2


def test_stop_at_boundary_makes_no_additional_request(tmp_path: Path) -> None:
    root = project(tmp_path)
    adapter = SequenceAdapter(
        proposal((".ai/next-steps.md", "Second logical step.\n")),
        proposal((".ai/next-steps.md", "# Complete\n")),
    )
    runner = loop(tmp_path)

    def stop_after_first(event) -> None:
        if event.kind == "attempt_finished":
            runner.request_stop()

    result = runner.run(root, adapter, on_event=stop_after_first)

    assert result.status is AutopilotRunStatus.STOPPED
    assert result.stop_reason is AutopilotStopReason.USER_STOPPED
    assert len(adapter.calls) == 1


def test_completed_next_step_makes_no_model_request(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / ".ai/next-steps.md").write_text("DONE Project complete.\n", encoding="utf-8")
    adapter = SequenceAdapter("must not be used")

    result = loop(tmp_path).run(root, adapter)

    assert result.stop_reason is AutopilotStopReason.COMPLETED
    assert result.status is AutopilotRunStatus.COMPLETED
    assert result.total_attempts == 0
    assert adapter.calls == []


def test_rollback_error_is_immediately_terminal(tmp_path: Path) -> None:
    root = project(tmp_path)

    class RollbackErrorRunner:
        calls = 0

        def run(self, root, adapter, selected_files, retry_context=None, on_progress=None):
            del root, adapter, selected_files, retry_context, on_progress
            self.calls += 1
            return AutopilotStepResult(AutopilotStepStatus.ROLLBACK_ERROR)

    step_runner = RollbackErrorRunner()
    result = AutopilotLoopRunner(step_runner).run(root, SequenceAdapter("unused"))

    assert result.stop_reason is AutopilotStopReason.ROLLBACK_ERROR
    assert result.total_attempts == 1
    assert step_runner.calls == 1
